#!/usr/bin/env python3
"""Preserve the local ASVspoof5 engines; train on 2019 LA and test on both 2019/2021 LA."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SPECS = [
    ('wavlm_base_lt', '06_wavlm_base_lt_aug', 'AST_ASVspoof5.conf', 'AST', 'microsoft/wavlm-base', 4),
    ('wavlm_large_lt', '07_wavlm_large_lt_aug', 'AST_WavLM_Large_LT_ASVspoof5.conf', 'AST', 'microsoft/wavlm-large', 4),
    ('wavlm_base_light_mamba_2l_k7_e2', '17_wavlm_large_light_mamba_2l_k7_e2',
     'AST_LightMamba2L_K7_WavLM_Large_ASVspoof5.conf', 'AST_LightMambaConfig', 'microsoft/wavlm-base', 2),
    ('wavlm_large_light_mamba_2l_k7_e2', '17_wavlm_large_light_mamba_2l_k7_e2',
     'AST_LightMamba2L_K7_WavLM_Large_ASVspoof5.conf', 'AST_LightMambaConfig', 'microsoft/wavlm-large', 2),
]
EXPECTED = {'train': 25380, 'dev': 24844, 'eval2019': 71237, 'eval2021': 148176}

def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def read_rows(path, year):
    rows = []
    excluded = Counter()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        p = line.split()
        if not p:
            continue
        if year == 2019:
            if len(p) != 5:
                raise ValueError(f'{path}:{line_no}: expected 5 columns')
            speaker, utterance, _, attack, label = p
            codec = condition = '-'
        else:
            if len(p) < 8:
                raise ValueError(f'{path}:{line_no}: expected at least 8 columns')
            speaker, utterance, codec, condition, attack, label, trim, subset = p[:8]
            if subset != 'eval':
                excluded[subset] += 1
                continue
        if label not in {'bonafide', 'spoof'}:
            raise ValueError(f'Invalid label at {path}:{line_no}')
        if label == 'bonafide':
            attack = 'bonafide'
        # Original engine's nine-column adapter, preserving speaker/attack/codec/condition/label.
        rows.append([speaker, utterance, '-', codec, '-', '-', condition, attack, label])
    return rows, dict(excluded)

def prepare(args):
    root19 = args.data2019.resolve()
    root21 = args.data2021.resolve()
    if (root19 / 'LA').is_dir():
        root19 = root19 / 'LA'
    protocols = root19 / 'ASVspoof2019_LA_cm_protocols'
    inputs = {
        'train': (protocols/'ASVspoof2019.LA.cm.train.trn.txt', 2019,
                  root19/'ASVspoof2019_LA_train/flac'),
        'dev': (protocols/'ASVspoof2019.LA.cm.dev.trl.txt', 2019,
                root19/'ASVspoof2019_LA_dev/flac'),
        'eval2019': (protocols/'ASVspoof2019.LA.cm.eval.trl.txt', 2019,
                     root19/'ASVspoof2019_LA_eval/flac'),
        'eval2021': (root21/'keys/LA/CM/trial_metadata.txt', 2021,
                 root21/'ASVspoof2021_LA_eval/flac'),
    }
    all_rows = {}
    split_manifest = {}
    for split, (protocol, year, audio_dir) in inputs.items():
        rows, excluded = read_rows(protocol, year)
        ids = [r[1] for r in rows]
        if len(rows) != EXPECTED[split] or len(set(ids)) != len(ids):
            raise ValueError(f'{split}: unexpected count or duplicate utterances: {len(rows)}')
        for utt in ids:
            audio = audio_dir / (utt + '.flac')
            if not audio.is_file() or audio.stat().st_size == 0:
                raise FileNotFoundError(f'Missing or empty audio: {audio}')
        all_rows[split] = rows
        split_manifest[split] = {
            'dataset_year': year, 'items': len(rows),
            'labels': dict(Counter(r[8] for r in rows)),
            'protocol': str(protocol), 'protocol_sha256': digest(protocol),
            'audio_directory': str(audio_dir), 'excluded_subsets': excluded,
        }
        print(f'{split}: {len(rows)} files checked; {split_manifest[split]["labels"]}', flush=True)
    for left, right in [('train', 'dev'), ('train', 'eval2019'), ('dev', 'eval2019'),
                        ('train', 'eval2021'), ('dev', 'eval2021')]:
        if {r[1] for r in all_rows[left]} & {r[1] for r in all_rows[right]}:
            raise ValueError(f'Utterance overlap: {left}/{right}')

    # Validate all four local source configurations before creating a run.
    configurations = []
    for name, folder, conf_name, architecture, frontend, layers in SPECS:
        source = args.source.resolve() / folder / 'code'
        conf = source / 'config' / conf_name
        cfg = json.loads(conf.read_text())
        mc = cfg['model_config']
        if name == 'wavlm_base_light_mamba_2l_k7_e2':
            if mc['pretrained_name'] != 'microsoft/wavlm-large' or mc['freeze_layers'] != 18:
                raise ValueError('Unexpected source Light Mamba frontend/freezing settings')
            mc['pretrained_name'] = 'microsoft/wavlm-base'
            mc['freeze_layers'] = 8
        if (mc['architecture'], mc['pretrained_name'], mc['lt_layers']) != (architecture, frontend, layers):
            raise ValueError(f'{conf}: model no longer matches the requested experiment')
        if architecture == 'AST_LightMambaConfig' and (mc['light_mamba_kernel'], mc['light_mamba_expansion']) != (7, 2):
            raise ValueError('Expected Light Mamba kernel 7 and expansion 2')
        for key, expected in [('track_eval_during_training', 'False'), ('allow_eval_tracking_leak', 'False'),
                              ('run_final_eval_after_training', 'True'), ('prefer_track1_tsv', 'True')]:
            if str(cfg.get(key)).lower() != expected.lower():
                raise ValueError(f'{conf}: {key} must be {expected}')
        if str(mc.get('use_attack_head')).lower() != 'false':
            raise ValueError('This adapter expects the original disabled auxiliary attack head')
        configurations.append((name, source, conf, cfg))

    destination = args.destination
    if destination is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        destination = Path.home() / 'projects/asvspoof2019-2021-la/runs' / stamp
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    adapters = {}
    for year in (2019, 2021):
        data = destination/f'data_adapter_{year}'
        data.mkdir()
        adapters[year] = data
        for split, source_split in [('train', 'train'), ('dev', 'dev'), ('eval', f'eval{year}')]:
            audio_dir = inputs[source_split][2]
            tsv = data/('ASVspoof5.train.tsv' if split == 'train' else f'ASVspoof5.{split}.track_1.tsv')
            tsv.write_text(''.join('\t'.join(row) + '\n' for row in all_rows[source_split]))
            split_manifest[source_split]['adapter_sha256'] = digest(tsv)
            (data / {'train': 'flac_T', 'dev': 'flac_D', 'eval': 'flac_E'}[split]).symlink_to(audio_dir, target_is_directory=True)
    data = adapters[2019]
    save_json(destination/'dataset_manifest.json', split_manifest)
    run_specs = []
    for name, source, conf, cfg in configurations:
        target = destination/'experiments'/name
        target.mkdir(parents=True)
        code = target/'code'
        shutil.copytree(source, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'exp_result'))
        shutil.copy2(conf, target/'source_config.conf')
        # Preserve originals; record dataset changes and the new base Light Mamba variant.
        cfg['database_path'] = str(data)
        cfg['model_path'] = None
        cfg['save_tsne'] = 'True'
        cfg['save_epoch_figures'] = 'True'
        save_json(target/'experiment.conf', cfg)
        eval_cfg = json.loads(json.dumps(cfg))
        eval_cfg['database_path'] = str(adapters[2021])
        save_json(target/'evaluation_2021.conf', eval_cfg)
        (target/'evaluate_selected.py').write_text(EVAL_HELPER)
        copied = {str(p.relative_to(code)): digest(p) for p in sorted(code.rglob('*')) if p.is_file()}
        for relative, sha in copied.items():
            if digest(source/relative) != sha:
                raise RuntimeError(f'Source copy mismatch: {relative}')
        original_cfg = json.loads(conf.read_text())
        changes = {key: [original_cfg.get(key), value] for key, value in cfg.items()
                   if value != original_cfg.get(key)}
        save_json(target/'source_manifest.json', {'source_code': str(source), 'source_config': str(conf),
                  'source_config_sha256': digest(conf), 'code_sha256': copied,
                  'config_changes': changes})
        run_specs.append({'name': name, 'directory': str(target), 'config': str(target/'experiment.conf'),
                          'eval2021_config': str(target/'evaluation_2021.conf')})
    save_json(destination/'experiments.json', run_specs)
    shutil.copy2(Path(__file__).resolve(), destination/'prepare_and_run.py')
    (destination/'README.md').write_text('''# ASVspoof 2019 LA training → ASVspoof 2021 LA evaluation

Training: all 25,380 official 2019 LA training trials.
Development: all 24,844 official 2019 LA development trials.
Test 2019: all 71,237 official 2019 LA evaluation trials.
Test 2021: all 148,176 trials marked `eval` in the 2021 LA metadata.
2021 progress and hidden subsets are excluded. There is no separate 2021 development set.

The copied ASVspoof 5 training engines and model implementations are unchanged.
The ASVspoof5 names in data_adapter are compatibility filenames, not the data source.
Audio symlinks point to the local 2019/2021 datasets; files are not duplicated.
The original three configs change only database_path and obsolete evaluation-only model_path.
The new base Light Mamba config derives from large Light Mamba, changing pretrained_name
to microsoft/wavlm-base and freeze_layers from 18 to 8 (matching the previous base experiment).
Each model starts from its Hugging Face pretrained WavLM frontend and a fresh backend.
No previous ASVspoof checkpoint is loaded for training.
Best checkpoint, early stopping, calibration and decision threshold use 2019 development data.
2019 evaluation follows training. The same best checkpoint then runs through the original
evaluation-only entry point on 2021, recalculating calibration/thresholds using 2019 development only.
The evaluation helper only records the returned metrics; it does not change scores or selection.
Both tests save t-SNE PNG/CSV, confusion-matrix PNG/JSON, scores, predictions and metrics.
The original tsne_max_samples setting is retained (0 means every trial).
Outputs: experiments/<model>/results_2019 and results_2021.
Each selected checkpoint SHA256 is recorded, and verified unchanged after 2021 evaluation.
The original balanced sampler samples with replacement, so individual epochs need not visit every training trial.

Seed: 1234. Maximum epochs: 20 each. Batch size: 8, accumulation: 4.
Early-stopping patience: 8 for the two Transformer models, 2 for Light Mamba.
The four experiments run sequentially in separate processes.
Every new launcher invocation creates a fresh run directory; it does not resume interrupted training.
Logs, source hashes, original/adapted configs and environment are saved here.
The engine's DCF output is its existing metric, not automatically the official 2021 min t-DCF.
''')
    print(f'Prepared: {destination}', flush=True)
    return destination, run_specs

EVAL_HELPER = '"""Evaluate a selected checkpoint using the unchanged source engine and record metrics."""\nimport argparse\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent/\'code\'))\nimport main as engine\n\ndef evaluate_selected(config, checkpoint, output_dir):\n    original = engine.evaluate_split\n    def recorded(*args, **kwargs):\n        result = original(*args, **kwargs)\n        engine.save_json(Path(kwargs[\'output_dir\'])/(kwargs[\'split_name\']+\'_metrics.json\'),\n                         {key: value for key, value in result.items() if key != \'rows\'})\n        return result\n    engine.evaluate_split = recorded\n    try:\n        engine.main(argparse.Namespace(config=config, output_dir=output_dir, seed=1234,\n                    eval=True, eval_minimal=False, comment=None, eval_model_weights=checkpoint))\n    finally:\n        engine.evaluate_split = original\n\nif __name__ == \'__main__\':\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\'--config\', required=True)\n    parser.add_argument(\'--checkpoint\', required=True)\n    parser.add_argument(\'--output-dir\', required=True)\n    args = parser.parse_args()\n    evaluate_selected(args.config, args.checkpoint, args.output_dir)\n'

def run_logged(command, cwd, log_path):
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f'Command failed (exit {returncode}). See {log_path}')

def verify_figures(run_dir, evaluation_folder, tag):
    for split in ('dev', 'eval'):
        for path in [run_dir/evaluation_folder/split/f'{split}_confusion_matrix.png',
                     run_dir/evaluation_folder/split/f'{split}_confusion_matrix.json',
                     run_dir/f'tsne_{tag}_{split}'/f'{split}_tsne.png',
                     run_dir/f'tsne_{tag}_{split}'/f'{split}_tsne_points.csv']:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Required output missing: {path}')

def execute(destination, run_specs):
    # Fail before lengthy training if the requested environment cannot use CUDA.
    subprocess.run([sys.executable, '-c',
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in this environment'; "
        "print('GPU:', torch.cuda.get_device_name(0)); "
        "x=torch.ones(2,device='cuda'); print('GPU check:', (x+x).tolist())"], check=True)
    with (destination/'environment.txt').open('w') as handle:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
    with (destination/'gpu.txt').open('w') as handle:
        subprocess.run(['nvidia-smi'], stdout=handle, check=True)
    for spec in run_specs:
        target = Path(spec['directory'])
        # Import the exact engine and selected model before starting the first experiment.
        mc = json.loads(Path(spec['config']).read_text())['model_config']
        subprocess.run([sys.executable, '-c',
            f"import main; import importlib; importlib.import_module('models.{mc['architecture']}'); print('Imports OK')"],
            cwd=target/'code', check=True)
    for i, spec in enumerate(run_specs, 1):
        target = Path(spec['directory'])
        command = [sys.executable, '-u', 'main.py', '--config', spec['config'],
                   '--output_dir', str(target/'results_2019'), '--seed', '1234']
        save_json(target/'command_train_2019.json', command)
        print(f'\nStarting experiment {i}/{len(run_specs)}: {spec["name"]} — training and 2019 evaluation', flush=True)
        run_logged(command, target/'code', target/'console_train_2019.log')
        summaries = list((target/'results_2019').glob('*/artifacts/final_summary.json'))
        if len(summaries) != 1 or not json.loads(summaries[0].read_text()).get('final_eval_metrics'):
            raise RuntimeError(f'{spec["name"]}: no final 2019 evaluation summary was produced')
        trained_run = summaries[0].parent.parent
        checkpoint = trained_run/'weights/best_ast.pth'
        checkpoint_sha = digest(checkpoint)
        verify_figures(trained_run, 'final_best_model_eval', 'final')
        command21 = [sys.executable, '-u', str(target/'evaluate_selected.py'),
                     '--config', spec['eval2021_config'], '--checkpoint', str(checkpoint),
                     '--output-dir', str(target/'results_2021')]
        save_json(target/'command_eval_2021.json', command21)
        print(f'\nTesting the same selected checkpoint on 2021 LA: {spec["name"]}', flush=True)
        run_logged(command21, target/'code', target/'console_eval_2021.log')
        metrics21 = list((target/'results_2021').glob('*/eval_loaded_model/eval/eval_metrics.json'))
        if len(metrics21) != 1:
            raise RuntimeError(f'{spec["name"]}: no 2021 evaluation metrics were produced')
        evaluation_run = metrics21[0].parent.parent.parent
        verify_figures(evaluation_run, 'eval_loaded_model', 'eval_only')
        if digest(checkpoint) != checkpoint_sha:
            raise RuntimeError('Selected checkpoint changed during evaluation')
        save_json(target/'completed.json', {'final_summary_2019': str(summaries[0]),
                  'metrics_2021': str(metrics21[0]), 'selected_checkpoint': str(checkpoint),
                  'checkpoint_sha256': checkpoint_sha})
    comparison = {}
    for spec in run_specs:
        target = Path(spec['directory'])
        completed = json.loads((target/'completed.json').read_text())
        comparison[spec['name']] = {
            'training_and_2019': json.loads(Path(completed['final_summary_2019']).read_text()),
            'evaluation_2021': json.loads(Path(completed['metrics_2021']).read_text()),
            'checkpoint_sha256': completed['checkpoint_sha256'],
        }
    save_json(destination/'comparison.json', comparison)
    print(f'\nAll four experiments completed. Results: {destination}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.home()/'projects/lightmamba-asvspoof5')
    parser.add_argument('--data2019', type=Path, default=Path.home()/'datasets/ASVspoof2019')
    parser.add_argument('--data2021', type=Path, default=Path.home()/'datasets/ASVspoof2021')
    parser.add_argument('--destination', type=Path, help='New, non-existing run directory')
    parser.add_argument('--run', action='store_true', help='Automatically train/evaluate all four after preparation')
    args = parser.parse_args()
    destination, specs = prepare(args)
    if args.run:
        execute(destination, specs)
    else:
        print('Preparation only; no training started.', flush=True)

if __name__ == '__main__':
    main()
