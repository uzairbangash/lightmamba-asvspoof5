State Space Modeling with Light Mamba for Efficient Deepfake Speech Detection

This repository contains experiment code for ASVspoof5 deepfake speech
detection systems based on self-supervised speech representations and compact
sequence backends, including Light Transformer, Transformer, official Mamba,
Light Mamba, and fusion variants.

Repository Structure

Each numbered folder corresponds to one experiment.

01_rawnet2/
02_aasist/
03_wavlm_base_cnn_lt_no_aug/
04_wavlm_base_cnn_lt_aug/
05_wavlm_base_lt_no_aug/
06_wavlm_base_lt_aug/
07_wavlm_large_lt_aug/
08_wavlm_base_transformer_aug/
09_wavlm_large_transformer_aug/
10_wavlm_large_cnn_lt_aug/
11_wavlm_large_lt_conditioned_pool/
12_wavlm_base_official_mamba_no_aug/
13_wavlm_base_official_mamba_aug/
14_wavlm_large_official_mamba_aug/
15_wavlm_large_light_mamba_4l_k7_e2/
16_wavlm_large_multikernel_bilight_mamba/
17_wavlm_large_light_mamba_2l_k7_e2/
18_wavlm_large_ultralight_mamba_2l_k5_e1/
19_wavlm_large_light_mamba_3l_k7_e2/
20_wavlm_large_light_mamba_2l_k5_e2/
21_wavlm_large_bilight_mamba_2l_k7_e2_gated/
22_wavlm_large_dualscale_light_mamba_fusion/
23_wavlm_large_cnn_lt_gated_fusion/
24_xlsr_lt/
25_xlsr_light_mamba_2l_k7_e2/
26_hubert_lt/
27_hubert_light_mamba_2l_k7_e2/
28_whisper_lt/
29_whisper_light_mamba_2l_k7_e2/

Inside each experiment folder:

code/
  Source code, configuration files, model files, and utility scripts.

README.txt
  Minimal run command for that experiment.

Results, trained checkpoints, embeddings, t-SNE files, confusion matrices, and
dataset audio are not included in this GitHub code package.

Software Requirements

Recommended environment:

Python 3.10
CUDA-enabled PyTorch environment
NVIDIA GPU for training and evaluation

Core Python packages used by the experiment code:

torch
torchaudio
numpy
scikit-learn
librosa
soundfile
matplotlib
tqdm
tensorboard
tensorboardX==2.5
transformers>=4.41.0
torchcontrib

Additional package for official Mamba experiments:

mamba-ssm

The lightweight Mamba variants implemented in this repository do not require
the official mamba-ssm package unless the selected experiment uses the official
mamba-ssm backend.

Example Environment Setup

conda create -n asv5 python=3.10
conda activate asv5

Install PyTorch and torchaudio with the CUDA build appropriate for the target
machine. For example, on a CUDA 11.8 environment:

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

Install the remaining packages:

pip install numpy scikit-learn librosa soundfile matplotlib tqdm tensorboard tensorboardX==2.5 transformers torchcontrib

For official Mamba experiments only:

pip install mamba-ssm

Dataset Setup

Download and prepare the ASVspoof5 dataset according to the official challenge
instructions. The experiment configs expect the ASVspoof5 protocol files and
audio folders to be available through the database_path field inside each
configuration file.

Before running an experiment, open the relevant config file in:

<experiment_folder>/code/config/

and set database_path to the local ASVspoof5 dataset location.

Running an Experiment

Open the README.txt inside the desired experiment folder and run the command
shown there.

General command pattern for AST-based experiments:

cd <experiment_folder>/code
python3.10 main.py \
  --config <config_file> \
  --output_dir <output_directory> \
  --comment <run_name>

Example:

cd 17_wavlm_large_light_mamba_2l_k7_e2/code
python3.10 main.py \
  --config config/AST_LightMamba2L_K7_WavLM_Large_ASVspoof5.conf \
  --output_dir ./exp_result \
  --comment wavlm_large_light_mamba_2l_k7_e2

Baseline folders may use their own entry scripts:

01_rawnet2 uses Main_RawNet2_baseline.py
02_aasist uses main.py with config/AASIST_ASVspoof5.conf

Experiment Outputs

By default, outputs are written under the directory passed to --output_dir.
Typical outputs include metric logs, checkpoints, best-metric files, final
summary files, score files, and optional visualization artifacts depending on
the selected configuration.

Reproducibility Notes

Use the configuration file supplied inside each experiment's code/config/
directory. The reported experiments use the settings in those configuration
files, including model architecture, SSL frontend, augmentation mode, training
schedule, and evaluation behavior.

When comparing results, use the best checkpoint selected by development-set
EER and then perform the final evaluation with the corresponding saved model.
