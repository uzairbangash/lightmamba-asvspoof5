# # """
# # Main script that trains, validates, and evaluates
# # various models including AASIST.

# # AASIST
# # Copyright (c) 2021-present NAVER Corp.
# # MIT license
# # """
# # import argparse
# # import json
# # import os
# # import sys
# # import warnings
# # from importlib import import_module
# # from pathlib import Path
# # from shutil import copy
# # from typing import Dict, List, Union

# # import torch
# # import torch.nn as nn
# # import numpy as np
# # from torch.utils.data import DataLoader
# # from torch.utils.tensorboard import SummaryWriter
# # from torchcontrib.optim import SWA

# # from data_utils import (TrainDataset,TestDataset, genSpoof_list)
# # from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
# # from utils import create_optimizer, seed_worker, set_seed, str_to_bool

# # warnings.filterwarnings("ignore", category=FutureWarning)
# # from tqdm import tqdm
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt
# # import csv

# # def main(args: argparse.Namespace) -> None:
# #     """
# #     Main function.
# #     Trains, validates, and evaluates the ASVspoof detection model.
# #     """
# #     # load experiment configurations
# #     with open(args.config, "r") as f_json:
# #         config = json.loads(f_json.read())
# #     model_config = config["model_config"]
# #     optim_config = config["optim_config"]
# #     optim_config["epochs"] = config["num_epochs"]
# #     if "eval_all_best" not in config:
# #         config["eval_all_best"] = "True"
# #     if "freq_aug" not in config:
# #         config["freq_aug"] = "False"

# #     # make experiment reproducible
# #     set_seed(args.seed, config)

# #     # define database related paths
# #     output_dir = Path(args.output_dir)
# #     database_path = Path(config["database_path"])
# #     dev_trial_path = (database_path /
# #                       "ASVspoof5.dev.metainfor.txt")
# #     # define model related paths
# #     model_tag = "{}_ep{}_bs{}".format(
# #         os.path.splitext(os.path.basename(args.config))[0],
# #         config["num_epochs"], config["batch_size"])
# #     if args.comment:
# #         model_tag = model_tag + "_{}".format(args.comment)
# #     model_tag = output_dir / model_tag
# #     model_save_path = model_tag / "weights"
# #     eval_score_path = model_tag / config["eval_output"]
# #     writer = SummaryWriter(model_tag)
# #     os.makedirs(model_save_path, exist_ok=True)
# #     copy(args.config, model_tag / "config.conf")

# #     # set device
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print("Device: {}".format(device))
# #     if device == "cpu":
# #         raise ValueError("GPU not detected!")

# #     # define model architecture
# #     model = get_model(model_config, device)

# #     # define dataloaders
# #     trn_loader, dev_loader = get_loader(
# #         database_path, args.seed, config)

# #     # evaluates pretrained model 
# #     # NOTE: Currently it is evaluated on the development set instead of the evaluation set
# #     if args.eval:
# #         model_path = args.eval_model_weights or config["model_path"]
# #         model.load_state_dict(
# #             torch.load(model_path, map_location=device))
# #         print("Model loaded : {}".format(model_path))
# #         print("Start evaluation...")
# #         produce_evaluation_file(dev_loader, model, device,
# #                                 eval_score_path, dev_trial_path)

# #         eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
# #             cm_scores_file=eval_score_path,
# #             output_file=model_tag/"loaded_model_result.txt")
# #         print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))

# #         if str_to_bool(config.get("save_tsne", "True")):
# #             save_embedding_tsne(
# #                 data_loader=dev_loader,
# #                 model=model,
# #                 device=device,
# #                 trial_path=dev_trial_path,
# #                 save_dir=model_tag / "tsne_eval",
# #                 max_samples=int(config.get("tsne_max_samples", 2000)),
# #             )
# #         sys.exit(0)

# #     # get optimizer and scheduler
# #     optim_config["steps_per_epoch"] = len(trn_loader)
# #     optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
# #     optimizer_swa = SWA(optimizer)

# #     best_dev_eer = 100.
# #     best_dev_dcf = 1.
# #     best_dev_cllr = 1.
# #     n_swa_update = 0  # number of snapshots of model to use in SWA
# #     f_log = open(model_tag / "metric_log.txt", "a")
# #     f_log.write("=" * 5 + "\n")

# #     # make directory for metric logging
# #     metric_path = model_tag / "metrics"
# #     os.makedirs(metric_path, exist_ok=True)

# #     # Training
# #     for epoch in range(config["num_epochs"]):
# #         print("training epoch{:03d}".format(epoch))
        
# #         running_loss = train_epoch(trn_loader, model, optimizer, device,
# #                                    scheduler, config)
        
# #         produce_evaluation_file(dev_loader, model, device,
# #                                 metric_path/"dev_score.txt", dev_trial_path)
# #         dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
# #             cm_scores_file=metric_path/"dev_score.txt",
# #             output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
# #             printout=False)
# #         print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
# #             running_loss, dev_eer, dev_dcf, dev_cllr))
# #         writer.add_scalar("loss", running_loss, epoch)
# #         writer.add_scalar("dev_eer", dev_eer, epoch)
# #         writer.add_scalar("dev_dcf", dev_dcf, epoch)
# #         writer.add_scalar("dev_cllr", dev_cllr, epoch)
# #         torch.save(model.state_dict(),
# #                        model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

# #         best_dev_dcf = min(dev_dcf, best_dev_dcf)
# #         best_dev_cllr = min(dev_cllr, best_dev_cllr)
# #         if best_dev_eer >= dev_eer:
# #             print("best model find at epoch", epoch)
# #             best_dev_eer = dev_eer
# #             torch.save(model.state_dict(), model_save_path / "best_ast.pth")

# #             print("Saving epoch {} for swa".format(epoch))
# #             optimizer_swa.update_swa()
# #             n_swa_update += 1
# #         writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
# #         writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
# #         writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

# #     if str_to_bool(config.get("save_tsne", "True")):
# #         best_weight = model_save_path / "best_ast.pth"
# #         if best_weight.exists():
# #             model.load_state_dict(torch.load(best_weight, map_location=device))
# #         save_embedding_tsne(
# #             data_loader=dev_loader,
# #             model=model,
# #             device=device,
# #             trial_path=dev_trial_path,
# #             save_dir=model_tag / "tsne_final",
# #             max_samples=int(config.get("tsne_max_samples", 2000)),
# #         )
    

# # def get_model(model_config: Dict, device: torch.device):
# #     """Define DNN model architecture"""
# #     module = import_module("models.{}".format(model_config["architecture"]))
# #     _model = getattr(module, "Model")
# #     model = _model(model_config).to(device)
# #     nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
# #     print("no. model params:{}".format(nb_params))

# #     return model


# # def get_loader(
# #         database_path: str,
# #         seed: int,
# #         config: dict) -> List[torch.utils.data.DataLoader]:
# #     """Make PyTorch DataLoaders for train / developement"""

# #     trn_database_path = database_path / "flac_T/"
# #     dev_database_path = database_path / "flac_D/"

# #     trn_list_path = (database_path /
# #                      "ASVspoof5.train.metainfor.txt")
# #     dev_trial_path = (database_path /
# #                       "ASVspoof5.dev.metainfor.txt")

# #     d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
# #                                             is_train=True,
# #                                             is_eval=False)
# #     print("no. training files:", len(file_train))

# #     train_set = TrainDataset(list_IDs=file_train,
# #                                            labels=d_label_trn,
# #                                            base_dir=trn_database_path)
# #     gen = torch.Generator()
# #     gen.manual_seed(seed)
# #     trn_loader = DataLoader(train_set,
# #                             batch_size=config["batch_size"],
# #                             shuffle=True,
# #                             drop_last=True,
# #                             pin_memory=True,
# #                             worker_init_fn=seed_worker,
# #                             generator=gen)

# #     _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
# #                                 is_train=False,
# #                                 is_eval=False)
# #     print("no. validation files:", len(file_dev))

# #     dev_set = TestDataset(list_IDs=file_dev[:2000],
# #                                             base_dir=dev_database_path)
# #     dev_loader = DataLoader(dev_set,
# #                             batch_size=config["batch_size"],
# #                             shuffle=False,
# #                             drop_last=False,
# #                             pin_memory=True)

# #     return trn_loader, dev_loader

# # def produce_evaluation_file(
# #     data_loader: DataLoader,
# #     model,
# #     device: torch.device,
# #     save_path: str,
# #     trial_path: str) -> None:
# #     """Perform evaluation and save the score to a file"""
# #     model.eval()
# #     with open(trial_path, "r") as f_trl:
# #         trial_lines = f_trl.readlines()
# #     fname_list = []
# #     score_list = []
# #     for batch_x, utt_id in tqdm(data_loader):
# #         with torch.no_grad():
# #             _, batch_out = model(batch_x)
# #             batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
# #         # add outputs
# #         fname_list.extend(utt_id)
# #         score_list.extend(batch_score.tolist())

# #     #assert len(trial_lines) == len(fname_list) == len(score_list)
# #     with open(save_path, "w") as fh:
# #         for fn, sco, trl in zip(fname_list, score_list, trial_lines):
# #             spk_id, utt_id, _, _, src, key = trl.strip().split(' ')
# #             assert fn == utt_id
# #             fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
# #     print("Scores saved to {}".format(save_path))


# # def _parse_trial_labels(trial_path: str):
# #     labels = {}
# #     with open(trial_path, "r") as f:
# #         for line in f:
# #             spk_id, utt_id, _, _, src, key = line.strip().split(" ")
# #             labels[utt_id] = 1 if key == "bonafide" else 0
# #     return labels


# # def save_embedding_tsne(
# #     data_loader: DataLoader,
# #     model,
# #     device: torch.device,
# #     trial_path: str,
# #     save_dir: Path,
# #     max_samples: int = 2000,
# # ) -> None:
# #     """Extract embeddings, run t-SNE, and save CSV/PNG artifacts."""
# #     del device
# #     save_dir.mkdir(parents=True, exist_ok=True)
# #     label_map = _parse_trial_labels(trial_path)

# #     model.eval()
# #     embeddings = []
# #     scores = []
# #     utt_ids = []

# #     for batch_x, batch_utt_id in tqdm(data_loader, desc="extract_tsne"):
# #         with torch.no_grad():
# #             batch_emb, batch_out = model(batch_x)
# #             batch_score = batch_out[:, 1].detach().cpu().numpy()
# #             batch_emb = batch_emb.detach().cpu().numpy()

# #         embeddings.append(batch_emb)
# #         scores.append(batch_score)
# #         utt_ids.extend(batch_utt_id)
# #         if len(utt_ids) >= max_samples:
# #             break

# #     if not utt_ids:
# #         print("No embeddings extracted for t-SNE.")
# #         return

# #     embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
# #     scores = np.concatenate(scores, axis=0)[:max_samples]
# #     utt_ids = utt_ids[:max_samples]
# #     labels = np.array([label_map.get(u, -1) for u in utt_ids])

# #     perplexity = min(30, max(5, len(utt_ids) - 1))
# #     tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
# #     coords = tsne.fit_transform(embeddings)

# #     csv_path = save_dir / "dev_tsne_points.csv"
# #     with open(csv_path, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["utt_id", "label", "bonafide_score", "tsne_x", "tsne_y"])
# #         for utt_id, label, score, (x, y) in zip(utt_ids, labels, scores, coords):
# #             writer.writerow([utt_id, label, float(score), float(x), float(y)])

# #     png_path = save_dir / "dev_tsne_plot.png"
# #     plt.figure(figsize=(8, 6))
# #     spoof_mask = labels == 0
# #     bona_mask = labels == 1
# #     if spoof_mask.any():
# #         plt.scatter(coords[spoof_mask, 0], coords[spoof_mask, 1], s=10, alpha=0.7, label="spoof")
# #     if bona_mask.any():
# #         plt.scatter(coords[bona_mask, 0], coords[bona_mask, 1], s=10, alpha=0.7, label="bonafide")
# #     plt.title("Development-set AST embeddings (t-SNE)")
# #     plt.xlabel("t-SNE 1")
# #     plt.ylabel("t-SNE 2")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(png_path, dpi=200)
# #     plt.close()
# #     print(f"Saved t-SNE CSV to {csv_path}")
# #     print(f"Saved t-SNE plot to {png_path}")


# # def train_epoch(
# #     trn_loader: DataLoader,
# #     model,
# #     optim: Union[torch.optim.SGD, torch.optim.Adam],
# #     device: torch.device,
# #     scheduler: torch.optim.lr_scheduler,
# #     config: argparse.Namespace):
# #     """Train the model for one epoch"""
# #     running_loss = 0
# #     num_total = 0.0
# #     ii = 0
# #     model.train()

# #     # set objective (Loss) functions
# #     weight = torch.FloatTensor([0.1, 0.9]).to(device)
# #     criterion = nn.CrossEntropyLoss(weight=weight)
# #     for batch_x, batch_y in tqdm(trn_loader):
# #         batch_size = batch_x.size(0)
# #         num_total += batch_size
# #         ii += 1
# #         batch_y = batch_y.view(-1).type(torch.int64).to(device)
# #         _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
# #         batch_loss = criterion(batch_out, batch_y)
# #         running_loss += batch_loss.item() * batch_size
# #         optim.zero_grad()
# #         batch_loss.backward()
# #         optim.step()

# #         if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
# #             scheduler.step()
# #         elif scheduler is None:
# #             pass
# #         else:
# #             raise ValueError("scheduler error, got:{}".format(scheduler))

# #     running_loss /= num_total
# #     return running_loss


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="ASVspoof detection system")
# #     parser.add_argument("--config",
# #                         dest="config",
# #                         type=str,
# #                         help="configuration file",
# #                         required=True)
# #     parser.add_argument(
# #         "--output_dir",
# #         dest="output_dir",
# #         type=str,
# #         help="output directory for results",
# #         default="./exp_result",
# #     )
# #     parser.add_argument("--seed",
# #                         type=int,
# #                         default=1234,
# #                         help="random seed (default: 1234)")
# #     parser.add_argument(
# #         "--eval",
# #         action="store_true",
# #         help="when this flag is given, evaluates given model and exit")
# #     parser.add_argument("--comment",
# #                         type=str,
# #                         default=None,
# #                         help="comment to describe the saved model")
# #     parser.add_argument("--eval_model_weights",
# #                         type=str,
# #                         default=None,
# #                         help="directory to the model weight file (can be also given in the config file)")
# #     main(parser.parse_args())



# import argparse
# import csv
# import json
# import os
# import sys
# import warnings
# from importlib import import_module
# from pathlib import Path
# from shutil import copy
# from typing import Dict, List, Optional, Union

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.manifold import TSNE
# from sklearn.metrics import roc_curve, auc
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchcontrib.optim import SWA
# from tqdm import tqdm

# from data_utils import TrainDataset, TestDataset, genSpoof_list
# from eval.calculate_metrics import calculate_minDCF_EER_CLLR
# from eval.calculate_modules import compute_eer, compute_det_curve
# from utils import create_optimizer, seed_worker, set_seed, str_to_bool

# warnings.filterwarnings("ignore", category=FutureWarning)


# # =========================================================
# # Main
# # =========================================================
# def main(args: argparse.Namespace) -> None:
#     with open(args.config, "r") as f_json:
#         config = json.loads(f_json.read())

#     model_config = config["model_config"]
#     optim_config = config["optim_config"]
#     optim_config["epochs"] = config["num_epochs"]

#     # ---------------- defaults ----------------
#     config.setdefault("eval_all_best", "True")
#     config.setdefault("freq_aug", "False")

#     config.setdefault("early_stop", "True")
#     config.setdefault("early_stop_patience", 5)
#     config.setdefault("early_stop_min_delta", 0.0)

#     config.setdefault("save_first_epoch_plots", "True")
#     config.setdefault("save_tsne", "True")
#     config.setdefault("tsne_max_samples", 2000)

#     # save score/prediction details every epoch
#     config.setdefault("save_epoch_scores", "True")
#     config.setdefault("save_epoch_predictions", "True")
#     config.setdefault("save_epoch_metric_json", "True")
#     config.setdefault("save_epoch_figures", "True")

#     # embedding dump for future custom plots
#     config.setdefault("save_embedding_archives", "True")
#     config.setdefault("embedding_archive_max_samples", 3000)

#     set_seed(args.seed, config)

#     output_dir = Path(args.output_dir)
#     database_path = Path(config["database_path"])

#     dev_trial_path = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     model_tag = "{}_ep{}_bs{}".format(
#         os.path.splitext(os.path.basename(args.config))[0],
#         config["num_epochs"],
#         config["batch_size"],
#     )
#     if args.comment:
#         model_tag = model_tag + f"_{args.comment}"

#     model_tag = output_dir / model_tag

#     # ---------------- paths ----------------
#     model_save_path = model_tag / "weights"
#     scores_dir = model_tag / "scores"
#     predictions_dir = model_tag / "predictions"
#     figures_dir = model_tag / "figures"
#     metrics_dir = model_tag / "metrics"
#     embedding_dir = model_tag / "embeddings"
#     archive_dir = model_tag / "artifacts"

#     history_csv = model_tag / "training_history.csv"
#     history_jsonl = model_tag / "training_history.jsonl"
#     best_json = model_tag / "best_metrics.json"
#     run_summary_json = model_tag / "run_summary.json"
#     metric_log_path = model_tag / "metric_log.txt"

#     writer = SummaryWriter(model_tag)

#     for p in [
#         model_save_path,
#         scores_dir,
#         predictions_dir,
#         figures_dir,
#         metrics_dir,
#         embedding_dir,
#         archive_dir,
#     ]:
#         p.mkdir(parents=True, exist_ok=True)

#     copy(args.config, model_tag / "config.conf")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")
#     if device == "cpu":
#         raise ValueError("GPU not detected!")

#     model = get_model(model_config, device)
#     trn_loader, dev_loader, eval_loader = get_loader(database_path, args.seed, config)

#     save_json(
#         run_summary_json,
#         {
#             "config_path": args.config,
#             "output_dir": str(model_tag),
#             "database_path": str(database_path),
#             "num_epochs_requested": int(config["num_epochs"]),
#             "batch_size": int(config["batch_size"]),
#             "seed": int(args.seed),
#             "device": device,
#             "comment": args.comment,
#         },
#     )

#     # =====================================================
#     # EVAL ONLY
#     # =====================================================
#     if args.eval:
#         model_path = args.eval_model_weights or config["model_path"]
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"Model loaded : {model_path}")
#         print("Start evaluation...")

#         eval_dir = model_tag / "eval_loaded_model"
#         eval_dir.mkdir(parents=True, exist_ok=True)

#         dev_metrics = evaluate_split(
#             split_name="dev",
#             data_loader=dev_loader,
#             model=model,
#             device=device,
#             trial_path=dev_trial_path,
#             score_txt_path=eval_dir / "dev_scores.txt",
#             prediction_csv_path=eval_dir / "dev_predictions.csv",
#             output_dir=eval_dir / "dev",
#             save_all_figures=True,
#             save_embeddings=str_to_bool(str(config.get("save_embedding_archives", "True"))),
#             embedding_save_path=eval_dir / "dev_embeddings_sampled.npz",
#             embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#         )

#         print(
#             "DONE. dev_eer: {:.3f}, dev_dcf:{:.5f}, dev_cllr:{:.5f}".format(
#                 dev_metrics["eer"], dev_metrics["dcf"], dev_metrics["cllr"]
#             )
#         )

#         if eval_loader is not None and eval_trial_path.exists():
#             eval_metrics = evaluate_split(
#                 split_name="eval",
#                 data_loader=eval_loader,
#                 model=model,
#                 device=device,
#                 trial_path=eval_trial_path,
#                 score_txt_path=eval_dir / "eval_scores.txt",
#                 prediction_csv_path=eval_dir / "eval_predictions.csv",
#                 output_dir=eval_dir / "eval",
#                 save_all_figures=True,
#                 save_embeddings=str_to_bool(str(config.get("save_embedding_archives", "True"))),
#                 embedding_save_path=eval_dir / "eval_embeddings_sampled.npz",
#                 embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#             )

#             print(
#                 "DONE. eval_eer: {:.3f}, eval_dcf:{:.5f}, eval_cllr:{:.5f}".format(
#                     eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]
#                 )
#             )

#         if str_to_bool(str(config.get("save_tsne", "True"))):
#             save_embedding_tsne(
#                 data_loader=dev_loader,
#                 model=model,
#                 device=device,
#                 trial_path=dev_trial_path,
#                 save_dir=eval_dir / "tsne_dev",
#                 max_samples=int(config.get("tsne_max_samples", 2000)),
#                 split_name="dev",
#             )
#             if eval_loader is not None and eval_trial_path.exists():
#                 save_embedding_tsne(
#                     data_loader=eval_loader,
#                     model=model,
#                     device=device,
#                     trial_path=eval_trial_path,
#                     save_dir=eval_dir / "tsne_eval",
#                     max_samples=int(config.get("tsne_max_samples", 2000)),
#                     split_name="eval",
#                 )
#         sys.exit(0)

#     # =====================================================
#     # TRAIN
#     # =====================================================
#     optim_config["steps_per_epoch"] = len(trn_loader)
#     optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
#     optimizer_swa = SWA(optimizer)

#     best_dev_eer = float("inf")
#     best_dev_dcf = float("inf")
#     best_dev_cllr = float("inf")
#     best_epoch = -1
#     n_swa_update = 0
#     epochs_no_improve = 0

#     history_rows: List[Dict] = []

#     early_stop_enabled = str_to_bool(str(config.get("early_stop", "True")))
#     early_stop_patience = int(config.get("early_stop_patience", 5))
#     early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))

#     save_tsne_flag = str_to_bool(str(config.get("save_tsne", "True")))
#     save_epoch_scores = str_to_bool(str(config.get("save_epoch_scores", "True")))
#     save_epoch_predictions = str_to_bool(str(config.get("save_epoch_predictions", "True")))
#     save_epoch_metric_json = str_to_bool(str(config.get("save_epoch_metric_json", "True")))
#     save_epoch_figures = str_to_bool(str(config.get("save_epoch_figures", "True")))
#     save_embedding_archives = str_to_bool(str(config.get("save_embedding_archives", "True")))
#     embedding_archive_max_samples = int(config.get("embedding_archive_max_samples", 3000))

#     f_log = open(metric_log_path, "a")
#     f_log.write("=" * 20 + "\n")

#     for epoch in range(config["num_epochs"]):
#         print(f"\n========== Epoch {epoch:03d} ==========")

#         running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
#         current_lr = float(optimizer.param_groups[0]["lr"])

#         epoch_score_dir = scores_dir / f"epoch_{epoch:03d}"
#         epoch_pred_dir = predictions_dir / f"epoch_{epoch:03d}"
#         epoch_fig_dir = figures_dir / f"epoch_{epoch:03d}"
#         epoch_metric_dir = metrics_dir / f"epoch_{epoch:03d}"
#         epoch_emb_dir = embedding_dir / f"epoch_{epoch:03d}"

#         for p in [epoch_score_dir, epoch_pred_dir, epoch_fig_dir, epoch_metric_dir, epoch_emb_dir]:
#             p.mkdir(parents=True, exist_ok=True)

#         dev_metrics = evaluate_split(
#             split_name="dev",
#             data_loader=dev_loader,
#             model=model,
#             device=device,
#             trial_path=dev_trial_path,
#             score_txt_path=epoch_score_dir / "dev_scores.txt" if save_epoch_scores else None,
#             prediction_csv_path=epoch_pred_dir / "dev_predictions.csv" if save_epoch_predictions else None,
#             output_dir=epoch_fig_dir / "dev" if save_epoch_figures else epoch_metric_dir / "dev",
#             save_all_figures=save_epoch_figures,
#             save_embeddings=save_embedding_archives,
#             embedding_save_path=epoch_emb_dir / "dev_embeddings.npz",
#             embedding_max_samples=embedding_archive_max_samples,
#         )

#         eval_metrics = None
#         if eval_loader is not None and eval_trial_path.exists():
#             eval_metrics = evaluate_split(
#                 split_name="eval",
#                 data_loader=eval_loader,
#                 model=model,
#                 device=device,
#                 trial_path=eval_trial_path,
#                 score_txt_path=epoch_score_dir / "eval_scores.txt" if save_epoch_scores else None,
#                 prediction_csv_path=epoch_pred_dir / "eval_predictions.csv" if save_epoch_predictions else None,
#                 output_dir=epoch_fig_dir / "eval" if save_epoch_figures else epoch_metric_dir / "eval",
#                 save_all_figures=save_epoch_figures,
#                 save_embeddings=save_embedding_archives,
#                 embedding_save_path=epoch_emb_dir / "eval_embeddings.npz",
#                 embedding_max_samples=embedding_archive_max_samples,
#             )

#         # First epoch dedicated t-SNE
#         if epoch == 0 and save_tsne_flag:
#             save_embedding_tsne(
#                 data_loader=dev_loader,
#                 model=model,
#                 device=device,
#                 trial_path=dev_trial_path,
#                 save_dir=model_tag / "tsne_epoch0_dev",
#                 max_samples=int(config.get("tsne_max_samples", 2000)),
#                 split_name="dev",
#             )
#             if eval_loader is not None and eval_trial_path.exists():
#                 save_embedding_tsne(
#                     data_loader=eval_loader,
#                     model=model,
#                     device=device,
#                     trial_path=eval_trial_path,
#                     save_dir=model_tag / "tsne_epoch0_eval",
#                     max_samples=int(config.get("tsne_max_samples", 2000)),
#                     split_name="eval",
#                 )

#         dev_eer = float(dev_metrics["eer"])
#         dev_dcf = float(dev_metrics["dcf"])
#         dev_cllr = float(dev_metrics["cllr"])

#         print(
#             "Train Loss: {:.6f} | LR: {:.8f} | dev_eer: {:.4f} | dev_dcf: {:.6f} | dev_cllr: {:.6f}".format(
#                 running_loss, current_lr, dev_eer, dev_dcf, dev_cllr
#             )
#         )
#         if eval_metrics is not None:
#             print(
#                 "eval_eer: {:.4f} | eval_dcf: {:.6f} | eval_cllr: {:.6f}".format(
#                     eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]
#                 )
#             )

#         # TensorBoard
#         writer.add_scalar("loss", running_loss, epoch)
#         writer.add_scalar("lr", current_lr, epoch)
#         writer.add_scalar("dev_eer", dev_eer, epoch)
#         writer.add_scalar("dev_dcf", dev_dcf, epoch)
#         writer.add_scalar("dev_cllr", dev_cllr, epoch)

#         if eval_metrics is not None:
#             writer.add_scalar("eval_eer", eval_metrics["eer"], epoch)
#             writer.add_scalar("eval_dcf", eval_metrics["dcf"], epoch)
#             writer.add_scalar("eval_cllr", eval_metrics["cllr"], epoch)

#         ckpt_name = f"epoch_{epoch:03d}_devEER_{dev_eer:.6f}.pth"
#         torch.save(model.state_dict(), model_save_path / ckpt_name)

#         row = {
#             "epoch": int(epoch),
#             "loss": float(running_loss),
#             "lr": float(current_lr),
#             "dev_eer": float(dev_eer),
#             "dev_dcf": float(dev_dcf),
#             "dev_cllr": float(dev_cllr),
#             "dev_auc": float(dev_metrics["auc"]),
#             "dev_threshold_eer": float(dev_metrics["threshold_eer"]),
#             "dev_tp": int(dev_metrics["tp"]),
#             "dev_tn": int(dev_metrics["tn"]),
#             "dev_fp": int(dev_metrics["fp"]),
#             "dev_fn": int(dev_metrics["fn"]),
#             "dev_accuracy": float(dev_metrics["accuracy"]),
#             "dev_precision_bonafide": float(dev_metrics["precision_bonafide"]),
#             "dev_recall_bonafide": float(dev_metrics["recall_bonafide"]),
#             "dev_f1_bonafide": float(dev_metrics["f1_bonafide"]),
#         }

#         if eval_metrics is not None:
#             row.update(
#                 {
#                     "eval_eer": float(eval_metrics["eer"]),
#                     "eval_dcf": float(eval_metrics["dcf"]),
#                     "eval_cllr": float(eval_metrics["cllr"]),
#                     "eval_auc": float(eval_metrics["auc"]),
#                     "eval_threshold_eer": float(eval_metrics["threshold_eer"]),
#                     "eval_tp": int(eval_metrics["tp"]),
#                     "eval_tn": int(eval_metrics["tn"]),
#                     "eval_fp": int(eval_metrics["fp"]),
#                     "eval_fn": int(eval_metrics["fn"]),
#                     "eval_accuracy": float(eval_metrics["accuracy"]),
#                     "eval_precision_bonafide": float(eval_metrics["precision_bonafide"]),
#                     "eval_recall_bonafide": float(eval_metrics["recall_bonafide"]),
#                     "eval_f1_bonafide": float(eval_metrics["f1_bonafide"]),
#                 }
#             )

#         history_rows.append(row)
#         append_history_csv(history_csv, row, write_header=(epoch == 0 and not history_csv.exists()))
#         append_jsonl(history_jsonl, row)

#         if save_epoch_metric_json:
#             save_json(epoch_metric_dir / "metrics.json", row)

#         # master curves updated every epoch
#         plot_metric_curves(history_rows, figures_dir / "loss_curve.png", ["loss"], ylabel="Loss")
#         plot_metric_curves(history_rows, figures_dir / "lr_curve.png", ["lr"], ylabel="Learning Rate")
#         plot_metric_curves(history_rows, figures_dir / "eer_curve.png", ["dev_eer", "eval_eer"], ylabel="EER")
#         plot_metric_curves(history_rows, figures_dir / "dcf_curve.png", ["dev_dcf", "eval_dcf"], ylabel="minDCF")
#         plot_metric_curves(history_rows, figures_dir / "cllr_curve.png", ["dev_cllr", "eval_cllr"], ylabel="CLLR")
#         plot_metric_curves(history_rows, figures_dir / "auc_curve.png", ["dev_auc", "eval_auc"], ylabel="AUC")
#         plot_metric_curves(
#             history_rows,
#             figures_dir / "accuracy_curve.png",
#             ["dev_accuracy", "eval_accuracy"],
#             ylabel="Accuracy",
#         )

#         best_dev_dcf = min(best_dev_dcf, dev_dcf)
#         best_dev_cllr = min(best_dev_cllr, dev_cllr)

#         improved = dev_eer < (best_dev_eer - early_stop_min_delta)
#         if improved:
#             print(f"Best model found at epoch {epoch}")
#             best_dev_eer = dev_eer
#             best_epoch = epoch
#             epochs_no_improve = 0

#             torch.save(model.state_dict(), model_save_path / "best_ast.pth")

#             print(f"Saving epoch {epoch} for SWA")
#             optimizer_swa.update_swa()
#             n_swa_update += 1

#             best_data = {
#                 "best_epoch": int(epoch),
#                 "best_dev_eer": float(dev_eer),
#                 "best_dev_dcf": float(dev_dcf),
#                 "best_dev_cllr": float(dev_cllr),
#                 "best_dev_auc": float(dev_metrics["auc"]),
#             }
#             if eval_metrics is not None:
#                 best_data.update(
#                     {
#                         "best_eval_eer": float(eval_metrics["eer"]),
#                         "best_eval_dcf": float(eval_metrics["dcf"]),
#                         "best_eval_cllr": float(eval_metrics["cllr"]),
#                         "best_eval_auc": float(eval_metrics["auc"]),
#                     }
#                 )
#             save_json(best_json, best_data)
#         else:
#             epochs_no_improve += 1

#         writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
#         writer.add_scalar("best_dev_dcf", best_dev_dcf, epoch)
#         writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

#         log_line = (
#             f"epoch={epoch}, loss={running_loss:.6f}, lr={current_lr:.8f}, "
#             f"dev_eer={dev_eer:.6f}, dev_dcf={dev_dcf:.6f}, dev_cllr={dev_cllr:.6f}, "
#             f"dev_auc={dev_metrics['auc']:.6f}, dev_acc={dev_metrics['accuracy']:.6f}"
#         )
#         if eval_metrics is not None:
#             log_line += (
#                 f", eval_eer={eval_metrics['eer']:.6f}, eval_dcf={eval_metrics['dcf']:.6f}, "
#                 f"eval_cllr={eval_metrics['cllr']:.6f}, eval_auc={eval_metrics['auc']:.6f}, "
#                 f"eval_acc={eval_metrics['accuracy']:.6f}"
#             )
#         f_log.write(log_line + "\n")
#         f_log.flush()

#         if early_stop_enabled and epochs_no_improve >= early_stop_patience:
#             print(
#                 f"Early stopping triggered at epoch {epoch}. "
#                 f"Best epoch = {best_epoch}, best dev_eer = {best_dev_eer:.6f}"
#             )
#             break

#     f_log.close()
#     writer.close()

#     # =====================================================
#     # FINAL BEST MODEL EVAL
#     # =====================================================
#     best_weight = model_save_path / "best_ast.pth"
#     if best_weight.exists():
#         model.load_state_dict(torch.load(best_weight, map_location=device))

#     final_dir = model_tag / "final_best_model_eval"
#     final_dir.mkdir(parents=True, exist_ok=True)

#     final_dev_metrics = evaluate_split(
#         split_name="dev",
#         data_loader=dev_loader,
#         model=model,
#         device=device,
#         trial_path=dev_trial_path,
#         score_txt_path=final_dir / "dev_best_scores.txt",
#         prediction_csv_path=final_dir / "dev_best_predictions.csv",
#         output_dir=final_dir / "dev",
#         save_all_figures=True,
#         save_embeddings=save_embedding_archives,
#         embedding_save_path=final_dir / "dev_best_embeddings.npz",
#         embedding_max_samples=embedding_archive_max_samples,
#     )

#     final_eval_metrics = None
#     if eval_loader is not None and eval_trial_path.exists():
#         final_eval_metrics = evaluate_split(
#             split_name="eval",
#             data_loader=eval_loader,
#             model=model,
#             device=device,
#             trial_path=eval_trial_path,
#             score_txt_path=final_dir / "eval_best_scores.txt",
#             prediction_csv_path=final_dir / "eval_best_predictions.csv",
#             output_dir=final_dir / "eval",
#             save_all_figures=True,
#             save_embeddings=save_embedding_archives,
#             embedding_save_path=final_dir / "eval_best_embeddings.npz",
#             embedding_max_samples=embedding_archive_max_samples,
#         )

#     if save_tsne_flag:
#         save_embedding_tsne(
#             data_loader=dev_loader,
#             model=model,
#             device=device,
#             trial_path=dev_trial_path,
#             save_dir=model_tag / "tsne_final_dev",
#             max_samples=int(config.get("tsne_max_samples", 2000)),
#             split_name="dev",
#         )
#         if eval_loader is not None and eval_trial_path.exists():
#             save_embedding_tsne(
#                 data_loader=eval_loader,
#                 model=model,
#                 device=device,
#                 trial_path=eval_trial_path,
#                 save_dir=model_tag / "tsne_final_eval",
#                 max_samples=int(config.get("tsne_max_samples", 2000)),
#                 split_name="eval",
#             )

#     save_json(
#         archive_dir / "final_summary.json",
#         {
#             "best_epoch": int(best_epoch),
#             "best_dev_eer": float(best_dev_eer),
#             "best_dev_dcf": float(best_dev_dcf),
#             "best_dev_cllr": float(best_dev_cllr),
#             "swa_updates": int(n_swa_update),
#             "final_dev_metrics": final_dev_metrics,
#             "final_eval_metrics": final_eval_metrics,
#         },
#     )

#     if n_swa_update > 0:
#         print(f"SWA updated {n_swa_update} time(s).")

#     print(f"\nTraining completed. Best epoch: {best_epoch}, Best dev EER: {best_dev_eer:.6f}")


# # =========================================================
# # Model / Loader
# # =========================================================
# def get_model(model_config: Dict, device: torch.device):
#     module = import_module(f"models.{model_config['architecture']}")
#     _model = getattr(module, "Model")
#     model = _model(model_config).to(device)
#     nb_params = sum(param.view(-1).size()[0] for param in model.parameters())
#     print(f"no. model params:{nb_params}")
#     return model


# def get_loader(database_path: Path, seed: int, config: dict):
#     trn_database_path = database_path / "flac_T"
#     dev_database_path = database_path / "flac_D"
#     eval_database_path = database_path / "flac_E"

#     trn_list_path = database_path / "ASVspoof5.train.metainfor.txt"
#     dev_trial_path = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
#     print("no. training files:", len(file_train))

#     train_set = TrainDataset(list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path)
#     gen = torch.Generator()
#     gen.manual_seed(seed)

#     trn_loader = DataLoader(
#         train_set,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=gen,
#     )

#     _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
#     print("no. validation files:", len(file_dev))
#     dev_set = TestDataset(list_IDs=file_dev, base_dir=dev_database_path)
#     dev_loader = DataLoader(
#         dev_set,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         drop_last=False,
#         pin_memory=True,
#     )

#     eval_loader = None
#     if eval_trial_path.exists() and eval_database_path.exists():
#         _, file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=False)
#         print("no. eval files:", len(file_eval))
#         eval_set = TestDataset(list_IDs=file_eval, base_dir=eval_database_path)
#         eval_loader = DataLoader(
#             eval_set,
#             batch_size=config["batch_size"],
#             shuffle=False,
#             drop_last=False,
#             pin_memory=True,
#         )
#     else:
#         print("Eval loader skipped: flac_E or ASVspoof5.eval.metainfor.txt not found.")

#     return trn_loader, dev_loader, eval_loader


# # =========================================================
# # Evaluation helpers
# # =========================================================
# def parse_trial_file(trial_path: Path) -> Dict[str, Dict]:
#     info = {}
#     with open(trial_path, "r") as f:
#         for line in f:
#             parts = line.strip().split(" ")
#             if len(parts) < 6:
#                 continue
#             spk_id, utt_id, p2, p3, p4, key = parts[:6]
#             info[utt_id] = {
#                 "speaker_id": spk_id,
#                 "utt_id": utt_id,
#                 "meta_2": p2,
#                 "meta_3": p3,
#                 "meta_4": p4,
#                 "label_text": key,
#                 "label_int": 1 if key == "bonafide" else 0,
#             }
#     return info


# def collect_predictions(
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     save_embeddings: bool = False,
#     embedding_max_samples: int = 3000,
# ):
#     model.eval()
#     trial_info = parse_trial_file(trial_path)

#     rows = []
#     sampled_embeddings = []
#     sampled_scores = []
#     sampled_labels = []
#     sampled_ids = []

#     use_all_embeddings = embedding_max_samples is None or int(embedding_max_samples) <= 0

#     for batch_x, utt_ids in tqdm(data_loader):
#         batch_x = batch_x.to(device)

#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)
#             probs = torch.softmax(batch_out, dim=1)

#         batch_emb_np = batch_emb.detach().cpu().numpy()
#         batch_logits_np = batch_out.detach().cpu().numpy()
#         batch_probs_np = probs.detach().cpu().numpy()

#         for i, utt_id in enumerate(utt_ids):
#             if utt_id not in trial_info:
#                 continue

#             meta = trial_info[utt_id]
#             prob_spoof = float(batch_probs_np[i, 0])
#             prob_bonafide = float(batch_probs_np[i, 1])
#             logit_spoof = float(batch_logits_np[i, 0])
#             logit_bonafide = float(batch_logits_np[i, 1])

#             pred_int = 1 if prob_bonafide >= prob_spoof else 0
#             true_int = int(meta["label_int"])

#             rows.append(
#                 {
#                     "speaker_id": meta["speaker_id"],
#                     "utt_id": utt_id,
#                     "true_label_text": meta["label_text"],
#                     "true_label_int": true_int,
#                     "pred_label_text": "bonafide" if pred_int == 1 else "spoof",
#                     "pred_label_int": pred_int,
#                     "is_correct": int(pred_int == true_int),
#                     "logit_spoof": logit_spoof,
#                     "logit_bonafide": logit_bonafide,
#                     "prob_spoof": prob_spoof,
#                     "prob_bonafide": prob_bonafide,
#                     "score": prob_bonafide,
#                 }
#             )

#             if save_embeddings and (use_all_embeddings or len(sampled_ids) < embedding_max_samples):
#                 sampled_embeddings.append(batch_emb_np[i])
#                 sampled_scores.append(prob_bonafide)
#                 sampled_labels.append(true_int)
#                 sampled_ids.append(utt_id)

#     return rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids


# def save_score_txt(score_txt_path: Path, rows: List[Dict]) -> None:
#     with open(score_txt_path, "w") as fh:
#         for r in rows:
#             fh.write(
#                 f"{r['speaker_id']} {r['utt_id']} {r['score']} {r['true_label_text']}\n"
#             )


# def save_prediction_csv(prediction_csv_path: Path, rows: List[Dict]) -> None:
#     if not rows:
#         return
#     with open(prediction_csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)


# def compute_binary_metrics_from_rows(rows: List[Dict]) -> Dict:
#     scores = np.array([r["score"] for r in rows], dtype=np.float64)
#     labels = np.array([r["true_label_int"] for r in rows], dtype=np.int64)

#     bona_scores = scores[labels == 1]
#     spoof_scores = scores[labels == 0]

#     eer, frr, far, thresholds = compute_eer(bona_scores, spoof_scores)
#     min_index = int(np.argmin(np.abs(frr - far)))
#     threshold_eer = float(thresholds[min_index])

#     pred = (scores >= threshold_eer).astype(np.int64)

#     tp = int(np.sum((pred == 1) & (labels == 1)))
#     tn = int(np.sum((pred == 0) & (labels == 0)))
#     fp = int(np.sum((pred == 1) & (labels == 0)))
#     fn = int(np.sum((pred == 0) & (labels == 1)))

#     accuracy = float((tp + tn) / max(1, len(labels)))
#     precision_bonafide = float(tp / max(1, tp + fp))
#     recall_bonafide = float(tp / max(1, tp + fn))
#     f1_bonafide = float(
#         (2 * precision_bonafide * recall_bonafide) / max(1e-12, precision_bonafide + recall_bonafide)
#     )

#     fpr, tpr, roc_thresholds = roc_curve(labels, scores)
#     roc_auc = float(auc(fpr, tpr))

#     return {
#         "scores": scores,
#         "labels": labels,
#         "bona_scores": bona_scores,
#         "spoof_scores": spoof_scores,
#         "eer": float(eer),
#         "frr": frr,
#         "far": far,
#         "thresholds": thresholds,
#         "threshold_eer": threshold_eer,
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#         "accuracy": accuracy,
#         "precision_bonafide": precision_bonafide,
#         "recall_bonafide": recall_bonafide,
#         "f1_bonafide": f1_bonafide,
#         "fpr": fpr,
#         "tpr": tpr,
#         "roc_thresholds": roc_thresholds,
#         "auc": roc_auc,
#     }


# def evaluate_split(
#     split_name: str,
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     score_txt_path: Optional[Path],
#     prediction_csv_path: Optional[Path],
#     output_dir: Path,
#     save_all_figures: bool = True,
#     save_embeddings: bool = False,
#     embedding_save_path: Optional[Path] = None,
#     embedding_max_samples: int = 3000,
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)

#     rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids = collect_predictions(
#         data_loader=data_loader,
#         model=model,
#         device=device,
#         trial_path=trial_path,
#         save_embeddings=save_embeddings,
#         embedding_max_samples=embedding_max_samples,
#     )

#     if score_txt_path is None:
#         score_txt_path = output_dir / f"{split_name}_scores.txt"
#     save_score_txt(score_txt_path, rows)

#     if prediction_csv_path is not None:
#         save_prediction_csv(prediction_csv_path, rows)

#     dcf, eer_from_metric_fn, cllr = calculate_minDCF_EER_CLLR(
#         cm_scores_file=score_txt_path,
#         output_file=output_dir / f"{split_name}_DCF_EER.txt",
#         printout=False,
#     )

#     stats = compute_binary_metrics_from_rows(rows)

#     metrics = {
#         "dcf": float(dcf),
#         "eer": float(eer_from_metric_fn),
#         "cllr": float(cllr),
#         "threshold_eer": float(stats["threshold_eer"]),
#         "tp": int(stats["tp"]),
#         "tn": int(stats["tn"]),
#         "fp": int(stats["fp"]),
#         "fn": int(stats["fn"]),
#         "accuracy": float(stats["accuracy"]),
#         "precision_bonafide": float(stats["precision_bonafide"]),
#         "recall_bonafide": float(stats["recall_bonafide"]),
#         "f1_bonafide": float(stats["f1_bonafide"]),
#         "auc": float(stats["auc"]),
#         "n_samples": int(len(rows)),
#         "n_bonafide": int(np.sum(stats["labels"] == 1)),
#         "n_spoof": int(np.sum(stats["labels"] == 0)),
#     }

#     save_json(output_dir / f"{split_name}_metrics.json", metrics)

#     if save_all_figures:
#         save_confusion_plot(
#             tp=stats["tp"],
#             tn=stats["tn"],
#             fp=stats["fp"],
#             fn=stats["fn"],
#             title=f"{split_name.upper()} confusion matrix",
#             save_path=output_dir / f"{split_name}_confusion_matrix.png",
#         )
#         save_confusion_json(
#             tp=stats["tp"],
#             tn=stats["tn"],
#             fp=stats["fp"],
#             fn=stats["fn"],
#             save_path=output_dir / f"{split_name}_confusion_matrix.json",
#         )
#         save_roc_curve(
#             fpr=stats["fpr"],
#             tpr=stats["tpr"],
#             roc_auc=stats["auc"],
#             save_png=output_dir / f"{split_name}_roc_curve.png",
#             save_csv=output_dir / f"{split_name}_roc_curve.csv",
#             split_name=split_name,
#         )
#         save_det_curve_plot(
#             bona_scores=stats["bona_scores"],
#             spoof_scores=stats["spoof_scores"],
#             save_png=output_dir / f"{split_name}_det_curve.png",
#             save_csv=output_dir / f"{split_name}_det_curve.csv",
#             split_name=split_name,
#         )
#         save_score_histogram(
#             bona_scores=stats["bona_scores"],
#             spoof_scores=stats["spoof_scores"],
#             threshold=stats["threshold_eer"],
#             save_path=output_dir / f"{split_name}_score_histogram.png",
#             split_name=split_name,
#         )
#         save_threshold_sweep_csv(
#             scores=stats["scores"],
#             labels=stats["labels"],
#             thresholds=stats["thresholds"],
#             save_path=output_dir / f"{split_name}_threshold_sweep.csv",
#         )

#     if save_embeddings and embedding_save_path is not None and len(sampled_ids) > 0:
#         np.savez_compressed(
#             embedding_save_path,
#             utt_ids=np.array(sampled_ids),
#             labels=np.array(sampled_labels),
#             scores=np.array(sampled_scores),
#             embeddings=np.array(sampled_embeddings),
#         )

#     return metrics


# # =========================================================
# # Plot helpers
# # =========================================================
# def save_confusion_plot(tp: int, tn: int, fp: int, fn: int, title: str, save_path: Path) -> None:
#     matrix = np.array([[tp, fn], [fp, tn]])

#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(matrix)
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["Pred Bonafide", "Pred Spoof"])
#     ax.set_yticklabels(["True Bonafide", "True Spoof"])
#     ax.set_title(title)

#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

#     fig.colorbar(im, ax=ax)
#     fig.tight_layout()
#     plt.savefig(save_path, dpi=220)
#     plt.close(fig)


# def save_confusion_json(tp: int, tn: int, fp: int, fn: int, save_path: Path) -> None:
#     save_json(save_path, {"tp": tp, "tn": tn, "fp": fp, "fn": fn})


# def save_roc_curve(fpr, tpr, roc_auc: float, save_png: Path, save_csv: Path, split_name: str) -> None:
#     with open(save_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["fpr", "tpr"])
#         for x, y in zip(fpr, tpr):
#             writer.writerow([float(x), float(y)])

#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
#     plt.plot([0, 1], [0, 1], linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"{split_name.upper()} ROC Curve")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_png, dpi=220)
#     plt.close()


# def save_det_curve_plot(bona_scores, spoof_scores, save_png: Path, save_csv: Path, split_name: str) -> None:
#     frr, far, thresholds = compute_det_curve(bona_scores, spoof_scores)

#     with open(save_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["threshold", "frr", "far"])
#         for thr, x, y in zip(thresholds, frr[1:], far[1:]):
#             writer.writerow([float(thr), float(x), float(y)])

#     plt.figure(figsize=(6, 5))
#     plt.plot(far[1:], frr[1:])
#     plt.xlabel("FAR")
#     plt.ylabel("FRR")
#     plt.title(f"{split_name.upper()} DET Curve")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_png, dpi=220)
#     plt.close()


# def save_score_histogram(bona_scores, spoof_scores, threshold: float, save_path: Path, split_name: str) -> None:
#     plt.figure(figsize=(8, 5))
#     plt.hist(spoof_scores, bins=60, alpha=0.6, label="spoof")
#     plt.hist(bona_scores, bins=60, alpha=0.6, label="bonafide")
#     plt.axvline(threshold, linestyle="--", label=f"EER thr={threshold:.4f}")
#     plt.xlabel("Bonafide score")
#     plt.ylabel("Count")
#     plt.title(f"{split_name.upper()} Score Histogram")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=220)
#     plt.close()


# def save_threshold_sweep_csv(scores, labels, thresholds, save_path: Path) -> None:
#     with open(save_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "threshold", "tp", "tn", "fp", "fn",
#             "accuracy", "precision_bonafide", "recall_bonafide", "f1_bonafide"
#         ])
#         for thr in thresholds:
#             pred = (scores >= thr).astype(np.int64)
#             tp = int(np.sum((pred == 1) & (labels == 1)))
#             tn = int(np.sum((pred == 0) & (labels == 0)))
#             fp = int(np.sum((pred == 1) & (labels == 0)))
#             fn = int(np.sum((pred == 0) & (labels == 1)))

#             acc = float((tp + tn) / max(1, len(labels)))
#             prec = float(tp / max(1, tp + fp))
#             rec = float(tp / max(1, tp + fn))
#             f1 = float((2 * prec * rec) / max(1e-12, prec + rec))
#             writer.writerow([float(thr), tp, tn, fp, fn, acc, prec, rec, f1])


# def plot_metric_curves(history_rows: List[Dict], save_path: Path, metric_keys: List[str], ylabel: str = "Value") -> None:
#     if not history_rows:
#         return

#     epochs = [row["epoch"] for row in history_rows]
#     plt.figure(figsize=(8, 5))
#     plotted = False

#     for key in metric_keys:
#         values = [row.get(key, None) for row in history_rows]
#         if all(v is None for v in values):
#             continue

#         x = [e for e, v in zip(epochs, values) if v is not None]
#         y = [v for v in values if v is not None]
#         if len(x) > 0:
#             plt.plot(x, y, marker="o", label=key)
#             plotted = True

#     if not plotted:
#         plt.close()
#         return

#     plt.xlabel("Epoch")
#     plt.ylabel(ylabel)
#     plt.title(save_path.stem.replace("_", " ").upper())
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=220)
#     plt.close()


# # =========================================================
# # Embeddings / t-SNE
# # =========================================================
# def _parse_trial_labels(trial_path: Path):
#     labels = {}
#     with open(trial_path, "r") as f:
#         for line in f:
#             parts = line.strip().split(" ")
#             if len(parts) < 6:
#                 continue
#             _, utt_id, _, _, _, key = parts[:6]
#             labels[utt_id] = 1 if key == "bonafide" else 0
#     return labels


# def save_embedding_tsne(
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     save_dir: Path,
#     max_samples: int = 2000,
#     split_name: str = "dev",
# ) -> None:
#     save_dir.mkdir(parents=True, exist_ok=True)
#     label_map = _parse_trial_labels(trial_path)

#     model.eval()
#     embeddings = []
#     scores = []
#     utt_ids = []

#     use_all = max_samples is None or int(max_samples) <= 0

#     for batch_x, batch_utt_id in tqdm(data_loader, desc=f"extract_tsne_{split_name}"):
#         batch_x = batch_x.to(device)
#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)
#             batch_score = torch.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
#             batch_emb = batch_emb.detach().cpu().numpy()

#         embeddings.append(batch_emb)
#         scores.append(batch_score)
#         utt_ids.extend(batch_utt_id)

#         if (not use_all) and len(utt_ids) >= max_samples:
#             break

#     if len(utt_ids) < 2:
#         print(f"Not enough samples for t-SNE on {split_name}.")
#         return

#     embeddings = np.concatenate(embeddings, axis=0)
#     scores = np.concatenate(scores, axis=0)

#     if not use_all:
#         embeddings = embeddings[:max_samples]
#         scores = scores[:max_samples]
#         utt_ids = utt_ids[:max_samples]

#     labels = np.array([label_map.get(u, -1) for u in utt_ids])

#     np.savez_compressed(
#         save_dir / f"{split_name}_tsne_source_embeddings.npz",
#         utt_ids=np.array(utt_ids),
#         labels=labels,
#         scores=scores,
#         embeddings=embeddings,
#     )

#     n_samples = len(utt_ids)
#     perplexity = min(30, max(5, n_samples - 1))

#     tsne = TSNE(
#         n_components=2,
#         perplexity=perplexity,
#         init="pca",
#         learning_rate="auto",
#         random_state=42,
#     )
#     coords = tsne.fit_transform(embeddings)

#     csv_path = save_dir / f"{split_name}_tsne_points.csv"
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["utt_id", "label", "bonafide_score", "tsne_x", "tsne_y"])
#         for utt_id, label, score, (x, y) in zip(utt_ids, labels, scores, coords):
#             writer.writerow([utt_id, int(label), float(score), float(x), float(y)])

#     png_path = save_dir / f"{split_name}_tsne_plot.png"
#     plt.figure(figsize=(8, 6))

#     spoof_mask = labels == 0
#     bona_mask = labels == 1
#     unknown_mask = labels == -1

#     if spoof_mask.any():
#         plt.scatter(coords[spoof_mask, 0], coords[spoof_mask, 1], s=10, alpha=0.7, label="spoof")
#     if bona_mask.any():
#         plt.scatter(coords[bona_mask, 0], coords[bona_mask, 1], s=10, alpha=0.7, label="bonafide")
#     if unknown_mask.any():
#         plt.scatter(coords[unknown_mask, 0], coords[unknown_mask, 1], s=10, alpha=0.7, label="unknown")

#     plt.title(f"{split_name.upper()} set AST embeddings (t-SNE)")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(png_path, dpi=220)
#     plt.close()

#     print(f"Saved t-SNE using {n_samples} samples for {split_name}")
#     print(f"Saved t-SNE CSV to {csv_path}")
#     print(f"Saved t-SNE plot to {png_path}")


# # =========================================================
# # Logging helpers
# # =========================================================
# def append_history_csv(csv_path: Path, row: Dict, write_header: bool = False) -> None:
#     with open(csv_path, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)


# def append_jsonl(path: Path, row: Dict) -> None:
#     with open(path, "a") as f:
#         f.write(json.dumps(row) + "\n")


# def save_json(path: Path, data: Dict) -> None:
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)


# # =========================================================
# # Train
# # =========================================================
# def train_epoch(
#     trn_loader: DataLoader,
#     model,
#     optim: Union[torch.optim.SGD, torch.optim.Adam],
#     device: torch.device,
#     scheduler,
#     config: dict,
# ):
#     running_loss = 0.0
#     num_total = 0.0
#     model.train()

#     weight = torch.FloatTensor([0.1, 0.9]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)

#     for batch_x, batch_y in tqdm(trn_loader):
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.view(-1).type(torch.int64).to(device)

#         batch_size = batch_x.size(0)
#         num_total += batch_size

#         _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
#         batch_loss = criterion(batch_out, batch_y)
#         running_loss += batch_loss.item() * batch_size

#         optim.zero_grad()
#         batch_loss.backward()
#         optim.step()

#         if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
#             if scheduler is not None:
#                 scheduler.step()
#         elif scheduler is None:
#             pass
#         else:
#             raise ValueError(f"scheduler error, got:{scheduler}")

#     running_loss /= num_total
#     return float(running_loss)


# # =========================================================
# # CLI
# # =========================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="ASVspoof detection system")
#     parser.add_argument("--config", dest="config", type=str, help="configuration file", required=True)
#     parser.add_argument(
#         "--output_dir",
#         dest="output_dir",
#         type=str,
#         help="output directory for results",
#         default="./exp_result",
#     )
#     parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
#     parser.add_argument("--eval", action="store_true", help="evaluate given model and exit")
#     parser.add_argument("--comment", type=str, default=None, help="comment to describe the saved model")
#     parser.add_argument(
#         "--eval_model_weights",
#         type=str,
#         default=None,
#         help="path to the model weight file (can also be given in config file)",
#     )
#     main(parser.parse_args())



# ====================
# epoch=0, loss=0.030815, lr=0.00000994, dev_eer=0.113930, dev_dcf=0.169651, dev_cllr=1.049690, dev_auc=0.945824, dev_acc=0.886356, dev_f1=0.776366, eval_eer=0.324981, eval_dcf=0.766844, eval_cllr=1.117895, eval_auc=0.757970, eval_acc=0.675041, eval_f1=0.458433
# epoch=1, loss=0.012977, lr=0.00000978, dev_eer=0.111930, dev_dcf=0.169378, dev_cllr=1.060325, dev_auc=0.946214, dev_acc=0.888555, dev_f1=0.780308, eval_eer=0.305830, eval_dcf=0.773373, eval_cllr=1.114550, eval_auc=0.766234, eval_acc=0.694217, eval_f1=0.480654
# epoch=2, loss=0.010452, lr=0.00000951, dev_eer=0.100787, dev_dcf=0.188173, dev_cllr=1.129529, dev_auc=0.957115, dev_acc=0.900333, dev_f1=0.801425, eval_eer=0.323683, eval_dcf=0.781762, eval_cllr=1.152685, eval_auc=0.749619, eval_acc=0.676400, eval_f1=0.461361
# epoch=3, loss=0.008747, lr=0.00000914, dev_eer=0.142502, dev_dcf=0.271126, dev_cllr=1.007717, dev_auc=0.913410, dev_acc=0.857665, dev_f1=0.728334, eval_eer=0.345459, eval_dcf=0.799497, eval_cllr=1.086807, eval_auc=0.720600, eval_acc=0.654599, eval_f1=0.435850
# epoch=4, loss=0.007902, lr=0.00000868, dev_eer=0.124469, dev_dcf=0.227305, dev_cllr=0.994335, dev_auc=0.924890, dev_acc=0.875728, dev_f1=0.758200, eval_eer=0.334154, eval_dcf=0.801107, eval_cllr=1.082606, eval_auc=0.734987, eval_acc=0.665896, eval_f1=0.448217
# epoch=5, loss=0.006950, lr=0.00000815, dev_eer=0.113846, dev_dcf=0.200087, dev_cllr=1.041947, dev_auc=0.936079, dev_acc=0.886399, dev_f1=0.776419, eval_eer=0.334233, eval_dcf=0.781113, eval_cllr=1.084168, eval_auc=0.741394, eval_acc=0.665789, eval_f1=0.448057
# epoch=6, loss=0.006709, lr=0.00000754, dev_eer=0.124907, dev_dcf=0.243550, dev_cllr=1.059538, dev_auc=0.928084, dev_acc=0.875196, dev_f1=0.757228, eval_eer=0.336583, eval_dcf=0.821877, eval_cllr=1.117982, eval_auc=0.730040, eval_acc=0.663475, eval_f1=0.445545
# epoch=7, loss=0.005736, lr=0.00000689, dev_eer=0.124017, dev_dcf=0.246542, dev_cllr=1.077916, dev_auc=0.931433, dev_acc=0.876040, dev_f1=0.759451, eval_eer=0.335177, eval_dcf=0.768727, eval_cllr=1.120931, eval_auc=0.734804, eval_acc=0.664575, eval_f1=0.447549
# ====================
# ====================

# for below


# import argparse
# import csv
# import json
# import os
# import sys
# import warnings
# from importlib import import_module
# from pathlib import Path
# from shutil import copy
# from typing import Dict, List, Optional, Union

# import matplotlib
# matplotlib.use("Agg")  # safe backend for remote/server training
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.manifold import TSNE
# from sklearn.metrics import roc_curve, auc
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchcontrib.optim import SWA
# from tqdm import tqdm

# from data_utils import TrainDataset, TestDataset, genSpoof_list
# from eval.calculate_metrics import calculate_minDCF_EER_CLLR
# from eval.calculate_modules import compute_eer, compute_det_curve
# from utils import create_optimizer, seed_worker, set_seed, str_to_bool

# warnings.filterwarnings("ignore", category=FutureWarning)


# # =========================================================
# # Main
# # =========================================================
# def main(args: argparse.Namespace) -> None:
#     with open(args.config, "r") as f_json:
#         config = json.loads(f_json.read())

#     model_config = config["model_config"]
#     optim_config = config["optim_config"]
#     optim_config["epochs"] = config["num_epochs"]

#     # ---------------- defaults ----------------
#     config.setdefault("eval_all_best", "True")
#     config.setdefault("freq_aug", "False")

#     config.setdefault("early_stop", "True")
#     config.setdefault("early_stop_patience", 5)
#     config.setdefault("early_stop_min_delta", 0.0)

#     config.setdefault("save_first_epoch_plots", "True")
#     config.setdefault("save_tsne", "True")
#     config.setdefault("tsne_max_samples", 2000)

#     # save score/prediction details every epoch
#     config.setdefault("save_epoch_scores", "True")
#     config.setdefault("save_epoch_predictions", "True")
#     config.setdefault("save_epoch_metric_json", "True")
#     config.setdefault("save_epoch_figures", "True")

#     # embedding dump for future custom plots
#     config.setdefault("save_embedding_archives", "True")
#     config.setdefault("embedding_archive_max_samples", 3000)

#     set_seed(args.seed, config)

#     output_dir = Path(args.output_dir)
#     database_path = Path(config["database_path"])

#     dev_trial_path = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     model_tag = "{}_ep{}_bs{}".format(
#         os.path.splitext(os.path.basename(args.config))[0],
#         config["num_epochs"],
#         config["batch_size"],
#     )
#     if args.comment:
#         model_tag = model_tag + f"_{args.comment}"

#     model_tag = output_dir / model_tag

#     # ---------------- paths ----------------
#     model_save_path = model_tag / "weights"
#     scores_dir = model_tag / "scores"
#     predictions_dir = model_tag / "predictions"
#     figures_dir = model_tag / "figures"
#     metrics_dir = model_tag / "metrics"
#     embedding_dir = model_tag / "embeddings"
#     archive_dir = model_tag / "artifacts"

#     history_csv = model_tag / "training_history.csv"
#     history_jsonl = model_tag / "training_history.jsonl"
#     best_json = model_tag / "best_metrics.json"
#     run_summary_json = model_tag / "run_summary.json"
#     metric_log_path = model_tag / "metric_log.txt"

#     writer = SummaryWriter(str(model_tag))

#     for p in [
#         model_save_path,
#         scores_dir,
#         predictions_dir,
#         figures_dir,
#         metrics_dir,
#         embedding_dir,
#         archive_dir,
#     ]:
#         p.mkdir(parents=True, exist_ok=True)

#     copy(args.config, model_tag / "config.conf")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")
#     if device == "cpu":
#         raise ValueError("GPU not detected!")

#     model = get_model(model_config, device)
#     trn_loader, dev_loader, eval_loader = get_loader(database_path, args.seed, config)

#     save_json(
#         run_summary_json,
#         {
#             "config_path": args.config,
#             "output_dir": str(model_tag),
#             "database_path": str(database_path),
#             "num_epochs_requested": int(config["num_epochs"]),
#             "batch_size": int(config["batch_size"]),
#             "seed": int(args.seed),
#             "device": device,
#             "comment": args.comment,
#         },
#     )

#     # =====================================================
#     # EVAL ONLY
#     # =====================================================
#     if args.eval:
#         model_path = args.eval_model_weights or config["model_path"]
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"Model loaded : {model_path}")
#         print("Start evaluation...")

#         eval_dir = model_tag / "eval_loaded_model"
#         eval_dir.mkdir(parents=True, exist_ok=True)

#         dev_metrics = evaluate_split(
#             split_name="dev",
#             data_loader=dev_loader,
#             model=model,
#             device=device,
#             trial_path=dev_trial_path,
#             score_txt_path=eval_dir / "dev_scores.txt",
#             prediction_csv_path=eval_dir / "dev_predictions.csv",
#             output_dir=eval_dir / "dev",
#             save_all_figures=True,
#             save_embeddings=str_to_bool(str(config.get("save_embedding_archives", "True"))),
#             embedding_save_path=eval_dir / "dev_embeddings_sampled.npz",
#             embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#         )

#         print(
#             "DONE. dev_eer: {:.3f}, dev_dcf:{:.5f}, dev_cllr:{:.5f}".format(
#                 dev_metrics["eer"], dev_metrics["dcf"], dev_metrics["cllr"]
#             )
#         )

#         if eval_loader is not None and eval_trial_path.exists():
#             eval_metrics = evaluate_split(
#                 split_name="eval",
#                 data_loader=eval_loader,
#                 model=model,
#                 device=device,
#                 trial_path=eval_trial_path,
#                 score_txt_path=eval_dir / "eval_scores.txt",
#                 prediction_csv_path=eval_dir / "eval_predictions.csv",
#                 output_dir=eval_dir / "eval",
#                 save_all_figures=True,
#                 save_embeddings=str_to_bool(str(config.get("save_embedding_archives", "True"))),
#                 embedding_save_path=eval_dir / "eval_embeddings_sampled.npz",
#                 embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#             )

#             print(
#                 "DONE. eval_eer: {:.3f}, eval_dcf:{:.5f}, eval_cllr:{:.5f}".format(
#                     eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]
#                 )
#             )

#         if str_to_bool(str(config.get("save_tsne", "True"))):
#             save_embedding_tsne(
#                 data_loader=dev_loader,
#                 model=model,
#                 device=device,
#                 trial_path=dev_trial_path,
#                 save_dir=eval_dir / "tsne_dev",
#                 max_samples=int(config.get("tsne_max_samples", 2000)),
#                 split_name="dev",
#             )
#             if eval_loader is not None and eval_trial_path.exists():
#                 save_embedding_tsne(
#                     data_loader=eval_loader,
#                     model=model,
#                     device=device,
#                     trial_path=eval_trial_path,
#                     save_dir=eval_dir / "tsne_eval",
#                     max_samples=int(config.get("tsne_max_samples", 2000)),
#                     split_name="eval",
#                 )
#         writer.close()
#         sys.exit(0)

#     # =====================================================
#     # TRAIN
#     # =====================================================
#     optim_config["steps_per_epoch"] = len(trn_loader)
#     optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
#     optimizer_swa = SWA(optimizer)

#     best_dev_eer = float("inf")
#     best_dev_dcf = float("inf")
#     best_dev_cllr = float("inf")
#     best_epoch = -1
#     n_swa_update = 0
#     epochs_no_improve = 0

#     history_rows: List[Dict] = []

#     early_stop_enabled = str_to_bool(str(config.get("early_stop", "True")))
#     early_stop_patience = int(config.get("early_stop_patience", 5))
#     early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))

#     save_tsne_flag = str_to_bool(str(config.get("save_tsne", "True")))
#     save_epoch_scores = str_to_bool(str(config.get("save_epoch_scores", "True")))
#     save_epoch_predictions = str_to_bool(str(config.get("save_epoch_predictions", "True")))
#     save_epoch_metric_json = str_to_bool(str(config.get("save_epoch_metric_json", "True")))
#     save_epoch_figures = str_to_bool(str(config.get("save_epoch_figures", "True")))
#     save_embedding_archives = str_to_bool(str(config.get("save_embedding_archives", "True")))
#     embedding_archive_max_samples = int(config.get("embedding_archive_max_samples", 3000))

#     f_log = open(metric_log_path, "a", encoding="utf-8")
#     f_log.write("=" * 20 + "\n")
#     f_log.flush()

#     try:
#         for epoch in range(config["num_epochs"]):
#             print(f"\n========== Epoch {epoch:03d} ==========")

#             running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
#             current_lr = float(optimizer.param_groups[0]["lr"])

#             epoch_score_dir = scores_dir / f"epoch_{epoch:03d}"
#             epoch_pred_dir = predictions_dir / f"epoch_{epoch:03d}"
#             epoch_fig_dir = figures_dir / f"epoch_{epoch:03d}"
#             epoch_metric_dir = metrics_dir / f"epoch_{epoch:03d}"
#             epoch_emb_dir = embedding_dir / f"epoch_{epoch:03d}"

#             for p in [epoch_score_dir, epoch_pred_dir, epoch_fig_dir, epoch_metric_dir, epoch_emb_dir]:
#                 p.mkdir(parents=True, exist_ok=True)

#             dev_metrics = evaluate_split(
#                 split_name="dev",
#                 data_loader=dev_loader,
#                 model=model,
#                 device=device,
#                 trial_path=dev_trial_path,
#                 score_txt_path=epoch_score_dir / "dev_scores.txt" if save_epoch_scores else None,
#                 prediction_csv_path=epoch_pred_dir / "dev_predictions.csv" if save_epoch_predictions else None,
#                 output_dir=epoch_fig_dir / "dev" if save_epoch_figures else epoch_metric_dir / "dev",
#                 save_all_figures=save_epoch_figures,
#                 save_embeddings=save_embedding_archives,
#                 embedding_save_path=epoch_emb_dir / "dev_embeddings.npz",
#                 embedding_max_samples=embedding_archive_max_samples,
#             )

#             eval_metrics = None
#             if eval_loader is not None and eval_trial_path.exists():
#                 eval_metrics = evaluate_split(
#                     split_name="eval",
#                     data_loader=eval_loader,
#                     model=model,
#                     device=device,
#                     trial_path=eval_trial_path,
#                     score_txt_path=epoch_score_dir / "eval_scores.txt" if save_epoch_scores else None,
#                     prediction_csv_path=epoch_pred_dir / "eval_predictions.csv" if save_epoch_predictions else None,
#                     output_dir=epoch_fig_dir / "eval" if save_epoch_figures else epoch_metric_dir / "eval",
#                     save_all_figures=save_epoch_figures,
#                     save_embeddings=save_embedding_archives,
#                     embedding_save_path=epoch_emb_dir / "eval_embeddings.npz",
#                     embedding_max_samples=embedding_archive_max_samples,
#                 )

#             # First epoch dedicated t-SNE
#             if epoch == 0 and save_tsne_flag:
#                 save_embedding_tsne(
#                     data_loader=dev_loader,
#                     model=model,
#                     device=device,
#                     trial_path=dev_trial_path,
#                     save_dir=model_tag / "tsne_epoch0_dev",
#                     max_samples=int(config.get("tsne_max_samples", 2000)),
#                     split_name="dev",
#                 )
#                 if eval_loader is not None and eval_trial_path.exists():
#                     save_embedding_tsne(
#                         data_loader=eval_loader,
#                         model=model,
#                         device=device,
#                         trial_path=eval_trial_path,
#                         save_dir=model_tag / "tsne_epoch0_eval",
#                         max_samples=int(config.get("tsne_max_samples", 2000)),
#                         split_name="eval",
#                     )

#             dev_eer = float(dev_metrics["eer"])
#             dev_dcf = float(dev_metrics["dcf"])
#             dev_cllr = float(dev_metrics["cllr"])

#             print(
#                 "Train Loss: {:.6f} | LR: {:.8f} | dev_eer: {:.4f} | dev_dcf: {:.6f} | dev_cllr: {:.6f}".format(
#                     running_loss, current_lr, dev_eer, dev_dcf, dev_cllr
#                 )
#             )
#             if eval_metrics is not None:
#                 print(
#                     "eval_eer: {:.4f} | eval_dcf: {:.6f} | eval_cllr: {:.6f}".format(
#                         eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]
#                     )
#                 )

#             # TensorBoard
#             writer.add_scalar("loss", running_loss, epoch)
#             writer.add_scalar("lr", current_lr, epoch)
#             writer.add_scalar("dev_eer", dev_eer, epoch)
#             writer.add_scalar("dev_dcf", dev_dcf, epoch)
#             writer.add_scalar("dev_cllr", dev_cllr, epoch)
#             writer.add_scalar("dev_auc", dev_metrics["auc"], epoch)
#             writer.add_scalar("dev_accuracy", dev_metrics["accuracy"], epoch)
#             writer.add_scalar("dev_f1_bonafide", dev_metrics["f1_bonafide"], epoch)

#             if eval_metrics is not None:
#                 writer.add_scalar("eval_eer", eval_metrics["eer"], epoch)
#                 writer.add_scalar("eval_dcf", eval_metrics["dcf"], epoch)
#                 writer.add_scalar("eval_cllr", eval_metrics["cllr"], epoch)
#                 writer.add_scalar("eval_auc", eval_metrics["auc"], epoch)
#                 writer.add_scalar("eval_accuracy", eval_metrics["accuracy"], epoch)
#                 writer.add_scalar("eval_f1_bonafide", eval_metrics["f1_bonafide"], epoch)

#             ckpt_name = f"epoch_{epoch:03d}_devEER_{dev_eer:.6f}.pth"
#             torch.save(model.state_dict(), model_save_path / ckpt_name)

#             row = {
#                 "epoch": int(epoch),
#                 "loss": float(running_loss),
#                 "lr": float(current_lr),
#                 "dev_eer": float(dev_eer),
#                 "dev_dcf": float(dev_dcf),
#                 "dev_cllr": float(dev_cllr),
#                 "dev_auc": float(dev_metrics["auc"]),
#                 "dev_threshold_eer": float(dev_metrics["threshold_eer"]),
#                 "dev_tp": int(dev_metrics["tp"]),
#                 "dev_tn": int(dev_metrics["tn"]),
#                 "dev_fp": int(dev_metrics["fp"]),
#                 "dev_fn": int(dev_metrics["fn"]),
#                 "dev_accuracy": float(dev_metrics["accuracy"]),
#                 "dev_precision_bonafide": float(dev_metrics["precision_bonafide"]),
#                 "dev_recall_bonafide": float(dev_metrics["recall_bonafide"]),
#                 "dev_f1_bonafide": float(dev_metrics["f1_bonafide"]),
#             }

#             if eval_metrics is not None:
#                 row.update(
#                     {
#                         "eval_eer": float(eval_metrics["eer"]),
#                         "eval_dcf": float(eval_metrics["dcf"]),
#                         "eval_cllr": float(eval_metrics["cllr"]),
#                         "eval_auc": float(eval_metrics["auc"]),
#                         "eval_threshold_eer": float(eval_metrics["threshold_eer"]),
#                         "eval_tp": int(eval_metrics["tp"]),
#                         "eval_tn": int(eval_metrics["tn"]),
#                         "eval_fp": int(eval_metrics["fp"]),
#                         "eval_fn": int(eval_metrics["fn"]),
#                         "eval_accuracy": float(eval_metrics["accuracy"]),
#                         "eval_precision_bonafide": float(eval_metrics["precision_bonafide"]),
#                         "eval_recall_bonafide": float(eval_metrics["recall_bonafide"]),
#                         "eval_f1_bonafide": float(eval_metrics["f1_bonafide"]),
#                     }
#                 )

#             history_rows.append(row)
#             append_history_csv(history_csv, row)
#             append_jsonl(history_jsonl, row)

#             if save_epoch_metric_json:
#                 save_json(epoch_metric_dir / "metrics.json", row)

#             # master curves updated every epoch
#             plot_metric_curves(history_rows, figures_dir / "loss_curve.png", ["loss"], ylabel="Loss")
#             plot_metric_curves(history_rows, figures_dir / "lr_curve.png", ["lr"], ylabel="Learning Rate")
#             plot_metric_curves(history_rows, figures_dir / "eer_curve.png", ["dev_eer", "eval_eer"], ylabel="EER")
#             plot_metric_curves(history_rows, figures_dir / "dcf_curve.png", ["dev_dcf", "eval_dcf"], ylabel="minDCF")
#             plot_metric_curves(history_rows, figures_dir / "cllr_curve.png", ["dev_cllr", "eval_cllr"], ylabel="CLLR")
#             plot_metric_curves(history_rows, figures_dir / "auc_curve.png", ["dev_auc", "eval_auc"], ylabel="AUC")
#             plot_metric_curves(
#                 history_rows,
#                 figures_dir / "accuracy_curve.png",
#                 ["dev_accuracy", "eval_accuracy"],
#                 ylabel="Accuracy",
#             )
#             plot_metric_curves(
#                 history_rows,
#                 figures_dir / "f1_curve.png",
#                 ["dev_f1_bonafide", "eval_f1_bonafide"],
#                 ylabel="F1 bonafide",
#             )

#             best_dev_dcf = min(best_dev_dcf, dev_dcf)
#             best_dev_cllr = min(best_dev_cllr, dev_cllr)

#             improved = dev_eer < (best_dev_eer - early_stop_min_delta)
#             if improved:
#                 print(f"Best model found at epoch {epoch}")
#                 best_dev_eer = dev_eer
#                 best_epoch = epoch
#                 epochs_no_improve = 0

#                 torch.save(model.state_dict(), model_save_path / "best_ast.pth")

#                 print(f"Saving epoch {epoch} for SWA")
#                 optimizer_swa.update_swa()
#                 n_swa_update += 1

#                 best_data = {
#                     "best_epoch": int(epoch),
#                     "best_dev_eer": float(dev_eer),
#                     "best_dev_dcf": float(dev_dcf),
#                     "best_dev_cllr": float(dev_cllr),
#                     "best_dev_auc": float(dev_metrics["auc"]),
#                 }
#                 if eval_metrics is not None:
#                     best_data.update(
#                         {
#                             "best_eval_eer": float(eval_metrics["eer"]),
#                             "best_eval_dcf": float(eval_metrics["dcf"]),
#                             "best_eval_cllr": float(eval_metrics["cllr"]),
#                             "best_eval_auc": float(eval_metrics["auc"]),
#                         }
#                     )
#                 save_json(best_json, best_data)
#             else:
#                 epochs_no_improve += 1

#             writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
#             writer.add_scalar("best_dev_dcf", best_dev_dcf, epoch)
#             writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

#             log_line = (
#                 f"epoch={epoch}, loss={running_loss:.6f}, lr={current_lr:.8f}, "
#                 f"dev_eer={dev_eer:.6f}, dev_dcf={dev_dcf:.6f}, dev_cllr={dev_cllr:.6f}, "
#                 f"dev_auc={dev_metrics['auc']:.6f}, dev_acc={dev_metrics['accuracy']:.6f}, "
#                 f"dev_f1={dev_metrics['f1_bonafide']:.6f}"
#             )
#             if eval_metrics is not None:
#                 log_line += (
#                     f", eval_eer={eval_metrics['eer']:.6f}, eval_dcf={eval_metrics['dcf']:.6f}, "
#                     f"eval_cllr={eval_metrics['cllr']:.6f}, eval_auc={eval_metrics['auc']:.6f}, "
#                     f"eval_acc={eval_metrics['accuracy']:.6f}, eval_f1={eval_metrics['f1_bonafide']:.6f}"
#                 )
#             f_log.write(log_line + "\n")
#             f_log.flush()

#             if early_stop_enabled and epochs_no_improve >= early_stop_patience:
#                 print(
#                     f"Early stopping triggered at epoch {epoch}. "
#                     f"Best epoch = {best_epoch}, best dev_eer = {best_dev_eer:.6f}"
#                 )
#                 break

#     finally:
#         f_log.close()
#         writer.close()

#     # =====================================================
#     # FINAL BEST MODEL EVAL
#     # =====================================================
#     best_weight = model_save_path / "best_ast.pth"
#     if best_weight.exists():
#         model.load_state_dict(torch.load(best_weight, map_location=device))

#     final_dir = model_tag / "final_best_model_eval"
#     final_dir.mkdir(parents=True, exist_ok=True)

#     final_dev_metrics = evaluate_split(
#         split_name="dev",
#         data_loader=dev_loader,
#         model=model,
#         device=device,
#         trial_path=dev_trial_path,
#         score_txt_path=final_dir / "dev_best_scores.txt",
#         prediction_csv_path=final_dir / "dev_best_predictions.csv",
#         output_dir=final_dir / "dev",
#         save_all_figures=True,
#         save_embeddings=save_embedding_archives,
#         embedding_save_path=final_dir / "dev_best_embeddings.npz",
#         embedding_max_samples=embedding_archive_max_samples,
#     )

#     final_eval_metrics = None
#     if eval_loader is not None and eval_trial_path.exists():
#         final_eval_metrics = evaluate_split(
#             split_name="eval",
#             data_loader=eval_loader,
#             model=model,
#             device=device,
#             trial_path=eval_trial_path,
#             score_txt_path=final_dir / "eval_best_scores.txt",
#             prediction_csv_path=final_dir / "eval_best_predictions.csv",
#             output_dir=final_dir / "eval",
#             save_all_figures=True,
#             save_embeddings=save_embedding_archives,
#             embedding_save_path=final_dir / "eval_best_embeddings.npz",
#             embedding_max_samples=embedding_archive_max_samples,
#         )

#     if save_tsne_flag:
#         save_embedding_tsne(
#             data_loader=dev_loader,
#             model=model,
#             device=device,
#             trial_path=dev_trial_path,
#             save_dir=model_tag / "tsne_final_dev",
#             max_samples=int(config.get("tsne_max_samples", 2000)),
#             split_name="dev",
#         )
#         if eval_loader is not None and eval_trial_path.exists():
#             save_embedding_tsne(
#                 data_loader=eval_loader,
#                 model=model,
#                 device=device,
#                 trial_path=eval_trial_path,
#                 save_dir=model_tag / "tsne_final_eval",
#                 max_samples=int(config.get("tsne_max_samples", 2000)),
#                 split_name="eval",
#             )

#     save_json(
#         archive_dir / "final_summary.json",
#         {
#             "best_epoch": int(best_epoch),
#             "best_dev_eer": float(best_dev_eer),
#             "best_dev_dcf": float(best_dev_dcf),
#             "best_dev_cllr": float(best_dev_cllr),
#             "swa_updates": int(n_swa_update),
#             "final_dev_metrics": final_dev_metrics,
#             "final_eval_metrics": final_eval_metrics,
#         },
#     )

#     if n_swa_update > 0:
#         print(f"SWA updated {n_swa_update} time(s).")

#     print(f"\nTraining completed. Best epoch: {best_epoch}, Best dev EER: {best_dev_eer:.6f}")


# # =========================================================
# # Model / Loader
# # =========================================================
# def get_model(model_config: Dict, device: torch.device):
#     module = import_module(f"models.{model_config['architecture']}")
#     _model = getattr(module, "Model")
#     model = _model(model_config).to(device)
#     nb_params = sum(param.numel() for param in model.parameters())
#     print(f"no. model params:{nb_params}")
#     return model


# def get_loader(database_path: Path, seed: int, config: dict):
#     trn_database_path = database_path / "flac_T"
#     dev_database_path = database_path / "flac_D"
#     eval_database_path = database_path / "flac_E"

#     trn_list_path = database_path / "ASVspoof5.train.metainfor.txt"
#     dev_trial_path = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
#     print("no. training files:", len(file_train))

#     train_set = TrainDataset(list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path)
#     gen = torch.Generator()
#     gen.manual_seed(seed)

#     trn_loader = DataLoader(
#         train_set,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=gen,
#     )

#     _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
#     print("no. validation files:", len(file_dev))
#     dev_set = TestDataset(list_IDs=file_dev, base_dir=dev_database_path)
#     dev_loader = DataLoader(
#         dev_set,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         drop_last=False,
#         pin_memory=True,
#     )

#     eval_loader = None
#     if eval_trial_path.exists() and eval_database_path.exists():
#         _, file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=False)
#         print("no. eval files:", len(file_eval))
#         eval_set = TestDataset(list_IDs=file_eval, base_dir=eval_database_path)
#         eval_loader = DataLoader(
#             eval_set,
#             batch_size=config["batch_size"],
#             shuffle=False,
#             drop_last=False,
#             pin_memory=True,
#         )
#     else:
#         print("Eval loader skipped: flac_E or ASVspoof5.eval.metainfor.txt not found.")

#     return trn_loader, dev_loader, eval_loader


# # =========================================================
# # Evaluation helpers
# # =========================================================
# def parse_trial_file(trial_path: Path) -> Dict[str, Dict]:
#     info = {}
#     with open(trial_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 6:
#                 continue
#             spk_id, utt_id, p2, p3, p4, key = parts[:6]
#             info[utt_id] = {
#                 "speaker_id": spk_id,
#                 "utt_id": utt_id,
#                 "meta_2": p2,
#                 "meta_3": p3,
#                 "meta_4": p4,
#                 "label_text": key,
#                 "label_int": 1 if key == "bonafide" else 0,
#             }
#     return info


# def collect_predictions(
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     save_embeddings: bool = False,
#     embedding_max_samples: int = 3000,
# ):
#     model.eval()
#     trial_info = parse_trial_file(trial_path)

#     rows = []
#     sampled_embeddings = []
#     sampled_scores = []
#     sampled_labels = []
#     sampled_ids = []

#     use_all_embeddings = embedding_max_samples is None or int(embedding_max_samples) <= 0

#     for batch_x, utt_ids in tqdm(data_loader):
#         batch_x = batch_x.to(device)

#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)
#             probs = torch.softmax(batch_out, dim=1)

#         batch_emb_np = batch_emb.detach().cpu().numpy()
#         batch_logits_np = batch_out.detach().cpu().numpy()
#         batch_probs_np = probs.detach().cpu().numpy()

#         for i, utt_id in enumerate(utt_ids):
#             utt_id = str(utt_id)
#             if utt_id not in trial_info:
#                 continue

#             meta = trial_info[utt_id]
#             prob_spoof = float(batch_probs_np[i, 0])
#             prob_bonafide = float(batch_probs_np[i, 1])
#             logit_spoof = float(batch_logits_np[i, 0])
#             logit_bonafide = float(batch_logits_np[i, 1])

#             pred_int = 1 if prob_bonafide >= prob_spoof else 0
#             true_int = int(meta["label_int"])

#             rows.append(
#                 {
#                     "speaker_id": meta["speaker_id"],
#                     "utt_id": utt_id,
#                     "true_label_text": meta["label_text"],
#                     "true_label_int": true_int,
#                     "pred_label_text": "bonafide" if pred_int == 1 else "spoof",
#                     "pred_label_int": pred_int,
#                     "is_correct": int(pred_int == true_int),
#                     "logit_spoof": logit_spoof,
#                     "logit_bonafide": logit_bonafide,
#                     "prob_spoof": prob_spoof,
#                     "prob_bonafide": prob_bonafide,
#                     "score": prob_bonafide,
#                 }
#             )

#             if save_embeddings and (use_all_embeddings or len(sampled_ids) < embedding_max_samples):
#                 sampled_embeddings.append(batch_emb_np[i])
#                 sampled_scores.append(prob_bonafide)
#                 sampled_labels.append(true_int)
#                 sampled_ids.append(utt_id)

#     return rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids


# def save_score_txt(score_txt_path: Path, rows: List[Dict]) -> None:
#     with open(score_txt_path, "w", encoding="utf-8") as fh:
#         for r in rows:
#             fh.write(f"{r['speaker_id']} {r['utt_id']} {r['score']} {r['true_label_text']}\n")


# def save_prediction_csv(prediction_csv_path: Path, rows: List[Dict]) -> None:
#     if not rows:
#         return
#     fieldnames = list(rows[0].keys())
#     with open(prediction_csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
#         writer.writeheader()
#         writer.writerows(rows)


# def compute_binary_metrics_from_rows(rows: List[Dict]) -> Dict:
#     if len(rows) == 0:
#         raise ValueError("No rows available for metric computation.")

#     scores = np.array([r["score"] for r in rows], dtype=np.float64)
#     labels = np.array([r["true_label_int"] for r in rows], dtype=np.int64)

#     bona_scores = scores[labels == 1]
#     spoof_scores = scores[labels == 0]

#     eer, frr, far, thresholds = compute_eer(bona_scores, spoof_scores)
#     min_index = int(np.argmin(np.abs(frr - far)))
#     threshold_eer = float(thresholds[min_index])

#     pred = (scores >= threshold_eer).astype(np.int64)

#     tp = int(np.sum((pred == 1) & (labels == 1)))
#     tn = int(np.sum((pred == 0) & (labels == 0)))
#     fp = int(np.sum((pred == 1) & (labels == 0)))
#     fn = int(np.sum((pred == 0) & (labels == 1)))

#     accuracy = float((tp + tn) / max(1, len(labels)))
#     precision_bonafide = float(tp / max(1, tp + fp))
#     recall_bonafide = float(tp / max(1, tp + fn))
#     f1_bonafide = float(
#         (2 * precision_bonafide * recall_bonafide) / max(1e-12, precision_bonafide + recall_bonafide)
#     )

#     # robust ROC for edge cases
#     if len(np.unique(labels)) < 2:
#         fpr = np.array([0.0, 1.0], dtype=np.float64)
#         tpr = np.array([0.0, 1.0], dtype=np.float64)
#         roc_thresholds = np.array([np.inf, -np.inf], dtype=np.float64)
#         roc_auc = 0.5
#     else:
#         fpr, tpr, roc_thresholds = roc_curve(labels, scores)
#         roc_auc = float(auc(fpr, tpr))

#     return {
#         "scores": scores,
#         "labels": labels,
#         "bona_scores": bona_scores,
#         "spoof_scores": spoof_scores,
#         "eer": float(eer),
#         "frr": frr,
#         "far": far,
#         "thresholds": thresholds,
#         "threshold_eer": threshold_eer,
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#         "accuracy": accuracy,
#         "precision_bonafide": precision_bonafide,
#         "recall_bonafide": recall_bonafide,
#         "f1_bonafide": f1_bonafide,
#         "fpr": fpr,
#         "tpr": tpr,
#         "roc_thresholds": roc_thresholds,
#         "auc": roc_auc,
#     }


# def evaluate_split(
#     split_name: str,
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     score_txt_path: Optional[Path],
#     prediction_csv_path: Optional[Path],
#     output_dir: Path,
#     save_all_figures: bool = True,
#     save_embeddings: bool = False,
#     embedding_save_path: Optional[Path] = None,
#     embedding_max_samples: int = 3000,
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)

#     rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids = collect_predictions(
#         data_loader=data_loader,
#         model=model,
#         device=device,
#         trial_path=trial_path,
#         save_embeddings=save_embeddings,
#         embedding_max_samples=embedding_max_samples,
#     )

#     if len(rows) == 0:
#         raise ValueError(f"No prediction rows generated for split={split_name}. Check trial IDs and dataset IDs.")

#     if score_txt_path is None:
#         score_txt_path = output_dir / f"{split_name}_scores.txt"
#     save_score_txt(score_txt_path, rows)

#     if prediction_csv_path is not None:
#         save_prediction_csv(prediction_csv_path, rows)

#     dcf, eer_from_metric_fn, cllr = calculate_minDCF_EER_CLLR(
#         cm_scores_file=score_txt_path,
#         output_file=output_dir / f"{split_name}_DCF_EER.txt",
#         printout=False,
#     )

#     stats = compute_binary_metrics_from_rows(rows)

#     metrics = {
#         "dcf": float(dcf),
#         "eer": float(eer_from_metric_fn),
#         "cllr": float(cllr),
#         "threshold_eer": float(stats["threshold_eer"]),
#         "tp": int(stats["tp"]),
#         "tn": int(stats["tn"]),
#         "fp": int(stats["fp"]),
#         "fn": int(stats["fn"]),
#         "accuracy": float(stats["accuracy"]),
#         "precision_bonafide": float(stats["precision_bonafide"]),
#         "recall_bonafide": float(stats["recall_bonafide"]),
#         "f1_bonafide": float(stats["f1_bonafide"]),
#         "auc": float(stats["auc"]),
#         "n_samples": int(len(rows)),
#         "n_bonafide": int(np.sum(stats["labels"] == 1)),
#         "n_spoof": int(np.sum(stats["labels"] == 0)),
#     }

#     save_json(output_dir / f"{split_name}_metrics.json", metrics)

#     if save_all_figures:
#         save_confusion_plot(
#             tp=stats["tp"],
#             tn=stats["tn"],
#             fp=stats["fp"],
#             fn=stats["fn"],
#             title=f"{split_name.upper()} confusion matrix",
#             save_path=output_dir / f"{split_name}_confusion_matrix.png",
#         )
#         save_confusion_json(
#             tp=stats["tp"],
#             tn=stats["tn"],
#             fp=stats["fp"],
#             fn=stats["fn"],
#             save_path=output_dir / f"{split_name}_confusion_matrix.json",
#         )
#         save_roc_curve(
#             fpr=stats["fpr"],
#             tpr=stats["tpr"],
#             roc_auc=stats["auc"],
#             save_png=output_dir / f"{split_name}_roc_curve.png",
#             save_csv=output_dir / f"{split_name}_roc_curve.csv",
#             split_name=split_name,
#         )
#         save_det_curve_plot(
#             bona_scores=stats["bona_scores"],
#             spoof_scores=stats["spoof_scores"],
#             save_png=output_dir / f"{split_name}_det_curve.png",
#             save_csv=output_dir / f"{split_name}_det_curve.csv",
#             split_name=split_name,
#         )
#         save_score_histogram(
#             bona_scores=stats["bona_scores"],
#             spoof_scores=stats["spoof_scores"],
#             threshold=stats["threshold_eer"],
#             save_path=output_dir / f"{split_name}_score_histogram.png",
#             split_name=split_name,
#         )
#         save_threshold_sweep_csv(
#             scores=stats["scores"],
#             labels=stats["labels"],
#             thresholds=stats["thresholds"],
#             save_path=output_dir / f"{split_name}_threshold_sweep.csv",
#         )

#     if save_embeddings and embedding_save_path is not None and len(sampled_ids) > 0:
#         np.savez_compressed(
#             embedding_save_path,
#             utt_ids=np.array(sampled_ids),
#             labels=np.array(sampled_labels),
#             scores=np.array(sampled_scores),
#             embeddings=np.array(sampled_embeddings),
#         )

#     return metrics


# # =========================================================
# # Plot helpers
# # =========================================================
# def save_confusion_plot(tp: int, tn: int, fp: int, fn: int, title: str, save_path: Path) -> None:
#     matrix = np.array([[tp, fn], [fp, tn]], dtype=np.int64)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(matrix)
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["Pred Bonafide", "Pred Spoof"])
#     ax.set_yticklabels(["True Bonafide", "True Spoof"])
#     ax.set_title(title)

#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

#     fig.colorbar(im, ax=ax)
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=220, bbox_inches="tight")
#     plt.close(fig)


# def save_confusion_json(tp: int, tn: int, fp: int, fn: int, save_path: Path) -> None:
#     save_json(save_path, {"tp": tp, "tn": tn, "fp": fp, "fn": fn})


# def save_roc_curve(fpr, tpr, roc_auc: float, save_png: Path, save_csv: Path, split_name: str) -> None:
#     with open(save_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["fpr", "tpr"])
#         for x, y in zip(fpr, tpr):
#             writer.writerow([float(x), float(y)])

#     fig = plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
#     plt.plot([0, 1], [0, 1], linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"{split_name.upper()} ROC Curve")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     fig.savefig(save_png, dpi=220, bbox_inches="tight")
#     plt.close(fig)


# def save_det_curve_plot(bona_scores, spoof_scores, save_png: Path, save_csv: Path, split_name: str) -> None:
#     frr, far, thresholds = compute_det_curve(bona_scores, spoof_scores)

#     frr = np.asarray(frr)
#     far = np.asarray(far)
#     thresholds = np.asarray(thresholds)

#     n = min(len(thresholds), max(0, len(frr) - 1), max(0, len(far) - 1))
#     thr_vals = thresholds[:n]
#     frr_vals = frr[1:1 + n]
#     far_vals = far[1:1 + n]

#     with open(save_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["threshold", "frr", "far"])
#         for thr, x, y in zip(thr_vals, frr_vals, far_vals):
#             writer.writerow([float(thr), float(x), float(y)])

#     fig = plt.figure(figsize=(6, 5))
#     if len(far_vals) > 0 and len(frr_vals) > 0:
#         plt.plot(far_vals, frr_vals)
#     plt.xlabel("FAR")
#     plt.ylabel("FRR")
#     plt.title(f"{split_name.upper()} DET Curve")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     fig.savefig(save_png, dpi=220, bbox_inches="tight")
#     plt.close(fig)


# def save_score_histogram(bona_scores, spoof_scores, threshold: float, save_path: Path, split_name: str) -> None:
#     fig = plt.figure(figsize=(8, 5))
#     plt.hist(spoof_scores, bins=60, alpha=0.6, label="spoof")
#     plt.hist(bona_scores, bins=60, alpha=0.6, label="bonafide")
#     plt.axvline(threshold, linestyle="--", label=f"EER thr={threshold:.4f}")
#     plt.xlabel("Bonafide score")
#     plt.ylabel("Count")
#     plt.title(f"{split_name.upper()} Score Histogram")
#     plt.legend()
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=220, bbox_inches="tight")
#     plt.close(fig)


# def save_threshold_sweep_csv(scores, labels, thresholds, save_path: Path) -> None:
#     with open(save_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "threshold", "tp", "tn", "fp", "fn",
#             "accuracy", "precision_bonafide", "recall_bonafide", "f1_bonafide"
#         ])
#         for thr in thresholds:
#             pred = (scores >= thr).astype(np.int64)
#             tp = int(np.sum((pred == 1) & (labels == 1)))
#             tn = int(np.sum((pred == 0) & (labels == 0)))
#             fp = int(np.sum((pred == 1) & (labels == 0)))
#             fn = int(np.sum((pred == 0) & (labels == 1)))

#             acc = float((tp + tn) / max(1, len(labels)))
#             prec = float(tp / max(1, tp + fp))
#             rec = float(tp / max(1, tp + fn))
#             f1 = float((2 * prec * rec) / max(1e-12, prec + rec))
#             writer.writerow([float(thr), tp, tn, fp, fn, acc, prec, rec, f1])


# def plot_metric_curves(history_rows: List[Dict], save_path: Path, metric_keys: List[str], ylabel: str = "Value") -> None:
#     if not history_rows:
#         return

#     epochs = [int(row["epoch"]) for row in history_rows]
#     fig = plt.figure(figsize=(8, 5))
#     plotted = False

#     for key in metric_keys:
#         values = [row.get(key, None) for row in history_rows]
#         if all(v is None for v in values):
#             continue

#         x = [e for e, v in zip(epochs, values) if v is not None]
#         y = [float(v) for v in values if v is not None]
#         if len(x) > 0:
#             plt.plot(x, y, marker="o", label=key)
#             plotted = True

#     if not plotted:
#         plt.close(fig)
#         return

#     plt.xlabel("Epoch")
#     plt.ylabel(ylabel)
#     plt.title(save_path.stem.replace("_", " ").upper())
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=220, bbox_inches="tight")
#     plt.close(fig)


# # =========================================================
# # Embeddings / t-SNE
# # =========================================================
# def _parse_trial_labels(trial_path: Path):
#     labels = {}
#     with open(trial_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 6:
#                 continue
#             _, utt_id, _, _, _, key = parts[:6]
#             labels[utt_id] = 1 if key == "bonafide" else 0
#     return labels


# def save_embedding_tsne(
#     data_loader: DataLoader,
#     model,
#     device: torch.device,
#     trial_path: Path,
#     save_dir: Path,
#     max_samples: int = 2000,
#     split_name: str = "dev",
# ) -> None:
#     save_dir.mkdir(parents=True, exist_ok=True)
#     label_map = _parse_trial_labels(trial_path)

#     model.eval()
#     embeddings = []
#     scores = []
#     utt_ids = []

#     use_all = max_samples is None or int(max_samples) <= 0

#     for batch_x, batch_utt_id in tqdm(data_loader, desc=f"extract_tsne_{split_name}"):
#         batch_x = batch_x.to(device)
#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)
#             batch_score = torch.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
#             batch_emb = batch_emb.detach().cpu().numpy()

#         embeddings.append(batch_emb)
#         scores.append(batch_score)
#         utt_ids.extend([str(u) for u in batch_utt_id])

#         if (not use_all) and len(utt_ids) >= max_samples:
#             break

#     if len(utt_ids) < 2:
#         print(f"Not enough samples for t-SNE on {split_name}.")
#         return

#     embeddings = np.concatenate(embeddings, axis=0)
#     scores = np.concatenate(scores, axis=0)

#     if not use_all:
#         embeddings = embeddings[:max_samples]
#         scores = scores[:max_samples]
#         utt_ids = utt_ids[:max_samples]

#     labels = np.array([label_map.get(u, -1) for u in utt_ids])

#     np.savez_compressed(
#         save_dir / f"{split_name}_tsne_source_embeddings.npz",
#         utt_ids=np.array(utt_ids),
#         labels=labels,
#         scores=scores,
#         embeddings=embeddings,
#     )

#     n_samples = len(utt_ids)
#     perplexity = min(30, max(5, n_samples - 1))

#     tsne = TSNE(
#         n_components=2,
#         perplexity=perplexity,
#         init="pca",
#         learning_rate="auto",
#         random_state=42,
#     )
#     coords = tsne.fit_transform(embeddings)

#     csv_path = save_dir / f"{split_name}_tsne_points.csv"
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["utt_id", "label", "bonafide_score", "tsne_x", "tsne_y"])
#         for utt_id, label, score, (x, y) in zip(utt_ids, labels, scores, coords):
#             writer.writerow([utt_id, int(label), float(score), float(x), float(y)])

#     png_path = save_dir / f"{split_name}_tsne_plot.png"
#     fig = plt.figure(figsize=(8, 6))

#     spoof_mask = labels == 0
#     bona_mask = labels == 1
#     unknown_mask = labels == -1

#     if spoof_mask.any():
#         plt.scatter(coords[spoof_mask, 0], coords[spoof_mask, 1], s=10, alpha=0.7, label="spoof")
#     if bona_mask.any():
#         plt.scatter(coords[bona_mask, 0], coords[bona_mask, 1], s=10, alpha=0.7, label="bonafide")
#     if unknown_mask.any():
#         plt.scatter(coords[unknown_mask, 0], coords[unknown_mask, 1], s=10, alpha=0.7, label="unknown")

#     plt.title(f"{split_name.upper()} set AST embeddings (t-SNE)")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.legend()
#     plt.tight_layout()
#     fig.savefig(png_path, dpi=220, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Saved t-SNE using {n_samples} samples for {split_name}")
#     print(f"Saved t-SNE CSV to {csv_path}")
#     print(f"Saved t-SNE plot to {png_path}")


# # =========================================================
# # Logging helpers
# # =========================================================
# def append_history_csv(csv_path: Path, row: Dict) -> None:
#     row = dict(row)

#     if csv_path.exists():
#         with open(csv_path, "r", newline="", encoding="utf-8") as f:
#             reader = csv.reader(f)
#             existing_header = next(reader, None)

#         if existing_header is None:
#             existing_header = list(row.keys())
#     else:
#         existing_header = list(row.keys())

#     for key in existing_header:
#         if key not in row:
#             row[key] = ""

#     for key in row.keys():
#         if key not in existing_header:
#             existing_header.append(key)

#     rewrite_needed = False
#     if csv_path.exists():
#         with open(csv_path, "r", newline="", encoding="utf-8") as f:
#             reader = csv.reader(f)
#             current_header = next(reader, None)
#         if current_header != existing_header:
#             rewrite_needed = True

#     if rewrite_needed:
#         with open(csv_path, "r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             old_rows = list(reader)

#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=existing_header)
#             writer.writeheader()
#             for old_row in old_rows:
#                 normalized = {k: old_row.get(k, "") for k in existing_header}
#                 writer.writerow(normalized)
#             writer.writerow({k: row.get(k, "") for k in existing_header})
#     else:
#         write_header = not csv_path.exists()
#         with open(csv_path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=existing_header)
#             if write_header:
#                 writer.writeheader()
#             writer.writerow({k: row.get(k, "") for k in existing_header})


# def append_jsonl(path: Path, row: Dict) -> None:
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row) + "\n")


# def save_json(path: Path, data: Dict) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)


# # =========================================================
# # Train
# # =========================================================
# def train_epoch(
#     trn_loader: DataLoader,
#     model,
#     optim: Union[torch.optim.SGD, torch.optim.Adam],
#     device: torch.device,
#     scheduler,
#     config: dict,
# ):
#     running_loss = 0.0
#     num_total = 0.0
#     model.train()

#     weight = torch.FloatTensor([0.1, 0.9]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)

#     for batch_x, batch_y in tqdm(trn_loader):
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.view(-1).type(torch.int64).to(device)

#         batch_size = batch_x.size(0)
#         num_total += batch_size

#         _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
#         batch_loss = criterion(batch_out, batch_y)
#         running_loss += batch_loss.item() * batch_size

#         optim.zero_grad()
#         batch_loss.backward()
#         optim.step()

#         if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
#             if scheduler is not None:
#                 scheduler.step()
#         elif scheduler is None:
#             pass
#         else:
#             raise ValueError(f"scheduler error, got:{scheduler}")

#     running_loss /= max(1.0, num_total)
#     return float(running_loss)


# # =========================================================
# # CLI
# # =========================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="ASVspoof detection system")
#     parser.add_argument("--config", dest="config", type=str, help="configuration file", required=True)
#     parser.add_argument(
#         "--output_dir",
#         dest="output_dir",
#         type=str,
#         help="output directory for results",
#         default="./exp_result",
#     )
#     parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
#     parser.add_argument("--eval", action="store_true", help="evaluate given model and exit")
#     parser.add_argument("--comment", type=str, default=None, help="comment to describe the saved model")
#     parser.add_argument(
#         "--eval_model_weights",
#         type=str,
#         default=None,
#         help="path to the model weight file (can also be given in config file)",
#     )
#     main(parser.parse_args())






# # below 
# # epoch=0, loss=0.044415, lr=0.00000994, dev_eer=0.158292, dev_dcf=0.340060, dev_cllr=0.853132, dev_auc=0.892489, dev_acc=0.841716, dev_f1=0.702771, eval_eer=0.352482, eval_dcf=0.874755, eval_cllr=0.986607, eval_auc=0.683514, eval_acc=0.647520, eval_f1=0.428079
# # epoch=1, loss=0.018833, lr=0.00000978, dev_eer=0.149958, dev_dcf=0.280087, dev_cllr=0.832393, dev_auc=0.898622, dev_acc=0.850052, dev_f1=0.715954, eval_eer=0.365857, eval_dcf=0.801733, eval_cllr=0.985473, eval_auc=0.670351, eval_acc=0.634141, eval_f1=0.413909
# # epoch=2, loss=0.012929, lr=0.00000951, dev_eer=0.147060, dev_dcf=0.278819, dev_cllr=0.841535, dev_auc=0.889301, dev_acc=0.852933, dev_f1=0.720562, eval_eer=0.380661, eval_dcf=0.871033, eval_cllr=1.001747, eval_auc=0.645627, eval_acc=0.619340, eval_f1=0.398649
# # epoch=3, loss=0.009708, lr=0.00000914, dev_eer=0.160217, dev_dcf=0.292375, dev_cllr=0.837994, dev_auc=0.889157, dev_acc=0.839786, dev_f1=0.699755, eval_eer=0.379685, eval_dcf=0.873594, eval_cllr=0.994244, eval_auc=0.649129, eval_acc=0.620317, eval_f1=0.399642
# # epoch=4, loss=0.008441, lr=0.00000868, dev_eer=0.162634, dev_dcf=0.293779, dev_cllr=0.839257, dev_auc=0.886242, dev_acc=0.837374, dev_f1=0.695995, eval_eer=0.382101, eval_dcf=0.851553, eval_cllr=1.000191, eval_auc=0.648669, eval_acc=0.617902, eval_f1=0.397188
# # epoch=5, loss=0.007483, lr=0.00000815, dev_eer=0.158812, dev_dcf=0.301323, dev_cllr=0.836880, dev_auc=0.893594, dev_acc=0.841191, dev_f1=0.701952, eval_eer=0.389362, eval_dcf=0.868065, eval_cllr=1.003845, eval_auc=0.648602, eval_acc=0.610640, eval_f1=0.389873
# # epoch=6, loss=0.007024, lr=0.00000754, dev_eer=0.171857, dev_dcf=0.323032, dev_cllr=0.849022, dev_auc=0.887976, dev_acc=0.828136, dev_f1=0.681774, eval_eer=0.386342, eval_dcf=0.860799, eval_cllr=1.007025, eval_auc=0.648656, eval_acc=0.613660, eval_f1=0.392903
# # epoch=7, loss=0.005688, lr=0.00000689, dev_eer=0.191576, dev_dcf=0.376760, dev_cllr=0.865170, dev_auc=0.876495, dev_acc=0.808434, dev_f1=0.652340, eval_eer=0.398477, eval_dcf=0.852694, eval_cllr=1.013185, eval_auc=0.636233, eval_acc=0.601525, eval_f1=0.380831



# """
# main.py — ASVspoof5 anti-spoofing training with novel approach.

# Key components added / changed vs baseline:
#   1. GAMOptimizer  — Gradient Norm Aware Minimization (CVPR 2023 / SZU-AFS)
#      Wraps AdamW.  Two forward+backward passes per step (like SAM).
#      Empirically closed the dev-eval gap to 4.04% EER on ASVspoof5.
#   2. Differential LR  — backbone / fusion / heads at separate LRs.
#   3. Layer-unfreeze schedule  — bottom encoder blocks unlocked after warm-up.
#   4. Combined CE + OC-Softmax loss  — from model.compute_loss().
#   5. Gradient accumulation  — effective batch without extra GPU memory.
#   6. All existing logging / eval / t-SNE helpers are retained unchanged.
# """

# import argparse
# import csv
# import json
# import os
# import sys
# import warnings
# from importlib import import_module
# from pathlib import Path
# from shutil import copy
# from typing import Dict, List, Optional, Union

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.manifold import TSNE
# from sklearn.metrics import roc_curve, auc
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchcontrib.optim import SWA
# from tqdm import tqdm

# from data_utils import TrainDataset, TestDataset, genSpoof_list
# from eval.calculate_metrics import calculate_minDCF_EER_CLLR
# from eval.calculate_modules import compute_eer, compute_det_curve
# from utils import create_optimizer, seed_worker, set_seed, str_to_bool

# warnings.filterwarnings("ignore", category=FutureWarning)


# # ============================================================
# #  GAM Optimizer  (Zhang et al. CVPR 2023)
# #  Gradient Norm Aware Minimization — first-order flatness seeking.
# #  Proven on ASVspoof5 to close the generalisation gap.
# # ============================================================
# class GAMOptimizer:
#     """
#     Wraps any base optimizer (AdamW recommended) and applies the GAM
#     perturbation step:
#       1. Compute gradient at θ.
#       2. Perturb θ → θ + ρ * grad / ||grad||  (move toward high-gradient region).
#       3. Compute gradient at θ_perturbed.
#       4. Restore θ and apply the perturbed gradient with the base optimizer.

#     rho          : perturbation radius (default 0.05)
#     grad_rho     : gradient norm perturbation radius (default 0.02)
#     adaptive     : use adaptive per-parameter rho (like ASAM)

#     Reference:
#         Zhang et al., "Gradient Norm Aware Minimization Seeks First-Order
#         Flatness and Improves Generalization", CVPR 2023.
#     """

#     def __init__(
#         self,
#         params,
#         base_optimizer: torch.optim.Optimizer,
#         rho:      float = 0.05,
#         grad_rho: float = 0.02,
#         adaptive: bool  = False,
#     ):
#         self.base_optimizer = base_optimizer
#         self.rho      = rho
#         self.grad_rho = grad_rho
#         self.adaptive = adaptive
#         self.param_groups = base_optimizer.param_groups

#     @torch.no_grad()
#     def _gradient_norm(self) -> torch.Tensor:
#         shared_device = self.param_groups[0]["params"][0].device
#         norm = torch.norm(
#             torch.stack([
#                 ((p.grad if not self.adaptive else p.grad * p) ** 2).norm(p=2)
#                 for group in self.param_groups
#                 for p in group["params"]
#                 if p.grad is not None
#             ]),
#             p=2,
#         )
#         return norm

#     @torch.no_grad()
#     def first_step(self) -> None:
#         """Perturb weights in the direction of gradient."""
#         grad_norm = self._gradient_norm() + 1e-12
#         for group in self.param_groups:
#             scale = self.rho / grad_norm
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 e_w = (p ** 2 * p.grad if self.adaptive else p.grad) * scale
#                 p.add_(e_w)               # perturb
#                 p.grad_store = e_w        # store perturbation for restoration

#     @torch.no_grad()
#     def second_step(self) -> None:
#         """Restore weights and apply update using perturbed gradient."""
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 if hasattr(p, "grad_store"):
#                     p.sub_(p.grad_store)  # restore
#                     del p.grad_store
#         self.base_optimizer.step()
#         self.base_optimizer.zero_grad()

#     def zero_grad(self) -> None:
#         self.base_optimizer.zero_grad()

#     def state_dict(self) -> dict:
#         return self.base_optimizer.state_dict()

#     def load_state_dict(self, state: dict) -> None:
#         self.base_optimizer.load_state_dict(state)


# # ============================================================
# #  Main
# # ============================================================
# def main(args: argparse.Namespace) -> None:
#     with open(args.config, "r") as f_json:
#         config = json.loads(f_json.read())

#     model_config = config["model_config"]
#     optim_config = config["optim_config"]
#     optim_config["epochs"] = config["num_epochs"]

#     # ---- defaults ----
#     config.setdefault("eval_all_best",    "True")
#     config.setdefault("freq_aug",         "False")   # no augmentation (novel model approach)

#     # GAM
#     config.setdefault("use_gam",          "True")
#     config.setdefault("gam_rho",           0.05)
#     config.setdefault("gam_grad_rho",      0.02)

#     # Gradient accumulation
#     config.setdefault("grad_accum_steps",  4)

#     # Differential LR
#     config.setdefault("backbone_lr",       1e-5)
#     config.setdefault("fusion_lr",         5e-5)
#     config.setdefault("head_lr",           1e-4)

#     # Layer unfreeze schedule
#     config.setdefault("unfreeze_epoch",    5)

#     # SWA start epoch
#     config.setdefault("swa_start_epoch",   5)

#     # Early stopping
#     config.setdefault("early_stop",          "True")
#     config.setdefault("early_stop_patience",  7)
#     config.setdefault("early_stop_min_delta", 0.001)

#     # Logging
#     config.setdefault("save_tsne",                   "True")
#     config.setdefault("tsne_max_samples",             2000)
#     config.setdefault("save_epoch_scores",           "True")
#     config.setdefault("save_epoch_predictions",      "True")
#     config.setdefault("save_epoch_metric_json",      "True")
#     config.setdefault("save_epoch_figures",          "True")
#     config.setdefault("save_embedding_archives",     "True")
#     config.setdefault("embedding_archive_max_samples", 3000)

#     set_seed(args.seed, config)

#     output_dir    = Path(args.output_dir)
#     database_path = Path(config["database_path"])
#     dev_trial_path  = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     model_tag = "{}_ep{}_bs{}".format(
#         os.path.splitext(os.path.basename(args.config))[0],
#         config["num_epochs"],
#         config["batch_size"],
#     )
#     if args.comment:
#         model_tag += f"_{args.comment}"
#     model_tag = output_dir / model_tag

#     # ---- paths ----
#     model_save_path = model_tag / "weights"
#     scores_dir      = model_tag / "scores"
#     predictions_dir = model_tag / "predictions"
#     figures_dir     = model_tag / "figures"
#     metrics_dir     = model_tag / "metrics"
#     embedding_dir   = model_tag / "embeddings"
#     archive_dir     = model_tag / "artifacts"
#     history_csv     = model_tag / "training_history.csv"
#     history_jsonl   = model_tag / "training_history.jsonl"
#     best_json       = model_tag / "best_metrics.json"
#     run_summary_json = model_tag / "run_summary.json"
#     metric_log_path  = model_tag / "metric_log.txt"

#     writer = SummaryWriter(str(model_tag))
#     for p in [model_save_path, scores_dir, predictions_dir, figures_dir,
#               metrics_dir, embedding_dir, archive_dir]:
#         p.mkdir(parents=True, exist_ok=True)

#     copy(args.config, model_tag / "config.conf")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")
#     if device == "cpu":
#         raise ValueError("GPU not detected!")

#     model = get_model(model_config, device)
#     trn_loader, dev_loader, eval_loader = get_loader(database_path, args.seed, config)

#     save_json(run_summary_json, {
#         "config_path":          args.config,
#         "output_dir":           str(model_tag),
#         "database_path":        str(database_path),
#         "num_epochs_requested": int(config["num_epochs"]),
#         "batch_size":           int(config["batch_size"]),
#         "grad_accum_steps":     int(config["grad_accum_steps"]),
#         "effective_batch":      int(config["batch_size"]) * int(config["grad_accum_steps"]),
#         "use_gam":              config["use_gam"],
#         "seed":                 int(args.seed),
#         "device":               device,
#         "comment":              args.comment,
#     })

#     # ============================================================
#     # EVAL ONLY
#     # ============================================================
#     if args.eval:
#         model_path = args.eval_model_weights or config["model_path"]
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"Model loaded : {model_path}")
#         eval_dir = model_tag / "eval_loaded_model"
#         eval_dir.mkdir(parents=True, exist_ok=True)

#         dev_metrics = evaluate_split(
#             split_name="dev", data_loader=dev_loader, model=model, device=device,
#             trial_path=dev_trial_path,
#             score_txt_path=eval_dir / "dev_scores.txt",
#             prediction_csv_path=eval_dir / "dev_predictions.csv",
#             output_dir=eval_dir / "dev", save_all_figures=True,
#             save_embeddings=True,
#             embedding_save_path=eval_dir / "dev_embeddings.npz",
#             embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#         )
#         print("dev_eer: {:.3f}  dev_dcf: {:.5f}  dev_cllr: {:.5f}".format(
#             dev_metrics["eer"], dev_metrics["dcf"], dev_metrics["cllr"]))

#         if eval_loader is not None and eval_trial_path.exists():
#             eval_metrics = evaluate_split(
#                 split_name="eval", data_loader=eval_loader, model=model, device=device,
#                 trial_path=eval_trial_path,
#                 score_txt_path=eval_dir / "eval_scores.txt",
#                 prediction_csv_path=eval_dir / "eval_predictions.csv",
#                 output_dir=eval_dir / "eval", save_all_figures=True,
#                 save_embeddings=True,
#                 embedding_save_path=eval_dir / "eval_embeddings.npz",
#                 embedding_max_samples=int(config.get("embedding_archive_max_samples", 3000)),
#             )
#             print("eval_eer: {:.3f}  eval_dcf: {:.5f}  eval_cllr: {:.5f}".format(
#                 eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]))

#         writer.close()
#         sys.exit(0)

#     # ============================================================
#     # TRAIN
#     # ============================================================
#     use_gam          = str_to_bool(str(config.get("use_gam", "True")))
#     grad_accum_steps = int(config.get("grad_accum_steps", 4))
#     unfreeze_epoch   = int(config.get("unfreeze_epoch", 5))
#     swa_start_epoch  = int(config.get("swa_start_epoch", 5))
#     backbone_lr      = float(config.get("backbone_lr", 1e-5))
#     fusion_lr        = float(config.get("fusion_lr", 5e-5))
#     head_lr          = float(config.get("head_lr", 1e-4))

#     optim_config["steps_per_epoch"] = len(trn_loader)

#     # Build base AdamW with differential LR groups
#     if hasattr(model, "param_groups"):
#         pg = model.param_groups(backbone_lr=backbone_lr, fusion_lr=fusion_lr, head_lr=head_lr)
#     else:
#         pg = model.parameters()

#     base_optimizer = torch.optim.AdamW(
#         pg,
#         weight_decay=float(optim_config.get("weight_decay", 1e-4)),
#     )

#     # Cosine LR scheduler over total steps
#     total_steps = len(trn_loader) * config["num_epochs"] // grad_accum_steps
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         base_optimizer, T_max=total_steps, eta_min=1e-7
#     )

#     # Wrap with GAM
#     if use_gam:
#         optimizer = GAMOptimizer(
#             params=pg,
#             base_optimizer=base_optimizer,
#             rho=float(config.get("gam_rho", 0.05)),
#             grad_rho=float(config.get("gam_grad_rho", 0.02)),
#         )
#         print(f"[Train] GAM optimizer enabled (rho={config['gam_rho']}).")
#     else:
#         optimizer = base_optimizer
#         print("[Train] Using plain AdamW (GAM disabled).")

#     optimizer_swa = SWA(base_optimizer)

#     # CE loss with class weights
#     spoof_w  = float(config.get("ce_weight_spoof",    0.1))
#     bona_w   = float(config.get("ce_weight_bonafide", 0.9))
#     ce_weight = torch.FloatTensor([spoof_w, bona_w]).to(device)
#     ce_criterion = nn.CrossEntropyLoss(weight=ce_weight)

#     best_dev_eer  = float("inf")
#     best_dev_dcf  = float("inf")
#     best_dev_cllr = float("inf")
#     best_epoch    = -1
#     n_swa_update  = 0
#     epochs_no_improve = 0
#     history_rows: List[Dict] = []

#     early_stop_enabled   = str_to_bool(str(config.get("early_stop", "True")))
#     early_stop_patience  = int(config.get("early_stop_patience", 7))
#     early_stop_min_delta = float(config.get("early_stop_min_delta", 0.001))
#     save_tsne_flag            = str_to_bool(str(config.get("save_tsne", "True")))
#     save_epoch_scores         = str_to_bool(str(config.get("save_epoch_scores", "True")))
#     save_epoch_predictions    = str_to_bool(str(config.get("save_epoch_predictions", "True")))
#     save_epoch_metric_json    = str_to_bool(str(config.get("save_epoch_metric_json", "True")))
#     save_epoch_figures        = str_to_bool(str(config.get("save_epoch_figures", "True")))
#     save_embedding_archives   = str_to_bool(str(config.get("save_embedding_archives", "True")))
#     embedding_archive_max     = int(config.get("embedding_archive_max_samples", 3000))

#     f_log = open(metric_log_path, "a", encoding="utf-8")
#     f_log.write("=" * 20 + "\n")
#     f_log.flush()

#     try:
#         for epoch in range(config["num_epochs"]):
#             print(f"\n========== Epoch {epoch:03d} ==========")

#             # Layer unfreeze after warm-up
#             if epoch == unfreeze_epoch and hasattr(model, "unfreeze_all"):
#                 model.unfreeze_all()
#                 for g in base_optimizer.param_groups:
#                     g["lr"] = g["lr"] * 0.1
#                 print(f"[Train] Epoch {epoch}: all layers unfrozen, LR × 0.1")

#             running_loss = train_epoch(
#                 trn_loader=trn_loader,
#                 model=model,
#                 optimizer=optimizer,
#                 device=device,
#                 scheduler=scheduler,
#                 config=config,
#                 ce_criterion=ce_criterion,
#                 grad_accum_steps=grad_accum_steps,
#                 use_gam=use_gam,
#             )
#             current_lr = float(base_optimizer.param_groups[0]["lr"])

#             # Per-epoch dirs
#             esd  = scores_dir      / f"epoch_{epoch:03d}"
#             epd  = predictions_dir / f"epoch_{epoch:03d}"
#             efd  = figures_dir     / f"epoch_{epoch:03d}"
#             emd  = metrics_dir     / f"epoch_{epoch:03d}"
#             eemd = embedding_dir   / f"epoch_{epoch:03d}"
#             for p in [esd, epd, efd, emd, eemd]:
#                 p.mkdir(parents=True, exist_ok=True)

#             dev_metrics = evaluate_split(
#                 split_name="dev", data_loader=dev_loader, model=model,
#                 device=device, trial_path=dev_trial_path,
#                 score_txt_path=esd / "dev_scores.txt"          if save_epoch_scores      else None,
#                 prediction_csv_path=epd / "dev_predictions.csv" if save_epoch_predictions else None,
#                 output_dir=efd / "dev" if save_epoch_figures else emd / "dev",
#                 save_all_figures=save_epoch_figures,
#                 save_embeddings=save_embedding_archives,
#                 embedding_save_path=eemd / "dev_embeddings.npz",
#                 embedding_max_samples=embedding_archive_max,
#             )

#             eval_metrics = None
#             if eval_loader is not None and eval_trial_path.exists():
#                 eval_metrics = evaluate_split(
#                     split_name="eval", data_loader=eval_loader, model=model,
#                     device=device, trial_path=eval_trial_path,
#                     score_txt_path=esd / "eval_scores.txt"          if save_epoch_scores      else None,
#                     prediction_csv_path=epd / "eval_predictions.csv" if save_epoch_predictions else None,
#                     output_dir=efd / "eval" if save_epoch_figures else emd / "eval",
#                     save_all_figures=save_epoch_figures,
#                     save_embeddings=save_embedding_archives,
#                     embedding_save_path=eemd / "eval_embeddings.npz",
#                     embedding_max_samples=embedding_archive_max,
#                 )

#             if epoch == 0 and save_tsne_flag:
#                 _save_tsne_both(
#                     model, device, dev_loader, dev_trial_path,
#                     eval_loader, eval_trial_path, model_tag, "epoch0",
#                     int(config.get("tsne_max_samples", 2000)),
#                 )

#             dev_eer  = float(dev_metrics["eer"])
#             dev_dcf  = float(dev_metrics["dcf"])
#             dev_cllr = float(dev_metrics["cllr"])

#             print("Loss:{:.6f} LR:{:.2e} dev_eer:{:.4f} dev_dcf:{:.6f} dev_cllr:{:.6f}".format(
#                 running_loss, current_lr, dev_eer, dev_dcf, dev_cllr))
#             if eval_metrics is not None:
#                 print("eval_eer:{:.4f} eval_dcf:{:.6f} eval_cllr:{:.6f}".format(
#                     eval_metrics["eer"], eval_metrics["dcf"], eval_metrics["cllr"]))

#             writer.add_scalar("loss",          running_loss, epoch)
#             writer.add_scalar("lr",            current_lr,   epoch)
#             writer.add_scalar("dev_eer",       dev_eer,      epoch)
#             writer.add_scalar("dev_dcf",       dev_dcf,      epoch)
#             writer.add_scalar("dev_cllr",      dev_cllr,     epoch)
#             writer.add_scalar("dev_auc",       dev_metrics["auc"],      epoch)
#             writer.add_scalar("dev_accuracy",  dev_metrics["accuracy"], epoch)
#             writer.add_scalar("dev_f1_bonafide", dev_metrics["f1_bonafide"], epoch)
#             if eval_metrics is not None:
#                 writer.add_scalar("eval_eer",  eval_metrics["eer"],   epoch)
#                 writer.add_scalar("eval_dcf",  eval_metrics["dcf"],   epoch)
#                 writer.add_scalar("eval_cllr", eval_metrics["cllr"],  epoch)
#                 writer.add_scalar("eval_auc",  eval_metrics["auc"],   epoch)
#                 writer.add_scalar("eval_accuracy", eval_metrics["accuracy"], epoch)
#                 writer.add_scalar("eval_f1_bonafide", eval_metrics["f1_bonafide"], epoch)

#             torch.save(model.state_dict(),
#                        model_save_path / f"epoch_{epoch:03d}_devEER_{dev_eer:.6f}.pth")

#             row = {
#                 "epoch":  int(epoch), "loss": float(running_loss), "lr": float(current_lr),
#                 "dev_eer": float(dev_eer), "dev_dcf": float(dev_dcf), "dev_cllr": float(dev_cllr),
#                 "dev_auc": float(dev_metrics["auc"]),
#                 "dev_threshold_eer":      float(dev_metrics["threshold_eer"]),
#                 "dev_tp":  int(dev_metrics["tp"]),   "dev_tn": int(dev_metrics["tn"]),
#                 "dev_fp":  int(dev_metrics["fp"]),   "dev_fn": int(dev_metrics["fn"]),
#                 "dev_accuracy":           float(dev_metrics["accuracy"]),
#                 "dev_precision_bonafide": float(dev_metrics["precision_bonafide"]),
#                 "dev_recall_bonafide":    float(dev_metrics["recall_bonafide"]),
#                 "dev_f1_bonafide":        float(dev_metrics["f1_bonafide"]),
#             }
#             if eval_metrics is not None:
#                 row.update({
#                     "eval_eer": float(eval_metrics["eer"]), "eval_dcf": float(eval_metrics["dcf"]),
#                     "eval_cllr": float(eval_metrics["cllr"]), "eval_auc": float(eval_metrics["auc"]),
#                     "eval_threshold_eer":      float(eval_metrics["threshold_eer"]),
#                     "eval_tp":  int(eval_metrics["tp"]),   "eval_tn": int(eval_metrics["tn"]),
#                     "eval_fp":  int(eval_metrics["fp"]),   "eval_fn": int(eval_metrics["fn"]),
#                     "eval_accuracy":           float(eval_metrics["accuracy"]),
#                     "eval_precision_bonafide": float(eval_metrics["precision_bonafide"]),
#                     "eval_recall_bonafide":    float(eval_metrics["recall_bonafide"]),
#                     "eval_f1_bonafide":        float(eval_metrics["f1_bonafide"]),
#                 })
#             history_rows.append(row)
#             append_history_csv(history_csv, row)
#             append_jsonl(history_jsonl, row)
#             if save_epoch_metric_json:
#                 save_json(emd / "metrics.json", row)

#             _plot_all_curves(history_rows, figures_dir)

#             best_dev_dcf  = min(best_dev_dcf,  dev_dcf)
#             best_dev_cllr = min(best_dev_cllr, dev_cllr)

#             improved = dev_eer < (best_dev_eer - early_stop_min_delta)
#             if improved:
#                 print(f"Best model found at epoch {epoch}")
#                 best_dev_eer = dev_eer
#                 best_epoch   = epoch
#                 epochs_no_improve = 0
#                 torch.save(model.state_dict(), model_save_path / "best_ast.pth")
#                 best_data = {
#                     "best_epoch":    int(epoch),
#                     "best_dev_eer":  float(dev_eer),
#                     "best_dev_dcf":  float(dev_dcf),
#                     "best_dev_cllr": float(dev_cllr),
#                     "best_dev_auc":  float(dev_metrics["auc"]),
#                 }
#                 if eval_metrics is not None:
#                     best_data.update({
#                         "best_eval_eer":  float(eval_metrics["eer"]),
#                         "best_eval_dcf":  float(eval_metrics["dcf"]),
#                         "best_eval_cllr": float(eval_metrics["cllr"]),
#                         "best_eval_auc":  float(eval_metrics["auc"]),
#                     })
#                 save_json(best_json, best_data)
#             else:
#                 epochs_no_improve += 1

#             if epoch >= swa_start_epoch:
#                 optimizer_swa.update_swa()
#                 n_swa_update += 1
#                 print(f"[SWA] updated (total: {n_swa_update})")

#             writer.add_scalar("best_dev_eer",  best_dev_eer,  epoch)
#             writer.add_scalar("best_dev_dcf",  best_dev_dcf,  epoch)
#             writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

#             log_line = (
#                 f"epoch={epoch}, loss={running_loss:.6f}, lr={current_lr:.8f}, "
#                 f"dev_eer={dev_eer:.6f}, dev_dcf={dev_dcf:.6f}, dev_cllr={dev_cllr:.6f}, "
#                 f"dev_auc={dev_metrics['auc']:.6f}, dev_acc={dev_metrics['accuracy']:.6f}, "
#                 f"dev_f1={dev_metrics['f1_bonafide']:.6f}"
#             )
#             if eval_metrics is not None:
#                 log_line += (
#                     f", eval_eer={eval_metrics['eer']:.6f}, eval_dcf={eval_metrics['dcf']:.6f}, "
#                     f"eval_cllr={eval_metrics['cllr']:.6f}, eval_auc={eval_metrics['auc']:.6f}, "
#                     f"eval_acc={eval_metrics['accuracy']:.6f}, eval_f1={eval_metrics['f1_bonafide']:.6f}"
#                 )
#             f_log.write(log_line + "\n")
#             f_log.flush()

#             if early_stop_enabled and epochs_no_improve >= early_stop_patience:
#                 print(f"Early stopping at epoch {epoch}. Best={best_epoch}, dev_eer={best_dev_eer:.6f}")
#                 break

#     finally:
#         f_log.close()
#         writer.close()

#     # Final best-model evaluation
#     best_weight = model_save_path / "best_ast.pth"
#     if best_weight.exists():
#         model.load_state_dict(torch.load(best_weight, map_location=device))
#         print(f"Loaded best checkpoint: {best_weight}")

#     final_dir = model_tag / "final_best_model_eval"
#     final_dir.mkdir(parents=True, exist_ok=True)

#     final_dev = evaluate_split(
#         split_name="dev", data_loader=dev_loader, model=model,
#         device=device, trial_path=dev_trial_path,
#         score_txt_path=final_dir / "dev_best_scores.txt",
#         prediction_csv_path=final_dir / "dev_best_predictions.csv",
#         output_dir=final_dir / "dev", save_all_figures=True,
#         save_embeddings=save_embedding_archives,
#         embedding_save_path=final_dir / "dev_best_embeddings.npz",
#         embedding_max_samples=embedding_archive_max,
#     )

#     final_eval = None
#     if eval_loader is not None and eval_trial_path.exists():
#         final_eval = evaluate_split(
#             split_name="eval", data_loader=eval_loader, model=model,
#             device=device, trial_path=eval_trial_path,
#             score_txt_path=final_dir / "eval_best_scores.txt",
#             prediction_csv_path=final_dir / "eval_best_predictions.csv",
#             output_dir=final_dir / "eval", save_all_figures=True,
#             save_embeddings=save_embedding_archives,
#             embedding_save_path=final_dir / "eval_best_embeddings.npz",
#             embedding_max_samples=embedding_archive_max,
#         )

#     if save_tsne_flag:
#         _save_tsne_both(
#             model, device, dev_loader, dev_trial_path,
#             eval_loader, eval_trial_path, model_tag, "final",
#             int(config.get("tsne_max_samples", 2000)),
#         )

#     save_json(archive_dir / "final_summary.json", {
#         "best_epoch":         int(best_epoch),
#         "best_dev_eer":       float(best_dev_eer),
#         "best_dev_dcf":       float(best_dev_dcf),
#         "best_dev_cllr":      float(best_dev_cllr),
#         "swa_updates":        int(n_swa_update),
#         "final_dev_metrics":  final_dev,
#         "final_eval_metrics": final_eval,
#     })

#     print(f"\nDone. Best epoch: {best_epoch}, best dev_eer: {best_dev_eer:.6f}")
#     if final_eval:
#         print(f"Final eval_eer: {final_eval['eer']:.6f}")


# # ============================================================
# #  train_epoch  (GAM-aware)
# # ============================================================
# def train_epoch(
#     trn_loader: DataLoader,
#     model,
#     optimizer,
#     device: torch.device,
#     scheduler,
#     config: dict,
#     ce_criterion: nn.Module,
#     grad_accum_steps: int = 4,
#     use_gam: bool = True,
# ) -> float:
#     running_loss = 0.0
#     num_total    = 0.0
#     model.train()

#     is_gam = use_gam and isinstance(optimizer, GAMOptimizer)
#     base_opt = optimizer.base_optimizer if is_gam else optimizer
#     base_opt.zero_grad()

#     for step, (batch_x, batch_y) in enumerate(tqdm(trn_loader)):
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.view(-1).long().to(device)
#         B       = batch_x.size(0)
#         num_total += B

#         if is_gam:
#             # ---- GAM first pass ----
#             emb, logits = model(batch_x)
#             if hasattr(model, "compute_loss"):
#                 loss, _ = model.compute_loss(emb, logits, batch_y, ce_criterion)
#             else:
#                 loss = ce_criterion(logits, batch_y)
#             (loss / grad_accum_steps).backward()

#             if (step + 1) % grad_accum_steps == 0:
#                 optimizer.first_step()       # perturb weights

#                 # ---- GAM second pass ----
#                 emb2, logits2 = model(batch_x)
#                 if hasattr(model, "compute_loss"):
#                     loss2, _ = model.compute_loss(emb2, logits2, batch_y, ce_criterion)
#                 else:
#                     loss2 = ce_criterion(logits2, batch_y)
#                 loss2.backward()
#                 optimizer.second_step()      # restore + update

#                 scheduler.step()

#         else:
#             # ---- Plain AdamW ----
#             emb, logits = model(batch_x)
#             if hasattr(model, "compute_loss"):
#                 loss, _ = model.compute_loss(emb, logits, batch_y, ce_criterion)
#             else:
#                 loss = ce_criterion(logits, batch_y)
#             (loss / grad_accum_steps).backward()

#             if (step + 1) % grad_accum_steps == 0:
#                 base_opt.step()
#                 base_opt.zero_grad()
#                 scheduler.step()

#         running_loss += loss.item() * B

#     running_loss /= max(1.0, num_total)
#     return float(running_loss)


# # ============================================================
# #  Model / Loader
# # ============================================================
# def get_model(model_config: Dict, device: torch.device):
#     module   = import_module(f"models.{model_config['architecture']}")
#     _model   = getattr(module, "Model")
#     model    = _model(model_config).to(device)
#     total    = sum(p.view(-1).size()[0] for p in model.parameters())
#     trainable = sum(p.view(-1).size()[0] for p in model.parameters() if p.requires_grad)
#     print(f"Total params: {total:,}  |  Trainable: {trainable:,}")
#     return model


# def get_loader(database_path: Path, seed: int, config: dict):
#     trn_database_path  = database_path / "flac_T"
#     dev_database_path  = database_path / "flac_D"
#     eval_database_path = database_path / "flac_E"
#     trn_list_path   = database_path / "ASVspoof5.train.metainfor.txt"
#     dev_trial_path  = database_path / "ASVspoof5.dev.metainfor.txt"
#     eval_trial_path = database_path / "ASVspoof5.eval.metainfor.txt"

#     d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
#     print("no. training files:", len(file_train))
#     train_set = TrainDataset(list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path)
#     gen = torch.Generator()
#     gen.manual_seed(seed)
#     trn_loader = DataLoader(
#         train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True,
#         pin_memory=True, worker_init_fn=seed_worker, generator=gen,
#         num_workers=int(config.get("num_workers", 4)),
#     )

#     _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
#     print("no. dev files:", len(file_dev))
#     dev_set = TestDataset(list_IDs=file_dev, base_dir=dev_database_path)
#     dev_loader = DataLoader(
#         dev_set, batch_size=config["batch_size"], shuffle=False, drop_last=False,
#         pin_memory=True, num_workers=int(config.get("num_workers", 4)),
#     )

#     eval_loader = None
#     if eval_trial_path.exists() and eval_database_path.exists():
#         _, file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=False)
#         print("no. eval files:", len(file_eval))
#         eval_set = TestDataset(list_IDs=file_eval, base_dir=eval_database_path)
#         eval_loader = DataLoader(
#             eval_set, batch_size=config["batch_size"], shuffle=False, drop_last=False,
#             pin_memory=True, num_workers=int(config.get("num_workers", 4)),
#         )
#     else:
#         print("Eval loader skipped.")

#     return trn_loader, dev_loader, eval_loader


# # ============================================================
# #  Evaluation helpers  (unchanged from baseline)
# # ============================================================
# def parse_trial_file(trial_path: Path) -> Dict:
#     info = {}
#     with open(trial_path, "r") as f:
#         for line in f:
#             parts = line.strip().split(" ")
#             if len(parts) < 6:
#                 continue
#             spk_id, utt_id, p2, p3, p4, key = parts[:6]
#             info[utt_id] = {
#                 "speaker_id": spk_id, "utt_id": utt_id,
#                 "label_text": key, "label_int": 1 if key == "bonafide" else 0,
#             }
#     return info


# def collect_predictions(data_loader, model, device, trial_path,
#                         save_embeddings=False, embedding_max_samples=3000):

#     model.eval()
#     trial_info = parse_trial_file(trial_path)

#     rows, s_emb, s_scores, s_labels, s_ids = [], [], [], [], []
#     use_all = embedding_max_samples is None or embedding_max_samples <= 0

#     for batch_x, utt_ids in tqdm(data_loader):
#         batch_x = batch_x.to(device)

#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)

#             # 🔥 NO SOFTMAX — DIRECT SCORE
#             score_tensor = batch_out[:, 1]

#         emb_np = batch_emb.detach().cpu().numpy()
#         score_np = score_tensor.detach().cpu().numpy()

#         for i, uid in enumerate(utt_ids):
#             if uid not in trial_info:
#                 continue

#             meta = trial_info[uid]
#             score = float(score_np[i])

#             pred_int = 1 if score >= 0.0 else 0
#             true_int = int(meta["label_int"])

#             rows.append({
#                 "speaker_id": meta["speaker_id"],
#                 "utt_id": uid,

#                 "true_label_text": meta["label_text"],
#                 "true_label_int": true_int,

#                 "pred_label_text": "bonafide" if pred_int == 1 else "spoof",
#                 "pred_label_int": pred_int,
#                 "is_correct": int(pred_int == true_int),

#                 # keep compatibility
#                 "logit_spoof": float(-score),
#                 "logit_bonafide": float(score),

#                 # not used anymore
#                 "prob_spoof": float("nan"),
#                 "prob_bonafide": float("nan"),

#                 # 🔥 THIS is used for EER/minDCF
#                 "score": score
#             })

#             if save_embeddings and (use_all or len(s_ids) < embedding_max_samples):
#                 s_emb.append(emb_np[i])
#                 s_scores.append(score)
#                 s_labels.append(true_int)
#                 s_ids.append(uid)

#     return rows, s_emb, s_scores, s_labels, s_ids


# def save_score_txt(score_txt_path, rows):
#     with open(score_txt_path, "w") as fh:
#         for r in rows:
#             fh.write(f"{r['speaker_id']} {r['utt_id']} {r['score']} {r['true_label_text']}\n")


# def save_prediction_csv(path, rows):
#     if not rows:
#         return
#     with open(path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader(); writer.writerows(rows)


# def compute_binary_metrics_from_rows(rows):
#     scores = np.array([r["score"]          for r in rows], dtype=np.float64)
#     labels = np.array([r["true_label_int"] for r in rows], dtype=np.int64)
#     bona_scores  = scores[labels == 1]
#     spoof_scores = scores[labels == 0]
#     eer, frr, far, thresholds = compute_eer(bona_scores, spoof_scores)
#     idx = int(np.argmin(np.abs(frr - far)))
#     thr = float(thresholds[idx])
#     pred = (scores >= thr).astype(np.int64)
#     tp = int(np.sum((pred == 1) & (labels == 1)))
#     tn = int(np.sum((pred == 0) & (labels == 0)))
#     fp = int(np.sum((pred == 1) & (labels == 0)))
#     fn = int(np.sum((pred == 0) & (labels == 1)))
#     acc  = float((tp + tn) / max(1, len(labels)))
#     prec = float(tp / max(1, tp + fp))
#     rec  = float(tp / max(1, tp + fn))
#     f1   = float(2 * prec * rec / max(1e-12, prec + rec))
#     fpr, tpr, _ = roc_curve(labels, scores)
#     roc_auc = float(auc(fpr, tpr))
#     return {
#         "scores": scores, "labels": labels, "bona_scores": bona_scores, "spoof_scores": spoof_scores,
#         "eer": float(eer), "frr": frr, "far": far, "thresholds": thresholds, "threshold_eer": thr,
#         "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc,
#         "precision_bonafide": prec, "recall_bonafide": rec, "f1_bonafide": f1,
#         "fpr": fpr, "tpr": tpr, "auc": roc_auc,
#     }


# def evaluate_split(split_name, data_loader, model, device, trial_path,
#                    score_txt_path, prediction_csv_path, output_dir,
#                    save_all_figures=True, save_embeddings=False,
#                    embedding_save_path=None, embedding_max_samples=3000):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     rows, s_emb, s_scores, s_labels, s_ids = collect_predictions(
#         data_loader, model, device, trial_path,
#         save_embeddings=save_embeddings, embedding_max_samples=embedding_max_samples,
#     )

#     if score_txt_path is None:
#         score_txt_path = output_dir / f"{split_name}_scores.txt"
#     save_score_txt(score_txt_path, rows)
#     if prediction_csv_path is not None:
#         save_prediction_csv(prediction_csv_path, rows)

#     dcf, eer_fn, cllr = calculate_minDCF_EER_CLLR(
#         cm_scores_file=score_txt_path,
#         output_file=output_dir / f"{split_name}_DCF_EER.txt",
#         printout=False,
#     )
#     stats = compute_binary_metrics_from_rows(rows)

#     if save_embeddings and s_emb and embedding_save_path is not None:
#         np.savez_compressed(
#             embedding_save_path,
#             embeddings=np.array(s_emb), scores=np.array(s_scores),
#             labels=np.array(s_labels), utt_ids=np.array(s_ids),
#         )

#     metrics = {
#         "dcf": float(dcf), "eer": float(eer_fn), "cllr": float(cllr),
#         "auc": stats["auc"], "threshold_eer": stats["threshold_eer"],
#         "tp": stats["tp"], "tn": stats["tn"], "fp": stats["fp"], "fn": stats["fn"],
#         "accuracy": stats["accuracy"],
#         "precision_bonafide": stats["precision_bonafide"],
#         "recall_bonafide":    stats["recall_bonafide"],
#         "f1_bonafide":        stats["f1_bonafide"],
#     }

#     if save_all_figures:
#         _save_det_roc(stats, output_dir, split_name)

#     return metrics


# def _save_det_roc(stats, output_dir, split_name):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].plot(stats["far"], stats["frr"])
#     axes[0].set_xlabel("FAR"); axes[0].set_ylabel("FRR"); axes[0].set_title(f"{split_name} DET")
#     axes[0].grid(True, alpha=0.3)
#     axes[1].plot(stats["fpr"], stats["tpr"])
#     axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
#     axes[1].set_title(f"{split_name} ROC  AUC={stats['auc']:.4f}")
#     axes[1].grid(True, alpha=0.3)
#     plt.tight_layout()
#     fig.savefig(output_dir / f"{split_name}_det_roc.png", dpi=150, bbox_inches="tight")
#     plt.close(fig)


# # ============================================================
# #  t-SNE
# # ============================================================
# def _save_tsne_both(model, device, dev_loader, dev_trial_path,
#                     eval_loader, eval_trial_path, model_tag, suffix, max_samples):
#     save_embedding_tsne(
#         data_loader=dev_loader, model=model, device=device,
#         trial_path=dev_trial_path,
#         save_dir=model_tag / f"tsne_{suffix}_dev",
#         max_samples=max_samples, split_name="dev",
#     )
#     if eval_loader is not None and eval_trial_path.exists():
#         save_embedding_tsne(
#             data_loader=eval_loader, model=model, device=device,
#             trial_path=eval_trial_path,
#             save_dir=model_tag / f"tsne_{suffix}_eval",
#             max_samples=max_samples, split_name="eval",
#         )


# def save_embedding_tsne(data_loader, model, device, trial_path, save_dir,
#                         max_samples=2000, split_name="dev"):
#     import csv as _csv
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     trial_info = parse_trial_file(trial_path)
#     label_map  = {uid: info["label_int"] for uid, info in trial_info.items()}

#     model.eval()
#     embeddings, scores, utt_ids = [], [], []
#     use_all = max_samples is None or max_samples <= 0

#     for batch_x, batch_utt_id in tqdm(data_loader, desc=f"tsne_{split_name}"):
#         batch_x = batch_x.to(device)
#         with torch.no_grad():
#             batch_emb, batch_out = model(batch_x)
#             batch_score = torch.softmax(batch_out, dim=1)[:, 1].cpu().numpy()
#             batch_emb   = batch_emb.cpu().numpy()
#         embeddings.append(batch_emb); scores.append(batch_score)
#         utt_ids.extend([str(u) for u in batch_utt_id])
#         if not use_all and len(utt_ids) >= max_samples:
#             break

#     if len(utt_ids) < 2:
#         print(f"Not enough samples for t-SNE on {split_name}.")
#         return

#     embeddings = np.concatenate(embeddings, axis=0)
#     scores     = np.concatenate(scores, axis=0)
#     if not use_all:
#         embeddings = embeddings[:max_samples]; scores = scores[:max_samples]
#         utt_ids    = utt_ids[:max_samples]

#     labels = np.array([label_map.get(u, -1) for u in utt_ids])
#     np.savez_compressed(
#         save_dir / f"{split_name}_tsne_source.npz",
#         utt_ids=np.array(utt_ids), labels=labels, scores=scores, embeddings=embeddings,
#     )

#     perplexity = min(30, max(5, len(utt_ids) - 1))
#     tsne   = TSNE(n_components=2, perplexity=perplexity, init="pca",
#                   learning_rate="auto", random_state=42)
#     coords = tsne.fit_transform(embeddings)

#     csv_path = save_dir / f"{split_name}_tsne_points.csv"
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         w = _csv.writer(f)
#         w.writerow(["utt_id", "label", "bonafide_score", "tsne_x", "tsne_y"])
#         for uid, lbl, sc, (x, y) in zip(utt_ids, labels, scores, coords):
#             w.writerow([uid, int(lbl), float(sc), float(x), float(y)])

#     fig = plt.figure(figsize=(8, 6))
#     for mask, lbl in [(labels == 0, "spoof"), (labels == 1, "bonafide"), (labels == -1, "unknown")]:
#         if mask.any():
#             plt.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=lbl)
#     plt.title(f"{split_name.upper()} set embeddings (t-SNE)")
#     plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.legend(); plt.tight_layout()
#     fig.savefig(save_dir / f"{split_name}_tsne_plot.png", dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved t-SNE for {split_name} ({len(utt_ids)} samples) → {save_dir}")


# # ============================================================
# #  Curve plotting
# # ============================================================
# def _plot_all_curves(rows: List[Dict], figures_dir: Path) -> None:
#     def _plot(keys, ylabel, fname):
#         fig, ax = plt.subplots(figsize=(8, 4))
#         for k in keys:
#             vals = [r[k] for r in rows if k in r]
#             if vals:
#                 ax.plot(range(len(vals)), vals, label=k)
#         ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)
#         plt.tight_layout()
#         fig.savefig(figures_dir / fname, dpi=120, bbox_inches="tight")
#         plt.close(fig)

#     _plot(["loss"],                       "Loss",    "loss_curve.png")
#     _plot(["lr"],                         "LR",      "lr_curve.png")
#     _plot(["dev_eer", "eval_eer"],        "EER",     "eer_curve.png")
#     _plot(["dev_dcf", "eval_dcf"],        "minDCF",  "dcf_curve.png")
#     _plot(["dev_cllr", "eval_cllr"],      "CLLR",    "cllr_curve.png")
#     _plot(["dev_auc", "eval_auc"],        "AUC",     "auc_curve.png")
#     _plot(["dev_accuracy", "eval_accuracy"], "Acc",  "accuracy_curve.png")
#     _plot(["dev_f1_bonafide", "eval_f1_bonafide"], "F1", "f1_curve.png")


# # ============================================================
# #  Logging helpers
# # ============================================================
# def append_history_csv(csv_path: Path, row: Dict) -> None:
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)


# def append_jsonl(path: Path, row: Dict) -> None:
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row) + "\n")


# def save_json(path: Path, data: Dict) -> None:
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)


# # ============================================================
# #  CLI
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="ASVspoof5 — Multi-resolution AST + GAM")
#     parser.add_argument("--config",     dest="config",   type=str, required=True)
#     parser.add_argument("--output_dir", dest="output_dir", type=str, default="./exp_result")
#     parser.add_argument("--seed",       type=int, default=1234)
#     parser.add_argument("--eval",       action="store_true")
#     parser.add_argument("--comment",    type=str, default=None)
#     parser.add_argument("--eval_model_weights", type=str, default=None)
#     main(parser.parse_args())




"""
Refactored training / evaluation entry point for ASVspoof5.

This file preserves the current experiment flow while upgrading the pipeline:
    - keeps the original files untouched
    - uses the new ``ast_v2`` model by default for legacy ``AST`` configs
    - reads richer Track-1 TSV metadata when available
    - supports full-waveform variable-length loading
    - fixes gradient accumulation for GAM
    - adds multi-view evaluation through the v2 model
    - saves richer artifacts for later analysis or AI-assisted tooling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval.calculate_metrics import calculate_minDCF_EER_CLLR
from eval.calculate_modules import compute_eer
try:
    from utils_v2 import count_parameters, create_optimizer, seed_worker, set_seed, str_to_bool
except ModuleNotFoundError:
    from utils import count_parameters, create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Constants
# ============================================================
TRAIN_META = "ASVspoof5.train.metainfor.txt"
DEV_META = "ASVspoof5.dev.metainfor.txt"
EVAL_META = "ASVspoof5.eval.metainfor.txt"

TRAIN_TSV = "ASVspoof5.train.tsv"
DEV_TSV = "ASVspoof5.dev.track_1.tsv"
EVAL_TSV = "ASVspoof5.eval.track_1.tsv"

TRAIN_AUDIO_DIR = "flac_T"
DEV_AUDIO_DIR = "flac_D"
EVAL_AUDIO_DIR = "flac_E"

BONAFIDE_ATTACK_TOKEN = "bonafide"
UNKNOWN_TOKEN = "-"


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class ProtocolEntry:
    """Structured metadata for one utterance."""

    speaker_id: str
    utt_id: str
    gender: str
    codec_id: str
    codec_quality: str
    source_utt_id: str
    acoustic_condition: str
    attack_id: str
    label_text: str
    label_int: int


@dataclass
class AudioBatch:
    """Variable-length batch container returned by the custom collate function."""

    waveforms: list[torch.Tensor]
    utt_ids: list[str]
    labels: torch.Tensor
    attack_labels: torch.Tensor
    entries: list[ProtocolEntry]


@dataclass
class DataBundle:
    """Train / dev / eval loaders plus their metadata."""

    train_entries: list[ProtocolEntry]
    dev_entries: list[ProtocolEntry]
    eval_entries: list[ProtocolEntry]
    attack_label_map: dict[str, int]
    train_loader: DataLoader
    dev_loader: DataLoader
    eval_loader: Optional[DataLoader]


@dataclass(frozen=True)
class ExperimentPaths:
    """Common output paths for one experiment run."""

    model_tag: Path
    model_save_path: Path
    scores_dir: Path
    predictions_dir: Path
    figures_dir: Path
    metrics_dir: Path
    embedding_dir: Path
    archive_dir: Path
    history_csv: Path
    history_jsonl: Path
    best_json: Path
    run_summary_json: Path
    metric_log_path: Path

    @classmethod
    def build(
        cls,
        output_dir: Path,
        config_path: str,
        num_epochs: int,
        batch_size: int,
        comment: Optional[str],
    ) -> "ExperimentPaths":
        model_tag_name = f"{Path(config_path).stem}_ep{num_epochs}_bs{batch_size}"
        if comment:
            model_tag_name = f"{model_tag_name}_{comment}"

        model_tag = output_dir / model_tag_name
        paths = cls(
            model_tag=model_tag,
            model_save_path=model_tag / "weights",
            scores_dir=model_tag / "scores",
            predictions_dir=model_tag / "predictions",
            figures_dir=model_tag / "figures",
            metrics_dir=model_tag / "metrics",
            embedding_dir=model_tag / "embeddings",
            archive_dir=model_tag / "artifacts",
            history_csv=model_tag / "training_history.csv",
            history_jsonl=model_tag / "training_history.jsonl",
            best_json=model_tag / "best_metrics.json",
            run_summary_json=model_tag / "run_summary.json",
            metric_log_path=model_tag / "metric_log.txt",
        )
        for path in (
            paths.model_save_path,
            paths.scores_dir,
            paths.predictions_dir,
            paths.figures_dir,
            paths.metrics_dir,
            paths.embedding_dir,
            paths.archive_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass
class TrainingObjects:
    """Optimizer-side objects used by the train loop."""

    base_optimizer: torch.optim.Optimizer
    optimizer: "GAMOptimizer | torch.optim.Optimizer"
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    use_gam: bool
    amp_enabled: bool
    scaler: Optional[torch.cuda.amp.GradScaler]


# ============================================================
# Protocol parsing
# ============================================================
def _label_to_int(label_text: str) -> int:
    return 1 if label_text == "bonafide" else 0


def parse_track1_entry(parts: Sequence[str]) -> ProtocolEntry:
    """Parse a Track-1 TSV-style row."""

    if len(parts) < 9:
        raise ValueError(f"expected at least 9 columns, got {len(parts)}: {parts}")

    speaker_id, utt_id, gender = parts[0], parts[1], parts[2]
    codec_id = parts[3] if len(parts) > 3 else UNKNOWN_TOKEN
    codec_quality = parts[4] if len(parts) > 4 else UNKNOWN_TOKEN
    source_utt_id = parts[5] if len(parts) > 5 else UNKNOWN_TOKEN
    acoustic_condition = parts[6] if len(parts) > 6 else UNKNOWN_TOKEN
    attack_id = parts[7] if len(parts) > 7 else UNKNOWN_TOKEN
    label_text = parts[8]

    return ProtocolEntry(
        speaker_id=speaker_id,
        utt_id=utt_id,
        gender=gender,
        codec_id=codec_id,
        codec_quality=codec_quality,
        source_utt_id=source_utt_id,
        acoustic_condition=acoustic_condition,
        attack_id=attack_id,
        label_text=label_text,
        label_int=_label_to_int(label_text),
    )


def parse_meta_entry(parts: Sequence[str]) -> ProtocolEntry:
    """Parse the reduced ``.metainfor.txt`` row format."""

    if len(parts) < 6:
        raise ValueError(f"expected at least 6 columns, got {len(parts)}: {parts}")

    speaker_id, utt_id, gender, meta3, meta4, label_text = parts[:6]

    return ProtocolEntry(
        speaker_id=speaker_id,
        utt_id=utt_id,
        gender=gender,
        codec_id=meta3,
        codec_quality=meta4,
        source_utt_id=UNKNOWN_TOKEN,
        acoustic_condition=UNKNOWN_TOKEN,
        attack_id=BONAFIDE_ATTACK_TOKEN if label_text == "bonafide" else UNKNOWN_TOKEN,
        label_text=label_text,
        label_int=_label_to_int(label_text),
    )


def load_protocol_entries(
    database_path: Path,
    split_name: str,
    prefer_track1_tsv: bool = True,
) -> list[ProtocolEntry]:
    """Load metadata for train / dev / eval with a TSV-first fallback."""

    split_name = split_name.lower()
    tsv_map = {"train": TRAIN_TSV, "dev": DEV_TSV, "eval": EVAL_TSV}
    meta_map = {"train": TRAIN_META, "dev": DEV_META, "eval": EVAL_META}

    tsv_path = database_path / tsv_map[split_name]
    meta_path = database_path / meta_map[split_name]

    entries: list[ProtocolEntry] = []

    if prefer_track1_tsv and tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                entries.append(parse_track1_entry(parts))
        return deduplicate_entries(entries, split_name=split_name, source_path=tsv_path)

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                entries.append(parse_meta_entry(parts))
        return deduplicate_entries(entries, split_name=split_name, source_path=meta_path)

    raise FileNotFoundError(f"no protocol file found for split '{split_name}' in {database_path}")


def deduplicate_entries(
    entries: Sequence[ProtocolEntry],
    split_name: str,
    source_path: Path,
) -> list[ProtocolEntry]:
    """Deduplicate protocol entries while keeping deterministic ordering."""

    deduplicated: list[ProtocolEntry] = []
    seen_ids: set[str] = set()
    duplicate_count = 0
    for entry in entries:
        if entry.utt_id in seen_ids:
            duplicate_count += 1
            continue
        deduplicated.append(entry)
        seen_ids.add(entry.utt_id)

    if duplicate_count > 0:
        print(f"[data] ignored {duplicate_count} duplicate rows in {source_path.name} for split={split_name}")
    return deduplicated


def build_attack_label_map(train_entries: Sequence[ProtocolEntry]) -> dict[str, int]:
    """Create the auxiliary attack-label mapping from the train split."""

    spoof_attacks = sorted(
        {
            entry.attack_id
            for entry in train_entries
            if entry.label_int == 0 and entry.attack_id not in {"", UNKNOWN_TOKEN}
        }
    )

    if not spoof_attacks:
        return {}

    mapping = {attack_id: index for index, attack_id in enumerate(spoof_attacks)}
    mapping[BONAFIDE_ATTACK_TOKEN] = len(mapping)
    return mapping


def protocol_lookup(entries: Sequence[ProtocolEntry]) -> dict[str, ProtocolEntry]:
    """Build a fast lookup by utterance ID."""

    return {entry.utt_id: entry for entry in entries}


# ============================================================
# Audio loading
# ============================================================
def load_audio_mono(path: Path, expected_sample_rate: int) -> torch.Tensor:
    """Load audio as a mono float tensor, resampling only if needed."""

    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "main_v2.py requires the 'soundfile' package for waveform loading."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    waveform_np, sample_rate = sf.read(str(path), always_2d=False)
    waveform = torch.as_tensor(np.asarray(waveform_np), dtype=torch.float32)

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {tuple(waveform.shape)} for {path}")

    if sample_rate != expected_sample_rate:
        try:
            waveform = torchaudio_resample(waveform, sample_rate, expected_sample_rate)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            raise RuntimeError(
                f"failed to resample {path} from {sample_rate} Hz to {expected_sample_rate} Hz"
            ) from exc

    return waveform.contiguous()


def torchaudio_resample(waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Resample with torchaudio when sample rates differ."""

    import torchaudio

    return torchaudio.functional.resample(waveform.unsqueeze(0), orig_sr, new_sr).squeeze(0)


# ============================================================
# Dataset / DataLoader
# ============================================================
class WaveformDataset(Dataset):
    """Full-waveform dataset that keeps utterance metadata available."""

    def __init__(
        self,
        entries: Sequence[ProtocolEntry],
        audio_dir: Path,
        sample_rate: int,
        attack_label_map: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.entries = list(entries)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.attack_label_map = dict(attack_label_map or {})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        waveform = load_audio_mono(self.audio_dir / f"{entry.utt_id}.flac", self.sample_rate)

        attack_token = BONAFIDE_ATTACK_TOKEN if entry.label_int == 1 else entry.attack_id
        attack_label = self.attack_label_map.get(attack_token, -1)

        return {
            "waveform": waveform,
            "utt_id": entry.utt_id,
            "label": entry.label_int,
            "attack_label": attack_label,
            "entry": entry,
        }


def collate_audio_batch(items: Sequence[dict[str, Any]]) -> AudioBatch:
    """Custom collate function for variable-length waveform batches."""

    waveforms = [item["waveform"] for item in items]
    utt_ids = [item["utt_id"] for item in items]
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    attack_labels = torch.tensor([item["attack_label"] for item in items], dtype=torch.long)
    entries = [item["entry"] for item in items]

    return AudioBatch(
        waveforms=waveforms,
        utt_ids=utt_ids,
        labels=labels,
        attack_labels=attack_labels,
        entries=entries,
    )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    """Build a DataLoader with deterministic worker seeding."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "collate_fn": collate_audio_batch,
        "worker_init_fn": seed_worker,
    }

    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    elif shuffle:
        kwargs["generator"] = generator

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def build_data_bundle(
    database_path: Path,
    seed: int,
    config: Mapping[str, Any],
    include_eval: bool = False,
) -> DataBundle:
    """Load train/dev splits and optionally the eval split."""

    prefer_track1_tsv = str_to_bool(config.get("prefer_track1_tsv", "True"))
    sample_rate = int(config["model_config"].get("sample_rate", 16_000))
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 4))
    pin_memory = bool(torch.cuda.is_available())
    balanced_sampling = str_to_bool(config.get("balanced_sampling", "True"))
    balanced_sampling_power = float(config.get("balanced_sampling_power", 0.5))

    train_entries = load_protocol_entries(database_path, "train", prefer_track1_tsv=prefer_track1_tsv)
    dev_entries = load_protocol_entries(database_path, "dev", prefer_track1_tsv=prefer_track1_tsv)
    eval_entries: list[ProtocolEntry] = []
    if include_eval:
        eval_entries = load_protocol_entries(database_path, "eval", prefer_track1_tsv=prefer_track1_tsv)

    required_audio_dirs = [TRAIN_AUDIO_DIR, DEV_AUDIO_DIR]
    if include_eval:
        required_audio_dirs.append(EVAL_AUDIO_DIR)

    for audio_dir_name in required_audio_dirs:
        audio_dir = database_path / audio_dir_name
        if not audio_dir.exists():
            raise FileNotFoundError(f"expected audio directory is missing: {audio_dir}")

    attack_label_map = build_attack_label_map(train_entries)

    train_dataset = WaveformDataset(
        entries=train_entries,
        audio_dir=database_path / TRAIN_AUDIO_DIR,
        sample_rate=sample_rate,
        attack_label_map=attack_label_map,
    )
    dev_dataset = WaveformDataset(
        entries=dev_entries,
        audio_dir=database_path / DEV_AUDIO_DIR,
        sample_rate=sample_rate,
        attack_label_map=attack_label_map,
    )
    eval_dataset: Optional[WaveformDataset] = None
    if include_eval:
        eval_dataset = WaveformDataset(
            entries=eval_entries,
            audio_dir=database_path / EVAL_AUDIO_DIR,
            sample_rate=sample_rate,
            attack_label_map=attack_label_map,
        )

    train_sampler: Optional[WeightedRandomSampler] = None
    if balanced_sampling:
        label_counts: dict[int, int] = defaultdict(int)
        for entry in train_entries:
            label_counts[int(entry.label_int)] += 1

        sample_weights = []
        total_items = float(len(train_entries))
        for entry in train_entries:
            class_count = max(1, label_counts[int(entry.label_int)])
            inverse_freq = total_items / float(class_count)
            sample_weights.append(inverse_freq ** balanced_sampling_power)

        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "[data] balanced sampling enabled "
            f"(power={balanced_sampling_power:.2f}, counts={dict(sorted(label_counts.items()))})"
        )

    train_loader = build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        drop_last=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    dev_loader = build_loader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader: Optional[DataLoader] = None
    if eval_dataset is not None:
        eval_loader = build_loader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    print(f"no. training files: {len(train_entries)}")
    print(f"no. dev files: {len(dev_entries)}")
    if include_eval:
        print(f"no. eval files: {len(eval_entries)}")
    else:
        print("no. eval files: skipped during training for research-safe isolation")
    print(f"attack label classes: {len(attack_label_map)}")

    return DataBundle(
        train_entries=train_entries,
        dev_entries=dev_entries,
        eval_entries=eval_entries,
        attack_label_map=attack_label_map,
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_loader=eval_loader,
    )


# ============================================================
# Optimizer helpers
# ============================================================
class GAMOptimizer:
    """Gradient norm aware minimization with proper micro-batch replays."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        grad_rho: float = 0.02,
        adaptive: bool = False,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.grad_rho = grad_rho
        self.adaptive = adaptive
        self.param_groups = base_optimizer.param_groups

    @torch.no_grad()
    def _gradient_norm(self) -> torch.Tensor:
        norms: list[torch.Tensor] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad if not self.adaptive else parameter.grad * parameter.abs().clamp_min(1e-12)
                norms.append(torch.norm(grad, p=2))

        if not norms:
            device = self.param_groups[0]["params"][0].device
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._gradient_norm().clamp_min(1e-12)
        denom = grad_norm + self.grad_rho

        for group in self.param_groups:
            scale = self.rho / denom
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                perturb = (parameter.pow(2) * parameter.grad if self.adaptive else parameter.grad) * scale
                parameter.add_(perturb)
                parameter._gam_perturb = perturb  # type: ignore[attr-defined]

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                perturb = getattr(parameter, "_gam_perturb", None)
                if perturb is None:
                    continue
                parameter.sub_(perturb)
                delattr(parameter, "_gam_perturb")

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)


class RunningSWA:
    """Very small dependency-free running parameter average."""

    def __init__(self) -> None:
        self.average_state: dict[str, torch.Tensor] | None = None
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()

        if self.average_state is None:
            self.average_state = {
                key: value.detach().cpu().clone()
                for key, value in state.items()
            }
            self.num_updates = 1
            return

        old_updates = self.num_updates
        new_updates = old_updates + 1
        for key, value in state.items():
            current_value = value.detach().cpu()
            if torch.is_floating_point(self.average_state[key]):
                self.average_state[key].mul_(old_updates / new_updates)
                self.average_state[key].add_(current_value, alpha=1.0 / new_updates)
            else:
                # Non-floating buffers such as BatchNorm counters are not
                # meaningfully averaged; we simply keep the latest value.
                self.average_state[key] = current_value.clone()
        self.num_updates = new_updates

    def save(self, path: Path) -> None:
        if self.average_state is None:
            return
        torch.save(self.average_state, path)


class ModelEMA:
    """CPU-backed exponential moving average for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in the open interval (0, 1)")

        self.decay = decay
        self.shadow_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            current_value = value.detach().cpu()
            if torch.is_floating_point(current_value):
                self.shadow_state[key].mul_(self.decay)
                self.shadow_state[key].add_(current_value, alpha=1.0 - self.decay)
            else:
                self.shadow_state[key] = current_value.clone()
        self.num_updates += 1

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterable[None]:
        backup_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow_state, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup_state, strict=True)

    def save(self, path: Path) -> None:
        torch.save(self.shadow_state, path)


class LogisticScoreCalibrator:
    """Simple Platt-style score calibrator fit on dev scores."""

    def __init__(self) -> None:
        self.scale = 1.0
        self.bias = 0.0
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> bool:
        if scores.size == 0 or np.unique(labels).size < 2:
            return False

        score_tensor = torch.as_tensor(scores, dtype=torch.float32)
        label_tensor = torch.as_tensor(labels, dtype=torch.float32)
        scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optimizer = torch.optim.LBFGS(
            [scale, bias],
            lr=0.2,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = scale * score_tensor + bias
            loss = criterion(logits, label_tensor)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except RuntimeError:
            return False

        self.scale = float(scale.detach().item())
        self.bias = float(bias.detach().item())
        self.is_fitted = True
        return True

    def transform_array(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return scores
        return scores * self.scale + self.bias

    def transform_scalar(self, score: float) -> float:
        if not self.is_fitted:
            return float(score)
        return float(score * self.scale + self.bias)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "is_fitted": self.is_fitted,
        }


# ============================================================
# Model construction
# ============================================================
def apply_default_config(config: dict[str, Any]) -> dict[str, Any]:
    """Mutate the config in-place with v2 defaults."""

    config.setdefault("eval_all_best", "True")
    config.setdefault("freq_aug", "False")

    config.setdefault("use_gam", "True")
    config.setdefault("use_amp", "True")
    config.setdefault("gam_rho", 0.03)
    config.setdefault("gam_grad_rho", 0.02)
    config.setdefault("grad_accum_steps", 8)
    config.setdefault("grad_clip_norm", 5.0)

    config.setdefault("backbone_lr", 1.5e-5)
    config.setdefault("fusion_lr", 2e-4)
    config.setdefault("head_lr", 3e-4)
    config.setdefault("layerwise_lr_decay", 0.9)
    config.setdefault("unfreeze_epoch", 5)

    config.setdefault("use_ema", "True")
    config.setdefault("ema_decay", 0.9995)
    config.setdefault("ema_update_interval", 4)
    config.setdefault("eval_use_ema", "True")
    config.setdefault("swa_start_epoch", 10)

    config.setdefault("early_stop", "True")
    config.setdefault("early_stop_patience", 10)
    config.setdefault("early_stop_min_delta", 0.0005)

    config.setdefault("prefer_track1_tsv", "True")
    config.setdefault("balanced_sampling", "True")
    config.setdefault("balanced_sampling_power", 0.5)
    config.setdefault("save_tsne", "False")
    config.setdefault("tsne_max_samples", 100000)
    config.setdefault("save_epoch_scores", "True")
    config.setdefault("save_epoch_predictions", "True")
    config.setdefault("save_epoch_metric_json", "True")
    config.setdefault("save_epoch_figures", "True")
    config.setdefault("save_embedding_archives", "True")
    config.setdefault("embedding_archive_max_samples", 3000)
    config.setdefault("save_group_metric_json", "True")
    config.setdefault("save_calibrated_metrics", "True")
    config.setdefault("enable_calibration", "False")
    config.setdefault("track_eval_during_training", "False")
    config.setdefault("allow_eval_tracking_leak", "False")
    config.setdefault("run_final_eval_after_training", "False")

    config.setdefault("eval_num_segments", 5)
    config.setdefault("eval_segment_overlap", 0.5)
    config.setdefault("eval_score_pooling", "mean_max_attention")

    config.setdefault("binary_loss", "bce")
    config.setdefault("ce_weight_spoof", 1.0)
    config.setdefault("ce_weight_bonafide", 1.0)
    config.setdefault("focal_gamma", 2.0)
    config.setdefault("focal_alpha", -1.0)

    model_config = config.setdefault("model_config", {})
    model_config.setdefault("architecture", "AST")
    model_config.setdefault("eval_num_segments", config["eval_num_segments"])
    model_config.setdefault("eval_segment_overlap", config["eval_segment_overlap"])
    model_config.setdefault("eval_score_pooling", config["eval_score_pooling"])
    model_config.setdefault("binary_loss", config["binary_loss"])
    model_config.setdefault("ce_weight_spoof", config["ce_weight_spoof"])
    model_config.setdefault("ce_weight_bonafide", config["ce_weight_bonafide"])
    model_config.setdefault("focal_gamma", config["focal_gamma"])
    model_config.setdefault("focal_alpha", config["focal_alpha"])

    optim_config = config.setdefault("optim_config", {})
    optim_config.setdefault("optimizer", "adamw")
    optim_config.setdefault("base_lr", float(config.get("backbone_lr", 1.5e-5)))
    optim_config.setdefault("lr_min", 1e-7)
    optim_config.setdefault("betas", [0.9, 0.98])
    optim_config.setdefault("weight_decay", 1e-4)
    optim_config.setdefault("scheduler", "cosine")
    optim_config.setdefault("amsgrad", "False")

    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early on invalid configuration values."""

    required_keys = ["database_path", "batch_size", "num_epochs", "model_config", "optim_config"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"missing required config keys: {missing}")

    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(config["num_epochs"]) <= 0:
        raise ValueError("num_epochs must be > 0")
    if int(config["grad_accum_steps"]) <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if float(config["grad_clip_norm"]) < 0.0:
        raise ValueError("grad_clip_norm must be >= 0")
    if not 0.0 < float(config["ema_decay"]) < 1.0:
        raise ValueError("ema_decay must be in the open interval (0, 1)")
    if int(config["ema_update_interval"]) <= 0:
        raise ValueError("ema_update_interval must be > 0")
    if not 0.0 <= float(config["eval_segment_overlap"]) < 1.0:
        raise ValueError("eval_segment_overlap must be in the range [0.0, 1.0)")
    if float(config["layerwise_lr_decay"]) <= 0.0:
        raise ValueError("layerwise_lr_decay must be > 0")


def resolve_eval_tracking_policy(config: Mapping[str, Any]) -> bool:
    """Gate eval-set tracking during training to avoid accidental leakage."""

    requested = str_to_bool(config.get("track_eval_during_training", "False"))
    explicitly_allowed = str_to_bool(config.get("allow_eval_tracking_leak", "False"))

    if requested and not explicitly_allowed:
        print(
            "[eval] disabled track_eval_during_training because it exposes labeled eval metrics "
            "during training. Set allow_eval_tracking_leak=True only if you intentionally want that."
        )
        return False

    return requested


def resolve_model_module_name(architecture: str) -> str:
    """Resolve legacy architecture keys to the new v2 model when appropriate."""

    normalized = architecture.strip()
    lower_name = normalized.lower()

    candidates: list[str] = []

    if lower_name in {"ast", "ast_v2"}:
        candidates.extend(["models.ast_v2", "ast_v2"])

    if "." in normalized:
        candidates.append(normalized)
    else:
        candidates.extend([f"models.{normalized}", normalized])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            import_module(candidate)
            return candidate
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        f"could not resolve architecture '{architecture}'. Tried: {', '.join(candidates)}"
    )


def get_model(model_config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Instantiate the requested model."""

    architecture = str(model_config["architecture"])
    module_name = resolve_model_module_name(architecture)
    module = import_module(module_name)
    model_cls = getattr(module, "Model")
    model = model_cls(model_config).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model module: {module_name}")
    print(f"Total params: {total_params:,}  |  Trainable: {trainable_params:,}")
    return model


# ============================================================
# Training helpers
# ============================================================
def build_training_objects(
    model: nn.Module,
    config: Mapping[str, Any],
    steps_per_epoch: int,
    device: torch.device,
) -> TrainingObjects:
    """Build the optimizer stack with safe AMP / GAM coordination."""

    backbone_lr = float(config.get("backbone_lr", 1e-5))
    fusion_lr = float(config.get("fusion_lr", 5e-5))
    head_lr = float(config.get("head_lr", 1e-4))
    layerwise_lr_decay = float(config.get("layerwise_lr_decay", 1.0))
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
    requested_use_gam = str_to_bool(config.get("use_gam", "True"))
    requested_amp = str_to_bool(config.get("use_amp", "True"))
    amp_enabled = device.type == "cuda" and requested_amp
    use_gam = requested_use_gam

    if hasattr(model, "param_groups"):
        model_parameters = model.param_groups(
            backbone_lr=backbone_lr,
            fusion_lr=fusion_lr,
            head_lr=head_lr,
            layerwise_lr_decay=layerwise_lr_decay,
            weight_decay=float(config["optim_config"].get("weight_decay", 1e-4)),
        )
    else:
        model_parameters = model.parameters()

    optim_config = dict(config["optim_config"])
    optim_config["epochs"] = int(config["num_epochs"])
    optim_config["steps_per_epoch"] = math.ceil(steps_per_epoch / grad_accum_steps)

    base_optimizer, scheduler = create_optimizer(model_parameters, optim_config)

    if use_gam and amp_enabled:
        print("[train] GAM requested with AMP. Falling back to base optimizer + AMP for stability.")
        use_gam = False

    if use_gam:
        optimizer: GAMOptimizer | torch.optim.Optimizer = GAMOptimizer(
            base_optimizer=base_optimizer,
            rho=float(config.get("gam_rho", 0.03)),
            grad_rho=float(config.get("gam_grad_rho", 0.02)),
            adaptive=False,
        )
        print(f"[train] GAM enabled (rho={config.get('gam_rho', 0.03)})")
    else:
        optimizer = base_optimizer
        if amp_enabled:
            print("[train] Using base optimizer with AMP.")
        else:
            print("[train] Using base optimizer without GAM.")

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    return TrainingObjects(
        base_optimizer=base_optimizer,
        optimizer=optimizer,
        scheduler=scheduler,
        use_gam=use_gam,
        amp_enabled=amp_enabled,
        scaler=scaler if amp_enabled else None,
    )


def compute_model_loss(
    model: nn.Module,
    batch: AudioBatch,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward pass and compute the model loss."""

    labels = batch.labels.to(device)
    attack_labels = batch.attack_labels.to(device)
    embeddings, logits = model(batch.waveforms)

    if hasattr(model, "compute_loss"):
        loss, components = model.compute_loss(
            embeddings=embeddings,
            logits=logits,
            labels=labels,
            ce_criterion=None,
            attack_labels=attack_labels,
        )
        return loss, components

    loss = F.cross_entropy(logits, labels)
    return loss, {"total": loss.detach()}


def maybe_clip_gradients(model: nn.Module, grad_clip_norm: float) -> None:
    """Apply gradient clipping if configured."""

    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)


def autocast_context(device: torch.device, enabled: bool) -> Any:
    """Return a device-appropriate autocast context."""

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def process_microbatch_group(
    group: Sequence[AudioBatch],
    model: nn.Module,
    training_objects: TrainingObjects,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, bool]:
    """Apply one optimizer update from a group of micro-batches."""

    divisor = float(len(group))
    weighted_loss_sum = 0.0
    step_completed = False
    base_optimizer = training_objects.base_optimizer

    base_optimizer.zero_grad(set_to_none=True)

    if training_objects.use_gam and isinstance(training_objects.optimizer, GAMOptimizer):
        # ----------------------------------------------------
        # First pass: accumulate gradients over the full group
        # ----------------------------------------------------
        for micro_batch in group:
            loss, _ = compute_model_loss(model, micro_batch, device)
            (loss / divisor).backward()
            weighted_loss_sum += loss.item() * len(micro_batch.waveforms)

        maybe_clip_gradients(model, grad_clip_norm)
        training_objects.optimizer.first_step(zero_grad=True)

        # ----------------------------------------------------
        # Second pass: replay the exact same micro-batches
        # ----------------------------------------------------
        for micro_batch in group:
            loss, _ = compute_model_loss(model, micro_batch, device)
            (loss / divisor).backward()

        maybe_clip_gradients(model, grad_clip_norm)
        training_objects.optimizer.second_step(zero_grad=True)
        step_completed = True
    else:
        # ----------------------------------------------------
        # Standard optimizer path
        # ----------------------------------------------------
        scaler = training_objects.scaler
        for micro_batch in group:
            with autocast_context(device, training_objects.amp_enabled):
                loss, _ = compute_model_loss(model, micro_batch, device)
                scaled_loss = loss / divisor

            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            weighted_loss_sum += loss.item() * len(micro_batch.waveforms)

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(base_optimizer)
        maybe_clip_gradients(model, grad_clip_norm)

        if scaler is not None and scaler.is_enabled():
            previous_scale = float(scaler.get_scale())
            scaler.step(base_optimizer)
            scaler.update()
            step_completed = float(scaler.get_scale()) >= previous_scale
        else:
            base_optimizer.step()
            step_completed = True
        base_optimizer.zero_grad(set_to_none=True)

    if scheduler is not None and step_completed:
        scheduler.step()

    return weighted_loss_sum, step_completed


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    training_objects: TrainingObjects,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    grad_accum_steps: int,
    grad_clip_norm: float,
    ema_tracker: Optional[ModelEMA] = None,
    ema_update_interval: int = 1,
) -> float:
    """Train for one epoch with correct GAM-aware accumulation."""

    running_loss = 0.0
    num_total = 0
    model.train()
    optimizer_updates = 0

    microbatch_group: list[AudioBatch] = []

    for batch in tqdm(train_loader, desc="train", leave=False):
        microbatch_group.append(batch)
        num_total += len(batch.waveforms)

        if len(microbatch_group) < grad_accum_steps:
            continue

        group_loss, step_completed = process_microbatch_group(
            group=microbatch_group,
            model=model,
            training_objects=training_objects,
            scheduler=scheduler,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        running_loss += group_loss
        if step_completed and ema_tracker is not None:
            optimizer_updates += 1
            if optimizer_updates % max(1, ema_update_interval) == 0:
                ema_tracker.update(model)
        microbatch_group = []

    if microbatch_group:
        group_loss, step_completed = process_microbatch_group(
            group=microbatch_group,
            model=model,
            training_objects=training_objects,
            scheduler=scheduler,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        running_loss += group_loss
        if step_completed and ema_tracker is not None:
            optimizer_updates += 1
            if optimizer_updates % max(1, ema_update_interval) == 0:
                ema_tracker.update(model)

    return float(running_loss / max(1, num_total))


# ============================================================
# Evaluation helpers
# ============================================================
def save_score_txt(score_txt_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with open(score_txt_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row['speaker_id']} {row['utt_id']} {row['score']} {row['true_label_text']}\n"
            )


def save_prediction_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_confusion_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute threshold-dependent metrics for a fixed score threshold."""

    predictions = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy = float((tp + tn) / max(1, len(labels)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision_bonafide": precision,
        "recall_bonafide": recall,
        "f1_bonafide": f1,
    }


def compute_binary_metrics_from_rows(
    rows: Sequence[dict[str, Any]],
    decision_threshold: Optional[float] = None,
    decision_source: str = "split_eer",
) -> dict[str, Any]:
    """Compute both ranking metrics and explicit threshold metrics."""

    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["true_label_int"] for row in rows], dtype=np.int64)

    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    eer, frr, far, thresholds = compute_eer(bona_scores, spoof_scores)
    threshold_index = int(np.argmin(np.abs(frr - far)))
    threshold_eer = float(thresholds[threshold_index])
    if decision_threshold is None:
        decision_threshold = threshold_eer
        decision_source = "split_eer"

    decision_metrics = compute_confusion_metrics(scores, labels, threshold=float(decision_threshold))
    zero_metrics = compute_confusion_metrics(scores, labels, threshold=0.0)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    return {
        "scores": scores,
        "labels": labels,
        "bona_scores": bona_scores,
        "spoof_scores": spoof_scores,
        "eer": float(eer),
        "frr": frr,
        "far": far,
        "thresholds": thresholds,
        "threshold_eer": threshold_eer,
        "decision_threshold": float(decision_threshold),
        "decision_source": decision_source,
        "decision_metrics": decision_metrics,
        "zero_metrics": zero_metrics,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
    }


def annotate_prediction_rows(
    rows: Sequence[dict[str, Any]],
    decision_threshold: float,
    decision_source: str,
) -> list[dict[str, Any]]:
    """Add explicit threshold decisions to prediction rows."""

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        score = float(row["score"])
        pred_zero = int(score >= 0.0)
        pred_decision = int(score >= decision_threshold)
        enriched = dict(row)
        enriched.update(
            {
                "decision_threshold": float(decision_threshold),
                "decision_source": decision_source,
                "pred_label_text_zero": "bonafide" if pred_zero == 1 else "spoof",
                "pred_label_int_zero": pred_zero,
                "is_correct_zero": int(pred_zero == row["true_label_int"]),
                "pred_label_text_decision": "bonafide" if pred_decision == 1 else "spoof",
                "pred_label_int_decision": pred_decision,
                "is_correct_decision": int(pred_decision == row["true_label_int"]),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def build_group_breakdown(
    rows: Sequence[dict[str, Any]],
    key: str,
    decision_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Summarize subgroup behavior for richer diagnostics."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, UNKNOWN_TOKEN))].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in grouped.items():
        scores = np.asarray([row["score"] for row in group_rows], dtype=np.float64)
        labels = np.asarray([row["true_label_int"] for row in group_rows], dtype=np.int64)
        zero_metrics = compute_confusion_metrics(scores, labels, threshold=0.0)
        decision_metrics = compute_confusion_metrics(scores, labels, threshold=decision_threshold)

        group_summary: dict[str, Any] = {
            "count": int(len(group_rows)),
            "bonafide_count": int(np.sum(labels == 1)),
            "spoof_count": int(np.sum(labels == 0)),
            "accuracy_at_zero": float(zero_metrics["accuracy"]),
            "accuracy_at_decision": float(decision_metrics["accuracy"]),
            "decision_threshold": float(decision_threshold),
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
        }

        if np.any(labels == 1) and np.any(labels == 0):
            eer, _, _, _ = compute_eer(scores[labels == 1], scores[labels == 0])
            group_summary["eer"] = float(eer)

        summary[group_name] = group_summary

    return summary


def collect_predictions(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    trial_lookup: Mapping[str, ProtocolEntry],
    save_embeddings: bool = False,
    embedding_max_samples: int = 3000,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[float], list[int], list[str]]:
    """Run evaluation and collect per-utterance predictions."""

    model.eval()
    rows: list[dict[str, Any]] = []
    sampled_embeddings: list[np.ndarray] = []
    sampled_scores: list[float] = []
    sampled_labels: list[int] = []
    sampled_ids: list[str] = []
    use_all_embeddings = embedding_max_samples is None or embedding_max_samples <= 0

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="eval", leave=False):
            embeddings, logits = model(batch.waveforms)
            embeddings_np = embeddings.detach().cpu().numpy()
            scores_np = logits[:, 1].detach().cpu().numpy()

            for index, utt_id in enumerate(batch.utt_ids):
                entry = trial_lookup[utt_id]
                score = float(scores_np[index])
                clipped_score = float(np.clip(score, -60.0, 60.0))

                row = {
                    "speaker_id": entry.speaker_id,
                    "utt_id": entry.utt_id,
                    "gender": entry.gender,
                    "codec_id": entry.codec_id,
                    "codec_quality": entry.codec_quality,
                    "source_utt_id": entry.source_utt_id,
                    "acoustic_condition": entry.acoustic_condition,
                    "attack_id": entry.attack_id,
                    "true_label_text": entry.label_text,
                    "true_label_int": entry.label_int,
                    "logit_spoof": float(-score),
                    "logit_bonafide": float(score),
                    "prob_spoof_uncalibrated": float(1.0 / (1.0 + np.exp(clipped_score))),
                    "prob_bonafide_uncalibrated": float(1.0 / (1.0 + np.exp(-clipped_score))),
                    "score": score,
                }
                rows.append(row)

                if save_embeddings and (use_all_embeddings or len(sampled_ids) < embedding_max_samples):
                    sampled_embeddings.append(embeddings_np[index])
                    sampled_scores.append(score)
                    sampled_labels.append(entry.label_int)
                    sampled_ids.append(entry.utt_id)

    return rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids


def save_group_metrics_json(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    decision_threshold: float,
    decision_source: str,
) -> None:
    """Save subgroup metrics to JSON for later analysis."""

    summary = {
        "decision_threshold": float(decision_threshold),
        "decision_source": decision_source,
        "codec_id": build_group_breakdown(rows, "codec_id", decision_threshold),
        "codec_quality": build_group_breakdown(rows, "codec_quality", decision_threshold),
        "acoustic_condition": build_group_breakdown(rows, "acoustic_condition", decision_threshold),
        "attack_id": build_group_breakdown(rows, "attack_id", decision_threshold),
        "gender": build_group_breakdown(rows, "gender", decision_threshold),
    }
    save_json(output_path, summary)


def _save_det_roc(stats: Mapping[str, Any], output_dir: Path, split_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(stats["far"], stats["frr"])
    axes[0].set_xlabel("FAR")
    axes[0].set_ylabel("FRR")
    axes[0].set_title(f"{split_name} DET")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stats["fpr"], stats["tpr"])
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title(f"{split_name} ROC  AUC={stats['auc']:.4f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{split_name}_det_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(
    split_name: str,
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    trial_entries: Sequence[ProtocolEntry],
    score_txt_path: Optional[Path],
    prediction_csv_path: Optional[Path],
    output_dir: Path,
    save_all_figures: bool = True,
    save_embeddings: bool = False,
    embedding_save_path: Optional[Path] = None,
    embedding_max_samples: int = 3000,
    save_group_metric_json: bool = True,
    decision_threshold: Optional[float] = None,
    decision_source: str = "split_eer",
    calibrator: Optional[LogisticScoreCalibrator] = None,
    save_calibrated_metrics: bool = True,
    return_rows: bool = False,
) -> dict[str, Any]:
    """Evaluate one split and save the standard artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = protocol_lookup(trial_entries)

    rows, sampled_embeddings, sampled_scores, sampled_labels, sampled_ids = collect_predictions(
        data_loader=data_loader,
        model=model,
        device=device,
        trial_lookup=lookup,
        save_embeddings=save_embeddings,
        embedding_max_samples=embedding_max_samples,
    )

    if score_txt_path is None:
        score_txt_path = output_dir / f"{split_name}_scores.txt"
    save_score_txt(score_txt_path, rows)

    dcf, eer_fn, cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=score_txt_path,
        output_file=output_dir / f"{split_name}_DCF_EER.txt",
        printout=False,
    )
    stats = compute_binary_metrics_from_rows(
        rows,
        decision_threshold=decision_threshold,
        decision_source=decision_source,
    )
    enriched_rows = annotate_prediction_rows(
        rows,
        decision_threshold=float(stats["decision_threshold"]),
        decision_source=str(stats["decision_source"]),
    )

    if prediction_csv_path is not None:
        save_prediction_csv(prediction_csv_path, enriched_rows)

    if save_embeddings and sampled_embeddings and embedding_save_path is not None:
        np.savez_compressed(
            embedding_save_path,
            embeddings=np.asarray(sampled_embeddings),
            scores=np.asarray(sampled_scores),
            labels=np.asarray(sampled_labels),
            utt_ids=np.asarray(sampled_ids),
        )

    if save_group_metric_json:
        save_group_metrics_json(
            enriched_rows,
            output_path=output_dir / f"{split_name}_group_metrics.json",
            decision_threshold=float(stats["decision_threshold"]),
            decision_source=str(stats["decision_source"]),
        )

    if save_all_figures:
        _save_det_roc(stats, output_dir, split_name)

    result: dict[str, Any] = {
        "dcf": float(dcf),
        "eer": float(eer_fn),
        "cllr": float(cllr),
        "auc": float(stats["auc"]),
        "threshold_eer": float(stats["threshold_eer"]),
        "decision_threshold": float(stats["decision_threshold"]),
        "decision_source": str(stats["decision_source"]),
        "tp": int(stats["decision_metrics"]["tp"]),
        "tn": int(stats["decision_metrics"]["tn"]),
        "fp": int(stats["decision_metrics"]["fp"]),
        "fn": int(stats["decision_metrics"]["fn"]),
        "accuracy": float(stats["decision_metrics"]["accuracy"]),
        "precision_bonafide": float(stats["decision_metrics"]["precision_bonafide"]),
        "recall_bonafide": float(stats["decision_metrics"]["recall_bonafide"]),
        "f1_bonafide": float(stats["decision_metrics"]["f1_bonafide"]),
        "accuracy_at_zero": float(stats["zero_metrics"]["accuracy"]),
        "f1_bonafide_at_zero": float(stats["zero_metrics"]["f1_bonafide"]),
    }

    if calibrator is not None and calibrator.is_fitted and save_calibrated_metrics:
        calibrated_rows = [dict(row, score=calibrator.transform_scalar(float(row["score"]))) for row in rows]
        calibrated_score_txt = output_dir / f"{split_name}_scores_calibrated.txt"
        save_score_txt(calibrated_score_txt, calibrated_rows)
        cal_dcf, cal_eer, cal_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=calibrated_score_txt,
            output_file=output_dir / f"{split_name}_DCF_EER_calibrated.txt",
            printout=False,
        )
        cal_stats = compute_binary_metrics_from_rows(
            calibrated_rows,
            decision_threshold=0.0,
            decision_source="calibrated_zero",
        )
        calibrated_prediction_rows = annotate_prediction_rows(
            calibrated_rows,
            decision_threshold=0.0,
            decision_source="calibrated_zero",
        )
        if prediction_csv_path is not None:
            calibrated_prediction_path = prediction_csv_path.with_name(
                f"{prediction_csv_path.stem}_calibrated{prediction_csv_path.suffix}"
            )
            save_prediction_csv(calibrated_prediction_path, calibrated_prediction_rows)
        calibrated_summary = {
            "dcf": float(cal_dcf),
            "eer": float(cal_eer),
            "cllr": float(cal_cllr),
            "auc": float(cal_stats["auc"]),
            "decision_threshold": 0.0,
            "decision_source": "calibrated_zero",
            "accuracy": float(cal_stats["decision_metrics"]["accuracy"]),
            "precision_bonafide": float(cal_stats["decision_metrics"]["precision_bonafide"]),
            "recall_bonafide": float(cal_stats["decision_metrics"]["recall_bonafide"]),
            "f1_bonafide": float(cal_stats["decision_metrics"]["f1_bonafide"]),
        }
        save_json(output_dir / f"{split_name}_calibrated_metrics.json", calibrated_summary)
        result["calibrated"] = calibrated_summary

    if return_rows:
        result["rows"] = enriched_rows

    return result


# ============================================================
# t-SNE helpers
# ============================================================
def build_tsne(**kwargs: Any) -> TSNE:
    """Create a TSNE instance compatible across sklearn versions."""

    try:
        return TSNE(n_iter=500, **kwargs)
    except TypeError:
        return TSNE(max_iter=500, **kwargs)


def save_embedding_tsne(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    trial_entries: Sequence[ProtocolEntry],
    save_dir: Path,
    max_samples: int = 500,
    split_name: str = "dev",
) -> None:
    """Extract embeddings, run t-SNE, and save a plot plus CSV."""

    lookup = protocol_lookup(trial_entries)
    save_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    scores: list[float] = []
    utt_ids: list[str] = []

    model.eval()
    collected = 0

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"tsne_{split_name}", leave=False):
            batch_embeddings, batch_logits = model(batch.waveforms)
            batch_scores = batch_logits[:, 1].detach().cpu().numpy()
            batch_embeddings_np = batch_embeddings.detach().cpu().numpy()

            for index, utt_id in enumerate(batch.utt_ids):
                if collected >= max_samples:
                    break
                embeddings.append(batch_embeddings_np[index])
                scores.append(float(batch_scores[index]))
                utt_ids.append(utt_id)
                collected += 1

            if collected >= max_samples:
                break

    if len(utt_ids) < 10:
        print(f"[tsne] not enough samples for {split_name}")
        return

    label_array = np.asarray([lookup[utt_id].label_int for utt_id in utt_ids], dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float32)
    embedding_array = np.asarray(embeddings, dtype=np.float32)

    perplexity = max(5, min(20, len(utt_ids) // 3))
    tsne = build_tsne(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords = tsne.fit_transform(embedding_array)

    csv_path = save_dir / f"{split_name}_tsne_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "label", "score", "tsne_x", "tsne_y"])
        for utt_id, label, score, (x_coord, y_coord) in zip(utt_ids, label_array, score_array, coords):
            writer.writerow([utt_id, int(label), float(score), float(x_coord), float(y_coord)])

    fig = plt.figure(figsize=(8, 6))
    for mask, label_name in ((label_array == 0, "spoof"), (label_array == 1, "bonafide")):
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=label_name)
    plt.legend()
    plt.title(f"{split_name} TSNE")
    plt.tight_layout()
    fig.savefig(save_dir / f"{split_name}_tsne.png", dpi=150)
    plt.close(fig)

    print(f"[tsne] saved {split_name} ({len(utt_ids)} samples)")


def save_tsne_pair(
    model: nn.Module,
    device: torch.device,
    dev_loader: DataLoader,
    dev_entries: Sequence[ProtocolEntry],
    eval_loader: Optional[DataLoader],
    eval_entries: Sequence[ProtocolEntry],
    model_tag: Path,
    tag: str,
    max_samples: int,
) -> None:
    """Save both dev and eval t-SNE plots."""

    save_embedding_tsne(
        data_loader=dev_loader,
        model=model,
        device=device,
        trial_entries=dev_entries,
        save_dir=model_tag / f"tsne_{tag}_dev",
        max_samples=max_samples,
        split_name="dev",
    )

    if eval_loader is not None and eval_entries:
        save_embedding_tsne(
            data_loader=eval_loader,
            model=model,
            device=device,
            trial_entries=eval_entries,
            save_dir=model_tag / f"tsne_{tag}_eval",
            max_samples=max_samples,
            split_name="eval",
        )


# ============================================================
# Plotting / logging helpers
# ============================================================
def plot_all_curves(rows: Sequence[dict[str, Any]], figures_dir: Path) -> None:
    """Plot the main training curves."""

    def _plot(keys: Sequence[str], ylabel: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 4))
        for key in keys:
            values = [row[key] for row in rows if key in row]
            if values:
                axis.plot(range(len(values)), values, label=key)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(figures_dir / filename, dpi=120, bbox_inches="tight")
        plt.close(fig)

    _plot(["loss"], "Loss", "loss_curve.png")
    _plot(["lr"], "LR", "lr_curve.png")
    _plot(["dev_eer", "eval_eer"], "EER", "eer_curve.png")
    _plot(["dev_dcf", "eval_dcf"], "minDCF", "dcf_curve.png")
    _plot(["dev_cllr", "eval_cllr"], "CLLR", "cllr_curve.png")
    _plot(["dev_auc", "eval_auc"], "AUC", "auc_curve.png")
    _plot(["dev_accuracy", "eval_accuracy"], "Accuracy", "accuracy_curve.png")
    _plot(["dev_f1_bonafide", "eval_f1_bonafide"], "F1", "f1_curve.png")


def append_history_csv(csv_path: Path, row: Mapping[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def build_calibrator_from_rows(rows: Sequence[dict[str, Any]]) -> Optional[LogisticScoreCalibrator]:
    """Fit a score calibrator from prediction rows when possible."""

    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["true_label_int"] for row in rows], dtype=np.int64)
    calibrator = LogisticScoreCalibrator()
    return calibrator if calibrator.fit(scores, labels) else None


@contextmanager
def maybe_ema_context(
    model: nn.Module,
    ema_tracker: Optional[ModelEMA],
    enabled: bool,
) -> Iterable[None]:
    """Evaluate under EMA weights when available."""

    if enabled and ema_tracker is not None and ema_tracker.num_updates > 0:
        with ema_tracker.average_parameters(model):
            yield
        return

    with nullcontext():
        yield


# ============================================================
# Main entry point
# ============================================================
def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.loads(handle.read())

    config = apply_default_config(config)
    validate_config(config)
    set_seed(args.seed, config)

    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    num_epochs = int(config["num_epochs"])
    batch_size = int(config["batch_size"])

    paths = ExperimentPaths.build(
        output_dir=output_dir,
        config_path=args.config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        comment=args.comment,
    )

    copy(args.config, paths.model_tag / "config.conf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] CUDA not detected. Running on CPU will be slow.")
    print(f"Device: {device}")

    run_final_eval_after_training = str_to_bool(config.get("run_final_eval_after_training", "False"))
    include_eval_split = bool(args.eval or run_final_eval_after_training)
    data_bundle = build_data_bundle(
        database_path,
        args.seed,
        config,
        include_eval=include_eval_split,
    )
    model = get_model(config["model_config"], device)

    writer = SummaryWriter(str(paths.model_tag))

    use_ema = str_to_bool(config.get("use_ema", "True"))
    eval_use_ema = str_to_bool(config.get("eval_use_ema", "True"))
    enable_calibration = str_to_bool(config.get("enable_calibration", "False"))
    save_calibrated_metrics = str_to_bool(config.get("save_calibrated_metrics", "True"))
    track_eval_during_training = resolve_eval_tracking_policy(config)
    ema_tracker = ModelEMA(model, decay=float(config.get("ema_decay", 0.9995))) if use_ema else None

    save_json(
        paths.run_summary_json,
        {
            "config_path": args.config,
            "output_dir": str(paths.model_tag),
            "database_path": str(database_path),
            "num_epochs_requested": num_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": int(config["grad_accum_steps"]),
            "effective_batch": batch_size * int(config["grad_accum_steps"]),
            "use_gam": str(config["use_gam"]),
            "use_amp": str(config["use_amp"]),
            "use_ema": use_ema,
            "ema_decay": float(config.get("ema_decay", 0.9995)),
            "ema_update_interval": int(config.get("ema_update_interval", 4)),
            "eval_use_ema": eval_use_ema,
            "seed": int(args.seed),
            "device": str(device),
            "comment": args.comment,
            "train_items": len(data_bundle.train_entries),
            "dev_items": len(data_bundle.dev_entries),
            "eval_items": len(data_bundle.eval_entries),
            "attack_label_classes": len(data_bundle.attack_label_map),
            "eval_num_segments": int(config["model_config"]["eval_num_segments"]),
            "eval_segment_overlap": float(config["model_config"]["eval_segment_overlap"]),
            "eval_score_pooling": str(config["model_config"]["eval_score_pooling"]),
            "binary_loss": str(config["model_config"]["binary_loss"]),
            "ce_weight_spoof": float(config["model_config"]["ce_weight_spoof"]),
            "ce_weight_bonafide": float(config["model_config"]["ce_weight_bonafide"]),
            "balanced_sampling": str(config["balanced_sampling"]),
            "balanced_sampling_power": float(config["balanced_sampling_power"]),
            "enable_calibration": enable_calibration,
            "track_eval_during_training_requested": str(config["track_eval_during_training"]),
            "track_eval_during_training_effective": track_eval_during_training,
            "allow_eval_tracking_leak": str(config["allow_eval_tracking_leak"]),
            "run_final_eval_after_training": run_final_eval_after_training,
            "eval_split_loaded": include_eval_split,
        },
    )

    save_tsne_flag = str_to_bool(config.get("save_tsne", "False"))
    save_epoch_scores = str_to_bool(config.get("save_epoch_scores", "True"))
    save_epoch_predictions = str_to_bool(config.get("save_epoch_predictions", "True"))
    save_epoch_metric_json = str_to_bool(config.get("save_epoch_metric_json", "True"))
    save_epoch_figures = str_to_bool(config.get("save_epoch_figures", "True"))
    save_embedding_archives = str_to_bool(config.get("save_embedding_archives", "True"))
    save_group_metric_json = str_to_bool(config.get("save_group_metric_json", "True"))
    embedding_archive_max = int(config.get("embedding_archive_max_samples", 3000))
    # --------------------------------------------------------
    # Eval-only path
    # --------------------------------------------------------
    if args.eval:
        model_path = args.eval_model_weights or config.get("model_path")
        if not model_path:
            raise ValueError("evaluation requested but no model_path / eval_model_weights provided")
        if data_bundle.eval_loader is None or not data_bundle.eval_entries:
            raise ValueError("evaluation requested but the eval split is not available")

        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")

        eval_dir = paths.model_tag / "eval_loaded_model"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with maybe_ema_context(model, ema_tracker, enabled=eval_use_ema):
            dev_metrics = evaluate_split(
                split_name="dev",
                data_loader=data_bundle.dev_loader,
                model=model,
                device=device,
                trial_entries=data_bundle.dev_entries,
                score_txt_path=eval_dir / "dev_scores.txt",
                prediction_csv_path=eval_dir / "dev_predictions.csv",
                output_dir=eval_dir / "dev",
                save_all_figures=True,
                save_embeddings=True,
                embedding_save_path=eval_dir / "dev_embeddings.npz",
                embedding_max_samples=embedding_archive_max,
                save_group_metric_json=save_group_metric_json,
                save_calibrated_metrics=save_calibrated_metrics,
                return_rows=enable_calibration,
            )
            calibrator = build_calibrator_from_rows(dev_metrics["rows"]) if enable_calibration else None
            eval_metrics = evaluate_split(
                split_name="eval",
                data_loader=data_bundle.eval_loader,
                model=model,
                device=device,
                trial_entries=data_bundle.eval_entries,
                score_txt_path=eval_dir / "eval_scores.txt",
                prediction_csv_path=eval_dir / "eval_predictions.csv",
                output_dir=eval_dir / "eval",
                save_all_figures=True,
                save_embeddings=True,
                embedding_save_path=eval_dir / "eval_embeddings.npz",
                embedding_max_samples=embedding_archive_max,
                save_group_metric_json=save_group_metric_json,
                decision_threshold=float(dev_metrics["threshold_eer"]),
                decision_source="dev_eer_threshold",
                calibrator=calibrator,
                save_calibrated_metrics=save_calibrated_metrics,
            )
        print(
            "dev_eer: {:.3f}  dev_dcf: {:.5f}  dev_cllr: {:.5f}".format(
                dev_metrics["eer"],
                dev_metrics["dcf"],
                dev_metrics["cllr"],
            )
        )
        print(
            "eval_eer: {:.3f}  eval_dcf: {:.5f}  eval_cllr: {:.5f}".format(
                eval_metrics["eer"],
                eval_metrics["dcf"],
                eval_metrics["cllr"],
            )
        )

        if save_tsne_flag:
            save_tsne_pair(
                model=model,
                device=device,
                dev_loader=data_bundle.dev_loader,
                dev_entries=data_bundle.dev_entries,
                eval_loader=data_bundle.eval_loader,
                eval_entries=data_bundle.eval_entries,
                model_tag=paths.model_tag,
                tag="eval_only",
                max_samples=int(config.get("tsne_max_samples", 100000)),
            )

        writer.close()
        return

    # --------------------------------------------------------
    # Train path
    # --------------------------------------------------------
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
    grad_clip_norm = float(config.get("grad_clip_norm", 5.0))
    unfreeze_epoch = int(config.get("unfreeze_epoch", 5))
    swa_start_epoch = int(config.get("swa_start_epoch", 10))
    unfreeze_strategy = str(config.get("unfreeze_strategy", "gradual")).strip().lower()
    unfreeze_layers_per_epoch = max(1, int(config.get("unfreeze_layers_per_epoch", 1)))

    training_objects = build_training_objects(
        model=model,
        config=config,
        steps_per_epoch=len(data_bundle.train_loader),
        device=device,
    )
    save_json(
        paths.archive_dir / "training_runtime.json",
        {
            "use_gam_requested": str(config.get("use_gam", "True")),
            "use_gam_actual": bool(training_objects.use_gam),
            "amp_enabled_actual": bool(training_objects.amp_enabled),
            "layerwise_lr_decay": float(config.get("layerwise_lr_decay", 1.0)),
            "binary_loss": str(config["model_config"]["binary_loss"]),
            "ema_update_interval": int(config.get("ema_update_interval", 4)),
            "eval_use_ema": bool(eval_use_ema),
        },
    )

    swa_tracker = RunningSWA()
    best_dev_eer = float("inf")
    best_dev_dcf = float("inf")
    best_dev_cllr = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history_rows: list[dict[str, Any]] = []
    n_swa_updates = 0

    early_stop_enabled = str_to_bool(config.get("early_stop", "True"))
    early_stop_patience = int(config.get("early_stop_patience", 10))
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0005))

    metric_log_handle = open(paths.metric_log_path, "a", encoding="utf-8")
    metric_log_handle.write("=" * 20 + "\n")
    metric_log_handle.flush()

    try:
        for epoch in range(num_epochs):
            print(f"\n========== Epoch {epoch:03d} ==========")

            # ------------------------------------------------
            # Optional layer unfreeze
            # ------------------------------------------------
            if epoch >= unfreeze_epoch:
                if unfreeze_strategy == "all" and epoch == unfreeze_epoch and hasattr(model, "unfreeze_all"):
                    model.unfreeze_all()
                    for group in training_objects.base_optimizer.param_groups:
                        if str(group.get("name", "")).startswith("wavlm_"):
                            group["lr"] *= 0.5
                    print(
                        f"[train] Epoch {epoch}: transformer fully unfrozen, "
                        "backbone LR groups scaled by 0.5"
                    )
                elif unfreeze_strategy == "gradual" and hasattr(model, "unfreeze_top_layers"):
                    newly_unfrozen = int(model.unfreeze_top_layers(unfreeze_layers_per_epoch))
                    if newly_unfrozen > 0:
                        print(
                            f"[train] Epoch {epoch}: gradually unfroze {newly_unfrozen} "
                            "previously frozen WavLM layers"
                        )

            # ------------------------------------------------
            # One training epoch
            # ------------------------------------------------
            running_loss = train_epoch(
                train_loader=data_bundle.train_loader,
                model=model,
                training_objects=training_objects,
                device=device,
                scheduler=training_objects.scheduler,
                grad_accum_steps=grad_accum_steps,
                grad_clip_norm=grad_clip_norm,
                ema_tracker=ema_tracker,
                ema_update_interval=int(config.get("ema_update_interval", 4)),
            )
            current_lr = float(training_objects.base_optimizer.param_groups[0]["lr"])

            # ------------------------------------------------
            # Per-epoch artifact directories
            # ------------------------------------------------
            epoch_scores_dir = paths.scores_dir / f"epoch_{epoch:03d}"
            epoch_predictions_dir = paths.predictions_dir / f"epoch_{epoch:03d}"
            epoch_figures_dir = paths.figures_dir / f"epoch_{epoch:03d}"
            epoch_metrics_dir = paths.metrics_dir / f"epoch_{epoch:03d}"
            epoch_embeddings_dir = paths.embedding_dir / f"epoch_{epoch:03d}"

            for path in (
                epoch_scores_dir,
                epoch_predictions_dir,
                epoch_figures_dir,
                epoch_metrics_dir,
                epoch_embeddings_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------
            # Dev and eval metrics
            # ------------------------------------------------
            eval_metrics: Optional[dict[str, Any]] = None
            with maybe_ema_context(model, ema_tracker, enabled=eval_use_ema):
                dev_metrics = evaluate_split(
                    split_name="dev",
                    data_loader=data_bundle.dev_loader,
                    model=model,
                    device=device,
                    trial_entries=data_bundle.dev_entries,
                    score_txt_path=epoch_scores_dir / "dev_scores.txt" if save_epoch_scores else None,
                    prediction_csv_path=epoch_predictions_dir / "dev_predictions.csv" if save_epoch_predictions else None,
                    output_dir=epoch_figures_dir / "dev" if save_epoch_figures else epoch_metrics_dir / "dev",
                    save_all_figures=save_epoch_figures,
                    save_embeddings=save_embedding_archives,
                    embedding_save_path=epoch_embeddings_dir / "dev_embeddings.npz",
                    embedding_max_samples=embedding_archive_max,
                    save_group_metric_json=save_group_metric_json,
                    save_calibrated_metrics=save_calibrated_metrics,
                    return_rows=enable_calibration,
                )
                if track_eval_during_training:
                    calibrator = build_calibrator_from_rows(dev_metrics["rows"]) if enable_calibration else None
                    eval_metrics = evaluate_split(
                        split_name="eval",
                        data_loader=data_bundle.eval_loader,
                        model=model,
                        device=device,
                        trial_entries=data_bundle.eval_entries,
                        score_txt_path=epoch_scores_dir / "eval_scores.txt" if save_epoch_scores else None,
                        prediction_csv_path=epoch_predictions_dir / "eval_predictions.csv" if save_epoch_predictions else None,
                        output_dir=epoch_figures_dir / "eval" if save_epoch_figures else epoch_metrics_dir / "eval",
                        save_all_figures=save_epoch_figures,
                        save_embeddings=save_embedding_archives,
                        embedding_save_path=epoch_embeddings_dir / "eval_embeddings.npz",
                        embedding_max_samples=embedding_archive_max,
                        save_group_metric_json=save_group_metric_json,
                        decision_threshold=float(dev_metrics["threshold_eer"]),
                        decision_source="dev_eer_threshold",
                        calibrator=calibrator,
                        save_calibrated_metrics=save_calibrated_metrics,
                    )

            if epoch == 0 and save_tsne_flag:
                with maybe_ema_context(model, ema_tracker, enabled=eval_use_ema):
                    save_tsne_pair(
                        model=model,
                        device=device,
                        dev_loader=data_bundle.dev_loader,
                        dev_entries=data_bundle.dev_entries,
                        eval_loader=data_bundle.eval_loader,
                        eval_entries=data_bundle.eval_entries,
                        model_tag=paths.model_tag,
                        tag="epoch0",
                        max_samples=int(config.get("tsne_max_samples", 100000)),
                    )

            dev_eer = float(dev_metrics["eer"])
            dev_dcf = float(dev_metrics["dcf"])
            dev_cllr = float(dev_metrics["cllr"])

            print(
                "Loss:{:.6f} LR:{:.2e} dev_eer:{:.4f} dev_dcf:{:.6f} dev_cllr:{:.6f}".format(
                    running_loss,
                    current_lr,
                    dev_eer,
                    dev_dcf,
                    dev_cllr,
                )
            )
            if eval_metrics is not None:
                print(
                    "eval_eer:{:.4f} eval_dcf:{:.6f} eval_cllr:{:.6f}".format(
                        eval_metrics["eer"],
                        eval_metrics["dcf"],
                        eval_metrics["cllr"],
                    )
                )

            # ------------------------------------------------
            # TensorBoard
            # ------------------------------------------------
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_dcf", dev_dcf, epoch)
            writer.add_scalar("dev_cllr", dev_cllr, epoch)
            writer.add_scalar("dev_auc", dev_metrics["auc"], epoch)
            writer.add_scalar("dev_accuracy", dev_metrics["accuracy"], epoch)
            writer.add_scalar("dev_f1_bonafide", dev_metrics["f1_bonafide"], epoch)
            if eval_metrics is not None:
                writer.add_scalar("eval_eer", eval_metrics["eer"], epoch)
                writer.add_scalar("eval_dcf", eval_metrics["dcf"], epoch)
                writer.add_scalar("eval_cllr", eval_metrics["cllr"], epoch)
                writer.add_scalar("eval_auc", eval_metrics["auc"], epoch)
                writer.add_scalar("eval_accuracy", eval_metrics["accuracy"], epoch)
                writer.add_scalar("eval_f1_bonafide", eval_metrics["f1_bonafide"], epoch)

            # ------------------------------------------------
            # Checkpoints and history
            # ------------------------------------------------
            checkpoint_name = f"epoch_{epoch:03d}_devEER_{dev_eer:.6f}.pth"
            torch.save(model.state_dict(), paths.model_save_path / checkpoint_name)

            row = {
                "epoch": int(epoch),
                "loss": float(running_loss),
                "lr": float(current_lr),
                "dev_eer": float(dev_eer),
                "dev_dcf": float(dev_dcf),
                "dev_cllr": float(dev_cllr),
                "dev_auc": float(dev_metrics["auc"]),
                "dev_threshold_eer": float(dev_metrics["threshold_eer"]),
                "dev_decision_threshold": float(dev_metrics["decision_threshold"]),
                "dev_decision_source": str(dev_metrics["decision_source"]),
                "dev_tp": int(dev_metrics["tp"]),
                "dev_tn": int(dev_metrics["tn"]),
                "dev_fp": int(dev_metrics["fp"]),
                "dev_fn": int(dev_metrics["fn"]),
                "dev_accuracy": float(dev_metrics["accuracy"]),
                "dev_accuracy_at_zero": float(dev_metrics["accuracy_at_zero"]),
                "dev_precision_bonafide": float(dev_metrics["precision_bonafide"]),
                "dev_recall_bonafide": float(dev_metrics["recall_bonafide"]),
                "dev_f1_bonafide": float(dev_metrics["f1_bonafide"]),
            }
            if eval_metrics is not None:
                row.update(
                    {
                        "eval_eer": float(eval_metrics["eer"]),
                        "eval_dcf": float(eval_metrics["dcf"]),
                        "eval_cllr": float(eval_metrics["cllr"]),
                        "eval_auc": float(eval_metrics["auc"]),
                        "eval_threshold_eer": float(eval_metrics["threshold_eer"]),
                        "eval_decision_threshold": float(eval_metrics["decision_threshold"]),
                        "eval_decision_source": str(eval_metrics["decision_source"]),
                        "eval_tp": int(eval_metrics["tp"]),
                        "eval_tn": int(eval_metrics["tn"]),
                        "eval_fp": int(eval_metrics["fp"]),
                        "eval_fn": int(eval_metrics["fn"]),
                        "eval_accuracy": float(eval_metrics["accuracy"]),
                        "eval_accuracy_at_zero": float(eval_metrics["accuracy_at_zero"]),
                        "eval_precision_bonafide": float(eval_metrics["precision_bonafide"]),
                        "eval_recall_bonafide": float(eval_metrics["recall_bonafide"]),
                        "eval_f1_bonafide": float(eval_metrics["f1_bonafide"]),
                    }
                )
            history_rows.append(row)
            append_history_csv(paths.history_csv, row)
            append_jsonl(paths.history_jsonl, row)

            if save_epoch_metric_json:
                save_json(epoch_metrics_dir / "metrics.json", row)

            plot_all_curves(history_rows, paths.figures_dir)

            # ------------------------------------------------
            # Best-model tracking
            # ------------------------------------------------
            best_dev_dcf = min(best_dev_dcf, dev_dcf)
            best_dev_cllr = min(best_dev_cllr, dev_cllr)

            improved = dev_eer < (best_dev_eer - early_stop_min_delta)
            if improved:
                print(f"Best model found at epoch {epoch}")
                best_dev_eer = dev_eer
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), paths.model_save_path / "best_ast_raw.pth")
                if ema_tracker is not None and eval_use_ema and ema_tracker.num_updates > 0:
                    ema_tracker.save(paths.model_save_path / "best_ast.pth")
                else:
                    torch.save(model.state_dict(), paths.model_save_path / "best_ast.pth")

                best_payload = {
                    "best_epoch": int(epoch),
                    "best_dev_eer": float(dev_eer),
                    "best_dev_dcf": float(dev_dcf),
                    "best_dev_cllr": float(dev_cllr),
                    "best_dev_auc": float(dev_metrics["auc"]),
                    "best_dev_decision_threshold": float(dev_metrics["decision_threshold"]),
                }
                if eval_metrics is not None:
                    best_payload.update(
                        {
                            "best_eval_eer": float(eval_metrics["eer"]),
                            "best_eval_dcf": float(eval_metrics["dcf"]),
                            "best_eval_cllr": float(eval_metrics["cllr"]),
                            "best_eval_auc": float(eval_metrics["auc"]),
                            "best_eval_decision_threshold": float(eval_metrics["decision_threshold"]),
                        }
                    )
                save_json(paths.best_json, best_payload)
            else:
                epochs_no_improve += 1

            # ------------------------------------------------
            # Optional running SWA
            # ------------------------------------------------
            if epoch >= swa_start_epoch:
                swa_tracker.update(model)
                n_swa_updates += 1
                print(f"[swa] updated (total: {n_swa_updates})")

            writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
            writer.add_scalar("best_dev_dcf", best_dev_dcf, epoch)
            writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

            log_line = (
                f"epoch={epoch}, loss={running_loss:.6f}, lr={current_lr:.8f}, "
                f"dev_eer={dev_eer:.6f}, dev_dcf={dev_dcf:.6f}, dev_cllr={dev_cllr:.6f}, "
                f"dev_auc={dev_metrics['auc']:.6f}, dev_acc={dev_metrics['accuracy']:.6f}, "
                f"dev_f1={dev_metrics['f1_bonafide']:.6f}"
            )
            if eval_metrics is not None:
                log_line += (
                    f", eval_eer={eval_metrics['eer']:.6f}, eval_dcf={eval_metrics['dcf']:.6f}, "
                    f"eval_cllr={eval_metrics['cllr']:.6f}, eval_auc={eval_metrics['auc']:.6f}, "
                    f"eval_acc={eval_metrics['accuracy']:.6f}, eval_f1={eval_metrics['f1_bonafide']:.6f}"
                )
            metric_log_handle.write(log_line + "\n")
            metric_log_handle.flush()

            if early_stop_enabled and epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best={best_epoch}, dev_eer={best_dev_eer:.6f}"
                )
                break

    finally:
        metric_log_handle.close()
        writer.close()

    # --------------------------------------------------------
    # Final best-model evaluation
    # --------------------------------------------------------
    best_weight = paths.model_save_path / "best_ast.pth"
    if best_weight.exists():
        model.load_state_dict(torch.load(best_weight, map_location=device))
        print(f"Loaded best checkpoint: {best_weight}")

    final_dir = paths.model_tag / "final_best_model_eval"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_dev = evaluate_split(
        split_name="dev",
        data_loader=data_bundle.dev_loader,
        model=model,
        device=device,
        trial_entries=data_bundle.dev_entries,
        score_txt_path=final_dir / "dev_best_scores.txt",
        prediction_csv_path=final_dir / "dev_best_predictions.csv",
        output_dir=final_dir / "dev",
        save_all_figures=True,
        save_embeddings=save_embedding_archives,
        embedding_save_path=final_dir / "dev_best_embeddings.npz",
        embedding_max_samples=embedding_archive_max,
        save_group_metric_json=save_group_metric_json,
        save_calibrated_metrics=save_calibrated_metrics,
        return_rows=enable_calibration,
    )
    final_calibrator = build_calibrator_from_rows(final_dev["rows"]) if enable_calibration else None

    final_eval: Optional[dict[str, Any]] = None
    if run_final_eval_after_training:
        if data_bundle.eval_loader is None or not data_bundle.eval_entries:
            raise ValueError("final eval after training was requested but the eval split is not available")
        final_eval = evaluate_split(
            split_name="eval",
            data_loader=data_bundle.eval_loader,
            model=model,
            device=device,
            trial_entries=data_bundle.eval_entries,
            score_txt_path=final_dir / "eval_best_scores.txt",
            prediction_csv_path=final_dir / "eval_best_predictions.csv",
            output_dir=final_dir / "eval",
            save_all_figures=True,
            save_embeddings=save_embedding_archives,
            embedding_save_path=final_dir / "eval_best_embeddings.npz",
            embedding_max_samples=embedding_archive_max,
            save_group_metric_json=save_group_metric_json,
            decision_threshold=float(final_dev["threshold_eer"]),
            decision_source="dev_eer_threshold",
            calibrator=final_calibrator,
            save_calibrated_metrics=save_calibrated_metrics,
        )
    else:
        print("[eval] final eval after training is skipped by default for research-safe isolation.")

    if save_tsne_flag:
        save_tsne_pair(
            model=model,
            device=device,
            dev_loader=data_bundle.dev_loader,
            dev_entries=data_bundle.dev_entries,
            eval_loader=data_bundle.eval_loader,
            eval_entries=data_bundle.eval_entries,
            model_tag=paths.model_tag,
            tag="final",
            max_samples=int(config.get("tsne_max_samples", 100000)),
        )

    if swa_tracker.num_updates > 0:
        swa_path = paths.model_save_path / "swa_ast.pth"
        swa_tracker.save(swa_path)
        print(f"[swa] saved running average weights to {swa_path}")

    final_dev_summary = {key: value for key, value in final_dev.items() if key != "rows"}
    final_eval_summary = None if final_eval is None else {key: value for key, value in final_eval.items() if key != "rows"}

    save_json(
        paths.archive_dir / "final_summary.json",
        {
            "best_epoch": int(best_epoch),
            "best_dev_eer": float(best_dev_eer),
            "best_dev_dcf": float(best_dev_dcf),
            "best_dev_cllr": float(best_dev_cllr),
            "swa_updates": int(n_swa_updates),
            "final_dev_metrics": final_dev_summary,
            "final_eval_metrics": final_eval_summary,
        },
    )

    print(f"\nDone. Best epoch: {best_epoch}, best dev_eer: {best_dev_eer:.6f}")
    if final_eval is not None:
        print(f"Final eval_eer: {final_eval['eer']:.6f}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof5 v2 training / evaluation pipeline")
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./exp_result")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--eval_model_weights", type=str, default=None)
    main(parser.parse_args())
