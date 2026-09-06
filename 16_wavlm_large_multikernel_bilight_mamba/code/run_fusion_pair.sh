#!/usr/bin/env bash
set -euo pipefail

cd "/home/uzair/Desktop/baseline/asvspoof5/AST: Audio Spectrogram Transformer/WavLM_LT_Aug"

export TMPDIR="/media/uzair/Data/tmp"
export TEMP="/media/uzair/Data/tmp"
export TMP="/media/uzair/Data/tmp"
mkdir -p "$TMPDIR"

PY="/home/uzair/miniconda3/envs/asv5/bin/python3.10"
OUT="/media/uzair/Data/asvspoof5_exp_result"

"$PY" main.py \
  --config config/AST_DualScaleLightMambaFusion_WavLM_Large_ASVspoof5.conf \
  --output_dir "$OUT" \
  --comment wavlm_large_dual_scale_light_mamba_fusion

"$PY" main.py \
  --config config/AST_CNNLightTransformerFusion_WavLM_Large_ASVspoof5.conf \
  --output_dir "$OUT" \
  --comment wavlm_large_cnn_light_transformer_fusion
