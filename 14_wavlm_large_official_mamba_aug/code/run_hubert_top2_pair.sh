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
  --config config/AST_HuBERT_LightTransformer_ASVspoof5.conf \
  --output_dir "$OUT" \
  --comment hubert_large_light_transformer

"$PY" main.py \
  --config config/AST_HuBERT_LightMamba2L_K7_ASVspoof5.conf \
  --output_dir "$OUT" \
  --comment hubert_large_light_mamba_2layer_k7_e2
