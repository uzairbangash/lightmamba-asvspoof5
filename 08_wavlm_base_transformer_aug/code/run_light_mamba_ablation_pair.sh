#!/usr/bin/env bash
set -euo pipefail

cd "/home/uzair/Desktop/baseline/asvspoof5/AST: Audio Spectrogram Transformer/WavLM_LT_Aug"

mkdir -p "/media/uzair/Data/tmp" "/media/uzair/Data/asvspoof5_exp_result"

export TMPDIR="/media/uzair/Data/tmp"
export TEMP="/media/uzair/Data/tmp"
export TMP="/media/uzair/Data/tmp"
export CUDA_HOME="/home/uzair/miniconda3/envs/asv5"

PYTHON="/home/uzair/miniconda3/envs/asv5/bin/python3.10"
OUTPUT_DIR="/media/uzair/Data/asvspoof5_exp_result"

"${PYTHON}" main.py \
  --config config/AST_LightMamba2L_K7_WavLM_Large_ASVspoof5.conf \
  --output_dir "${OUTPUT_DIR}" \
  --comment wavlm_large_light_mamba_2layer_k7_e2_aug

"${PYTHON}" main.py \
  --config config/AST_UltraLightMamba2L_K5_E1_WavLM_Large_ASVspoof5.conf \
  --output_dir "${OUTPUT_DIR}" \
  --comment wavlm_large_ultralight_mamba_2layer_k5_e1_aug
