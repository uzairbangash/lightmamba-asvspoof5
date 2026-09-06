from __future__ import annotations

from . import AST_LightMambaConfig as light_mamba_module
from .whisper_frontend import WhisperEncoderFrontend

light_mamba_module.WavLMModel = WhisperEncoderFrontend

Model = light_mamba_module.Model
