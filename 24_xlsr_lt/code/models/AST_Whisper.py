from __future__ import annotations

from . import AST as ast_module
from .whisper_frontend import WhisperEncoderFrontend

ast_module.WavLMModel = WhisperEncoderFrontend

Model = ast_module.Model
