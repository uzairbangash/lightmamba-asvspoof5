from __future__ import annotations

from transformers import AutoModel

from . import AST as ast_module

ast_module.WavLMModel = AutoModel

Model = ast_module.Model
