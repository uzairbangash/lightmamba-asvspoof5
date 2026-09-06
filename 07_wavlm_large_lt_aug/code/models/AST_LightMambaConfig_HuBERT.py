from __future__ import annotations

from transformers import AutoModel

from . import AST_LightMambaConfig as light_mamba_module

light_mamba_module.WavLMModel = AutoModel

Model = light_mamba_module.Model
