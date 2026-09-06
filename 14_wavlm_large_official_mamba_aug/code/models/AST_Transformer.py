from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .AST import Model as BaseModel
from .AST import SinusoidalPositionalEncoding


class TransformerBackend(nn.Module):
    """Standard Transformer backend replacing the lightweight Transformer."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        return self.norm(self.encoder(x))


class Model(BaseModel):
    """WavLM-large with standard Transformer backend."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        ff_dim = int(config.get("transformer_ff_dim", 2048))
        self.light_transformer = TransformerBackend(
            dim=256,
            num_layers=self.config.lt_layers,
            heads=self.config.lt_heads,
            dropout=self.config.dropout,
            ff_dim=ff_dim,
        )
        print(
            "[transformer_backend] "
            f"layers={self.config.lt_layers}, heads={self.config.lt_heads}, ff_dim={ff_dim}"
        )
