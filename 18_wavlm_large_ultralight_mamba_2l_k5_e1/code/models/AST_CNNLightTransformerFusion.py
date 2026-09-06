from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .AST import LightweightTransformerBackend, Model as BaseModel


class LocalCNNBranch(nn.Module):
    """Depthwise-separable local artifact branch for WavLM sequences."""

    def __init__(self, dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            _TransposeToChannels(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            _TransposeToTime(),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _TransposeToChannels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class _TransposeToTime(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class CNNLightTransformerFusionBackend(nn.Module):
    """Global Light Transformer fused with a local CNN artifact branch."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.transformer = LightweightTransformerBackend(
            dim=dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.cnn = LocalCNNBranch(dim=dim, dropout=dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_x = self.transformer(x)
        local_x = self.cnn(x)
        gate = self.gate(torch.cat([global_x, local_x], dim=-1))
        fused = gate * global_x + (1.0 - gate) * local_x
        return self.norm(x + self.fuse(fused))


class Model(BaseModel):
    """WavLM-large with CNN + Light Transformer gated backend fusion."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.light_transformer = CNNLightTransformerFusionBackend(
            dim=256,
            num_layers=self.config.lt_layers,
            heads=self.config.lt_heads,
            dropout=self.config.dropout,
        )
        print("[cnn_light_transformer_fusion] CNN local branch + Light Transformer gated fusion")
