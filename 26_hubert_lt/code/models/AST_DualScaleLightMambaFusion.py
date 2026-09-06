from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .AST_LightMambaConfig import LightMambaBackend, Model as BaseModel


class DualScaleLightMambaBackend(nn.Module):
    """Fuse two complementary Light Mamba temporal scales."""

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.2,
        k7_layers: int = 2,
        k7_expansion: int = 2,
        k5_layers: int = 2,
        k5_expansion: int = 1,
    ) -> None:
        super().__init__()
        self.k7_branch = LightMambaBackend(
            dim=dim,
            num_layers=k7_layers,
            kernel_size=7,
            expansion=k7_expansion,
            dropout=dropout,
        )
        self.k5_branch = LightMambaBackend(
            dim=dim,
            num_layers=k5_layers,
            kernel_size=5,
            expansion=k5_expansion,
            dropout=dropout,
        )
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
        k7 = self.k7_branch(x)
        k5 = self.k5_branch(x)
        gate = self.gate(torch.cat([k7, k5], dim=-1))
        fused = gate * k7 + (1.0 - gate) * k5
        return self.norm(x + self.fuse(fused))


class Model(BaseModel):
    """WavLM-large with dual-scale Light Mamba gated fusion backend."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.sequence_backend = DualScaleLightMambaBackend(
            dim=256,
            dropout=self.config.dropout,
            k7_layers=int(config.get("dual_k7_layers", 2)),
            k7_expansion=int(config.get("dual_k7_expansion", 2)),
            k5_layers=int(config.get("dual_k5_layers", 2)),
            k5_expansion=int(config.get("dual_k5_expansion", 1)),
        )
        print(
            "[dual_scale_light_mamba] "
            "k7=2-layer expansion2 + k5=2-layer expansion1 gated fusion"
        )
