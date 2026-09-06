from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AST import LightweightTransformerBackend, Model as BaseModel


class DepthwiseSubsampler(nn.Module):
    """Temporal reduction block used by the CNN + Light Transformer backend."""

    def __init__(self, dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        if residual.size(1) == x.size(1):
            x = x + residual
        return x


class MultiScaleConvBranch(nn.Module):
    """Multi-kernel local artifact branch from the CNN + Light Transformer baseline."""

    def __init__(self, dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm1d(dim),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(dim * 3, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        multi_scale = [branch(x_t) for branch in self.branches]
        fused = self.fuse(torch.cat(multi_scale, dim=1))
        return fused.transpose(1, 2)


class GatedFusionBlock(nn.Module):
    """Learned fusion of local CNN cues and global Light Transformer cues."""

    def __init__(self, dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, conv_x: torch.Tensor, trans_x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([conv_x, trans_x], dim=-1))
        fused = gate * trans_x + (1.0 - gate) * conv_x
        return self.out_norm(fused + 0.5 * (conv_x + trans_x))


class CNNLightTransformerBackend(nn.Module):
    """Depthwise CNN branch plus Light Transformer branch with gated fusion."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.subsampler = DepthwiseSubsampler(dim=dim, dropout=dropout)
        self.conv_branch = MultiScaleConvBranch(dim=dim, dropout=dropout)
        self.transformer_branch = LightweightTransformerBackend(
            dim=dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.hybrid_fusion = GatedFusionBlock(dim=dim, dropout=dropout)

    def subsample(self, x: torch.Tensor) -> torch.Tensor:
        return self.subsampler(x)

    def refine(self, x: torch.Tensor) -> torch.Tensor:
        conv_x = self.conv_branch(x)
        trans_x = self.transformer_branch(x)
        return self.hybrid_fusion(conv_x, trans_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.subsample(x))


class Model(BaseModel):
    """WavLM front-end with the original CNN + Light Transformer backend pattern."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.light_transformer = CNNLightTransformerBackend(
            dim=256,
            num_layers=self.config.lt_layers,
            heads=self.config.lt_heads,
            dropout=self.config.dropout,
        )
        print(
            "[cnn_light_transformer] "
            f"subsample + multiscale CNN + {self.config.lt_layers}-layer Light Transformer gated fusion"
        )

    def _encode_preprocessed_batch(self, wav_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.wavlm(
            wav_batch,
            output_hidden_states=True,
            return_dict=True,
        )
        x = self.layer_gating(outputs.hidden_states)
        x = self.bottleneck(x)
        x = self.light_transformer.subsample(x)
        x = self.sequence_mask(x)
        x = self.light_transformer.refine(x)
        x = self.pool(x)
        emb = self.embedding(x)
        return F.normalize(emb, dim=1)
