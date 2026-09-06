

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


# ============================================================
# Constants and small helpers
# ============================================================
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SCORE_POOLING = "mean"


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse a config boolean with backward-compatible semantics."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if normalized in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise ValueError(f"invalid truth value {value}")


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ModelConfig:
    """Typed view over the loose JSON config dictionary."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    pretrained_name: str = "microsoft/wavlm-base"

    freeze_feature_extractor: bool = True
    freeze_layers: int = 2

    lt_layers: int = 4
    lt_heads: int = 4
    dropout: float = 0.1

    num_attack_classes: int = 9
    use_attack_head: bool = False

    binary_loss: str = "bce"
    pos_weight: float = 1.0
    ce_weight_spoof: float = 1.0
    ce_weight_bonafide: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = -1.0
    w_bce: float = 1.0
    w_supcon: float = 0.15
    w_aam: float = 0.05
    w_gate_entropy: float = 1e-4
    w_l2_anchor: float = 1e-4

    aam_margin: float = 0.2
    aam_scale: float = 30.0
    supcon_temperature: float = 0.07

    eval_score_pooling: str = DEFAULT_SCORE_POOLING

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelConfig":
        instance = cls(
            sample_rate=int(config.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            pretrained_name=str(config.get("pretrained_name", "microsoft/wavlm-base")),
            freeze_feature_extractor=_as_bool(
                config.get("freeze_feature_extractor", True),
                default=True,
            ),
            freeze_layers=int(config.get("freeze_layers", 2)),
            lt_layers=int(config.get("lt_layers", 4)),
            lt_heads=int(config.get("lt_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
            num_attack_classes=int(config.get("num_attack_classes", 9)),
            use_attack_head=_as_bool(config.get("use_attack_head", False), default=False),
            binary_loss=str(config.get("binary_loss", "bce")).strip().lower(),
            pos_weight=float(config.get("pos_weight", 1.0)),
            ce_weight_spoof=float(config.get("ce_weight_spoof", 1.0)),
            ce_weight_bonafide=float(config.get("ce_weight_bonafide", 1.0)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            focal_alpha=float(config.get("focal_alpha", -1.0)),
            w_bce=float(config.get("w_bce", 1.0)),
            w_supcon=float(config.get("w_supcon", 0.15)),
            w_aam=float(config.get("w_aam", 0.05)),
            w_gate_entropy=float(config.get("w_gate_entropy", 1e-4)),
            w_l2_anchor=float(config.get("w_l2_anchor", 1e-4)),
            aam_margin=float(config.get("aam_margin", 0.2)),
            aam_scale=float(config.get("aam_scale", 30.0)),
            supcon_temperature=float(config.get("supcon_temperature", 0.07)),
            eval_score_pooling=str(config.get("eval_score_pooling", DEFAULT_SCORE_POOLING)).strip().lower(),
        )
        valid_losses = {"bce", "weighted_bce", "focal", "ce"}
        if instance.binary_loss not in valid_losses:
            raise ValueError(
                f"unsupported binary_loss '{instance.binary_loss}'. Expected one of {sorted(valid_losses)}"
            )

        valid_pooling = {
            "mean",
            "median",
            "max",
            "meanmax",
            "attention",
            "mean_max_attention",
            "mma",
        }
        if instance.eval_score_pooling not in valid_pooling:
            raise ValueError(
                f"unsupported eval_score_pooling '{instance.eval_score_pooling}'. "
                f"Expected one of {sorted(valid_pooling)}"
            )
        return instance


# ============================================================
# Waveform preprocessing helpers
# ============================================================
# ============================================================
# Encoder fusion and pooling blocks
# ============================================================
class LayerGating(nn.Module):
    """Learned convex combination of WavLM hidden states."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(num_layers))

        with torch.no_grad():
            for index in range(num_layers):
                if 1 <= index <= 8:
                    self.gate_logits[index] = 1.0
                elif index > 8:
                    self.gate_logits[index] = -0.5

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.gate_logits, dim=0)
        stacked = torch.stack(list(hidden_states), dim=0)
        return torch.sum(weights[:, None, None, None] * stacked, dim=0)

    def entropy_loss(self) -> torch.Tensor:
        probabilities = torch.softmax(self.gate_logits, dim=0)
        return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the lightweight transformer."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.dim)
        )
        encoding = torch.zeros(x.size(1), self.dim, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].size(1)])
        return x + encoding.unsqueeze(0)


class LightweightTransformerBackend(nn.Module):
    """Small transformer that refines the WavLM sequence."""

    def __init__(self, dim: int = 256, num_layers: int = 4, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        return self.encoder(x)


class AttentiveStatsPooling(nn.Module):
    """Attention-weighted mean and standard-deviation pooling."""

    def __init__(self, dim: int = 256, bottleneck: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.attention(x), dim=1)
        mean = torch.sum(attention * x, dim=1)
        variance = torch.sum(attention * (x - mean.unsqueeze(1)).pow(2), dim=1)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=-1)


# ============================================================
# Output heads and losses
# ============================================================
class BinarySpoofHead(nn.Module):
    """Direct bonafide score head."""

    def __init__(self, emb_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.network(emb).squeeze(1)


class AAMSoftmaxHead(nn.Module):
    """Angular-margin auxiliary classifier for attack families."""

    def __init__(self, emb_dim: int = 256, num_classes: int = 9, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)

        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes

    def forward(self, emb: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(emb, weight)

        if labels is None:
            return self.scale * cosine

        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, self.num_classes).float().to(emb.device)
        logits = cosine * (1 - one_hot) + target * one_hot
        return self.scale * logits


class SupConProjectionHead(nn.Module):
    """Projection head for supervised contrastive learning."""

    def __init__(self, emb_dim: int = 256, proj_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(x), dim=1)


class SupConLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-8))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1.0)
        return -mean_log_prob_pos.mean()


# ============================================================
# Main model
# ============================================================
class Model(nn.Module):
    """WavLM front-end with lightweight anti-spoof heads."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = ModelConfig.from_mapping(config)

        self.sample_rate = self.config.sample_rate

        # ----------------------------------------------------
        # WavLM front-end
        # ----------------------------------------------------
        self.wavlm = WavLMModel.from_pretrained(
            self.config.pretrained_name,
            output_hidden_states=True,
        )

        hidden_size = self.wavlm.config.hidden_size
        num_hidden_states = self.wavlm.config.num_hidden_layers + 1

        # ----------------------------------------------------
        # Sequence refinement blocks
        # ----------------------------------------------------
        self.layer_gating = LayerGating(num_hidden_states)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.light_transformer = LightweightTransformerBackend(
            dim=256,
            num_layers=self.config.lt_layers,
            heads=self.config.lt_heads,
            dropout=self.config.dropout,
        )
        self.pool = AttentiveStatsPooling(dim=256, bottleneck=128)

        # ----------------------------------------------------
        # Embedding and heads
        # ----------------------------------------------------
        self.embedding = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
        )
        self.binary_head = BinarySpoofHead(emb_dim=256)
        self.attack_head = AAMSoftmaxHead(
            emb_dim=256,
            num_classes=self.config.num_attack_classes,
            margin=self.config.aam_margin,
            scale=self.config.aam_scale,
        )
        self.supcon_head = SupConProjectionHead(emb_dim=256, proj_dim=128)
        self.supcon_loss = SupConLoss(temperature=self.config.supcon_temperature)
        self.register_buffer("bce_pos_weight", torch.tensor(self.config.pos_weight))
        self.register_buffer("spoof_class_weight", torch.tensor(self.config.ce_weight_spoof))
        self.register_buffer("bonafide_class_weight", torch.tensor(self.config.ce_weight_bonafide))

        # ----------------------------------------------------
        # Fine-tuning controls
        # ----------------------------------------------------
        self._apply_freezing()

        # L2-SP anchor for trainable WavLM parameters only.
        self.anchor: dict[str, torch.Tensor] = {}
        for name, parameter in self.wavlm.named_parameters():
            if parameter.requires_grad:
                self.anchor[name] = parameter.detach().clone()

    # --------------------------------------------------------
    # Freezing / optimization helpers
    # --------------------------------------------------------
    def _apply_freezing(self) -> None:
        if self.config.freeze_feature_extractor and hasattr(self.wavlm, "feature_extractor"):
            for parameter in self.wavlm.feature_extractor.parameters():
                parameter.requires_grad = False

        if self.config.freeze_layers > 0:
            for layer_index, layer in enumerate(self.wavlm.encoder.layers):
                if layer_index < self.config.freeze_layers:
                    for parameter in layer.parameters():
                        parameter.requires_grad = False

        print(f"[ast_v2] WavLM feature extractor frozen: {self.config.freeze_feature_extractor}")
        print(f"[ast_v2] Frozen WavLM transformer layers: {self.config.freeze_layers}")

    def unfreeze_all(self) -> None:
        """Unfreeze all transformer layers while keeping the feature extractor frozen if requested."""

        for parameter in self.wavlm.encoder.parameters():
            parameter.requires_grad = True

        if self.config.freeze_feature_extractor and hasattr(self.wavlm, "feature_extractor"):
            for parameter in self.wavlm.feature_extractor.parameters():
                parameter.requires_grad = False

        for name, parameter in self.wavlm.named_parameters():
            if parameter.requires_grad and name not in self.anchor:
                self.anchor[name] = parameter.detach().clone()

        print("[ast_v2] WavLM transformer unfrozen; feature extractor remains frozen.")

    def unfreeze_top_layers(self, num_layers: int = 1) -> int:
        """Gradually unfreeze previously frozen backbone layers from top to bottom."""

        if num_layers <= 0 or self.config.freeze_layers <= 0:
            return 0

        unfrozen = 0
        max_frozen_index = min(self.config.freeze_layers, len(self.wavlm.encoder.layers)) - 1
        for layer_index in range(max_frozen_index, -1, -1):
            layer = self.wavlm.encoder.layers[layer_index]
            if any(not parameter.requires_grad for parameter in layer.parameters()):
                for parameter in layer.parameters():
                    parameter.requires_grad = True
                unfrozen += 1
                if unfrozen >= num_layers:
                    break

        if self.config.freeze_feature_extractor and hasattr(self.wavlm, "feature_extractor"):
            for parameter in self.wavlm.feature_extractor.parameters():
                parameter.requires_grad = False

        for name, parameter in self.wavlm.named_parameters():
            if parameter.requires_grad and name not in self.anchor:
                self.anchor[name] = parameter.detach().clone()

        return unfrozen

    def param_groups(
        self,
        backbone_lr: float = 1.5e-5,
        fusion_lr: float = 2e-4,
        head_lr: float = 3e-4,
        layerwise_lr_decay: float = 1.0,
        weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return optimizer parameter groups with optional layer-wise LR decay."""

        def is_no_decay(name: str) -> bool:
            lower = name.lower()
            return lower.endswith("bias") or "layernorm" in lower or ".norm" in lower or "layer_norm" in lower

        def add_named_group(
            groups: list[dict[str, Any]],
            named_params: Sequence[tuple[str, nn.Parameter]],
            lr: float,
            group_name: str,
        ) -> None:
            decay_params: list[nn.Parameter] = []
            no_decay_params: list[nn.Parameter] = []

            for name, parameter in named_params:
                if is_no_decay(name):
                    no_decay_params.append(parameter)
                else:
                    decay_params.append(parameter)

            if decay_params:
                groups.append(
                    {
                        "params": decay_params,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "name": f"{group_name}_decay",
                    }
                )
            if no_decay_params:
                groups.append(
                    {
                        "params": no_decay_params,
                        "lr": lr,
                        "weight_decay": 0.0,
                        "name": f"{group_name}_nodecay",
                    }
                )

        groups: list[dict[str, Any]] = []
        num_layers = len(self.wavlm.encoder.layers)

        base_named_params = [
            (name, parameter)
            for name, parameter in self.wavlm.named_parameters()
            if "encoder.layers." not in name
        ]
        base_lr = backbone_lr * (layerwise_lr_decay ** max(1, num_layers))
        add_named_group(groups, base_named_params, base_lr, "wavlm_base")

        for reverse_depth, layer_index in enumerate(range(num_layers - 1, -1, -1)):
            layer_lr = backbone_lr * (layerwise_lr_decay ** reverse_depth)
            layer_named_params = [
                (name, parameter)
                for name, parameter in self.wavlm.encoder.layers[layer_index].named_parameters()
            ]
            layer_named_params = [
                (f"encoder.layers.{layer_index}.{name}", parameter)
                for name, parameter in layer_named_params
            ]
            add_named_group(groups, layer_named_params, layer_lr, f"wavlm_layer_{layer_index:02d}")

        backend_named_params = list(self.layer_gating.named_parameters())
        backend_named_params += [(f"bottleneck.{name}", parameter) for name, parameter in self.bottleneck.named_parameters()]
        backend_named_params += [
            (f"light_transformer.{name}", parameter)
            for name, parameter in self.light_transformer.named_parameters()
        ]
        backend_named_params += [(f"pool.{name}", parameter) for name, parameter in self.pool.named_parameters()]

        head_named_params = [(f"embedding.{name}", parameter) for name, parameter in self.embedding.named_parameters()]
        head_named_params += [(f"binary_head.{name}", parameter) for name, parameter in self.binary_head.named_parameters()]
        head_named_params += [(f"attack_head.{name}", parameter) for name, parameter in self.attack_head.named_parameters()]
        head_named_params += [(f"supcon_head.{name}", parameter) for name, parameter in self.supcon_head.named_parameters()]

        add_named_group(groups, backend_named_params, fusion_lr, "backend")
        add_named_group(groups, head_named_params, head_lr, "heads")
        return groups

    # --------------------------------------------------------
    # Waveform input
    # --------------------------------------------------------
    def preprocess_one(self, wav: torch.Tensor) -> torch.Tensor:
        """Use one complete waveform without audio preprocessing."""

        wav = wav.float()

        if wav.dim() > 1:
            wav = wav.squeeze()

        return wav

    def make_eval_views(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """Prepare one full-waveform evaluation view."""

        return [self.preprocess_one(wav)]

    def preprocess_batch(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """Legacy path for fixed-length tensor batches."""

        if wav_batch.dim() == 3:
            wav_batch = wav_batch.squeeze(1)
        processed = [self.preprocess_one(wav_batch[index]) for index in range(wav_batch.size(0))]
        return torch.stack(processed, dim=0).to(wav_batch.device)

    # --------------------------------------------------------
    # Encoding and aggregation
    # --------------------------------------------------------
    def _encode_preprocessed_batch(self, wav_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.wavlm(
            wav_batch,
            output_hidden_states=True,
            return_dict=True,
        )
        x = self.layer_gating(outputs.hidden_states)
        x = self.bottleneck(x)
        x = self.light_transformer(x)
        x = self.pool(x)
        emb = self.embedding(x)
        return F.normalize(emb, dim=1)

    def encode(self, wav_batch: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess_batch(wav_batch)
        return self._encode_preprocessed_batch(processed)

    def _fuse_segment_views(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooling = self.config.eval_score_pooling

        if scores.numel() == 1:
            return embeddings[0], scores[0]

        if pooling == "median":
            fused_embedding = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=1).squeeze(0)
            return fused_embedding, scores.median()

        if pooling == "max":
            index = int(torch.argmax(scores).item())
            return embeddings[index], scores[index]

        if pooling == "meanmax":
            mean_embedding = embeddings.mean(dim=0)
            max_index = int(torch.argmax(scores).item())
            fused_embedding = F.normalize(
                0.5 * (mean_embedding + embeddings[max_index]),
                dim=0,
            )
            fused_score = 0.5 * (scores.mean() + scores.max())
            return fused_embedding, fused_score

        attention_weights = torch.softmax(scores, dim=0).unsqueeze(-1)
        attn_embedding = torch.sum(attention_weights * embeddings, dim=0)
        attn_score = torch.sum(attention_weights.squeeze(-1) * scores, dim=0)

        if pooling in {"attention"}:
            return F.normalize(attn_embedding, dim=0), attn_score

        if pooling in {"mean_max_attention", "mma"}:
            mean_embedding = embeddings.mean(dim=0)
            max_index = int(torch.argmax(scores).item())
            fused_embedding = F.normalize(
                (mean_embedding + embeddings[max_index] + attn_embedding) / 3.0,
                dim=0,
            )
            fused_score = (scores.mean() + scores.max() + attn_score) / 3.0
            return fused_embedding, fused_score

        fused_embedding = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        return fused_embedding, scores.mean()

    def forward_utterance_batch(self, utterances: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward path for full-waveform variable-length batches."""

        device = next(self.parameters()).device
        prepared_views: list[torch.Tensor] = []
        view_counts: list[int] = []

        for wav in utterances:
            if self.training:
                views = [self.preprocess_one(wav)]
            else:
                views = self.make_eval_views(wav)

            prepared_views.extend(views)
            view_counts.append(len(views))

        segment_embeddings = torch.cat(
            [
                self._encode_preprocessed_batch(view.unsqueeze(0).to(device))
                for view in prepared_views
            ],
            dim=0,
        )
        segment_scores = self.binary_head(segment_embeddings)

        utterance_embeddings: list[torch.Tensor] = []
        utterance_scores: list[torch.Tensor] = []
        offset = 0

        for count in view_counts:
            emb_chunk = segment_embeddings[offset : offset + count]
            score_chunk = segment_scores[offset : offset + count]

            fused_emb, fused_score = self._fuse_segment_views(emb_chunk, score_chunk)

            utterance_embeddings.append(fused_emb)
            utterance_scores.append(fused_score)
            offset += count

        embeddings = torch.stack(utterance_embeddings, dim=0)
        scores = torch.stack(utterance_scores, dim=0)
        logits = torch.stack([-scores, scores], dim=1)
        return embeddings, logits

    def forward(
        self,
        wav: torch.Tensor | Sequence[torch.Tensor],
        Freq_aug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute utterance embeddings and 2-logit spoof scores."""

        del Freq_aug

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                return self.forward_utterance_batch([wav])

            emb = self.encode(wav)
            score = self.binary_head(emb)
            logits = torch.stack([-score, score], dim=1)
            return emb, logits

        if isinstance(wav, (list, tuple)):
            return self.forward_utterance_batch(wav)

        raise TypeError(f"unsupported waveform batch type: {type(wav)!r}")

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------
    def l2_anchor_loss(self) -> torch.Tensor:
        if not self.anchor:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0

        for name, parameter in self.wavlm.named_parameters():
            if name in self.anchor and parameter.requires_grad:
                total = total + (parameter - self.anchor[name].to(parameter.device)).pow(2).mean()
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return total / count

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ce_criterion: Optional[nn.Module] = None,
        attack_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the multi-objective anti-spoof loss."""

        labels = labels.long()
        score = logits[:, 1]
        binary_mode = self.config.binary_loss
        sample_weights = torch.where(
            labels == 1,
            self.bonafide_class_weight.to(score.device),
            self.spoof_class_weight.to(score.device),
        )

        # ----------------------------------------------------
        # Primary binary objective
        # ----------------------------------------------------
        if binary_mode == "ce":
            if ce_criterion is not None:
                bce = ce_criterion(logits, labels)
            else:
                class_weights = torch.stack(
                    [
                        self.spoof_class_weight.to(score.device),
                        self.bonafide_class_weight.to(score.device),
                    ]
                )
                bce = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            per_sample_bce = F.binary_cross_entropy_with_logits(
                score,
                labels.float(),
                pos_weight=self.bce_pos_weight.to(score.device),
                reduction="none",
            )

            if binary_mode == "focal":
                probabilities = torch.sigmoid(score)
                pt = torch.where(labels == 1, probabilities, 1.0 - probabilities).clamp_min(1e-6)
                focal_factor = (1.0 - pt).pow(self.config.focal_gamma)
                if 0.0 <= self.config.focal_alpha <= 1.0:
                    alpha_tensor = torch.where(
                        labels == 1,
                        torch.full_like(probabilities, self.config.focal_alpha),
                        torch.full_like(probabilities, 1.0 - self.config.focal_alpha),
                    )
                    focal_factor = focal_factor * alpha_tensor
                per_sample_bce = per_sample_bce * focal_factor

            weighted_losses = per_sample_bce * sample_weights
            bce = weighted_losses.sum() / sample_weights.sum().clamp_min(1e-8)

        # ----------------------------------------------------
        # Embedding regularization losses
        # ----------------------------------------------------
        proj = self.supcon_head(embeddings)
        supcon = self.supcon_loss(proj, labels) if embeddings.size(0) > 1 else torch.tensor(0.0, device=score.device)
        gate_reg = self.layer_gating.entropy_loss()
        l2sp = self.l2_anchor_loss()

        # ----------------------------------------------------
        # Optional attack-family auxiliary task
        # ----------------------------------------------------
        aam = torch.tensor(0.0, device=score.device)
        if self.config.use_attack_head and attack_labels is not None:
            attack_labels = attack_labels.long()
            valid_mask = attack_labels >= 0
            if valid_mask.any():
                attack_logits = self.attack_head(embeddings[valid_mask], attack_labels[valid_mask])
                aam = F.cross_entropy(attack_logits, attack_labels[valid_mask])

        # ----------------------------------------------------
        # Final weighted objective
        # ----------------------------------------------------
        total = (
            self.config.w_bce * bce
            + self.config.w_supcon * supcon
            + self.config.w_aam * aam
            + self.config.w_gate_entropy * gate_reg
            + self.config.w_l2_anchor * l2sp
        )

        components = {
            "total": total.detach(),
            "bce": bce.detach(),
            "supcon": supcon.detach(),
            "aam": aam.detach(),
            "gate_entropy": gate_reg.detach(),
            "l2sp": l2sp.detach(),
            "score_mean": score.detach().mean(),
            "spoof_weight": self.spoof_class_weight.detach(),
            "bonafide_weight": self.bonafide_class_weight.detach(),
        }
        return total, components
