

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel


# ============================================================
# Constants and small helpers
# ============================================================
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_EVAL_SEGMENTS = 5
DEFAULT_EVAL_SEGMENT_OVERLAP = 0.5
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


def _clamp_nonempty_waveform(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Replace pathological empty tensors with one second of silence."""

    if wav.numel() > 0:
        return wav
    return torch.zeros(sample_rate, dtype=torch.float32, device=wav.device)


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ModelConfig:
    """Typed view over the loose JSON config dictionary."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    crop_seconds: float = 4.0
    pretrained_name: str = "microsoft/wavlm-base"

    trim_silence: bool = True
    use_augment: bool = True
    freeze_feature_extractor: bool = True
    freeze_layers: int = 2

    lt_layers: int = 4
    lt_heads: int = 4
    light_mamba_kernel: int = 7
    light_mamba_expansion: int = 2
    dropout: float = 0.1
    backend_time_mask_prob: float = 0.3
    backend_time_mask_width: int = 24
    backend_channel_mask_prob: float = 0.2
    backend_channel_mask_width: int = 24

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

    silence_threshold: float = 0.005
    silence_hangover: int = 400
    eval_num_segments: int = DEFAULT_EVAL_SEGMENTS
    eval_segment_overlap: float = DEFAULT_EVAL_SEGMENT_OVERLAP
    eval_score_pooling: str = DEFAULT_SCORE_POOLING

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelConfig":
        instance = cls(
            sample_rate=int(config.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            crop_seconds=float(config.get("crop_seconds", 4.0)),
            pretrained_name=str(config.get("pretrained_name", "microsoft/wavlm-base")),
            trim_silence=_as_bool(config.get("trim_silence", True), default=True),
            use_augment=_as_bool(config.get("use_augment", True), default=True),
            freeze_feature_extractor=_as_bool(
                config.get("freeze_feature_extractor", True),
                default=True,
            ),
            freeze_layers=int(config.get("freeze_layers", 2)),
            lt_layers=int(config.get("lt_layers", 4)),
            lt_heads=int(config.get("lt_heads", 4)),
            light_mamba_kernel=int(config.get("light_mamba_kernel", 7)),
            light_mamba_expansion=int(config.get("light_mamba_expansion", 2)),
            dropout=float(config.get("dropout", 0.1)),
            backend_time_mask_prob=float(config.get("backend_time_mask_prob", 0.3)),
            backend_time_mask_width=int(config.get("backend_time_mask_width", 24)),
            backend_channel_mask_prob=float(config.get("backend_channel_mask_prob", 0.2)),
            backend_channel_mask_width=int(config.get("backend_channel_mask_width", 24)),
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
            silence_threshold=float(config.get("silence_threshold", 0.005)),
            silence_hangover=int(config.get("silence_hangover", 400)),
            eval_num_segments=max(1, int(config.get("eval_num_segments", DEFAULT_EVAL_SEGMENTS))),
            eval_segment_overlap=float(config.get("eval_segment_overlap", DEFAULT_EVAL_SEGMENT_OVERLAP)),
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
        if not 0.0 <= instance.eval_segment_overlap < 1.0:
            raise ValueError("eval_segment_overlap must be in the range [0.0, 1.0)")
        return instance


# ============================================================
# Waveform preprocessing helpers
# ============================================================
def trim_outer_silence(
    wav: torch.Tensor,
    threshold: float = 0.005,
    hangover: int = 400,
) -> torch.Tensor:
    """Trim low-energy regions at the beginning and end only."""

    if wav.dim() > 1:
        wav = wav.squeeze()

    idx = torch.where(wav.abs() > threshold)[0]
    if idx.numel() == 0:
        return wav

    start = max(0, int(idx[0]) - hangover)
    end = min(wav.numel(), int(idx[-1]) + hangover)
    return wav[start:end]


def crop_or_repeat(
    wav: torch.Tensor,
    target_len: int,
    random_crop: bool = True,
) -> torch.Tensor:
    """Crop a long utterance or repeat-pad a short one."""

    total = wav.numel()
    if total == target_len:
        return wav

    if total < target_len:
        repeat = math.ceil(target_len / max(1, total))
        return wav.repeat(repeat)[:target_len]

    if random_crop:
        start = random.randint(0, total - target_len)
    else:
        start = (total - target_len) // 2
    return wav[start:start + target_len]


def make_sliding_views(
    wav: torch.Tensor,
    target_len: int,
    max_segments: int,
    overlap: float,
) -> list[torch.Tensor]:
    """Create deterministic overlapping evaluation views over the whole utterance."""

    if wav.numel() <= target_len or max_segments <= 1:
        return [crop_or_repeat(wav, target_len=target_len, random_crop=False)]

    max_start = wav.numel() - target_len
    if max_start <= 0:
        return [crop_or_repeat(wav, target_len=target_len, random_crop=False)]

    hop = max(1, int(round(target_len * (1.0 - overlap))))
    starts = list(range(0, max_start + 1, hop))
    if starts[-1] != max_start:
        starts.append(max_start)

    if len(starts) > max_segments:
        sampled = torch.linspace(0, len(starts) - 1, steps=max_segments)
        starts = [starts[int(round(index.item()))] for index in sampled]

    return [wav[start : start + target_len] for start in starts]


# ============================================================
# Online augmentation
# ============================================================
class CodecAwareAugmentBank(nn.Module):
    """Spoof-domain corruption bank with codec / channel proxy degradations."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.max_ir_length = 129

    def add_noise(self, wav: torch.Tensor, snr_min: float = 5.0, snr_max: float = 20.0) -> torch.Tensor:
        noise = torch.randn_like(wav)
        snr = random.uniform(snr_min, snr_max)

        wav_power = wav.pow(2).mean().clamp_min(1e-8)
        noise_power = noise.pow(2).mean().clamp_min(1e-8)
        scale = torch.sqrt(wav_power / (10 ** (snr / 10) * noise_power))
        return wav + scale * noise

    def bandlimit_telephone(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torchaudio.functional.highpass_biquad(wav, self.sample_rate, 300)
        wav = torchaudio.functional.lowpass_biquad(wav, self.sample_rate, 3400)
        return wav

    def bandlimit_bluetooth(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torchaudio.functional.highpass_biquad(wav, self.sample_rate, 120)
        wav = torchaudio.functional.lowpass_biquad(wav, self.sample_rate, 7200)
        return wav

    def narrowband_roundtrip(self, wav: torch.Tensor) -> torch.Tensor:
        down = torchaudio.functional.resample(wav.unsqueeze(0), self.sample_rate, 8_000).squeeze(0)
        up = torchaudio.functional.resample(down.unsqueeze(0), 8_000, self.sample_rate).squeeze(0)
        return crop_or_repeat(up, target_len=wav.numel(), random_crop=False)

    def codec_proxy_roundtrip(self, wav: torch.Tensor) -> torch.Tensor:
        target_rate = random.choice([8_000, 12_000, 14_000])
        lowpass_hz = min(target_rate // 2 - 200, 6_800)
        wav = torchaudio.functional.lowpass_biquad(wav, self.sample_rate, max(1_600, lowpass_hz))
        down = torchaudio.functional.resample(wav.unsqueeze(0), self.sample_rate, target_rate).squeeze(0)
        down = self.quantize(down, bits=random.choice([6, 7, 8]))
        down = self.packet_loss(down, frame_ms=random.choice([10, 20]), zero_fill=random.random() < 0.5)
        up = torchaudio.functional.resample(down.unsqueeze(0), target_rate, self.sample_rate).squeeze(0)
        return crop_or_repeat(up, target_len=wav.numel(), random_crop=False)

    def gain_clip(self, wav: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(-6.0, 6.0)
        wav = wav * (10 ** (gain_db / 20.0))

        if random.random() < 0.5:
            threshold = random.uniform(0.92, 0.99)
            wav = torch.clamp(wav, -threshold, threshold) / threshold
        return wav

    def quantize(self, wav: torch.Tensor, bits: int = 8) -> torch.Tensor:
        levels = float(max(2, 2**bits - 1))
        clipped = torch.clamp(wav, -1.0, 1.0)
        return torch.round(clipped * levels) / levels

    def random_channel_ir(self, wav: torch.Tensor) -> torch.Tensor:
        ir_length = random.choice([17, 33, 65, 129])
        ir_length = min(ir_length, self.max_ir_length)
        taps = torch.zeros(ir_length, dtype=wav.dtype, device=wav.device)
        taps[0] = 1.0

        reflection_count = random.randint(2, 6)
        for _ in range(reflection_count):
            index = random.randint(1, ir_length - 1)
            amplitude = random.uniform(-0.25, 0.25) * math.exp(-index / max(1.0, ir_length / 5.0))
            taps[index] += amplitude

        taps = taps * torch.hann_window(ir_length, periodic=False, device=wav.device, dtype=wav.dtype)
        taps = taps / taps.abs().sum().clamp_min(1e-6)
        filtered = F.conv1d(
            wav.view(1, 1, -1),
            taps.view(1, 1, -1),
            padding=ir_length // 2,
        ).view(-1)
        return crop_or_repeat(filtered, target_len=wav.numel(), random_crop=False)

    def packet_loss(
        self,
        wav: torch.Tensor,
        frame_ms: int = 20,
        zero_fill: bool = True,
    ) -> torch.Tensor:
        frame_len = max(1, int(self.sample_rate * frame_ms / 1000))
        if wav.numel() < frame_len * 2:
            return wav

        corrupted = wav.clone()
        drop_ratio = random.uniform(0.02, 0.15)
        num_frames = max(1, wav.numel() // frame_len)
        num_drops = max(1, int(round(num_frames * drop_ratio)))
        frame_indices = random.sample(range(num_frames), k=min(num_drops, num_frames))

        for frame_index in frame_indices:
            start = frame_index * frame_len
            end = min(start + frame_len, wav.numel())
            if zero_fill or start == 0:
                corrupted[start:end] = 0.0
            else:
                corrupted[start:end] = corrupted[start - frame_len : start][: end - start]
        return corrupted

    def temporal_dropout(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() < 512:
            return wav
        drop_width = random.randint(max(64, wav.numel() // 40), max(128, wav.numel() // 12))
        drop_width = min(drop_width, wav.numel())
        start = random.randint(0, wav.numel() - drop_width)
        wav = wav.clone()
        wav[start : start + drop_width] = 0.0
        return wav

    def bluetooth_proxy(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self.bandlimit_bluetooth(wav)
        wav = self.codec_proxy_roundtrip(wav)
        wav = self.add_noise(wav, snr_min=8.0, snr_max=22.0)
        return self.gain_clip(wav)

    def compression_proxy(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self.quantize(wav, bits=random.choice([5, 6, 7]))
        wav = self.packet_loss(wav, frame_ms=random.choice([10, 20]), zero_fill=False)
        return wav

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return wav

        draw = random.random()

        if draw < 0.25:
            return wav
        if draw < 0.45:
            return self.add_noise(wav)
        if draw < 0.60:
            return self.bandlimit_telephone(wav)
        if draw < 0.75:
            return self.narrowband_roundtrip(wav)
        if draw < 0.82:
            return self.codec_proxy_roundtrip(wav)
        if draw < 0.88:
            wav = self.random_channel_ir(wav)
            return self.gain_clip(wav)
        if draw < 0.94:
            return self.bluetooth_proxy(wav)

        wav = self.add_noise(wav, snr_min=0.0, snr_max=10.0)
        wav = self.compression_proxy(wav)
        wav = self.temporal_dropout(wav)
        return self.gain_clip(wav)


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
        fused = None
        for weight, state in zip(weights, hidden_states):
            term = weight * state
            fused = term if fused is None else fused + term
        if fused is None:
            raise ValueError("LayerGating received no hidden states")
        return fused

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


class LightMambaBlock(nn.Module):
    """Lightweight Mamba-style temporal mixer for WavLM sequences."""

    def __init__(
        self,
        dim: int = 256,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        content, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        content = content.transpose(1, 2)
        content = self.depthwise_conv(content).transpose(1, 2)
        content = F.silu(content) * torch.sigmoid(gate)
        content = self.out_proj(content)
        return residual + self.dropout(content)


class LightMambaBackend(nn.Module):
    """Drop-in lightweight Mamba-style backend replacing the Light Transformer."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LightMambaBlock(
                    dim=dim,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SequenceMasking(nn.Module):
    """SpecAugment-style masking on the latent sequence."""

    def __init__(
        self,
        time_mask_prob: float = 0.3,
        max_time_masks: int = 2,
        max_time_width: int = 24,
        channel_mask_prob: float = 0.2,
        max_channel_width: int = 24,
    ) -> None:
        super().__init__()
        self.time_mask_prob = time_mask_prob
        self.max_time_masks = max_time_masks
        self.max_time_width = max_time_width
        self.channel_mask_prob = channel_mask_prob
        self.max_channel_width = max_channel_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        masked = x.clone()
        batch_size, time_steps, channels = masked.shape

        if time_steps > 1 and self.time_mask_prob > 0.0:
            for batch_index in range(batch_size):
                if random.random() >= self.time_mask_prob:
                    continue
                num_masks = random.randint(1, max(1, self.max_time_masks))
                for _ in range(num_masks):
                    width = min(time_steps, random.randint(1, max(1, self.max_time_width)))
                    start = random.randint(0, max(0, time_steps - width))
                    masked[batch_index, start : start + width, :] = 0.0

        if channels > 1 and self.channel_mask_prob > 0.0:
            for batch_index in range(batch_size):
                if random.random() >= self.channel_mask_prob:
                    continue
                width = min(channels, random.randint(1, max(1, self.max_channel_width)))
                start = random.randint(0, max(0, channels - width))
                masked[batch_index, :, start : start + width] = 0.0

        return masked


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
        self.crop_len = int(self.sample_rate * self.config.crop_seconds)
        self.eval_num_segments = self.config.eval_num_segments

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
        # Preprocessing / augmentation
        # ----------------------------------------------------
        self.augment = CodecAwareAugmentBank(sample_rate=self.sample_rate)

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
        self.sequence_mask = SequenceMasking(
            time_mask_prob=self.config.backend_time_mask_prob,
            max_time_masks=2,
            max_time_width=self.config.backend_time_mask_width,
            channel_mask_prob=self.config.backend_channel_mask_prob,
            max_channel_width=self.config.backend_channel_mask_width,
        )
        self.sequence_backend = LightMambaBackend(
            dim=256,
            num_layers=self.config.lt_layers,
            kernel_size=self.config.light_mamba_kernel,
            expansion=self.config.light_mamba_expansion,
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
        print(
            "[light_mamba_backend] "
            f"layers={self.config.lt_layers}, "
            f"kernel={self.config.light_mamba_kernel}, "
            f"expansion={self.config.light_mamba_expansion}"
        )

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
            (f"sequence_backend.{name}", parameter)
            for name, parameter in self.sequence_backend.named_parameters()
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
    # Waveform preprocessing
    # --------------------------------------------------------
    def _prepare_base_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.float()

        if wav.dim() > 1:
            wav = wav.squeeze()

        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        wav = _clamp_nonempty_waveform(wav, sample_rate=self.sample_rate)
        wav = wav - wav.mean()

        if self.config.trim_silence:
            wav = trim_outer_silence(
                wav,
                threshold=self.config.silence_threshold,
                hangover=self.config.silence_hangover,
            )

        return _clamp_nonempty_waveform(wav, sample_rate=self.sample_rate)

    def _normalize_view(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav - wav.mean()
        wav = wav / wav.std().clamp_min(1e-5)
        return wav

    def preprocess_one(self, wav: torch.Tensor, random_crop: Optional[bool] = None) -> torch.Tensor:
        """Prepare a single train-style view."""

        wav = self._prepare_base_waveform(wav)

        if self.training and self.config.use_augment:
            wav = self.augment(wav)

        crop_randomly = self.training if random_crop is None else random_crop
        wav = crop_or_repeat(wav, target_len=self.crop_len, random_crop=crop_randomly)
        return self._normalize_view(wav)

    def make_eval_views(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """Prepare multiple deterministic evaluation views."""

        wav = self._prepare_base_waveform(wav)
        views = make_sliding_views(
            wav,
            target_len=self.crop_len,
            max_segments=self.eval_num_segments,
            overlap=self.config.eval_segment_overlap,
        )
        return [self._normalize_view(view) for view in views]

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
        x = self.sequence_mask(x)
        x = self.sequence_backend(x)
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
                views = [self.preprocess_one(wav, random_crop=True)]
            else:
                views = self.make_eval_views(wav)

            prepared_views.extend(views)
            view_counts.append(len(views))

        preprocessed = torch.stack(prepared_views, dim=0).to(device)
        segment_embeddings = self._encode_preprocessed_batch(preprocessed)
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
