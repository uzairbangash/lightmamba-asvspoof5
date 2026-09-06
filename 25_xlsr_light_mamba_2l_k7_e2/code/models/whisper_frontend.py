from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel


class WhisperEncoderFrontend(nn.Module):
    """Whisper encoder wrapper with the same call shape as WavLM-style encoders."""

    def __init__(self, pretrained_name: str) -> None:
        super().__init__()
        self.model = WhisperModel.from_pretrained(pretrained_name)
        self.processor = WhisperFeatureExtractor.from_pretrained(pretrained_name)
        self.encoder = self.model.encoder
        self.config = SimpleNamespace(
            hidden_size=self.model.config.d_model,
            num_hidden_layers=self.model.config.encoder_layers,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name: str, *args: Any, **kwargs: Any) -> "WhisperEncoderFrontend":
        del args, kwargs
        return cls(pretrained_name)

    def forward(
        self,
        wav_batch: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        if wav_batch.dim() == 1:
            wav_batch = wav_batch.unsqueeze(0)

        device = wav_batch.device
        wav_list = [wav.detach().float().cpu().numpy() for wav in wav_batch]
        features = self.processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device=device)

        return self.encoder(
            features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
