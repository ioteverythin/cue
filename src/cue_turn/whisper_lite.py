"""Whisper's audio encoder in plain PyTorch, for running Cue v4/v5 without the transformers library.

Reads the same encoder/ folder (config.json + model.safetensors) and gives the same output as
transformers' WhisperEncoder (checked to float tolerance in the tests). The mel filterbank comes
from encoder/mel_filters.npy when transformers is not installed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Layer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d)
        self.self_attn.k_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d)
        self.self_attn.out_proj = nn.Linear(d, d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        h = self.self_attn_layer_norm(x)
        a = self.self_attn
        split = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(split(a.q_proj(h)), split(a.k_proj(h)), split(a.v_proj(h)))
        x = x + a.out_proj(o.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperEncoderLite(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, n = cfg["d_model"], cfg["max_source_positions"]
        self.conv1 = nn.Conv1d(cfg["num_mel_bins"], d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(n, d)
        self.layers = nn.ModuleList(_Layer(d, cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"])
                                    for _ in range(cfg["encoder_layers"]))
        self.layer_norm = nn.LayerNorm(d)

    @classmethod
    def from_folder(cls, folder: Path, device="cpu", dtype=torch.float32) -> "WhisperEncoderLite":
        from safetensors.torch import load_file
        folder = Path(folder)
        m = cls(json.loads((folder / "config.json").read_text()))
        m.load_state_dict(load_file(str(folder / "model.safetensors")))
        return m.to(device=device, dtype=dtype).eval()

    def forward(self, input_features: torch.Tensor):
        x = F.gelu(self.conv2(F.gelu(self.conv1(input_features)))).transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        out = self.layer_norm(x)
        return type("Out", (), {"last_hidden_state": out})()


def mel_filters(folder: Path) -> np.ndarray:
    """(n_freq, n_mels) filterbank: transformers' WhisperFeatureExtractor if available, else the saved copy."""
    try:
        from transformers import WhisperFeatureExtractor
        return WhisperFeatureExtractor.from_pretrained(folder).mel_filters
    except ImportError:
        return np.load(Path(folder) / "mel_filters.npy")
