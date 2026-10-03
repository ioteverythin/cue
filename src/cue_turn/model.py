"""Cue v4 and v3: a frozen speech encoder and a causal Transformer head (needs torch).

v4: the encoder of OpenAI's whisper-small (Apache-2.0) on 2-second windows.
v3: WavLM-Base+ (CC BY-SA 3.0). The model decides every 160 ms; a GPU is recommended.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .audio import SR, StreamResampler, as_float
from .policy import Policy, defaults


class CausalBlock(nn.Module):
    def __init__(self, d, heads, window, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class CueHead(nn.Module):
    """Causal frame model over encoder frames + an assistant-speaking flag; two heads."""

    def __init__(self, in_dim, d, layers, heads, window, n_classes, pos="abs"):
        super().__init__()
        self.proj = nn.Linear(in_dim + 1, d)
        self.pos = nn.Embedding(4096, d)
        self.pos_mode = pos          # "abs" (v3) or "none" (v4: order only from the causal mask)
        self.blocks = nn.ModuleList(CausalBlock(d, heads, window) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, n_classes)
        self.turn_head = nn.Linear(d, 1)
        self.window = window

    def forward(self, x, speaking):
        T = x.shape[1]
        h = self.proj(torch.cat([x, speaking[..., None]], dim=-1))
        if self.pos_mode == "abs":
            h = h + self.pos(torch.arange(T, device=x.device) % 4096)
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.window)
        for b in self.blocks:
            h = b(h, mask)
        h = self.ln(h)
        return self.head(h), self.turn_head(h).squeeze(-1)


class CueModel:
    """Cue v4 or v3 from a folder (config.json, cue.safetensors, encoder/)."""

    def __init__(self, folder: str | Path, device: str | None = None):
        from safetensors.torch import load_file
        folder = Path(folder)
        self.folder = folder
        self.cfg = c = json.loads((folder / "config.json").read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_type = c.get("encoder_type", "wavlm")
        if self.encoder_type == "whisper":
            from transformers import WhisperFeatureExtractor
            from transformers.models.whisper.modeling_whisper import WhisperEncoder
            self.encoder = WhisperEncoder.from_pretrained(folder / "encoder").to(self.device).eval()
            fe = WhisperFeatureExtractor.from_pretrained(folder / "encoder")
            self.mel_fb = torch.tensor(fe.mel_filters, dtype=torch.float32, device=self.device)
            self.win = torch.hann_window(400, device=self.device)
        else:
            from transformers import WavLMModel
            self.encoder = WavLMModel.from_pretrained(folder / "encoder").to(self.device).eval()
        self.head = CueHead(c["encoder_dim"], c["d"], c["layers"], c["heads"], c["window_frames"], len(c["labels"]),
                            pos=c.get("positions", "abs"))
        self.head.load_state_dict(load_file(str(folder / "cue.safetensors")))
        self.head.to(self.device).eval()
        self.labels = c["labels"]

    @torch.no_grad()
    def encode(self, window: torch.Tensor) -> torch.Tensor:
        """(B, context samples) at 16 kHz -> (B, frames, encoder_dim)."""
        if self.encoder_type != "whisper":
            return self.encoder(window).last_hidden_state
        s = torch.stft(window, 400, 160, window=self.win, return_complex=True)[..., :-1].abs() ** 2   # Whisper's log-mel
        m = torch.clamp(self.mel_fb.T @ s, min=1e-10).log10()
        m = torch.maximum(m, m.amax(dim=(1, 2), keepdim=True) - 8.0)
        return self.encoder((m + 4.0) / 4.0).last_hidden_state

    @classmethod
    def from_pretrained(cls, name_or_path: str = "IOTEverythin/cue-v4", device: str | None = None,
                        revision: str | None = None) -> "CueModel":
        from .hub import fetch
        return cls(fetch(name_or_path, revision=revision), device)

    def stream(self, sample_rate: int = SR, **settings) -> "CueStream":
        """A new call. sample_rate: the caller audio's own rate (for feed())."""
        return CueStream(self, sample_rate, **settings)


class CueStream:
    """One call. feed() any amount of caller audio at the stream's rate, or step() exactly
    one step (cfg chunk_ms) of 16 kHz audio at a time."""

    def __init__(self, model: CueModel, sample_rate: int = SR, **settings):
        self.m, c = model, model.cfg
        self.policy = Policy(model.labels, c["hop_ms"], **defaults(c, **settings))
        self.chunk_ms, self.hop = c["chunk_ms"], c["hop_ms"]
        self.step_samples = self.chunk_ms * SR // 1000
        self.buf = np.zeros(c["context_ms"] * SR // 1000, dtype=np.float32)
        # each layer looks back window_frames, so stacked layers see layers x window: keep all of it
        self.keep = c["layers"] * c["window_frames"] + c["chunk_ms"] // c["hop_ms"]
        self.x = np.zeros((0, c["encoder_dim"]), np.float32)
        self.s = np.zeros(0, np.float32)
        self.t_ms = 0
        self.res = StreamResampler(sample_rate)
        self.pending = np.zeros(0, np.float32)
        self.last_probs, self.last_p_done = None, 0.0

    @property
    def p(self) -> dict:
        """The decision settings in use; can be changed during the call."""
        return self.policy.p

    def feed(self, audio: np.ndarray, assistant_speaking) -> list[dict]:
        """audio: caller samples at the stream's rate (float in [-1, 1], or int16).
        assistant_speaking: a bool for this audio, or a callable t_ms -> bool.
        Returns [{"t_ms": ..., "decision": "STOP" | "PAUSE" | "RESPOND"}]."""
        self.pending = np.concatenate([self.pending, self.res(as_float(audio))])
        out, n = [], self.step_samples
        while len(self.pending) >= n:
            frames = [self.t_ms + i * self.hop + self.hop // 2 for i in range(self.chunk_ms // self.hop)]
            spk = [assistant_speaking(f) for f in frames] if callable(assistant_speaking) else assistant_speaking
            out += self._run(self.pending[:n], spk)
            self.pending = self.pending[n:]
        return out

    def step(self, audio: np.ndarray, assistant_speaking) -> dict:
        """One step of 16 kHz audio. assistant_speaking: one bool for the step, or one per
        20 ms frame of it. Returns {"decision", "t_ms", "p_done", "probs"}; a decision is
        reported at the end of its step."""
        assert len(audio) == self.step_samples, f"expected {self.step_samples} samples (16 kHz, {self.chunk_ms} ms)"
        acts = self._run(as_float(audio), assistant_speaking)
        return {"decision": acts[0]["decision"] if acts else None, "t_ms": self.t_ms, "p_done": self.last_p_done,
                "probs": dict(zip(self.m.labels, self.last_probs.round(3).tolist()))}

    @torch.no_grad()
    def _run(self, audio: np.ndarray, assistant_speaking) -> list[dict]:
        self.t_ms += self.chunk_ms
        self.buf = np.concatenate([self.buf, audio])[-len(self.buf):]
        h = self.m.encode(torch.from_numpy(self.buf)[None].to(self.m.device))[0]
        n = self.chunk_ms // self.hop
        self.x = np.concatenate([self.x, h[-n:].float().cpu().numpy()])[-self.keep:]
        spk = np.broadcast_to(np.asarray(assistant_speaking, dtype=np.float32), (n,))
        self.s = np.concatenate([self.s, spk])[-self.keep:]
        dev = self.m.device
        logits, tl = self.m.head(torch.from_numpy(self.x)[None].to(dev), torch.from_numpy(self.s)[None].to(dev))
        probs = torch.softmax(logits[0, -n:].float(), -1).cpu().numpy()
        done = torch.sigmoid(tl[0, -n:].float()).cpu().numpy()
        self.last_probs, self.last_p_done = probs.mean(0), float(done[-1])
        out = []
        for k in range(n):
            t = self.t_ms - self.chunk_ms + (k + 1) * self.hop
            act = self.policy(probs[k], done[k], spk[k], t)
            if act:
                out.append({"t_ms": t, "decision": act})
        return out

    def reset_turn(self):
        """Forget the current turn's counters (keeps the audio context)."""
        self.policy.reset_turn()
