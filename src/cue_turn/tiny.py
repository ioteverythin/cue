"""Cue Tiny: numpy + onnxruntime only (no torch), about 10% of one CPU core per call."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .audio import StreamResampler, as_float
from .policy import Policy, defaults


class CausalMel:
    """Log-mel at 10 ms, frame i covering the 25 ms ending at (i+1)*10 ms; pairs into 20 ms frames of 160."""

    def __init__(self, fb: np.ndarray, cfg: dict):
        self.fb, self.n_fft, self.hop = fb.astype(np.float32), cfg["n_fft"], cfg["hop"]
        self.off, self.scale, self.floor = cfg["log_offset"], cfg["log_scale"], cfg["floor"]
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)   # periodic Hann
        self.buf = np.zeros(self.n_fft - self.hop, np.float32)
        self.pending = np.zeros((0, fb.shape[1]), np.float32)

    def __call__(self, a16: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, a16])
        n = (len(self.buf) - self.n_fft) // self.hop + 1 if len(self.buf) >= self.n_fft else 0
        if n > 0:
            idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n)[:, None]
            spec = np.abs(np.fft.rfft(self.buf[idx] * self.win, axis=1)) ** 2
            m = (np.log10(np.maximum(spec.astype(np.float32) @ self.fb, self.floor)) + self.off) / self.scale
            self.pending = np.concatenate([self.pending, m.astype(np.float32)])
            self.buf = self.buf[n * self.hop:]
        k = len(self.pending) // 2
        out = self.pending[: 2 * k].reshape(k, 2 * self.fb.shape[1])
        self.pending = self.pending[2 * k:]
        return out


class CueTiny:
    """The Cue Tiny model: load once per process, then one stream() per call."""

    def __init__(self, folder: str | Path, threads: int = 1):
        import onnxruntime as ort
        folder = Path(folder)
        self.folder = folder
        self.cfg = json.loads((folder / "config.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(folder / "cue_tiny.onnx"), so, providers=["CPUExecutionProvider"])
        self.fb = np.load(folder / "mel_filters.npy")
        self.labels = self.cfg["labels"]

    def stream(self, sample_rate: int = 16000, step_ms: int | None = None, **settings) -> "TinyStream":
        """A new call. sample_rate: the caller audio's own rate. step_ms: how often the
        model runs (default 40; 20 costs about twice the CPU, 160 about half)."""
        return TinyStream(self, sample_rate, step_ms or self.cfg["chunk_ms"], **settings)


class TinyStream:
    """One call. feed() any amount of caller audio; it returns the decisions made in it."""

    def __init__(self, m: CueTiny, sample_rate: int, step_ms: int, **settings):
        c = m.cfg
        self.m, self.hop, self.step_frames = m, c["hop_ms"], max(1, step_ms // c["hop_ms"])
        self.policy = Policy(m.labels, self.hop, **defaults(c, **settings))
        self.res, self.mel = StreamResampler(sample_rate), CausalMel(m.fb, c["mel"])
        d, L, W = c["d"], c["layers"], c["window_frames"]
        self.state = [np.zeros((1, c["in_dim"], 2), np.float32), np.zeros((1, d, 2), np.float32),
                      np.zeros((L, 1, W - 1, d), np.float32), np.zeros(1, np.int64)]
        self.frames = np.zeros((0, c["in_dim"]), np.float32)
        self.spk = np.zeros(0, np.float32)
        self.t_frames = 0
        self.last_probs, self.last_p_done = None, 0.0

    @property
    def p(self) -> dict:
        """The decision settings in use; can be changed during the call."""
        return self.policy.p

    def feed(self, audio: np.ndarray, assistant_speaking) -> list[dict]:
        """audio: caller samples at the stream's rate (float in [-1, 1], or int16).
        assistant_speaking: a bool for this audio, or a callable t_ms -> bool.
        Returns [{"t_ms": ..., "decision": "STOP" | "PAUSE" | "RESPOND"}]."""
        f = self.mel(self.res(as_float(audio)))
        if len(f):
            t0 = self.t_frames + len(self.frames)
            spk = np.array([float(assistant_speaking((t0 + i) * self.hop + self.hop // 2)) if callable(assistant_speaking)
                            else float(assistant_speaking) for i in range(len(f))], np.float32)
            self.frames = np.concatenate([self.frames, f])
            self.spk = np.concatenate([self.spk, spk])
        out = []
        while len(self.frames) >= self.step_frames:
            out += self._step(self.frames[: self.step_frames], self.spk[: self.step_frames])
            self.frames, self.spk = self.frames[self.step_frames:], self.spk[self.step_frames:]
        return out

    def _step(self, mel: np.ndarray, spk: np.ndarray) -> list[dict]:
        o = self.m.sess.run(None, {"mel": mel[None], "speaking": spk[None], "conv1": self.state[0], "conv2": self.state[1],
                                   "cache": self.state[2], "t0": self.state[3]})
        probs, done, self.state = o[0][0], o[1][0], list(o[2:])
        self.last_probs, self.last_p_done = probs[-1], float(done[-1])
        out = []
        for k in range(len(probs)):
            self.t_frames += 1
            t = self.t_frames * self.hop
            act = self.policy(probs[k], done[k], spk[k], t)
            if act:
                out.append({"t_ms": t, "decision": act})
        return out

    def reset_turn(self):
        """Forget the current turn's counters (keeps the audio context)."""
        self.policy.reset_turn()
