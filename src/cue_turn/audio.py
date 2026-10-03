"""Audio helpers in numpy: streaming resampling to 16 kHz, and reading a recording."""
from __future__ import annotations
import math
from math import gcd

import numpy as np

SR = 16000


class StreamResampler:
    """torchaudio.functional.resample (sinc_interp_hann, width 6, rolloff 0.99), streamed.

    Cue was trained on audio resampled with that filter; a different one (scipy's
    polyphase, for example) moves some frame probabilities by up to ~0.2. Output is held
    back by the filter's right half-width, and equals resampling the whole signal at once."""

    def __init__(self, orig: int, new: int = SR, width: int = 6, rolloff: float = 0.99):
        g = gcd(orig, new)
        self.o, self.n = orig // g, new // g
        self.identity = self.o == self.n
        if self.identity:
            return
        base = min(self.o, self.n) * rolloff
        self.w = math.ceil(width * self.o / base)
        idx = np.arange(-self.w, self.w + self.o, dtype=np.float64)[None, :] / self.o
        t = (np.arange(0, -self.n, -1, dtype=np.float64)[:, None] / self.n + idx) * base
        t = np.clip(t, -width, width)
        win = np.cos(t * math.pi / width / 2) ** 2
        t = t * math.pi
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(t == 0, 1.0, np.sin(t) / t)
        self.k = (k * win * base / self.o).astype(np.float32)          # (new, 2w + orig)
        self.buf = np.zeros(self.w, np.float32)                         # left zero padding, as torchaudio pads
        self.n_in = self.n_out = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.identity:
            return x.astype(np.float32)
        self.n_in += len(x)
        return self._run(x)

    def _run(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        L = self.k.shape[1]
        n_blocks = (len(self.buf) - L) // self.o + 1 if len(self.buf) >= L else 0
        if n_blocks <= 0:
            return np.zeros(0, np.float32)
        idx = np.arange(L)[None, :] + self.o * np.arange(n_blocks)[:, None]
        y = (self.buf[idx] @ self.k.T).reshape(-1)                     # (blocks * new,)
        self.buf = self.buf[n_blocks * self.o:]
        self.n_out += len(y)
        return y

    def flush(self) -> np.ndarray:
        """The held-back tail, as if the signal ended here (zero padding, as torchaudio)."""
        if self.identity:
            return np.zeros(0, np.float32)
        want = math.ceil(self.n_in * self.n / self.o) - self.n_out
        y = self._run(np.zeros(self.w + self.o, np.float32))
        return y[: max(0, want)]


def as_float(audio: np.ndarray) -> np.ndarray:
    """int16 PCM or float audio -> float32 in [-1, 1]."""
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample a whole signal to 16 kHz with the training filter."""
    r = StreamResampler(sr)
    a = as_float(audio)
    return np.concatenate([r(a), r.flush()])


def read(path, channel: int = 0) -> tuple[np.ndarray, int]:
    """One channel of a recording, as float32, and its sample rate."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if channel >= x.shape[1]:
        raise ValueError(f"{path} has {x.shape[1]} channel(s); channel {channel} asked for")
    return x[:, channel], sr


def activity(a16: np.ndarray, hop_ms: int = 20, hang_ms: int = 300, margin_db: float = 12.0,
             floor_win_s: float = 5.0) -> np.ndarray:
    """Per frame: is this channel speaking? Energy above a running noise floor (the
    minimum frame energy over the last few seconds, causal) plus a margin, held for
    hang_ms after the last loud frame. Used to read the assistant's channel of a
    two-channel recording."""
    from collections import deque
    h = SR * hop_ms // 1000
    n = len(a16) // h
    e = 10 * np.log10((a16[: n * h].reshape(n, h).astype(np.float64) ** 2).mean(1) + 1e-10)
    w = int(floor_win_s * 1000 / hop_ms)
    floor = np.empty(n)
    dq: deque = deque()                            # monotone deque: running minimum
    for i in range(n):
        while dq and e[dq[-1]] >= e[i]:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        floor[i] = e[dq[0]]
    loud = e > np.maximum(floor + margin_db, -60.0)
    act = np.zeros(n, bool)
    last = -10 ** 9
    for i in range(n):
        if loud[i]:
            last = i
        act[i] = i - last <= hang_ms // hop_ms
    return act
