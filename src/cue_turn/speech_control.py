"""The bot's speech under Cue's control: real-time playout with graceful stops, a real PAUSE and
resume, and the caller-reply rules that decide what happens after a pause. No Pipecat imports here,
so the logic is testable on its own; cue_turn.pipecat.CueSpeechController wires it into a pipeline.
"""
from __future__ import annotations
import re
from collections import deque

import numpy as np

HOLD = ("wait", "one second", "one sec", "a second", "a sec", "one minute", "a minute", "hold on", "hang on",
        "just a second", "just a minute", "give me a second", "give me a minute", "ek second", "ek minute",
        "ek min", "ruko", "rukiye", "ruk jao", "thoda ruko")
CONTINUE_WORDS = {"ok", "okay", "k", "yes", "yeah", "yep", "yup", "sure", "go", "on", "ahead", "continue", "carry",
                  "please", "sorry", "right", "alright", "fine", "done", "got", "it", "haan", "han", "ha", "ji",
                  "theek", "thik", "hai", "accha", "acha", "achha", "bolo", "boliye", "aage", "and", "so"}


def reply_kind(text: str) -> str:
    """After a PAUSE, what the caller's words ask for: "hold" (still waiting), "continue" (go on)
    or "turn" (they are saying something: answer it)."""
    t = re.sub(r"[^\w\s']", " ", text.lower()).split()
    if not t:
        return "hold"
    s = " ".join(t)
    if len(t) <= 6 and any(re.search(rf"\b{re.escape(h)}\b", s) for h in HOLD):
        return "hold"
    if len(t) <= 5 and all(w in CONTINUE_WORDS for w in t):
        return "continue"
    return "turn"


def fade(pcm: bytes, n: int, out: bool) -> bytes:
    """Linear fade over the first n samples (fade-in) or the whole chunk ending at zero (fade-out)."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if out:
        a *= np.linspace(1.0, 0.0, len(a), dtype=np.float32)
    else:
        k = min(n, len(a)); a[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


class Playout:
    """Ordered outgoing frames, audio released in real time with a small lookahead.

    items: ("audio", pcm bytes, sample_rate, num_channels, frame) or ("other", frame).
    The owner calls next(now) repeatedly; it returns the items to send now."""

    def __init__(self, lookahead_ms: int = 60, fade_ms: int = 30):
        self.q: deque = deque()
        self.lookahead, self.fade_ms = lookahead_ms / 1000, fade_ms
        self.paused = False
        self.fade_in_next = False
        self.played_until = 0.0              # wall time the released audio runs until

    def put(self, item):
        self.q.append(item)

    def clear(self):
        self.q.clear(); self.paused = False; self.fade_in_next = False

    def next(self, now: float) -> list:
        out = []
        self.played_until = max(self.played_until, now)
        while self.q and not self.paused:
            item = self.q[0]
            if item[0] == "audio":
                if self.played_until - now >= self.lookahead:
                    break
                _, pcm, sr, ch, frame = self.q.popleft()
                if self.fade_in_next:
                    pcm = fade(pcm, int(sr * self.fade_ms / 1000) * ch, out=False); self.fade_in_next = False
                self.played_until += len(pcm) / 2 / ch / sr
                out.append(("audio", pcm, sr, ch, frame))
            else:
                out.append(self.q.popleft())
        return out

    def _fade_out_head(self, consume: bool):
        """The next fade_ms of queued audio, faded to silence (removed from the queue if consume)."""
        if not self.q or self.q[0][0] != "audio":
            return None
        _, pcm, sr, ch, frame = self.q[0]
        n = int(sr * self.fade_ms / 1000) * ch * 2
        if consume:
            self.q.popleft()
            if pcm[n:]:
                self.q.appendleft(("audio", pcm[n:], sr, ch, frame))
        return ("audio", fade(pcm[:n], 0, out=True), sr, ch, frame)

    def pause(self):
        """Stop speaking with a short fade; everything from the faded slice on is kept for resume()."""
        tail = self._fade_out_head(consume=False)
        self.paused = True
        return [tail] if tail else []

    def resume(self, now: float):
        self.paused = False; self.fade_in_next = True; self.played_until = now

    def stop(self):
        """A graceful end for an interruption: the faded head, everything else dropped."""
        tail = self._fade_out_head(consume=True)
        self.clear()
        return [tail] if tail else []
