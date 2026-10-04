"""The decision rule: turns Cue's per-frame probabilities into STOP / PAUSE / RESPOND.

This is the rule Cue's published scores were computed with, applied one 20 ms frame
at a time. Every model (Tiny, v4, v3) uses it unchanged.
"""
from __future__ import annotations

SPEECH = ("USER_TURN", "SOFT_INTERRUPT", "HARD_INTERRUPT", "TAKEOVER")

#: Settings and what they do (the defaults come from each model's config.json).
SETTINGS = {
    "stop_p": "probability of an interruption needed to STOP (or of 'wait' to PAUSE)",
    "hold": "20 ms frames that probability must hold",
    "cooldown_ms": "no new STOP / PAUSE for this long after one",
    "p_done": "end-of-turn probability needed to RESPOND",
    "turn_hold": "20 ms frames it must hold",
    "min_speech": "20 ms frames of caller speech before an end of turn can be called",
    "fallback_ms": "answer anyway after this much silence once the caller has spoken (0: off)",
}


class Policy:
    """Per-call decision state. Call once per 20 ms frame, in order."""

    def __init__(self, labels: list[str], hop_ms: int, **settings):
        unknown = set(settings) - set(SETTINGS)
        if unknown:
            raise TypeError(f"unknown Cue settings {sorted(unknown)}; known: {sorted(SETTINGS)}")
        self.p = dict(settings)
        self.hop = hop_ms
        self.i_stop = [labels.index("HARD_INTERRUPT"), labels.index("TAKEOVER")]
        self.i_pause = labels.index("SOFT_INTERRUPT")
        self.i_speech = [labels.index(n) for n in SPEECH]
        self.last_act = -10 ** 9
        self.reset_turn()

    def reset_turn(self):
        """Forget the current turn's counters."""
        self.run_stop = self.run_pause = self.heard = self.run_done = self.quiet = 0

    def __call__(self, probs, p_done: float, speaking: float, t_ms: int) -> str | None:
        """probs: the frame's class probabilities; p_done: its end-of-turn probability;
        speaking: whether the assistant is speaking during it; t_ms: the frame's end."""
        p = self.p
        act = None
        if speaking >= 0.5:
            self.heard = self.run_done = self.quiet = 0
            if t_ms - self.last_act <= p["cooldown_ms"]:
                self.run_stop = self.run_pause = 0
                return None
            self.run_stop = self.run_stop + 1 if probs[self.i_stop].sum() >= p["stop_p"] else 0
            self.run_pause = self.run_pause + 1 if probs[self.i_pause] >= p["stop_p"] else 0
            if self.run_stop >= p["hold"]:
                act = "STOP"
            elif self.run_pause >= p["hold"]:
                act = "PAUSE"
            if act:
                self.last_act = t_ms
                self.run_stop = self.run_pause = 0
            return act
        self.run_stop = self.run_pause = 0
        if probs[self.i_speech].sum() >= 0.5:
            self.heard += 1
            self.run_done = self.quiet = 0
            return None
        self.quiet += 1
        if self.heard >= p["min_speech"]:
            self.run_done = self.run_done + 1 if p_done >= p["p_done"] else 0
            if self.run_done >= p["turn_hold"]:
                act = "RESPOND"
            # fallback: caller speech followed by this much silence gets an answer
            elif p.get("fallback_ms", 0) and self.quiet * self.hop >= p["fallback_ms"]:
                act = "RESPOND"
        if act:
            self.heard = self.run_done = self.quiet = 0
        return act


def defaults(cfg: dict, profile: str | None = None, **overrides) -> dict:
    """A model's default settings, then a named profile from its config (v5: "responsive",
    "balanced", "cautious"), then overrides (None values are ignored)."""
    out = {**cfg["policy"]["barge_in"], **cfg["policy"]["end_of_turn"]}
    if profile:
        profiles = cfg["policy"].get("profiles", {})
        if profile not in profiles:
            raise ValueError(f"no profile {profile!r} for this model; it has {sorted(profiles) or 'none'}")
        out.update(profiles[profile])
    return {**out, **{k: v for k, v in overrides.items() if v is not None}}
