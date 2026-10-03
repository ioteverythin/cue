"""Cue: when a voice agent should stop talking, and when it should answer.

Cue listens to the caller's side of a call and decides, causally, every few tens of ms:

  while the assistant is speaking   nothing (backchannel, noise) / PAUSE ("wait") / STOP (interruption)
  after the caller stops talking    RESPOND (turn is over) / nothing yet (caller is mid-thought)

    import cue_turn
    cue = cue_turn.load()                          # Cue Tiny, CPU; or load("v4")
    stream = cue.stream(sample_rate=8000)          # one per call, at the audio's own rate
    for chunk in caller_audio:                     # int16 or float32 numpy, any length
        for d in stream.feed(chunk, assistant_speaking=player.is_playing()):
            handle(d)                              # {"t_ms": 1840, "decision": "STOP"}
"""
from .hub import MODELS, fetch, load
from .policy import SETTINGS, Policy

__version__ = "0.1.0"


def __getattr__(name):
    if name in ("CueTiny", "TinyStream"):
        from . import tiny
        return getattr(tiny, name)
    if name in ("CueModel", "CueStream"):
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module 'cue_turn' has no attribute {name!r}")


__all__ = ["load", "fetch", "MODELS", "SETTINGS", "Policy", "CueTiny", "TinyStream", "CueModel", "CueStream"]
