"""The decision rule on hand-made probability sequences (no model needed)."""
import numpy as np
import pytest

from cue_turn import Policy

LABELS = ["NONE", "USER_TURN", "BACKCHANNEL", "ACCIDENTAL", "SOFT_INTERRUPT", "HARD_INTERRUPT", "TAKEOVER"]
SET = dict(stop_p=0.9, hold=2, cooldown_ms=1500, p_done=0.95, turn_hold=3, min_speech=4, fallback_ms=200)


def frame(label, p=1.0):
    v = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1), np.float32)
    v[LABELS.index(label)] = p
    return v


def run(pol, seq, start_ms=0):
    """seq: (label, p_done, speaking) per 20 ms frame -> [(t_ms, decision)]."""
    out = []
    for i, (lab, done, spk) in enumerate(seq):
        t = start_ms + (i + 1) * 20
        a = pol(frame(lab), done, spk, t)
        if a:
            out.append((t, a))
    return out


def test_stop_needs_hold_frames():
    pol = Policy(LABELS, 20, **SET)
    assert run(pol, [("HARD_INTERRUPT", 0, 1)] * 1) == []
    pol = Policy(LABELS, 20, **SET)
    assert run(pol, [("HARD_INTERRUPT", 0, 1)] * 2) == [(40, "STOP")]


def test_backchannel_never_stops():
    pol = Policy(LABELS, 20, **SET)
    assert run(pol, [("BACKCHANNEL", 0, 1)] * 50) == []


def test_pause_and_cooldown():
    pol = Policy(LABELS, 20, **SET)
    seq = [("SOFT_INTERRUPT", 0, 1)] * 2 + [("TAKEOVER", 0, 1)] * 10
    assert run(pol, seq) == [(40, "PAUSE")]                 # the takeover falls inside the cooldown


def test_takeover_and_hard_add_up():
    pol = Policy(LABELS, 20, **SET)
    v = np.zeros(len(LABELS), np.float32)
    v[LABELS.index("HARD_INTERRUPT")] = v[LABELS.index("TAKEOVER")] = 0.46
    assert pol(v, 0, 1, 20) is None and pol(v, 0, 1, 40) == "STOP"


def test_respond_after_speech_and_hold():
    pol = Policy(LABELS, 20, **{**SET, "fallback_ms": 0})
    seq = [("USER_TURN", 0, 0)] * 4 + [("NONE", 0.99, 0)] * 3
    assert run(pol, seq) == [(140, "RESPOND")]


def test_no_respond_without_enough_speech():
    pol = Policy(LABELS, 20, **{**SET, "fallback_ms": 0})
    assert run(pol, [("USER_TURN", 0, 0)] * 3 + [("NONE", 0.99, 0)] * 20) == []


def test_mid_thought_pause_waits_then_fallback():
    pol = Policy(LABELS, 20, **SET)
    seq = [("USER_TURN", 0, 0)] * 4 + [("NONE", 0.3, 0)] * 12
    assert run(pol, seq) == [(80 + 200, "RESPOND")]         # fallback after 200 ms of silence


def test_bot_speaking_resets_turn():
    pol = Policy(LABELS, 20, **{**SET, "fallback_ms": 0})
    seq = [("USER_TURN", 0, 0)] * 4 + [("NONE", 0, 1)] + [("NONE", 0.99, 0)] * 5
    assert run(pol, seq) == []


def test_unknown_setting():
    with pytest.raises(TypeError):
        Policy(LABELS, 20, **SET, stop_prob=0.5)
