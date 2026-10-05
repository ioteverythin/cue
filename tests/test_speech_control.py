"""The bot-speech controller: pacing, graceful stop, pause / resume, and the reply rules."""
import numpy as np
import pytest

from cue_turn.speech_control import Playout, fade, reply_kind


def chunk(ms=20, sr=16000, v=1000):
    return ("audio", (np.ones(sr * ms // 1000) * v).astype(np.int16).tobytes(), sr, 1, None)


@pytest.mark.parametrize("text,kind", [
    ("Wait.", "hold"), ("One second please", "hold"), ("haan ruko", "hold"), ("hold on, okay", "hold"),
    ("Okay, go on.", "continue"), ("yes", "continue"), ("haan ji, boliye", "continue"), ("sorry, go ahead", "continue"),
    ("No, I said Thursday not Tuesday", "turn"), ("what is the salary", "turn"), ("", "hold"),
])
def test_reply_kind(text, kind):
    assert reply_kind(text) == kind


def test_pacing_keeps_a_small_lookahead():
    p = Playout(lookahead_ms=60)
    for _ in range(10):
        p.put(chunk())
    assert len(p.next(0.0)) == 3                       # 60 ms ahead
    assert len(p.next(0.0)) == 0
    assert len(p.next(0.02)) == 1                      # one more as time passes


def test_other_frames_keep_their_order():
    p = Playout(lookahead_ms=20)
    p.put(chunk()); p.put(("other", "text")); p.put(chunk())
    out = p.next(0.0)
    assert [o[0] for o in out] == ["audio", "other"]


def test_pause_fades_and_keeps_the_rest_then_resume_fades_in():
    p = Playout(lookahead_ms=20, fade_ms=10)
    for _ in range(3):
        p.put(chunk())
    tail = p.pause()
    a = np.frombuffer(tail[0][1], np.int16)
    assert len(a) == 160 and a[0] > 900 and abs(a[-1]) < 10      # 10 ms fading to silence
    assert len(p.q) == 3 and p.next(1.0) == []                     # nothing lost, nothing sent while paused
    p.resume(2.0)
    first = np.frombuffer(p.next(2.0)[0][1], np.int16)
    assert first[0] == 0 and first[-1] == 1000                     # fades back in


def test_stop_sends_a_faded_tail_and_drops_the_rest():
    p = Playout(fade_ms=30)
    for _ in range(5):
        p.put(chunk())
    tail = p.stop()
    assert len(tail) == 1 and not p.q and not p.paused


def test_fade_in_and_out():
    pcm = (np.ones(100) * 1000).astype(np.int16).tobytes()
    out = np.frombuffer(fade(pcm, 0, out=True), np.int16)
    inn = np.frombuffer(fade(pcm, 50, out=False), np.int16)
    assert out[0] == 1000 and out[-1] == 0 and inn[0] == 0 and inn[60] == 1000
