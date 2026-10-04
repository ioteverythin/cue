"""Runs the real models. They are downloaded from Hugging Face, or taken from a folder
given in $CUE_TINY_DIR / $CUE_V4_DIR; the tests are skipped when neither is available."""
import os

import numpy as np
import pytest

import cue_turn


def _load(name, env):
    try:
        return cue_turn.load(os.getenv(env) or name)
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:                       # no network / no access
        pytest.skip(f"model {name} unavailable: {e}")


def _call(sr, seconds=12, seed=0):
    """Noise-floor 'call' with bursts of a voiced buzz (a stand-in for speech)."""
    rng = np.random.default_rng(seed)
    t = np.arange(sr * seconds) / sr
    a = 0.002 * rng.standard_normal(len(t))
    for s0 in (1.0, 4.0, 7.5):
        m = (t >= s0) & (t < s0 + 1.5)
        a[m] += 0.2 * np.sign(np.sin(2 * np.pi * 140 * t[m])) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    return (a * 32767).astype(np.int16)


@pytest.fixture(scope="module")
def tiny():
    return _load("tiny", "CUE_TINY_DIR")


def run(model, audio, sr, chunk, speaking, **settings):
    s = model.stream(sample_rate=sr, **settings)
    out = []
    for k in range(0, len(audio), chunk):
        out += s.feed(audio[k:k + chunk], speaking)
    return out, s


def test_tiny_chunking_does_not_change_decisions(tiny):
    a = _call(8000)
    spk = lambda t: 3500 <= t < 6000
    ref, s = run(tiny, a, 8000, 160, spk, fallback_ms=600)
    assert abs(s.t_frames - len(a) // 160) <= 2        # one decision frame per 20 ms of audio
    for chunk in (80, 333, 4000):
        assert run(tiny, a, 8000, chunk, spk, fallback_ms=600)[0] == ref


def test_tiny_outputs_are_probabilities(tiny):
    s = tiny.stream(sample_rate=8000)
    s.feed(_call(8000), False)
    assert abs(float(s.last_probs.sum()) - 1) < 1e-4 and 0 <= s.last_p_done <= 1


def test_tiny_settings_can_change_mid_call(tiny):
    s = tiny.stream(sample_rate=16000)
    s.p["fallback_ms"] = 1234
    assert s.policy.p["fallback_ms"] == 1234


def test_v4_feed_matches_step():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    v4 = _load("v4", "CUE_V4_DIR")
    a = _call(16000, seconds=6).astype(np.float32) / 32768
    s1 = v4.stream()
    stepped = [s1.step(a[k:k + s1.step_samples], False) for k in range(0, len(a) - s1.step_samples + 1, s1.step_samples)]
    fed, s2 = run(v4, a, 16000, 320, False)
    assert [d["decision"] for d in stepped if d["decision"]] == [d["decision"] for d in fed]
    np.testing.assert_allclose(s1.last_probs, s2.last_probs, atol=1e-5)


def test_v5_agent_audio_feed_matches_step():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    v5 = _load("v5", "CUE_V5_DIR")
    a = _call(16000, seconds=6).astype(np.float32) / 32768
    agent = _call(16000, seconds=6, seed=1).astype(np.float32) / 32768
    s1 = v5.stream()
    n = s1.step_samples
    stepped = [s1.step(a[k:k + n], True, agent_audio=agent[k:k + n]) for k in range(0, len(a) - n + 1, n)]
    s2 = v5.stream(sample_rate=16000, agent_sample_rate=16000)
    for k in range(0, len(a), 320):
        s2.feed(a[k:k + 320], True, agent_audio=agent[k:k + 320])
    np.testing.assert_allclose(s1.last_probs, s2.last_probs, atol=1e-5)
    s3 = v5.stream()                                  # without agent audio the input is zero: different
    for k in range(0, len(a) - n + 1, n):
        s3.step(a[k:k + n], True)
    assert not np.allclose(s1.last_probs, s3.last_probs, atol=1e-4)


def test_profiles():
    v5 = _load("v5", "CUE_V5_DIR")
    assert v5.stream(profile="cautious").p["stop_p"] == 0.95
    with pytest.raises(ValueError):
        v5.stream(profile="nope")
