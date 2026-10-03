import numpy as np
import pytest

from cue_turn.audio import StreamResampler, activity, to_16k


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_streaming_equals_whole(sr):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    whole = to_16k(x, sr)
    r, parts, k = StreamResampler(sr), [], 0
    while k < len(x):
        n = int(rng.integers(1, 900))
        parts.append(r(x[k:k + n]))
        k += n
    parts.append(r.flush())
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-6)
    assert len(whole) == int(np.ceil(len(x) * 16000 / sr))


@pytest.mark.parametrize("sr", [8000, 44100])
def test_matches_torchaudio(sr):
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchaudio.functional")
    x = np.random.default_rng(1).standard_normal(sr).astype(np.float32) * 0.1
    ref = F.resample(torch.from_numpy(x), sr, 16000).numpy()
    np.testing.assert_allclose(to_16k(x, sr), ref, atol=1e-5)


def test_activity_finds_a_tone():
    t = np.arange(16000 * 3) / 16000
    a = 0.001 * np.random.default_rng(2).standard_normal(len(t))
    a[16000:32000] += 0.3 * np.sin(2 * np.pi * 200 * t[16000:32000])
    act = activity(a.astype(np.float32))
    assert not act[10:45].any() and act[55:95].all()
