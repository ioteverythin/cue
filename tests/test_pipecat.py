"""The Pipecat strategies, with a stand-in model that returns scripted decisions."""
import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult

from cue_turn.pipecat import CueSession, CueTurnAnalyzer, CueUserTurnStartStrategy, CueUserTurnStopStrategy


class FakeStream:
    def __init__(self, script, fallback_ms):
        self.script, self.p, self.last_p_done, self.fed = script, {"fallback_ms": fallback_ms}, 0.5, []

    def feed(self, audio, speaking):
        self.fed.append((len(audio), speaking))
        return self.script.pop(0) if self.script else []


class FakeModel:
    cfg = {"policy": {"end_of_turn": {"fallback_ms": 2000}}}

    def __init__(self, *script):
        self.script = list(script)

    def stream(self, sample_rate, **settings):
        self.last = FakeStream(self.script, settings.get("fallback_ms", 2000))
        return self.last


def audio(n=160):
    return InputAudioRawFrame(audio=b"\x00\x00" * n, sample_rate=8000, num_channels=1)


def watch(strategy):
    seen = []

    async def started(_strategy, params):
        seen.append("started")

    async def reset(_strategy):
        seen.append("reset")

    strategy.add_event_handler("on_user_turn_started", started)
    strategy.add_event_handler("on_reset_aggregation", reset)
    return seen


async def test_stop_while_bot_speaks_interrupts():
    s = CueSession(FakeModel([{"t_ms": 40, "decision": "STOP"}]))
    start = CueUserTurnStartStrategy(s)
    seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.STOP
    assert seen == ["started"]
    assert s.model.last.fed == [(160, True)]


async def test_backchannel_while_bot_speaks_is_dropped():
    s = CueSession(FakeModel([]))
    start = CueUserTurnStartStrategy(s)
    seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.CONTINUE
    await start.process_frame(TranscriptionFrame(text="mm-hm", user_id="u", timestamp="0"))
    assert seen == ["reset"]


async def test_stop_after_bot_finished_is_ignored():
    s = CueSession(FakeModel([{"t_ms": 40, "decision": "STOP"}]))
    start = CueUserTurnStartStrategy(s)
    seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    await start.process_frame(BotStoppedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.CONTINUE
    assert seen == []


async def test_vad_starts_a_turn_when_bot_is_quiet():
    start = CueUserTurnStartStrategy(CueSession(FakeModel()))
    seen = watch(start)
    assert await start.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.STOP
    assert seen == ["started"]


def test_respond_completes_the_turn_and_audio_is_processed_once():
    s = CueSession(FakeModel([], [{"t_ms": 80, "decision": "RESPOND"}]))
    an = CueTurnAnalyzer(s)
    an.set_sample_rate(8000)
    f1, f2 = audio(), audio()
    assert an.append_audio(f1.audio, is_speech=True) == EndOfTurnState.INCOMPLETE
    s.feed(f2.audio, 8000)                         # the start strategy saw this frame first
    assert an.append_audio(f2.audio, is_speech=False) == EndOfTurnState.COMPLETE
    assert len(s.model.last.fed) == 2


def test_stale_respond_is_discarded_by_new_speech():
    s = CueSession(FakeModel([{"t_ms": 80, "decision": "RESPOND"}]))
    s.feed(audio().audio, 8000)
    assert s.pending_respond
    an = CueTurnAnalyzer(s)
    an.set_sample_rate(8000)
    assert an.append_audio(audio().audio, is_speech=True) == EndOfTurnState.INCOMPLETE


def test_widening_the_timeout_widens_the_fallback():
    s = CueSession(FakeModel())
    stop = CueUserTurnStopStrategy(s)
    stop._user_speech_timeout = 5.0
    assert s.fallback_ms() == 5000
    s.feed(audio().audio, 8000)
    assert s.model.last.p["fallback_ms"] == 5000 and stop._user_speech_timeout == 5.0


class FakeStream2(FakeStream):
    def feed(self, audio, speaking, agent_audio=None):
        self.fed.append((len(audio), speaking, None if agent_audio is None else agent_audio.copy()))
        return []


class FakeModel2(FakeModel):
    two_channel = True

    def stream(self, sample_rate, agent_sample_rate=None, **settings):
        self.last = FakeStream2(self.script, settings.get("fallback_ms", 2000))
        return self.last


def test_v5_gets_the_bot_audio_as_it_plays():
    s = CueSession(FakeModel2())
    s.feed(audio(160).audio, 8000)                         # no tap yet: no agent audio at all
    assert s.model.last.fed[-1][2] is None
    s.observe(BotStartedSpeakingFrame())
    s.feed_agent((np.ones(480) * 1000).astype(np.int16).tobytes(), 24000)    # 20 ms of bot audio
    s.feed(audio(160).audio, 8000)                         # 20 ms of caller audio
    a = s.model.last.fed[-1][2]
    assert len(a) == 480 and np.allclose(a, 1000 / 32768)
    s.feed(audio(160).audio, 8000)                         # queue empty: silence for the rest
    assert not s.model.last.fed[-1][2].any()


def test_v5_agent_audio_is_silence_when_the_bot_is_quiet_and_cleared_on_interruption():
    from pipecat.frames.frames import InterruptionFrame
    s = CueSession(FakeModel2())
    s.observe(BotStartedSpeakingFrame())
    s.feed_agent((np.ones(4800) * 1000).astype(np.int16).tobytes(), 24000)
    s.observe(InterruptionFrame())
    assert not s.agent_queue
    s.observe(BotStoppedSpeakingFrame())
    s.feed(audio(160).audio, 8000)
    assert not s.model.last.fed[-1][2].any()


async def test_tap_passes_frames_on_and_copies_bot_audio():
    from pipecat.frames.frames import OutputAudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection
    from cue_turn.pipecat import CueAgentAudioTap
    s = CueSession(FakeModel2())
    tap = CueAgentAudioTap(s)
    pushed = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)
    tap.push_frame = push
    f = OutputAudioRawFrame(audio=b"\x10\x00" * 480, sample_rate=24000, num_channels=1)
    await tap.process_frame(f, FrameDirection.DOWNSTREAM)
    assert pushed == [f] and s.agent_rate == 24000 and len(s.agent_queue) == 1
