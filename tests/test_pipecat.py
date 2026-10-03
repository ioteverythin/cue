"""The Pipecat strategies, with a stand-in model that returns scripted decisions."""
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
