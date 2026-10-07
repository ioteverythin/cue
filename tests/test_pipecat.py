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


class FakeController:
    def __init__(self, session):
        self.calls = []; session.controller = self

    async def pause(self, auto_resume=True): self.calls.append("pause" if auto_resume else "duck")
    async def resume(self): self.calls.append("resume")
    async def stop(self): self.calls.append("stop")


async def test_pause_then_continue_resumes_without_a_turn():
    s = CueSession(FakeModel([{"t_ms": 40, "decision": "PAUSE"}]))
    c = FakeController(s); start = CueUserTurnStartStrategy(s); seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.CONTINUE
    assert s.paused and c.calls == ["pause"] and seen == []
    await start.process_frame(BotStoppedSpeakingFrame())           # the transport notices the silence
    assert await start.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.CONTINUE
    await start.process_frame(TranscriptionFrame(text="wait", user_id="u", timestamp="0"))
    assert s.paused and seen == ["reset"]
    await start.process_frame(TranscriptionFrame(text="okay go on", user_id="u", timestamp="0"))
    assert not s.paused and c.calls == ["pause", "resume"] and seen == ["reset", "reset"]


async def test_pause_then_a_question_becomes_a_turn():
    s = CueSession(FakeModel([{"t_ms": 40, "decision": "PAUSE"}]))
    c = FakeController(s); start = CueUserTurnStartStrategy(s); seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    await start.process_frame(audio())
    r = await start.process_frame(TranscriptionFrame(text="actually what is the salary for this role", user_id="u", timestamp="0"))
    assert r == ProcessFrameResult.STOP and not s.paused and c.calls == ["pause", "stop"] and seen == ["started"]


async def test_stop_fades_out_then_interrupts():
    s = CueSession(FakeModel([{"t_ms": 40, "decision": "STOP"}]))
    c = FakeController(s); start = CueUserTurnStartStrategy(s); seen = watch(start)
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.STOP
    assert c.calls == ["stop"] and seen == ["started"]


async def test_controller_in_a_pipeline_paces_audio_in_order():
    import time
    from pipecat.frames.frames import OutputAudioRawFrame, TextFrame
    from pipecat.tests.utils import run_test
    from cue_turn.pipecat import CueSpeechController
    s = CueSession(FakeModel2())
    ctl = CueSpeechController(s, lookahead_ms=40)
    frames = [OutputAudioRawFrame(audio=(np.ones(320) * (i + 1)).astype(np.int16).tobytes(), sample_rate=16000, num_channels=1)
              for i in range(10)]                               # 10 x 20 ms
    frames.insert(5, TextFrame(text="mid"))
    t0 = time.perf_counter()
    down, _ = await run_test(ctl, frames_to_send=frames, expected_down_frames=[OutputAudioRawFrame] * 5 + [TextFrame] + [OutputAudioRawFrame] * 5)
    took = time.perf_counter() - t0
    vals = [int(np.frombuffer(f.audio, np.int16)[0]) for f in down if isinstance(f, OutputAudioRawFrame)]
    assert vals == list(range(1, 11))                           # in order, nothing lost
    assert took >= 0.15                                         # released in real time, not all at once
    assert s.agent_rate == 16000                                # Cue v5 heard what was played


class QuietModel(FakeModel):
    labels = ["NONE", "USER_TURN", "BACKCHANNEL", "ACCIDENTAL", "SOFT_INTERRUPT", "HARD_INTERRUPT", "TAKEOVER"]

    def __init__(self, *script, quiet_from=99):
        super().__init__(*script); self.quiet_from = quiet_from

    def stream(self, sample_rate, **settings):
        st = super().stream(sample_rate, **settings); model = self
        feed = st.feed

        def fed(audio, speaking):
            out = feed(audio, speaking)
            st.last_probs = np.array([1.0 if len(st.fed) >= model.quiet_from else 0.0, 1.0, 0, 0, 0, 0, 0])
            return out
        st.feed = fed
        return st


def ducking(model):
    s = CueSession(model)
    c = FakeController(s); c.duck_ms, c.duck_resume_ms = 1200, 240
    start = CueUserTurnStartStrategy(s)
    return s, c, start, watch(start)


async def test_duck_then_backchannel_carries_on():
    s, c, start, seen = ducking(QuietModel(quiet_from=3))       # caller audible for 2 chunks, then quiet
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.CONTINUE
    assert c.calls == ["duck"] and s.ducked_ms == 0
    await start.process_frame(BotStoppedSpeakingFrame())        # the transport notices the silence: no turn
    for _ in range(14):                                         # 20 ms chunks: 2 speech, then 12 quiet = 240 ms
        r = await start.process_frame(audio())
    assert r == ProcessFrameResult.CONTINUE and s.ducked_ms is None
    assert c.calls == ["duck", "resume"] and seen == ["reset"]


async def test_duck_then_caller_keeps_talking_is_their_turn():
    s, c, start, seen = ducking(QuietModel())                   # never quiet
    await start.process_frame(BotStartedSpeakingFrame())
    await start.process_frame(VADUserStartedSpeakingFrame())
    results = [await start.process_frame(audio()) for _ in range(60)]    # 60 x 20 ms = 1200 ms
    assert results[-1] == ProcessFrameResult.STOP and c.calls == ["duck", "stop"] and seen == ["started"]


async def test_duck_then_stop_is_their_turn_at_once():
    s, c, start, seen = ducking(QuietModel([], [{"t_ms": 40, "decision": "STOP"}]))
    await start.process_frame(BotStartedSpeakingFrame())
    await start.process_frame(VADUserStartedSpeakingFrame())
    assert await start.process_frame(audio()) == ProcessFrameResult.CONTINUE
    assert await start.process_frame(audio()) == ProcessFrameResult.STOP
    assert c.calls == ["duck", "stop"] and seen == ["started"]


async def test_no_duck_without_duck_ms():
    s = CueSession(FakeModel()); c = FakeController(s); start = CueUserTurnStartStrategy(s)
    await start.process_frame(BotStartedSpeakingFrame())
    assert await start.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.CONTINUE
    assert c.calls == [] and s.ducked_ms is None
