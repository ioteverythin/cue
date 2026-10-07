"""Cue in a Pipecat pipeline: when to interrupt the bot, and when the caller is done.

    pip install "cue-turn[pipecat]"

    from cue_turn.pipecat import CueSession, CueUserTurnStartStrategy, CueUserTurnStopStrategy

    cue = CueSession()                             # one per call (the model is loaded once per process)
    params = LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[CueUserTurnStartStrategy(cue)],  # only Cue's STOP / PAUSE interrupts the bot
            stop=[CueUserTurnStopStrategy(cue)],    # Cue's RESPOND ends the caller's turn
        ),
    )

Either strategy can be used alone. Both share the session, so each audio frame is
processed once. Keep a VAD analyzer in the pipeline: the caller's turn still starts on
VAD while the bot is quiet.

Cue v5 also hears the bot's own audio. Put a CueAgentAudioTap just before the transport's
output so it can see what the bot plays:

    pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts,
                         CueAgentAudioTap(cue), transport.output(), assistant_aggregator])

Without the tap, v5 runs with that input at zero (as it was also trained); v4 and Tiny ignore it.

CueSpeechController, in the same place, does what the tap does and also controls the bot's speech:
a STOP ends it with a short fade instead of a cut; a PAUSE ("wait", "one second") stops it but keeps
the rest, which resumes where it stopped when the caller says "okay, go on" (or after a silence),
or is dropped for a normal reply when the caller says something else. Use one or the other.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import numpy as np
from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    SystemFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData, TurnMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

from .hub import load
from .speech_control import Playout, reply_kind

_MODELS: dict = {}


def shared_model(name: str | None = None):
    """The model, loaded once per process and shared by all calls."""
    if name not in _MODELS:
        t0 = time.perf_counter()
        _MODELS[name] = load(name)
        logger.info(f"Cue model {name or 'default'} loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    return _MODELS[name]


class CueSession:
    """Per-call Cue state: one audio stream, whether the bot is speaking, the latest decisions.

    model: a loaded model (cue_turn.load(...)), a name ("tiny", "v4", a Hugging Face repo
    or a folder), or None for $CUE_MODEL / Cue Tiny. settings: decision settings
    (cue_turn.SETTINGS), e.g. {"stop_p": 0.99, "fallback_ms": 2000}."""

    def __init__(self, model=None, settings: dict | None = None):
        self.model = model if hasattr(model, "stream") else shared_model(model)
        self.settings = {k: v for k, v in (settings or {}).items() if v is not None}
        self.stream = None
        self.sample_rate = 0
        self.bot_speaking = False
        self.vad_user_speaking = False
        self.pending_respond = False
        self.pending_barge: dict | None = None
        self._last_buffer = None
        self.agent_queue: deque = deque()            # bot audio not yet played (v5), float32 chunks
        self.agent_rate = 0
        self.controller = None                       # a CueSpeechController, if in the pipeline
        self.paused = False                          # the bot is paused, waiting for the caller
        self.pause_text: list[str] = []
        self.ducked_ms: float | None = None          # the bot ducked for the caller: ms of caller audio since
        self.duck_quiet_ms = 0.0

    @property
    def wants_agent_audio(self) -> bool:
        return bool(getattr(self.model, "two_channel", False))

    def feed_agent(self, buffer: bytes, sample_rate: int, num_channels: int = 1):
        """Bot audio on its way to the caller (from CueAgentAudioTap)."""
        if not self.wants_agent_audio or not sample_rate:
            return
        a = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        if num_channels > 1:
            a = a.reshape(-1, num_channels).mean(1)
        if sample_rate != self.agent_rate:
            self.agent_queue.clear()
            self.agent_rate = sample_rate
            if self.stream is not None:               # the stream started before the bot first spoke
                from .audio import StreamResampler
                self.stream.ares = StreamResampler(sample_rate)
        self.agent_queue.append(a)

    def _agent_span(self, seconds: float):
        """The bot audio played over the next `seconds` (silence when the bot is quiet), or None
        if no bot audio has ever been seen (no tap in the pipeline)."""
        if not self.agent_rate:
            return None
        n = int(round(seconds * self.agent_rate))
        if not self.bot_speaking:
            return np.zeros(n, np.float32)
        out, got = [], 0
        while got < n and self.agent_queue:
            c = self.agent_queue[0]
            take = min(len(c), n - got)
            out.append(c[:take]); got += take
            if take == len(c):
                self.agent_queue.popleft()
            else:
                self.agent_queue[0] = c[take:]
        out.append(np.zeros(n - got, np.float32))
        return np.concatenate(out)

    @property
    def p_done(self) -> float:
        return self.stream.last_p_done if self.stream else 0.0

    def observe(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking = False
            self.pending_barge = None
            self.agent_queue.clear()
        elif isinstance(frame, InterruptionFrame):
            self.agent_queue.clear()                  # what was queued will not be played
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.vad_user_speaking = True
            self.pending_respond = False      # a RESPOND from before this speech is stale
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.vad_user_speaking = False

    def feed(self, buffer: bytes, sample_rate: int, num_channels: int = 1):
        """Feed one chunk of caller audio (int16 PCM). The same chunk reaches both
        strategies; it is processed once."""
        if buffer is self._last_buffer or not sample_rate:
            return
        self._last_buffer = buffer
        if self.stream is None or sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            extra = {"agent_sample_rate": self.agent_rate or 24000} if self.wants_agent_audio else {}
            self.stream = self.model.stream(sample_rate=sample_rate, **extra, **self.settings)
        audio = np.frombuffer(buffer, dtype=np.int16)
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels)[:, 0]
        if self.wants_agent_audio:
            agent = self._agent_span(len(audio) / sample_rate)
            decisions = self.stream.feed(audio, self.bot_speaking, agent_audio=agent)
        else:
            decisions = self.stream.feed(audio, self.bot_speaking)
        for d in decisions:
            logger.debug(f"Cue: {d['decision']} at {d['t_ms']} ms (bot speaking: {self.bot_speaking})")
            if d["decision"] == "RESPOND":
                self.pending_respond = True
            else:
                self.pending_barge = d

    def fallback_ms(self) -> int:
        if self.stream is not None:
            return int(self.stream.p.get("fallback_ms", 0))
        return int(self.settings.get("fallback_ms", self.model.cfg["policy"]["end_of_turn"].get("fallback_ms", 0)))

    def set_fallback_ms(self, ms: int):
        self.settings["fallback_ms"] = ms
        if self.stream is not None:
            self.stream.p["fallback_ms"] = ms

    def take_barge(self) -> dict | None:
        d, self.pending_barge = self.pending_barge, None
        return d

    def caller_quiet(self) -> bool:
        """Cue's own reading of the last chunk: no caller speech (a model without NONE: unknown, False)."""
        probs = getattr(self.stream, "last_probs", None)
        labels = getattr(self.model, "labels", None)
        if probs is None or not labels or "NONE" not in labels:
            return False
        return float(probs[labels.index("NONE")]) >= 0.5


class CueUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start with Cue deciding interruptions.

    Bot quiet: the caller's turn starts on VAD, as usual. Bot speaking: only Cue's
    STOP / PAUSE starts a turn (and so interrupts the bot); a backchannel, cough or
    background voice does not, and its transcript is dropped. PAUSE also interrupts
    (Pipecat has no pause / resume for the bot's audio).
    """

    def __init__(self, session: CueSession, **kwargs):
        super().__init__(**kwargs)
        self._s = session

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        s = self._s
        was_speaking = s.bot_speaking
        s.observe(frame)
        if s.ducked_ms is not None:
            return await self._ducked(frame)
        if (isinstance(frame, VADUserStartedSpeakingFrame) and s.bot_speaking and not s.paused
                and s.controller is not None and getattr(s.controller, "duck_ms", 0)):
            # the caller started over the bot: quiet now, and let Cue say what it was
            logger.info("Cue: caller started over the bot, ducking")
            s.ducked_ms, s.duck_quiet_ms = 0.0, 0.0
            await s.controller.pause(auto_resume=False)
            return ProcessFrameResult.CONTINUE
        if isinstance(frame, InputAudioRawFrame):
            s.feed(frame.audio, frame.sample_rate, frame.num_channels)
            d = s.take_barge()
            if d and s.bot_speaking:
                if d["decision"] == "PAUSE" and s.controller is not None:
                    logger.info("Cue PAUSE: pausing the bot")
                    s.paused, s.pause_text = True, []
                    await s.controller.pause()
                    return ProcessFrameResult.CONTINUE
                logger.info(f"Cue {d['decision']}: interrupting the bot")
                if s.controller is not None:
                    await s.controller.stop()
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        elif isinstance(frame, TranscriptionFrame) and s.paused:
            # the caller's reply to a pause: still waiting, go on, or a new turn
            s.pause_text.append(frame.text)
            kind = reply_kind(frame.text)                 # each phrase on its own: "wait" ... "okay, go on"
            logger.info(f"Cue: reply during pause '{' '.join(s.pause_text)}' -> {kind}")
            if kind == "turn":
                s.paused = False
                if s.controller is not None:
                    await s.controller.stop()
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            await self.trigger_reset_aggregation()
            if kind == "continue":
                s.paused = False
                if s.controller is not None:
                    await s.controller.resume()
            return ProcessFrameResult.CONTINUE
        elif s.paused:
            return ProcessFrameResult.CONTINUE           # VAD / interim words while paused: wait for the words
        elif isinstance(frame, VADUserStartedSpeakingFrame) and not s.bot_speaking:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        elif isinstance(frame, BotStoppedSpeakingFrame) and was_speaking and s.vad_user_speaking:
            # the caller started talking over the bot's last words: their turn starts now
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and s.bot_speaking:
            # words heard while the bot talks and Cue did not call an interruption
            # (a backchannel, an echo, a side conversation): not a turn
            await self.trigger_reset_aggregation()
        return ProcessFrameResult.CONTINUE

    async def _ducked(self, frame: Frame) -> ProcessFrameResult:
        """The bot is ducked: STOP makes it the caller's turn, PAUSE a pause, the caller going quiet
        (Cue or the VAD) lets the bot carry on, and talking past duck_ms makes it the caller's turn.
        Words heard meanwhile are kept until it is clear which."""
        s, c = self._s, self._s.controller

        async def turn():
            s.ducked_ms = None
            await c.stop()
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        async def carry_on(why):
            logger.info(f"Cue: {why}; the bot carries on")
            s.ducked_ms = None
            await self.trigger_reset_aggregation()
            await c.resume()
            return ProcessFrameResult.CONTINUE

        if isinstance(frame, InputAudioRawFrame):
            s.feed(frame.audio, frame.sample_rate, frame.num_channels)
            ms = 1000 * len(frame.audio) / 2 / max(1, frame.num_channels) / frame.sample_rate
            s.ducked_ms += ms
            d = s.take_barge()
            if d and d["decision"] == "PAUSE":
                s.ducked_ms, s.paused, s.pause_text = None, True, []      # stays paused, as for a PAUSE
                return ProcessFrameResult.CONTINUE
            if d:
                logger.info(f"Cue {d['decision']} while ducked: the caller's turn")
                return await turn()
            s.duck_quiet_ms = s.duck_quiet_ms + ms if s.caller_quiet() else 0.0
            if s.duck_quiet_ms >= c.duck_resume_ms:
                return await carry_on("the caller went quiet")
            if s.ducked_ms >= c.duck_ms:
                logger.info("Cue: the caller kept talking; their turn")
                return await turn()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            return await carry_on("the caller stopped (VAD)")
        return ProcessFrameResult.CONTINUE


class CueTurnParams(BaseTurnParams):
    stop_secs: float = 2.0          # informational: Cue's own fallback is the fallback_ms setting


class CueTurnAnalyzer(BaseTurnAnalyzer):
    """End of turn from Cue's RESPOND decision (for TurnAnalyzerUserTurnStopStrategy)."""

    def __init__(self, session: CueSession, params: CueTurnParams | None = None):
        super().__init__()
        self._s = session
        self._params = params or CueTurnParams()
        self._speech_triggered = False

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    @property
    def params(self) -> CueTurnParams:
        return self._params

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        if is_speech and not self._speech_triggered:
            self._speech_triggered = True
            self._s.pending_respond = False   # only a RESPOND after this speech counts
        self._s.feed(buffer, self.sample_rate)
        if self._s.pending_respond and self._speech_triggered:
            self._s.pending_respond = False
            self._speech_triggered = False
            return EndOfTurnState.COMPLETE
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        done = self._s.pending_respond
        if done:
            self._s.pending_respond = False
            self._speech_triggered = False
        return (EndOfTurnState.COMPLETE if done else EndOfTurnState.INCOMPLETE,
                TurnMetricsData(processor="CueTurnAnalyzer", is_complete=done,
                                probability=round(self._s.p_done, 4), e2e_processing_time_ms=0.0))

    def clear(self):
        self._speech_triggered = False
        self._s.pending_respond = False


class CueUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """TurnAnalyzerUserTurnStopStrategy with Cue; also tracks whether the bot is speaking.

    Code that widens the strategy's silence timeout mid-call (for example while the
    caller reads out a number) widens Cue's fallback with it."""

    def __init__(self, session: CueSession, **kwargs):
        super().__init__(turn_analyzer=CueTurnAnalyzer(session), **kwargs)
        self._s = session

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        self._s.observe(frame)
        return await super().process_frame(frame)

    @property
    def _user_speech_timeout(self) -> float:
        return float(self._s.fallback_ms()) / 1000.0

    @_user_speech_timeout.setter
    def _user_speech_timeout(self, seconds: float):
        self._s.set_fallback_ms(int(seconds * 1000))


class CueAgentAudioTap(FrameProcessor):
    """Passes every frame on; copies the bot's outgoing audio into the Cue session (v5)."""

    def __init__(self, session: CueSession, **kwargs):
        super().__init__(**kwargs)
        self._s = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._s.feed_agent(frame.audio, frame.sample_rate, frame.num_channels)
        await self.push_frame(frame, direction)


class CueSpeechController(FrameProcessor):
    """Between the TTS and transport.output(): releases the bot's audio in real time (a small
    lookahead), so Cue can fade it out on STOP and pause / resume it on PAUSE; also feeds Cue v5
    exactly the audio being played. Frames keep their order; system and upstream frames pass at once.

    duck_ms > 0 adds a duck: when the caller starts speaking over the bot, the bot goes quiet at once;
    Cue's STOP (or the caller still talking after duck_ms) makes it the caller's turn, the caller
    falling quiet for duck_resume_ms lets the bot carry on where it stopped. On Voxi-Duo this cut
    talk-over from 15% to 4% without making the bot choppier (1200 ms works well)."""

    def __init__(self, session: CueSession, lookahead_ms: int = 60, fade_ms: int = 30,
                 resume_after_ms: int = 20000, duck_ms: int = 0, duck_resume_ms: int = 240, **kwargs):
        super().__init__(**kwargs)
        self._s = session
        session.controller = self
        self._play = Playout(lookahead_ms, fade_ms)
        self._resume_after = resume_after_ms / 1000
        self.duck_ms, self.duck_resume_ms = duck_ms, duck_resume_ms     # 0: no duck
        self._paused_at = None
        self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._task = self.create_task(self._run())
            return
        if isinstance(frame, (CancelFrame, EndFrame)):
            if isinstance(frame, EndFrame):                  # let queued speech finish first
                while self._play.q and not self._play.paused:
                    await asyncio.sleep(0.02)
            if self._task:
                await self.cancel_task(self._task); self._task = None
            await self.push_frame(frame, direction)
            return
        if direction != FrameDirection.DOWNSTREAM or isinstance(frame, SystemFrame):
            if isinstance(frame, InterruptionFrame):
                self._play.clear(); self._s.paused = False; self._paused_at = None
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, OutputAudioRawFrame):
            self._play.put(("audio", frame.audio, frame.sample_rate, frame.num_channels, frame))
        else:
            self._play.put(("other", frame))

    async def _send(self, items):
        for it in items:
            if it[0] == "audio":
                _, pcm, sr, ch, orig = it
                self._s.feed_agent(pcm, sr, ch)
                await self.push_frame(OutputAudioRawFrame(audio=pcm, sample_rate=sr, num_channels=ch))
            else:
                await self.push_frame(it[1])

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            if self._play.paused and self._paused_at is not None and now - self._paused_at >= self._resume_after:
                logger.info("Cue: no reply after the pause; resuming")
                await self.resume()
            await self._send(self._play.next(now))
            await asyncio.sleep(0.01)

    async def pause(self, auto_resume: bool = True):
        self._paused_at = asyncio.get_running_loop().time() if auto_resume else None
        await self._send(self._play.pause())

    async def resume(self):
        self._s.paused = False; self._paused_at = None
        self._play.resume(asyncio.get_running_loop().time())

    async def stop(self):
        self._paused_at = None
        await self._send(self._play.stop())
