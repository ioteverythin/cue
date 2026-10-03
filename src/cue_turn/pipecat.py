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
"""
from __future__ import annotations

import time

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
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData, TurnMetricsData
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

from .hub import load

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

    @property
    def p_done(self) -> float:
        return self.stream.last_p_done if self.stream else 0.0

    def observe(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking = False
            self.pending_barge = None
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
            self.stream = self.model.stream(sample_rate=sample_rate, **self.settings)
        audio = np.frombuffer(buffer, dtype=np.int16)
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels)[:, 0]
        for d in self.stream.feed(audio, self.bot_speaking):
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
        if isinstance(frame, InputAudioRawFrame):
            s.feed(frame.audio, frame.sample_rate, frame.num_channels)
            d = s.take_barge()
            if d and s.bot_speaking:
                logger.info(f"Cue {d['decision']}: interrupting the bot")
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
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
