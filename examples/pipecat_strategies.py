"""Cue's turn-taking in a Pipecat pipeline (pip install "cue-turn[pipecat]" pipecat-ai[silero]).

Build the user aggregator's parameters once per call and pass them to
LLMContextAggregatorPair; the rest of the pipeline (transport, STT, LLM, TTS) is unchanged:

    context = LLMContext(messages)
    aggregators = LLMContextAggregatorPair(context, user_params=cue_user_params())
    pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts,
                         transport.output(), aggregators.assistant()])
"""
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from cue_turn.pipecat import CueSession, CueUserTurnStartStrategy, CueUserTurnStopStrategy


def cue_user_params(model: str | None = None, **settings) -> LLMUserAggregatorParams:
    """model: "tiny" (default, CPU), "v4", a Hugging Face repo or a folder.
    settings: Cue decision settings, e.g. stop_p=0.97, fallback_ms=1500."""
    cue = CueSession(model, settings)               # one per call; the model itself is loaded once per process
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),           # still starts the caller's turn while the bot is quiet
        user_turn_strategies=UserTurnStrategies(
            start=[CueUserTurnStartStrategy(cue)],  # while the bot speaks, only Cue's STOP / PAUSE interrupts it
            stop=[CueUserTurnStopStrategy(cue)],    # Cue's RESPOND ends the caller's turn
        ),
    )
