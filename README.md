# Cue

**When a voice agent should stop talking, and when it should answer.**

Cue listens to the caller's side of a call and decides, causally, as the audio arrives:

| While the agent is speaking, the caller… | Cue says |
|---|---|
| says "mm-hm", "okay", "right" (a backchannel) | nothing: keep talking |
| coughs, types, has a TV on, someone else in the room talks | nothing: keep talking |
| says "wait", "one second", "hold on" | **PAUSE** |
| objects or corrects ("no, that's wrong"), or starts a long turn of their own | **STOP** |

| After the caller stops talking… | Cue says |
|---|---|
| they are mid-thought: "umm, … it should be, … eighteen lakhs" | nothing yet: keep waiting |
| they have finished | **RESPOND** |

In a cascaded voice agent (speech recognition → language model → speech synthesis),
Cue takes the place of the voice-activity detector that decides interruptions and the
silence timer that decides end of turn.

```bash
pip install cue-turn
```

```python
import cue_turn

cue = cue_turn.load()                              # Cue Tiny: CPU, numpy + onnxruntime
stream = cue.stream(sample_rate=8000)              # one stream per call, at the audio's own rate

for chunk in caller_audio:                         # int16 or float32 numpy, any length
    for d in stream.feed(chunk, assistant_speaking=player.is_playing()):
        print(d)                                   # {"t_ms": 18420, "decision": "STOP"}
```

## Models

| | Cue Tiny | Cue v4 |
|---|---|---|
| Load | `cue_turn.load()` | `cue_turn.load("v4")` |
| Install | `pip install cue-turn` | `pip install "cue-turn[full]"` |
| Runs on | CPU: about 10% of one core per call | GPU recommended (Whisper-small encoder) |
| Decides every | 40 ms (configurable) | 160 ms |
| Size | 6.2M parameters | 12.5M head + 87M encoder |
| Best for | interruptions | interruptions and end of turn |
| Weights | [IOTEverythin/cue-tiny](https://huggingface.co/IOTEverythin/cue-tiny) | [IOTEverythin/cue-v4](https://huggingface.co/IOTEverythin/cue-v4) |

The weights download from Hugging Face on first use and are cached. `load()` also takes
a Hugging Face repo id or a local folder, and `$CUE_MODEL` sets the default.

## How good is it

Held-out synthetic Indian English calls (different scripts and voices from training),
caller channel only. Full tables and method are in the model cards.

**Interruptions** (120 calls, 450 events while the agent speaks; clean audio):

| System | Stops on backchannels | Stops on noise | Hard interrupts caught | Takeovers caught | Reaction p50 |
|---|---|---|---|---|---|
| VAD with a 200 ms hold (Silero) | 70.2% | 41.4% | 100% | 90.6% | 295 ms |
| VAD with a 400 ms hold (Silero) | 35.1% | 32.7% | 96.8% | 31.2% | 559 ms |
| **Cue v4** (default) | **0.9%** | **4.3%** | 87.1% | 81.2% | 334 ms |

**End of turn** (120 calls, 728 turn ends and 134 mid-thought pauses; clean audio):

| System | Answers into mid-thought pauses | Turn ends answered | Delay p50 |
|---|---|---|---|
| Silence timer 500 ms | 95.5% | 94.2% | 531 ms |
| Silence timer 800 ms | 74.6% | 93.0% | 819 ms |
| **Cue v4** | **6.0%** | **95.6%** | **174 ms** |
| Cue Tiny | 11.2% | 92.7% | 197 ms |

Compared under one rule (stop on at most 8% of backchannels) and averaged over the
clean and realistic test sets (with echo, reverb, noise and a phone codec), Cue Tiny
catches 84% of hard interrupts and 68% of takeovers, against Cue v4's 87% and 71%.
Use Tiny for interruptions; prefer v4 for end of turn.

## Settings

Each model ships tuned defaults. Override them per stream, for example
`cue.stream(sample_rate=8000, stop_p=0.9, fallback_ms=1500)`, or during a call through
`stream.p`.

| Setting | Tiny | v4 | Effect |
|---|---|---|---|
| `stop_p` | 0.93 | 0.99 | probability needed to STOP (or PAUSE); lower reacts sooner, stops on more backchannels |
| `hold` | 2 | 4 | 20 ms frames it must hold |
| `cooldown_ms` | 1500 | 1500 | no new STOP / PAUSE for this long after one |
| `p_done` | 0.99 | 0.97 | end-of-turn probability needed to RESPOND |
| `turn_hold` | 8 | 4 | 20 ms frames it must hold |
| `min_speech` | 10 | 10 | 20 ms frames of caller speech before an end of turn can be called |
| `fallback_ms` | 2000 | 2000 | answer anyway after this much silence once the caller has spoken |

## Pipecat

```bash
pip install "cue-turn[pipecat]"                    # Python 3.11+, as Pipecat
```

```python
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from cue_turn.pipecat import CueSession, CueUserTurnStartStrategy, CueUserTurnStopStrategy

cue = CueSession()                                   # one per call; the model loads once per process
params = LLMUserAggregatorParams(
    user_turn_strategies=UserTurnStrategies(
        start=[CueUserTurnStartStrategy(cue)],       # only Cue's STOP / PAUSE interrupts the bot
        stop=[CueUserTurnStopStrategy(cue)],         # Cue's RESPOND ends the caller's turn
    ),
)
```

- **Start:** while the bot is quiet, the caller's turn starts on VAD as usual. While the
  bot speaks, only Cue's STOP or PAUSE starts a turn and interrupts it; words transcribed
  during a backchannel are dropped from the turn.
- **Stop:** Pipecat's `TurnAnalyzerUserTurnStopStrategy` with Cue as the turn analyzer.
- Either can be used alone. Both share the `CueSession`, so audio is processed once.
- Keep a VAD analyzer in the pipeline, and feed Cue the caller's leg with echo
  cancellation on: the agent's own voice leaking back can look like an interruption.

See [examples/pipecat_strategies.py](examples/pipecat_strategies.py).

## Command line

```bash
cue-turn call.wav --bot-channel 1                  # two-channel recording: caller on 0, bot on 1
cue-turn caller.wav --bot-spans bot.json           # [[start_ms, end_ms], ...] when the bot spoke
cue-turn call.wav --model v4 --set stop_p=0.97 --json
```

## Limits

- **Trained on synthetic speech:** Indian English recruiter calls with TTS voices.
  It was checked on three real calls; other languages, accents and domains are untested.
- **Takeovers** (the caller starting a long turn of their own) are the hardest class.
- **Echo:** feed the caller's leg only, with echo cancellation.
- **Short standalone turns** like "Hello?" can get a low end-of-turn probability; the
  `fallback_ms` timer answers them.
- Cue decides *when* to stop or answer, not *what* was said.

## Licence

The code in this repository is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
The model weights are separate and carry their own licences, stated in each model card.

## Development

```bash
pip install -e ".[dev,pipecat]"
pytest                                               # model tests download Cue Tiny, or set CUE_TINY_DIR
```
