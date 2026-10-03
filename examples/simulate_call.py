"""Play a recorded call through Cue as if it were live, and print what Cue decides.

    python examples/simulate_call.py call.wav --bot-channel 1

The caller's channel is fed in 20 ms chunks, the way a phone line delivers it; the bot
is taken to be speaking whenever its own channel is (from the channel's energy).
"""
import argparse
import time

import cue_turn
from cue_turn.audio import activity, read, to_16k

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--caller-channel", type=int, default=0)
ap.add_argument("--bot-channel", type=int, default=1)
ap.add_argument("--model", default="tiny")
a = ap.parse_args()

caller, sr = read(a.wav, a.caller_channel)
bot = activity(to_16k(read(a.wav, a.bot_channel)[0], sr))          # per 20 ms: is the bot speaking?
bot_speaking = lambda t_ms: bool(bot[min(int(t_ms) // 20, len(bot) - 1)])

cue = cue_turn.load(a.model)
stream = cue.stream(sample_rate=sr)
chunk = sr // 50
t0 = time.perf_counter()
for k in range(0, len(caller), chunk):
    for d in stream.feed(caller[k:k + chunk], bot_speaking):
        state = "bot speaking" if bot_speaking(d["t_ms"]) else "bot quiet"
        print(f"{d['t_ms'] / 1000:8.2f} s  {d['decision']:<8} ({state})")
took = time.perf_counter() - t0
print(f"{len(caller) / sr:.0f} s of audio in {took:.1f} s ({100 * took / (len(caller) / sr):.1f}% of real time)")
