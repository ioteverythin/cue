"""Run Cue on a recording and print its decisions.

    cue-turn call.wav --bot-channel 1              # two-channel call: caller on 0, bot on 1
    cue-turn caller.wav --bot-spans spans.json     # mono caller audio + when the bot spoke ([[start_ms, end_ms], ...])
    cue-turn call.wav --model v4 --set stop_p=0.9 --json
"""
from __future__ import annotations
import argparse
import json
import sys

from .audio import activity, read, to_16k
from .hub import load
from .policy import SETTINGS


def _value(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cue-turn", description="Run Cue on a recording and print its decisions.",
                                 epilog="settings: " + "; ".join(f"{k}: {v}" for k, v in SETTINGS.items()))
    ap.add_argument("audio", help="a recording (wav, flac, ...)")
    ap.add_argument("--model", default=None, help='"tiny" (default), "v5", "v4", "v3", a Hugging Face repo or a folder')
    ap.add_argument("--channel", type=int, default=0, help="the caller's channel (default 0)")
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--bot-channel", type=int, help="the bot's channel; when it speaks is read from its energy (v5 also hears it)")
    who.add_argument("--bot-spans", help="JSON file: [[start_ms, end_ms], ...] when the bot speaks")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a decision setting (repeatable)")
    ap.add_argument("--profile", help='a named setting profile of the model (v5: "responsive", "balanced", "cautious")')
    ap.add_argument("--json", action="store_true", help="print the decisions as JSON")
    a = ap.parse_args(argv)

    settings = {"profile": a.profile} if a.profile else {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in SETTINGS or not v:
            ap.error(f"--set {kv}: expected NAME=VALUE with NAME one of {', '.join(SETTINGS)}")
        settings[k] = _value(v)

    caller, sr = read(a.audio, a.channel)
    if a.bot_channel is not None:
        act = activity(to_16k(read(a.audio, a.bot_channel)[0], sr))
        speaking = lambda t: bool(act[min(int(t) // 20, len(act) - 1)]) if len(act) else False
    elif a.bot_spans:
        spans = json.loads(open(a.bot_spans, encoding="utf-8").read())
        speaking = lambda t: any(s0 <= t < s1 for s0, s1 in spans)
    else:
        speaking = False
        print("note: no --bot-channel or --bot-spans, so the bot is taken to be silent: "
              "only end-of-turn decisions (RESPOND) can be made", file=sys.stderr)

    cue = load(a.model)
    two = getattr(cue, "two_channel", False) and a.bot_channel is not None
    if two:                                         # v5 also hears the bot's own channel
        bot = read(a.audio, a.bot_channel)[0]
        stream = cue.stream(sample_rate=sr, agent_sample_rate=sr, **settings)
    else:
        stream = cue.stream(sample_rate=sr, **settings)
    step = sr // 50                                 # feed 20 ms at a time, as a live call would
    out = []
    for k in range(0, len(caller), step):
        if two:
            out += stream.feed(caller[k:k + step], speaking, agent_audio=bot[k:k + step])
        else:
            out += stream.feed(caller[k:k + step], speaking)
    if a.json:
        print(json.dumps(out))
    else:
        for d in out:
            print(f"{d['t_ms'] / 1000:9.2f} s  {d['decision']}")
        print(f"{len(out)} decisions over {len(caller) / sr:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
