# Changelog

All notable changes to `cue-turn`. Versions follow [semantic versioning](https://semver.org/).

## Unreleased

- `CueSpeechController(duck_ms=...)`: a duck. The bot goes quiet the moment the caller starts over it;
  Cue's STOP (or the caller still talking after `duck_ms`) makes it their turn, the caller falling
  quiet lets the bot carry on. Off by default.
- `CueSpeechController` (Pipecat): releases the bot's audio in real time so Cue can end it with a
  short fade on STOP, and pause it on PAUSE ("wait", "one second") and resume where it stopped
  when the caller says "okay, go on" (or after a silence); other replies drop it for a new turn.
- `cue_turn.speech_control`: `reply_kind`, `fade`, `Playout`, usable without Pipecat.
- Cue v4 / v5 run without `transformers`: a plain-PyTorch Whisper encoder is used when it is not
  installed (identical outputs).
- Contributor guide, code of conduct, security policy, issue and pull request templates, CI.

## 0.2.0

- Cue v5: the agent's own audio as a second input (`agent_audio`, `agent_sample_rate`).
- Decision profiles: `responsive`, `balanced` (default), `cautious`.
- CLI: `--profile`; v5 hears `--bot-channel`.

## 0.1.0

- First release: Cue Tiny (numpy + onnxruntime), Cue v4 and v3 (torch), the decision rule,
  streaming resampler, Pipecat turn strategies, CLI.
