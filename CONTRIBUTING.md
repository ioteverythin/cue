# Contributing to Cue

Thanks for helping. Cue decides, from a caller's audio, when a voice agent should stop talking
and when it should answer; this repository is its runtime (`cue-turn`): loading the models,
streaming audio through them, the decision rule, and integrations such as Pipecat. The model
weights live on Hugging Face and are not part of this repository.

## Ways to help

- **Report a wrong decision.** A STOP on a backchannel, a missed interruption, an answer in the
  middle of a thought. Say which model and profile, the sample rate, and what Cue did versus what
  it should have done. Please do not attach recordings of real people unless they agreed to it
  being shared publicly; a description, or a recording of yourself, is enough.
- **Integrations.** Pipecat is supported; LiveKit Agents, Vapi-style webhooks, plain WebRTC or
  Twilio Media Streams examples are welcome.
- **Languages and accents.** Tell us where Cue gets backchannels or hesitations wrong in your
  language (for example "haan", "achha", "ji" in Hindi and Hinglish). Short lists of real
  backchannel words and hold phrases ("ek second", "ruko") help directly.
- **Speed.** CPU latency of Cue Tiny, ONNX export of the larger models, batching.
- **Docs and examples.** Anything that took you more than five minutes to figure out.

Issues labelled `good first issue` are small and self-contained.

## Setting up

```bash
git clone https://github.com/ioteverythin/cue.git
cd cue
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # numpy + onnxruntime: Cue Tiny, policy, audio
pip install -e ".[dev,full]"                            # + torch, transformers: Cue v4 / v5
pip install -e ".[dev,pipecat]"                         # + Pipecat integration (Python 3.11+)
```

## Running the tests

```bash
pytest
```

Tests that need a model, torch or Pipecat skip themselves when those are missing, so a plain
`.[dev]` install runs the policy, audio and speech-control tests in a few seconds. To run the
model tests against local copies instead of downloading from Hugging Face:

```bash
CUE_TINY_DIR=/path/to/cue-tiny CUE_V4_DIR=/path/to/cue-v4 CUE_V5_DIR=/path/to/cue-v5 pytest
```

## Making a change

1. Open an issue first for anything larger than a bug fix, so we can agree on the approach.
2. Branch from `main`, keep the change focused, and add or update a test.
3. Match the surrounding code: plain Python, short functions, comments that say *why*.
   The decision rule (`policy.py`) is what published scores were computed with; a change to its
   behaviour needs numbers before and after on the same audio.
4. Update `README.md` and `CHANGELOG.md` when users would notice the change.
5. Open a pull request and fill in the template. CI runs the tests on Python 3.10 to 3.13.

## Licence

The code is Apache-2.0 (see `LICENSE` and `NOTICE`). By submitting a contribution you agree it is
licensed under the same terms, as described in section 5 of the licence. The models on Hugging
Face carry their own licences; check the model card before using a model commercially.

## Conduct

Be kind and assume good intent. See `CODE_OF_CONDUCT.md`.
