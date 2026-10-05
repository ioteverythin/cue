# Security

Please report security problems privately, not in a public issue: use
[GitHub's private vulnerability reporting](https://github.com/ioteverythin/cue/security/advisories/new)
for this repository. Include what is affected, how to reproduce it, and its impact.

We aim to acknowledge a report within a week and to agree a disclosure date with you once a fix
is ready.

Things in scope: code execution or file access through a model download or config
(`cue_turn.load`), denial of service from crafted audio, and anything that leaks call audio.
The models decide when a voice agent speaks; a wrong decision on ordinary audio is a bug, not a
security issue, so please open a normal issue for those.
