# Residual-v8 model sweep

Status: complete train-only diagnostic; rejected as an R1b source.

All six preregistered low-capacity recovery heads trained successfully on the
three residual-v8 corpora. The nominal offline winner is
`v03-mlp-h8-w64-d025` (1,285,844-byte weight file, selected at step 750) with a
calibrated option margin of 2.4296875. Its held-out partition has 45 labeled
states, of which only 23 have a strict shaped-return preference: 20 favor an
override and 3 favor `RETRY_CURRENT`. At a zero measured false-override rate on
those three negative states, beneficial shaped-return recovery is 30%.

This ranking is not promoted to R1b. The underlying v8 corpus audit shows that
72/80 apparent override gains come only from step/call penalties, so the model
metrics measure efficiency-target prediction rather than physical recovery.
The checkpoints and registries are retained as the frozen 75-step/24-call,
efficiency-shaped ablation. No rollout development or final seed was consumed.

## Integrity

- Completion registry SHA-256: `bb4d4b690a734d18b288c4cba770a125979b0993ea8c8c2368f29afefb4f7275`
- Residual registry SHA-256: `bdd2a8a0b8d4e605dbb15ef9f66adccdbe98e1e2ca3c5701f4f0f7b08f941401`
