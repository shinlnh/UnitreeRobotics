# Ours R0 equal-horizon counterfactual corpus v3

This is a train-seed-only data-generation artifact. It does not consume the
rollout development seeds (`20007`, `21007`) or the held-out final seed (`7`).
The source is the 60-case H16 live rollout with base seed `10007`.

The v3 generator corrects two semantic defects in the retired v2 corpus:

- every option receives the same 75-simulator-step / 24-policy-call
  continuation budget, including confirmed `STOP` transitions;
- `ADVANCE` is valid only when a `STOP` transition is causally pending, which
  matches the online option mask.

## Corpus audit

- Source samples replayed: 25,369
- Labeled states: 322
- Valid option roll-forwards: 1,552
- Strict preferences: 300/322 (93.17%)
- Mean winning margin: 0.19714 return units
- Replay mismatch rate: 0.678%, below the frozen 5% ceiling
- Train/option-validation split: 48/12 episodes, using episode index modulo 5

Strict winners were `ACCEPT_B` 170, `BACKTRACK_ONE` 67, `REOBSERVE` 25,
`RETRY_CURRENT` 23, and `CONSENSUS_PREFIX` 15. Thus 130/300 strict labels
(43.33%) prefer a recovery over accepting B. `ADVANCE` had no strict wins and
was valid at only six naturally pending-stop states; no invalid action was
invented to balance the labels.

The corpus manifest SHA-256 after the deterministic split is
`5f02bcd537a37961d5d53b384d4c021dbe6d30eeb9b0dcdfac19f2805fa2ffdf`.
The branch-log SHA-256 is
`e8aae2f96faa9ec50405cbb687ddd39fd7c68e4aefddb478c25a19c79063ddaa`.

## Decision

The corpus passes the structural counterfactual audit. It is accepted as valid
training evidence, but the single-seed models trained from it are not accepted
for online selection because the held-out episode audit in the paired R0
search report exposes substantial generalization error. Additional frozen
train seeds must be collected before an R1 rollout.
