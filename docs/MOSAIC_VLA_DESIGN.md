# MOSAIC-VLA: ablation-identified recovery-program induction

Status: credit/certification component of RESOLVE-VLA, 2026-09-09. MOSAIC
abbreviates **Minimal Ordered Sufficient Ablation-Identified Continuations**.
It is not the complete method and is not a standalone novelty claim.

## Target: minimal sufficient adaptive programs

Let `B` be the exact frozen baseline and let

```text
sigma = (psi_1, ..., psi_d)
```

be an ordered recovery program. Each `psi_i` is a decision rule fixed before
the counterfactual experiment. It maps the causal observation at an absolute
simulator-step anchor to a subgoal transform, a continuous frozen-GR00T action
noise code, and a prefix. Thus the program is closed-loop even though the rule
and its random tape are fixed before treatment.

Let `sigma_-i` replace `psi_i` by B and retain all other rules in their original
order. For physical milestone `m`, define paired effects

```text
Delta_B,m(sigma) = Y_m(sigma) - Y_m(B)
Delta_i,m(sigma) = Y_m(sigma) - Y_m(sigma_-i).
```

The conservative certificate is

```text
C_m(sigma) = min(
    LCB[Delta_B,m(sigma)],
    min_i LCB[Delta_i,m(sigma)],
).
```

`sigma` is **deletion-minimal sufficient** for milestone `m` only when
`C_m(sigma)>0`: it beats B, and deleting any one rule destroys a statistically
clear part of the gain. This needs only `d+2` arms—B, the full program, and `d`
one-deletion programs—rather than an exponential ablation cube.

Deletion-minimal does not imply globally shortest. Prefix minimality is reported
separately; global minimality is not claimed. For depth two, the classical
factorial interaction contrast is an additional diagnostic, not the definition
or contribution.

Milestones remain separate binary physical outcomes: final success, ordered
subtask completion, and goal predicates. The generator is conditioned on one
target milestone at a time. No tunable scalar mixes a step/call preference into
task success.

## Proposed learning rule: deletion-necessity odds update

The contribution candidate is a counterfactual policy-improvement operator for
an adaptive program generator, not the certificate by itself. At iteration `k`,
recurrent policy `pi_k` samples a variable-length program online. Its per-anchor
random tape is sampled up front, so every retained `psi_i` denotes the same
stochastic decision rule in the full and deleted worlds even though their
observations can differ.

For one paired group, define the **deletion-necessity advantage**

```text
A^DN_i,m(sigma)
  = min{Delta_B,m(sigma), Delta_i,m(sigma)}
  = Y_m(sigma) - max{Y_m(B), Y_m(sigma_-i)}.
```

The second equality is exact because both contrasts share `Y_m(sigma)`. Thus
`A^DN_i,m>0` if and only if the program beats B and rule `i` is necessary for
that gain. It is zero for a redundant rule and negative for a harmful program
or deletion. Unlike assigning the same episodic return to every token, it is an
action-specific paired counterfactual advantage.

Each decision has an explicit exact-B atom. MOSAIC applies an exponentiated
mirror step to the sampled rule's odds against that atom:

```text
log [pi_{k+1}(psi_i | h_i,m) / pi_{k+1}(B | h_i,m)]
  = log [pi_k(psi_i | h_i,m) / pi_k(B | h_i,m)]
    + eta A^DN_i,m(sigma).
```

This is the **deletion-necessity odds update (DNOU)**. A neural recurrent
generator realizes it by a KL projection of the tilted per-anchor distributions
back into `pi_theta`; the evaluated counterfactual advantages are stop-gradient
targets. Calling this a policy gradient would be inaccurate: with shared
recurrent parameters, the token-wise update is a mirror-descent policy-
improvement operator, not an asserted unbiased gradient of one scalar episodic
objective.

Across paired episode groups, the training advantage uses held-out grouped
effect estimates. Once enough groups exist, its conservative version replaces
each term by its group lower bound. The hard certificate `C_m>0` is used only to
admit a program to the reported positive set. It is not used as a sparse on/off
training reward during bootstrap.

The generator emits a STOP token as well as continuous steering codes. A
redundant intervention receives no positive odds update, whereas a shorter
certified program does. A small registered maximum depth bounds compute; no
step/call reward is mixed into physical success.

DNOU must be compared with ordinary REINFORCE, history-complete BranchQ, and
successful-trajectory behavior cloning. The project will not call it a novel
optimizer or claim global policy improvement unless its fixed-point and shared-
parameter projection conditions survive a separate theoretical audit.

## Discovery without enumerating programs

Initial programs come from a registered low-discrepancy design in a
low-dimensional noise sphere. Later programs are sampled from `pi_theta`, not
from a finite option table. Each proposal costs only `d+2` paired rollouts.

An active allocation rule maintains a confidence interval for `Delta_B` and
every deletion effect. Additional simulator groups are assigned to programs
whose minimum bound is closest to zero and whose optimistic bound is positive.
Programs whose optimistic baseline or deletion bound is below zero are retired.
This spends replication on unresolved certificates rather than uniformly
enumerating action sequences.

The generated action is new: a continuous code moves the exact B flow-matching
noise along an equal-norm sphere and the frozen denoiser maps it to a physical
chunk. The radius is a safety trust region, not a list of sampled hypotheses.
If the diagnostic finds no useful program inside that reachable family, the
evidence rejects frozen-noise steering and requires learning an action adapter.

## Causal execution contract

- Anchors are absolute physical simulator steps and are fixed before any arm
  runs. STOP changes controller state but cannot move an anchor.
- The complete simulator/environment RNG state, dynamic-disturbance random
  state, schedule cursor, and toggle are restored. Random events attach to
  absolute simulator time; reusing only a numeric seed is invalid.
- B, full, and deletion arms have the same initial snapshot, global remaining
  budget, and absolute outcome horizon.
- `psi_i` is a state-conditioned rule with a preregistered random tape. It may
  react to its arm's observation, but its parameters and anchor cannot be chosen
  after observing an earlier intervention.
- Confidence intervals group by source episode/seed. The many arms from one
  snapshot are paired observations, not independent samples.

At deployment the recurrent generator observes the real state at each fixed
anchor and emits the next rule. It does not restore state, inspect simulator
predicates, or branch online. Exact B is the zero intervention and is used when
the generated program lacks a calibrated baseline-improvement bound.

## Falsification sequence

### M0: mechanism diagnostic

Run the depth-two quartet pilot specified in
`RECOVERY_CURVATURE_DESIGN.md`, with no learned controller. Proceed only if at
least three independent episode/seed groups across more than one case show a
useful interaction and at least one advances a subtask.

### M1: learning-rule pilot

On training seeds only, compare equal simulator-arm budgets for:

1. deletion-necessity odds update;
2. ordinary episodic REINFORCE;
3. behavior cloning of successful generated programs;
4. scalar-return BranchQ; and
5. random and cross-entropy-method noise search.

Primary criterion is certified new-program yield on held-out episode groups:
the fraction of generated, previously unevaluated programs with positive
`C_m`. Secondary criteria are subtask reach, harmful-program rate, simulator
arms per certificate, and action-noise radius. Prediction loss alone cannot pass
the gate.

### M2: benchmark evidence

Only one generator surviving M1 runs full development seed `23007`. Seed
`21007` is independent confirmation, not a new search. The final seed `7`
remains untouched until a single method has a positive paired task-macro lower
bound, real final-success evidence or a preregistered strong subtask effect, and
compliant recovery safety and runtime overhead.

## Claim boundary

Factorial causal effects, dynamic treatment regimes, relative action critics,
noise selection, latent VLA steering, and counterfactual policy-gradient
baselines all have prior art. The working novelty hypothesis is narrower:

> Learn an adaptive VLA recovery-program generator with a linear-cost paired
> deletion design, where every action receives a counterfactual necessity
> update and every reported recovery program is certified to outperform B and
> each one-rule deletion under continuous/no-restore execution.

This remains a hypothesis until the theoretical audit finds no equivalent
estimator and the matched-arm program-generation experiment is positive.
