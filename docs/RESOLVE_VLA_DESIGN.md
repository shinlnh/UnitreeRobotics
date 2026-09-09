# RESOLVE-VLA: counterfactual reachability RL for learned recovery policies

Status: active algorithm hypothesis, 2026-09-09. RESOLVE abbreviates **REcovery
through Sufficiency-Ordered Latent Value Estimation**. MOSAIC is retained as a
credit/certification layer inside RESOLVE; it is no longer the complete method.

## Why the method must be an RL policy

CTR can only choose hand-written recovery options. BranchQ can compose choices
but remains limited to frozen-GR00T candidates. The curvature pilot can expose
ordered complementarity but cannot create a recovery skill. RESOLVE instead
learns a dedicated closed-loop recovery policy:

```text
pi_B(a_t | h_t)
pi_R(c_t, z_{t+1}, handoff_t | h_t, z_t, a^B_t).
```

`pi_B` is the exact frozen GR00T-RC/B policy. `pi_R` is a recovery transformer
conditioned on causal VLA history, the current latent recovery state `z_t`, and
B's proposed chunk `a^B_t`. It emits a new residual action chunk `c_t`, advances
or terminates a learned latent program, and can hand control back to B. The
latent states are not supervised with a fixed taxonomy such as retry/backtrack;
their semantics must emerge from physical recovery rollouts.

The recovery actor is not merely a score head. A transformer predicts a
state-conditioned residual distribution over the entire action chunk:

```text
a^R_t = clip(a^B_t + s_theta(h_t,z_t) tanh(delta_theta(h_t,z_t,a^B_t))).
```

The scale `s_theta` is learned under a constrained trust region rather than
chosen as a fixed list of action options. The first implementation freezes the
3B VLA and trains an independent 8-layer, width-512 recovery transformer plus
twin critics. If mechanism and learning-rule gates pass, the registered scale-up
is 12 layers at width 768; model size is not swept on benchmark seeds.

## New learning object: counterfactual rescue Bellman triplet

Residual RL and separate recovery policies already have prior art. RESOLVE's
candidate contribution is a reachability Bellman operator that asks two
different causal questions for every recovery macro:

1. Is the learned recovery continuation sufficient to beat pure B?
2. Is this macro necessary, or would replacing it with B work just as well?

For one preregistered physical milestone `m`, use absorbing reachability rather
than a shaped scalar return. The Markov state is explicitly augmented as
`x_i=(h_i,z_i,i,t_i,b_i)`: causal observation history, latent program state,
program index, absolute physical anchor, and remaining global budget. Omitting
these variables makes the deletion estimand ill-defined. `r_m` is one exactly
when the macro first reaches `m`. From the same state and exogenous random tape,
execute three worlds:

- `R`: current recovery macro, then `pi_R` until handoff and `pi_B` afterward;
- `D`: replace the physical intervention in the current structural program slot
  by exact B, advance the slot, then resume the same `pi_R` from the observation
  actually produced in this arm;
- `B`: exact B for the full remaining horizon.

For a fixed recovery policy, let `P_R`, `P_D`, and `P_B` be the paired physical
transition kernels of those worlds. The coupled operator is

```text
(T_R V_R)_m(x,c) = r^R_m + (1-r^R_m) E_{P_R}[V_R,m(x'_R)]
(T_D V_R)_m(x,c) = r^D_m + (1-r^D_m) E_{P_D}[V_R,m(x'_D)]
(T_B V_B)_m(x)   = r^B_m + (1-r^B_m) E_{P_B}[V_B,m(x'_B)]

T^RDB(V_R,V_B) = (T_R V_R, T_D V_R, T_B V_B).
```

`D` is deliberately not a free third value function and not a renamed
`Q_R(h,B)`: it is the shared recovery continuation evaluated after a paired
`do(current physical macro := B)` transition at the same absolute slot. In code,
R and D share one recovery-continuation head; only the present residual and
transition sample differ. B has its own self-continuation head.

With `H` remaining anchors, terminal reachability zero, and absolute anchor in
`x`, `T^RDB` has a unique finite-horizon solution by backward induction. More
strongly, after `H` synchronous applications, its output is independent of the
initial value tables because every dependency has reached the terminal layer.
The repository includes a two-state fixed-point test where the first recovery
macro has zero immediate reward but receives positive CRB because it alone moves
the system into a state recoverable at the next anchor. This is a finite-horizon
fixed-point result, not a claimed one-step discounted contraction.

The first comparison contains whole-controller sufficiency; the second is a
one-position deletion while retaining the adaptive downstream policy. Define
the **counterfactual rescue bottleneck (CRB)**

```text
A^CRB_m(h,z,c)
  = min{Q^R_m(h,z,c) - V^B_m(h),
        Q^R_m(h,z,c) - Q^D_m(h,z,c)}.
```

This quantity is positive only when the recovery continuation beats pure B and
the present macro is causally needed. A macro early in `undo -> reposition ->
retry` can receive credit even when it makes no immediate progress because all
three critics evaluate the same long physical horizon. A history-complete scalar
Q can represent this behavior, but it does not directly estimate these two
paired contrasts or impose their conjunctive bottleneck.

With critic ensembles, policy improvement uses

```text
lower_A^CRB_m
  = min{LCB[Q^R_m - V^B_m], LCB[Q^R_m - Q^D_m]}.

log pi_{k+1}(c | h,z,m) / pi_{k+1}(B | h,z,m)
  = log pi_k(c | h,z,m) / pi_k(B | h,z,m)
    + eta lower_A^CRB_m.
```

The explicit B atom makes this an exponentiated counterfactual policy-
improvement step, implemented by KL projection into the recovery transformer.
The paper must not call it an unbiased policy gradient without proving the
shared-parameter projection result.

## Program discovery and certification

During RL, `pi_R` samples adaptive latent programs; it does not enumerate token
sequences. Replay contains paired `R/D/B` transitions, physical milestone
vectors, random-tape hashes, and policy versions. Uncertainty-targeted branching
spends `D` continuations only where the necessity sign is unresolved.

For a final program candidate `sigma=(psi_1,...,psi_d)`, the MOSAIC audit runs B,
the full program, and every one-rule deletion. It reports deletion-minimality:

```text
LCB[Y_m(sigma)-Y_m(B)] > 0
LCB[Y_m(sigma)-Y_m(sigma_-i)] > 0  for every i.
```

This costs `d+2` arms per paired group. It does not prove global shortestness.
The factorial curvature contrast remains a diagnostic only.

## Constrained rather than shaped optimization

Milestones are learned separate reachability heads: final success, ordered subtask
completion, and registered goal predicates. The active head is the furthest
currently unreached milestone; no coefficient trades task success for fewer
steps or policy calls.

Safety is a constrained MDP, not a negative reward mixed into success:

```text
maximize    E[lower_A^CRB_m]
subject to  P(regress an achieved milestone) <= epsilon_regress
            P(invalid or clipped action)      <= epsilon_action
            E[recovery physical steps]        <= B_recovery.
```

Dual variables are learned from constraint violations. The global benchmark
budget remains identical to B. At deployment, recovery activates only when the
recovery lower bound beats B's upper bound and all cost upper bounds are
feasible; uncertainty falls back to exact B.

## Architecture and training schedule

The pilot actor consumes the last eight frozen 2048-D GR00T context tokens, B's
16-by-7 proposed action chunk, normalized proprioceptive scalars, a learned
milestone token, and recurrent latent state. Its 8-layer causal transformer has
width 512, 8 heads, and a 2048 feed-forward width. Heads produce:

- a categorical latent transition including `HANDOFF`;
- mean/log-scale for a 16-by-7 residual chunk;
- a learned element-wise residual trust scale;
- twin reachability critics with one head shared by R/D continuations and a
  separate pure-B head; and
- regression/action/budget cost logits.

Training is staged but the algorithm is not changed between stages:

1. **R0 mechanics:** exact-zero actor equals B; paired R/D/B replay and absolute-
   time RNG restore pass deterministic tests.
2. **R1 critic warm start:** fit reachability critics on existing failed B
   anchors plus new paired train-seed branches. No actor benchmark claim.
3. **R2 off-policy RL:** alternate simulator collection and CRB actor/critic
   updates on seeds `10007`--`15007`; use prioritized replay by critic uncertainty
   and rare milestone transition.
4. **R3 development:** freeze one checkpoint before seed `23007`, confirm once on
   `21007`, and keep final seed `7` untouched until all gates pass.

The 32 CPU threads run simulator shards while the RTX 5070 Ti serves frozen B
and trains the recovery network in alternating phases. Concurrency is selected
from measured transitions/second and memory headroom, not from an arbitrary
worker count.

## Mandatory baselines and claim boundary

A separate recovery policy is not new: RecoveryChaining learns one with
hierarchical RL. Residual VLA RL is not new: PLD trains lightweight residual
specialists from base-policy failure regions. Recovery-trained VLAs are not new:
RePO-VLA learns from correction trajectories. Deletion counterfactual credit is
also not broadly new, including recent long-horizon agent work.

The narrow working novelty hypothesis is:

> Learn a dedicated VLA recovery policy with a paired reachability Bellman
> triplet whose conjunctive counterfactual bottleneck requires both superiority
> to the frozen base policy and long-horizon necessity of each adaptive recovery
> macro, then certify resulting latent programs by physical one-deletion trials.

Mandatory compute-matched ablations are SAC residual RL/PLD-style specialists,
RecoveryChaining-style nominal handoff, RePO-style success/recovery supervised
refinement, standard distributional actor-critic, BranchQ, and MOSAIC sequence-
level credit without Bellman critics. This remains a hypothesis until both the
theory audit and held-out physical experiments are positive.
