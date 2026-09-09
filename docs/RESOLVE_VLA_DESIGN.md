# RESOLVE-VLA: counterfactual reachability RL for learned recovery policies

Status: active algorithm hypothesis, 2026-09-09. RESOLVE abbreviates **REcovery
through Sufficiency-Ordered Latent Value Estimation**. MOSAIC is retained as a
credit/certification layer inside RESOLVE; it is no longer the complete method.

## Novelty exclusions fixed before experimentation

The paper must not claim a separate recovery policy, residual RL on a VLA, or
recovery-trajectory learning: those mechanisms are already represented by
[RecoveryChaining](https://arxiv.org/abs/2410.13979),
[PLD](https://arxiv.org/abs/2511.00091), and
[RePO-VLA](https://arxiv.org/abs/2605.09410). Probability of necessity and
sufficiency is classical, and 2026 CPTE already connects preference-based
counterfactual effects—including conditional PNS—to policy learning
([Parnas et al.](https://proceedings.mlr.press/v300/parnas26a.html)). Therefore
neither the `min` bottleneck nor paired-potential-outcome policy learning alone
is a novelty claim. The live algorithmic hypothesis is now narrower: propagate
the effect of the *least necessary state-contingent future recovery macro* by a
finite-horizon deletion Bellman recursion. Its novelty remains provisional
until a broader theory audit and positive held-out experiments.

The multi-world Bellman construction is not claimed by itself either. Joint
MDPs already formalize shared-exogenous-randomness transitions and Bellman
operators for joint return quantities ([Kaya et al., UAI
2026](https://proceedings.mlr.press/v337/kaya26b.html)). RESOLVE must therefore
show a recovery-specific learning or sampling result beyond merely placing the
R/D/B worlds in a product state space.

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
state-conditioned residual distribution over the entire action chunk. The
residual is applied in a bounded action-logit chart, rather than raw Euclidean
action space:

```text
u_B = atanh(normalize_to_unit_box(clip(a^B)))
u_R = u_B + s_theta(h_t,z_t) delta_theta(h_t,z_t,a^B_t)
a^R = denormalize_from_unit_box(tanh(u_R)).
```

The numerical implementation subtracts and restores the exact clipped B point,
so a zero shift is bitwise B even at an action boundary while retaining the
straight-through chart derivative. This fixes an R0 expressivity failure: only
`4.6%` of 20,025 audited expert H16 chunks were reachable by the old raw
`+/-0.25` residual box. The learned scale now changes distance in the logit
chart; every executable action remains in bounds and the complete action box is
reachable. This parameterization is an implementation prerequisite, not a
novelty claim.

The first implementation freezes the 3B VLA and trains an independent 8-layer,
width-512 recovery transformer plus twin critics. If mechanism and learning-rule
gates pass, the registered scale-up is 12 layers at width 768; model size is not
swept on benchmark seeds.

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

### All-position deletion-necessity recursion

The local CRB above is insufficient as the final learning object. It checks the
current structural slot but does not tell an earlier state whether a later macro
is dispensable. Enumerating every future deletion at every update costs
quadratically many continuations and still treats the deletion time as an
open-loop index.

For fixed `pi_R`, define the paired delete-now effect

```text
G_i^pi(x)
  = Q_i^R,pi(x, c_i) - Q_i^D,pi(x, c_i).
```

Both terms use the same long horizon and adaptive downstream `pi_R`; only the
current physical macro differs. Now introduce an auditor with one deletion
token. Before using it, the auditor observes the current recovery state and may
replace the current macro by B or defer the token. It chooses the deletion with
the *smallest* causal cost to recovery. Its finite-horizon necessity value is

```text
N_H^pi(x) = G_H^pi(x)
N_i^pi(x) = min { G_i^pi(x),
                  E_{x' ~ P_R^pi(.|x)}[N_{i+1}^pi(x')] }.
```

The deletion must occur by the final registered slot. The active recovery
advantage is therefore

```text
A_i^all-del(x) = min { V_i^R,pi(x) - V_i^B(x), N_i^pi(x) }.
```

This is not a renamed third critic. `G` evaluates a coupled physical
counterfactual after deleting now; `N` is the lower Bellman envelope over all
state-contingent future uses of the deletion token. A positive lower confidence
bound on `N` implies that even the least damaging admissible one-macro deletion
reduces reachability. It therefore implies every fixed-position deletion test,
while allowing the weakest position to depend on observations encountered
during recovery.

For a fixed policy and finite absolute horizon, the recursion has a unique
solution by backward induction. Its operator is monotone and non-expansive in
the sup norm because stochastic expectation and pointwise `min` are both
non-expansive; no discounted contraction is claimed. After `H` synchronous
backups, every dependency reaches the forced-deletion boundary, so the result
is independent of initialization. The repository test includes a stochastic
two-state example where the auditor deletes immediately on one branch and
waits on the other; the recurrence obtains a strictly tighter necessity value
than any one open-loop deletion time.

This certificate is stronger than deletion-minimality of one sampled open-loop
sequence, but it does not establish globally shortest recovery or rule out a
needlessly long sequence whose every step has been made necessary. The physical
step-budget constraint remains essential. Joint/product MDP machinery and
robust pointwise minima exist in prior work; any eventual novelty claim must be
about this recovery-specific one-deletion recursion, its sample-efficient
estimation, and empirical advantage—not those ingredients separately.

With critic ensembles, policy improvement uses the all-deletion envelope

```text
lower_A^all-del_m
  = min{LCB[V^R_m - V^B_m], LCB[N_m]}.

log pi_{k+1}(c | h,z,m) / pi_{k+1}(B | h,z,m)
  = log pi_k(c | h,z,m) / pi_k(B | h,z,m)
    + eta lower_A^all-del_m.
```

The explicit B atom makes this an exponentiated counterfactual policy-
improvement step, implemented by KL projection into the recovery transformer.
The paper must not call it an unbiased policy gradient without proving the
shared-parameter projection result.

### R0 causal-execution correction

The first snapshot-based counterfactual artifacts are invalid and superseded.
Teleporting MuJoCo `qpos/qvel` did not restore the OSC controller, evaluator,
observable caches, warm-start forces, or the solver's numerical path. Even
after expanding the snapshot, two separately replayed copies of the exact same
policy (`D_last` and `B_last`) could diverge after many identical actions. Such
divergence is simulator path noise, not a treatment effect.

The corrected collector therefore resets one environment and exactly replays
the logged source-action prefix before every *distinct* arm. It also applies a
canonical-estimand rule:

```text
B_0 is executed once per physical anchor and reused for every program.
B_last := D_last because their complete policies are mathematically identical.
```

This rule is more than an optimization: it prevents repeated evaluation of an
identical estimand from manufacturing causal labels. Truly distinct R, D, and B
arms remain paired by initial state, global budget, episode seed, and
request-local diffusion seed. Final effects still require repeated physical
seeds and grouped confidence intervals.

The R1a learner is deliberately an anchor-time Monte Carlo program pilot, not
yet the full closed-loop TD learner. Its actor and potential-outcome critics may
see the current anchor context, the complete proposed latent code sequence, and
the first VLA/residual chunk available at that decision. They may not see later
contexts or chunks produced after the intervention. R1a tests whether paired
R/D/B supervision improves held-out program selection over an ordinary
return-only actor-critic with identical programs, initialization, architecture,
optimizer, and update count. Passing R1a does not by itself validate the
Bellman contribution; the later closed-loop learner must train on per-anchor
transitions and exercise `T^RDB` directly.

### R0 result and resulting data correction

The fixed-program mechanism did not pass the ordered-progress gate. In the
canonical depth-2 corpus, 24 physical programs produced one predicate hit that
was deletion-redundant and zero ordered-subtask or final-success hits. Three
long-horizon, budget-capped subgoal-backtrack trials then produced `0/3`
ordered-subtask rescues and `0/3` final rescues. One trial reached two milk
predicates while the first ordered bowl subtask remained incomplete; that is
evidence that unordered "any predicate" is a misleading recovery target, not a
success.

Consequently fixed latent-noise programs and fixed subgoal rewind are frozen as
negative ablations. R1 begins from controlled failures sampled near successful
expert continuations and uses `next_ordered_subtask` as the first optimization
head. The 203,410 existing frozen-GR00T/subgoal feature states are used to
warm-start the action class on expert continuations (`183,430` train and
`19,980` development) before paired sparse RL. This warm-start is supervised
initialization only; it cannot validate the counterfactual operator.

### R1 frontier source and warm-start result

The executable frontier collector restores the exact demonstration XML and
state, then compares the logged expert continuation with frozen B under the
same physical horizon and policy-call budget. Anchor ordering is a registered
SHA-256 permutation rather than manifest order. The complete train-only source
contains 599 selected anchors: 597 valid paired arms and two exclusions whose
goal predicate became true during executable reset post-processing. The valid
arms divide into 75 expert-only rescues, 101 B-only outcomes, 74 both-success
outcomes, and 347 unresolved both-failure outcomes. The 75 rescues span 70
source-demonstration groups and all three scenes. Raw corpus hash, arm counts,
and exclusion reasons are recorded in the R1 artifact report.

The registered 4-layer, width-256 supervised warm-start did **not** pass its
predefined development gate even after the positive source grew from 15 to 75
pairs. The best admissible checkpoint remained exact B at step zero. At the
final step, the learned proposal intervened on 50% of positive development
pairs but also on 28.125% of B-safe pairs, while its positive action MAE was
1.231 times B's MAE. The gate required at least 50% positive intervention, at
most 10% false intervention, and an MAE ratio below one. This result rejects
frontier behavior cloning as the recovery learner; it is not a reason to tune
the deployment threshold. The actor checkpoint is retained only as a negative
initialization artifact. The next method stage must learn from paired physical
R/D/B transitions and the all-position deletion target.

Collection throughput is an implementation concern, not an algorithmic claim.
Eight simulator workers were empirically the useful operating point: a
16-anchor timing probe took 20 seconds and reached a sampled 94% GPU
utilization, versus 31 seconds with four workers. The reusable runner uses two
micro-shards per worker and a dynamic worker pool so variable episode lengths
do not leave the GPU idle behind one long static shard.

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
   time RNG restore pass deterministic tests. Fixed flow-noise and subgoal-
   backtrack programs are retained as failed ablations.
2. **R1 actor/critic warm start:** initialize the reachable action class on
   successful expert continuations, then fit reachability critics on controlled
   failures plus paired train-seed branches. No actor benchmark claim.
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
