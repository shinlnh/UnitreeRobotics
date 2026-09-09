# Cross-temporal recovery-curvature diagnostic

Status: diagnostic for the MOSAIC-VLA hypothesis, 2026-09-09. The factorial
contrast in this note is classical causal machinery and is not claimed as the
paper's new algorithm.

## Observation that motivates a different objective

CTR v9 corrected the source behavior, budget, proposal, and physical labels, but
its best complete R1b variant changed only one of 537 subtasks and achieved 0/60
final success. That controller asks whether one intervention is better than B.
A history-complete Q-function trained on adequate multi-step transitions can
represent this dependence in principle. The narrower problem is that one-shot
return ranking does not supervise complementary intervention effects directly.

The new hypothesis is narrower and testable:

> Useful recovery is often an interaction between interventions at different
> times. Neither intervention is beneficial alone; the ordered pair is. A mixed
> causal contrast may expose this complementarity directly enough to learn it.

For example, reopening the gripper can look harmful under a B continuation and
moving back can also look useless under B, while `open -> move back -> regrasp`
is the only sequence that restores controllability. A one-shot label assigns no
credit to either first step. A scalar return model can represent the sequence in
principle, but it does not identify or exploit the paired experiment that makes
this rare interaction statistically visible.

## Intervention programs

Let `B` be the exact frozen baseline continuation. An intervention `u` is not a
fixed recovery option. It is a small state-conditioned program

```text
u = (subgoal transform, action-generation steering code, prefix length)
```

applied to the observation present at its anchor. The zero program means exact
`B`, including B's sampled initial flow noise and STOP/prefix decision.

The steering code acts on the frozen GR00T flow-matching action generator. If
`epsilon_B` is B's initial action noise and `U` is a registered low-dimensional
orthogonal basis, first project `Ua` onto the tangent space of `epsilon_B`. The
code is decoded with a spherical exponential map

```text
e = epsilon_B / ||epsilon_B||
d = tangent_e(Ua) / ||tangent_e(Ua)||
epsilon(a) = ||epsilon_B|| [cos(||a||) e + sin(||a||) d],  ||a|| <= r.
```

Zero is exactly B. Nonzero codes preserve B's sampled noise norm, so the pilot
changes a direction on the typical-noise sphere instead of introducing an
out-of-distribution additive norm. All VLA weights remain frozen. Unlike
choosing among a fixed list, continuous codes can synthesize action chunks not
present in CTR. The first pilot uses a fixed signed basis; the basis is learnable
only after physical evidence shows that cross-temporal interaction exists.

## Classical diagnostic contrast

Let `Y_m(u -> v; xi)` be a binary physical milestone outcome under first-anchor
program `u`, second-anchor program `v`, and exogenous simulator randomness
`xi`. `m` ranges over ordered goal predicates, completed subtasks, and final
task success. From one snapshot and one `xi`, run the factorial quartet

```text
B -> B,  u -> B,  B -> v,  u -> v.
```

The **cross-temporal recovery curvature** is the mixed discrete derivative

```text
kappa_m(s; u, v, xi)
  = Y_m(u -> v; xi) - Y_m(u -> B; xi)
    - Y_m(B -> v; xi) + Y_m(B -> B; xi).
```

It removes the baseline outcome and both unary intervention effects. What
remains is the non-additive contribution of ordering `u` before `v`. This is a
standard 2-by-2 factorial interaction contrast, including for sequential
treatment rules. Its formula and the concept of interaction are not novel.

Curvature alone can be positive in edge cases where the baseline already
succeeds. Therefore the training record also contains the stricter
**irreducible rescue gain**

```text
g_m(s; u, v, xi)
  = Y_m(u -> v; xi)
    - max{Y_m(u -> B; xi), Y_m(B -> v; xi), Y_m(B -> B; xi)}.
```

For binary milestones, `g_m = +1` exactly when the joint ordered intervention
reaches a milestone that no proper one-intervention ablation reaches. A useful
synergy must simultaneously beat B, beat both unary arms, and have positive
curvature; the grouped confidence interval of every contrast must separate from
zero.

For an ordered intervention set `I` of size `d`, the higher-order extension is
the sequential Möbius difference

```text
kappa_m(I) = sum_{J subseteq I} (-1)^(|I|-|J|) Y_m(J),
```

where omitted positions execute B and subsequences preserve temporal order.
Depth two is the initial scientific claim. Depth three is allowed only if the
depth-two pilot is positive; otherwise a higher-order search would be an
unfalsifiable compute sink.

## Diagnostic model and steering feasibility

The model predicts milestone-wise unary effects and cross-time curvature, not
absolute Q-values. A low-rank bilinear parameterization makes the interaction
explicit:

```text
logit P(Y_m(u -> v)=1 | h)
  = b_m(h) + p_m(h)^T u + q_m(h)^T v + u^T M_m(h) v.
```

`h` is causal frozen-VLA history. The bilinear term is the learned curvature
field. With rank-`r` factors for `M`, its leading singular directions propose a
pair of steering codes rather than ranking a finite option list. Training uses
the complete quartet jointly:

- milestone likelihood for all four arms;
- a difference-in-differences loss matching `kappa_m`;
- an irreducible-gain ranking loss that separates `u -> v` from every proper
  ablation when `g_m=1`;
- zero-curvature regularization on quartets whose physical interaction is zero.

No step, VLA-call, option-type, or intervention-count reward is introduced.
The milestone vector is kept vector-valued. Selection is lexicographic: final
success, furthest completed subtask, then predicate reachability. This prevents
a tunable scalar reward from trading away task completion.

At runtime the model solves a small trust-region bilinear problem over `u` and a
contingent next-anchor code `v`. It executes only `u`, observes the real next
state, re-estimates the curvature field, and solves again. Thus the second code
is a continuation plan, not an open-loop action. Exact B wins ties and is used
when the paired lower confidence bound does not improve the physical milestone
vector. That guard is an operational constraint, not the claimed innovation.

## What the diagnostic distinguishes

| BranchQ-style formulation | Curvature diagnostic |
| --- | --- |
| learns absolute scalar return | learns a paired mixed causal derivative |
| credits a branch through Bellman backup | credits only the interaction left after both unary ablations |
| ranks a finite candidate lattice | optimizes continuous action-generation codes |
| multi-step behavior is implicit in Q | two-step irreducibility is a directly observed label |
| reward weights collapse outcomes | ordered physical milestones remain vector-valued |

BranchQ remains a required baseline if the curvature pilot succeeds. The
diagnostic is not the active contribution either; it tests whether the target
needed by MOSAIC-VLA exists in this environment.

## BQ-free pilot: falsify the new mechanism first

The first experiment uses only registered training seeds and does not train a
neural controller.

1. Select 12 snapshots from failed or stalled B trajectories, stratified over
   task type and seed.
2. Pre-register every `u` and `v` as a state-conditioned decision rule before
   any arm runs. Use four signed steering directions and a fixed
   eight-physical-step second-anchor spacing. Subgoal transform is initially
   zero so action curvature is not confounded with language changes.
3. Per snapshot, share one `B -> B` rollout and the unary arms, then execute all
   16 ordered pairs: `1 + 4 + 4 + 16 = 25` physical continuations.
4. Give every arm the same outcome horizon and remaining global step budget.
   Snapshot and restore every environment RNG state, disturbance cursor, and
   toggle. Random events are indexed by absolute simulator step; equal seeds
   without equal RNG state are insufficient.
5. Record the full milestone vector, not a shaped return.

The second anchor is an absolute physical-step boundary. It cannot be selected
after observing `u`, and a STOP/subgoal decision cannot move it. This prevents
post-treatment selection from redefining the regime.

The hypothesis proceeds only if:

- at least three independent snapshot/seed groups exhibit `g_m=1` for some
  physical milestone;
- the positive effect repeats outside a single task case;
- an interaction model improves leave-one-episode-out joint-rescue ranking over
  both an additive unary model and a BranchQ/scalar-return baseline; and
- at least one positive pair advances a completed subtask, rather than changing
  only a low-level predicate.

If no irreducible pair exists, the method is rejected before architecture or
threshold work. Then the evidence says the frozen VLA's reachable action family,
not credit assignment, is the limiting factor and the next algorithm must learn
a new action generator. If interaction exists but the fixed steering basis
cannot generalize, the basis—not the curvature target—is redesigned on training
seeds.

## Evidence and seed contract

Seeds `20007` and `22007` are consumed CTR diagnostics. MOSAIC development
must not validate on them. Training and simulator branches may use `10007`,
`11007`, `12007`, then newly registered `13007`--`15007`. `23007` remains the
first rollout-development seed, `21007` the independent confirmation seed,
`30007` implementation smoke only, and final seed `7` remains untouched until a
single method is frozen.

Large factorial branch trees and checkpoints belong on the tracked Hugging Face
dataset/model branches. Git stores code, compact quartet manifests, exact hashes,
pilot statistics, negative results, source commit, candidate basis, noise codes,
global budgets, and hardware/latency accounting.

## Claim boundary

The contrast itself cannot support a novelty claim. It only answers whether
irreducible ordered rescue effects exist and are predictable. The paper
hypothesis must be an algorithm that uses ablation-identified effects to
discover and generate previously unseen adaptive intervention programs.
