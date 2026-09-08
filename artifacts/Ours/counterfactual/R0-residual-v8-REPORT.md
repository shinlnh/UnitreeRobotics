# R0 residual-v8 counterfactual collection report

Status: complete, valid, and retained as a train-only ablation.

The collection uses exact abstaining B-retry behavior sources at train seeds
`10007`, `11007`, and `12007`. Every selected state is the first confirmed STOP
after at least 75 causal simulator steps. The five valid branch policies share
common random numbers, a 75-step / 24-policy-call continuation budget, and the
same B-retry continuation contract. No development or final seed was used.

| Seed | States | Branches | Strict preferences | Physical-benefit states | Return-benefit states | Efficiency-only states | Replay mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10007 | 62 | 303 | 27 | 3 | 24 | 21 | 1.2632% |
| 11007 | 72 | 353 | 36 | 2 | 29 | 27 | 2.6596% |
| 12007 | 55 | 273 | 35 | 3 | 27 | 24 | 4.8687% |
| **Total** | **189** | **929** | **98** | **8** | **80** | **72** | — |

All three corpus audits pass. Only 8/189 states (4.2328%) contain an override
whose final-success, completed-subtask, or success-predicate tuple is strictly
better than `RETRY_CURRENT`. Of the 80 states where the shaped return favors an
override, 72 are explained only by policy-call or executed-step cost. This is a
training-target failure rather than evidence of physical recovery, so v8 must
not consume the frozen development seeds. It remains an ablation for testing
whether efficiency-shaped counterfactual supervision learns the wrong control
objective.

The prospective successor is outcome-first residual supervision: remove
call/step costs from option targets, keep those quantities as evaluation-only
overhead metrics, and use a longer continuation horizon so delayed physical
recovery is observable. This amendment is made before seed `22007` is touched.

## Integrity

| Seed | Manifest SHA-256 | Branch log SHA-256 | Audit SHA-256 |
| ---: | --- | --- | --- |
| 10007 | `63ffee23f540e26ada2ce5b39a6e5ce708aeb23c2ec063f900424343b4c3e0e9` | `ad465e4bbb5b6053322dff7c0f6aa78fd538781a247d40413bb43e18ad9c7859` | `5d5b412b853000ce8b3fb0a697537e116476ec92433c402d76620cb0a2505bc9` |
| 11007 | `9c1688472dc920a909aa7fc6d94c584a75e024814c945e420c2d507128cbe964` | `fef40c28d0055642bcc6d6edf5595e03586ca5185026f47f10308164ddcfd209` | `fbb43a0a64b48764eaf44ad4ed285b8f62a9f53f8b5ca4273c7623e472961603` |
| 12007 | `9b78f4837fb540abdbcbc1e63e477675241634fd2bc19ee403b0a5ba307f889b` | `00d34d961486b4006f2222825f6892127f09db04d91e97643511dca604531362` | `f404019a28749f0cf43716945ca0e5abba1a2dc837afd44b9c8f249510c9861b` |
