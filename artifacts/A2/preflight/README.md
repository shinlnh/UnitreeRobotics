# A2 preflight

This reviewer-safe preflight records the frozen hierarchy audit before any A2
benchmark outcome is inspected.

- Branch: `feat(A2)/implement-GR00T-RC-fixed-hierarchy`
- Variant: `GR00T-RC-fixed-hierarchy`
- Low-level checkpoint source: frozen A1 `GR00T-RC`
- Planner: `RoboCerebra-HPE-fixed-anchor-reimplementation`
- Plan source: canonical `task_description.txt` step annotations
- Switch rule: exactly 150 executed control steps per subgoal
- Coverage: 60 benchmark cases, 563 subgoals
- Outcome-aware stop, re-planning, retry, state restoration, recovery: disabled

An end-to-end H16 pilot on `Ideal/case1`, seed 7 completed all 900 scheduled
control steps. The 60 policy calls visited all six subgoals, produced 60 frame
bundles, and truncated exactly five chunks at the five internal 150-step anchors.
The pilot is a plumbing/trace check only and was not used to tune the schedule.

The 31 unique plan hashes reflect benchmark reuse across paired conditions such
as Ideal, Observation Mismatching, and Random Disturbance. This is expected and
does not indicate missing cases.
