# Invalid residual-v6 collection

Residual-v6 was interrupted after 18 completed episode feature files and 490
counterfactual branches on training seed `10007`.  It has no manifest and is
inadmissible for training.

Review found that the first residual action was correct, but its closed-loop
continuation used B's immediate confirmed-STOP advance.  Runtime residual mode
instead falls back to B-retry, including a fresh one-retry allowance after
`ADVANCE` or `BACKTRACK_ONE` enters a different subtask.  Consequently those
v6 branch returns did not estimate the deployed policy and cannot be compared
as residual advantages.

Corrected v7 applies B-retry's per-subtask attempt state throughout every
equal-budget branch.  The incomplete 21 MiB v6 directory remains ignored in
local storage as debugging provenance; no v6 labels or models advance.
