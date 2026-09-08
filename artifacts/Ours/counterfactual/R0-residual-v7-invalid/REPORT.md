# Invalid residual-v7 collection

Residual-v7 was interrupted after 14 completed episode feature files and 342
counterfactual branches on training seed `10007`.  It has no manifest and is
inadmissible for training.

V7 correctly continued every option with B-retry, fixing v6.  A second runtime
comparison then found that its `CONSENSUS_PREFIX` branch queried four fresh
policy samples.  Deployable consensus includes the already-observed source STOP
proposal as hypothesis one and queries only three fresh samples for C4.  This
changed both the selected medoid and measured policy-call cost.

V8 preserves B-retry continuation and seeds consensus with the source proposal.
The incomplete v7 directory remains ignored in local storage as debugging
provenance; no v7 labels or models advance.
