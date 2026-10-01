# Codex implementation prompt — CMR-V1 X1 only

Repository: `N:\one_shot_frontier_public_code`
Branch: `main`

You are implementing only **CMR-V1 X1 — Exact kernel-feature prototype layer-selection collision**.

## Authorities

Read and obey:

1. `docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md`
2. `configs/cross_mechanism_replication_protocol_v1.yaml`
3. `docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv`

The scientific protocol is frozen. Do not change thresholds, formulas, feature patterns, expected regrets, layer semantics, seeds, paths or the final CMR decision matrix.

## Scope

Implement:

- `src/frontier/cmr_kernel_layer.py`
- `tools/run_cmr_v1.py` with `--mode x1`
- `tests/test_cmr_v1_x1.py`
- protocol/schema validation tests if needed
- formal X1 outputs under `runs/cmr_v1/x1`
- frozen X1 evidence under `evidence/cmr_v1/x1`

Do **not** run X2, X3, X4 or X5.
Do **not** extract real dataset features.
Do **not** create WILDS or external pair manifests.
Do **not** modify the manuscript.

## X1 analytic authority

Use:

\[
a=1/5,\quad q=\sqrt{24/25}.
\]

Good pattern, each class N=6:

- class 0: `(-a,+q),(-a,-q)` repeated three times;
- class 1: `(+a,+q),(+a,-q)` repeated three times.

Bad pattern:

- class 0: two `(+a,+q)`, two `(+a,-q)`, two `(-1,0)`;
- class 1: two `(-a,+q)`, two `(-a,-q)`, two `(+1,0)`.

Task A:

- layer1 = Good
- layer2 = Bad

Task B:

- layer1 = Bad
- layer2 = Good

Message per layer/class = `(count, vector_sum)`.
Prototype = `vector_sum / count`.
Classifier score = inner product with class prototype.

Expected:

- exact same message for A and B;
- exact same reconstructed prototype family;
- BACC(A, layer1)=1;
- BACC(A, layer2)=1/3;
- BACC(B, layer1)=1/3;
- BACC(B, layer2)=1;
- optimal sets disjoint;
- deterministic minimax regret = 2/3;
- randomized minimax regret = 1/3.

Numeric tolerance = `1e-12`.

## Required tests

At minimum test:

1. class counts;
2. class vector sums;
3. prototype equality;
4. BACC values;
5. optimal sets;
6. deterministic minimax regret;
7. randomized minimax regret;
8. deterministic rerun;
9. protocol version/status unchanged.

Use float64.

## Formal output

Write a machine-readable X1 summary with:

- protocol ID/version;
- git HEAD;
- exact pattern specification;
- message equality max abs;
- prototype equality max abs;
- four BACC values;
- optimal sets;
- deterministic regret;
- randomized regret;
- each X1 gate;
- final X1 decision.

Expected decision:

`X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION`

If any expected invariant fails, return non-zero and write:

`X1_IMPLEMENTATION_BLOCKED`

Do not weaken the tolerance.

## Validation before commit

Run:

```bat
python -m pytest tests\test_cmr_v1_x1.py -q
python tools\run_cmr_v1.py --mode x1
git diff --check
git status --short
```

Then show:

- test result;
- formal X1 JSON;
- changed-file list;
- exact proposed commit scope.

Do not commit unrelated untracked files.
