# CMR-V1 — Final Report: X2 Action Relevance and CMR-D Closeout

**Protocol:** CMR-V1 `1.0.0-FROZEN`
**Status:** `FROZEN_BEFORE_CMR_OUTCOME_ACCESS`
**Baseline HEAD at closeout:** `135f19d6c5146081740a990f82a18ce3fc3ecb83`
**Final machine decision:** `ACTION_FAMILY_NOT_DECISION_RELEVANT`
**Final protocol decision:** `CMR-D`
**Protocol stop:** `true` — X3, X4 and X5 were never run.

---

## 1. Executive decision

The frozen CMR-V1 X2 gate evaluated the representation-layer action family on eight
prespecified structural conditions (4 datasets × 2 encoders) with a unified CUDA
extraction backend. The result is a **prespecified hard-stop failure**:

| Gate | Criterion | Result |
| --- | --- | --- |
| X2-G1 | ≥ 6/8 conditions ACTION_RELEVANT (layer spread ≥ 0.01) | **PASS** — 8/8 |
| X2-G2 | ≥ 2 distinct ResNet-18 UNIQUE_BEST layers | **FAIL** — 1 distinct (`layer4`) |
| X2-G3 | ≥ 2 distinct ViT-B/16 UNIQUE_BEST blocks | **PASS** — 3 distinct (B6, B9, B12) |
| X2-G4 | all aggregation recovery checks ≤ 1e-8 | **PASS** — max 1.30e-11 |
| X2-G5 | all values finite | **PASS** |

Because X2 is the only hard scientific continuation gate, this frozen G2 failure
triggers the prespecified stop:

```
CMR_X2_DECISION = ACTION_FAMILY_NOT_DECISION_RELEVANT
FINAL_CMR_DECISION = CMR-D
PROTOCOL_STOP = true
```

The correct scientific reading is stated precisely in section 9: the action was
**empirically relevant**, but its **prespecified identity breadth criterion failed**.

---

## 2. Protocol authority

The closeout changes no protocol field. Authority verified by hash at closeout:

| Authority file | SHA-256 |
| --- | --- |
| `docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md` | `58132ae03f3ab70bb82443a30d7c313a39d068eeb2fb2533d68b5a4b2ab1025a` |
| `configs/cross_mechanism_replication_protocol_v1.yaml` | `8004ae02b690163ce41494fbf6f5464c6aee1712709d05428727fa9d658a1e4b` |
| `docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv` | `91e4834f5f671062c6f45ed87b6ef72235aeada0eef187eb8df33f66c0940001` |
| `docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json` | `77ab2bad46aa8d25630ccc1cb38d8aba44bc3c45fd2b772cdd37f4e39afb1545` |

Frozen and unchanged throughout: datasets, dataset splits and versions, encoders,
pretrained weight enums, preprocessing, candidate layers, feature-extraction
semantics, RFF dimension (`m = 256`), RFF sigma (`1.0`), RFF seed derivation,
client partition (20 clients, Dirichlet α = 0.10, seed 20260908), primary metric
(balanced accuracy), all X2 thresholds (0.01 / 0.002), and the final CMR-A/B/C/D
decision matrix. The manuscript was not modified.

---

## 3. X1 exact-collision result

X1 remains a valid exact witness and is unaffected by this closeout.

- Decision: `X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION`
- Message equality: `max_abs = 0.0` (tolerance 1e-12); class counts and class vector sums symbolically equal
- Prototype equality: `max_abs = 0.0`
- Optimal-set conflict: `A = [layer1]`, `B = [layer2]`, intersection empty
- Deterministic pair minimax regret: `0.6666666666666667` (expected 2/3 ± 1e-12)
- Randomized regret: `0.33333333333333337` (expected 1/3 ± 1e-12)
- In-process rerun identical: `true`; X1-G7 unit tests: 16 passed

Evidence: `evidence/cmr_v1/x1/gate_summary.json`, `evidence/cmr_v1/x1/evidence_manifest.json`.

---

## 4. CUDA migration and numerical parity

The X2 formal extraction was migrated from a CPU-only backend to an RTX 4060 CUDA
backend as an **implementation-backend migration**, with audited parity evidence.

**Migration gate — all eight gates PASS (`CUDA_MIGRATION_STATUS = PASS`):**

| Gate | Meaning | Result |
| --- | --- | --- |
| M1 | CUDA environment valid | PASS — RTX 4060, torch 2.7.0+cu128, CUDA 12.8, cuDNN 90701, driver 616.56, TF32 disabled |
| M2 | weights identity valid | PASS — state hashes identical across backends and equal to the Phase-1 authority |
| M3 | preprocessing identity valid | PASS — identical transform config and hash |
| M4 | layer hook identity valid | PASS — ResNet post-stage, ViT post-block CLS **before** final encoder LayerNorm |
| M5 | RFF arrays identical | PASS — byte-identical on regeneration |
| M6 | CPU↔GPU feature parity | PASS — worst normalized cosine median 0.999999999976, min 0.999999999925 |
| M7 | prediction agreement | PASS — 100% (32/32 layers), BACC difference 0.0 |
| M8 | no scientific protocol field changed | PASS |

Additional engineering facts:

- CPU↔GPU raw layer feature difference: `max_abs ≤ 2.41e-05`, `mean_abs ≤ 2.31e-06` (raw features are not required to be bitwise identical)
- Determinism: two complete CUDA extraction reruns for one ResNet-18 and one ViT-B/16 condition produced identical sample order, RFF features, predictions and BACC; the two formal evaluations produced a byte-identical `condition_results.csv`
- Throughput (engineering only, never used to select a batch size by model risk): ResNet-18 ~351 img/s (7.4× CPU), ViT-B/16 ~151 img/s (19.6× CPU)
- One implementation defect was found and fixed before any X2 outcome existed: the RFF input width had been built from a hard-coded 512/768 instead of the frozen per-layer width (64/128/256/512 and 768). The first attempt aborted with a shape error and **no X2 BACC was ever computed**; the fix derives widths from the frozen action family and a `verify_rff_dimensions()` guard now fails closed. Recorded in `runs/cmr_v1/x2_cuda_migration/engineering_correction_rff_width.json`.
- The CPU pre-migration cache is retained solely as `CPU_PREMIGRATION_REFERENCE_ONLY` engineering evidence and is explicitly excluded from the formal feature bank; no CPU/GPU mixed feature bank was used.

---

## 5. X2 real-representation results

Formal run: `runs/cmr_v1/x2/run_summary.json`, `runs/cmr_v1/x2/condition_results.csv`
Backend: CUDA (RTX 4060). Primary metric: balanced accuracy on the frozen official test set.

| Dataset | Encoder | act 1 | act 2 | act 3 | act 4 | Spread Δ | Best | 2nd best | Margin | UNIQUE_BEST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | ResNet-18 | 0.058500 | 0.149300 | 0.236500 | 0.390800 | 0.332300 | layer4 | layer3 | 0.154300 | layer4 |
| CIFAR-100 | ViT-B/16 | 0.080400 | 0.115400 | 0.365100 | 0.534500 | 0.454100 | B12 | B9 | 0.169400 | B12 |
| EuroSAT | ResNet-18 | 0.276533 | 0.620150 | 0.732917 | 0.760733 | 0.484200 | layer4 | layer3 | 0.027817 | layer4 |
| EuroSAT | ViT-B/16 | 0.420750 | 0.615483 | 0.754567 | 0.633417 | 0.333817 | B9 | B12 | 0.121150 | B9 |
| PathMNIST | ResNet-18 | 0.365843 | 0.547057 | 0.582139 | 0.600814 | 0.234970 | layer4 | layer3 | 0.018675 | layer4 |
| PathMNIST | ViT-B/16 | 0.329508 | 0.597793 | 0.413105 | 0.494788 | 0.268286 | B6 | B12 | 0.103006 | B6 |
| DermaMNIST | ResNet-18 | 0.263693 | 0.301980 | 0.363780 | 0.382187 | 0.118495 | layer4 | layer3 | 0.018407 | layer4 |
| DermaMNIST | ViT-B/16 | 0.278901 | 0.413371 | 0.457144 | 0.425478 | 0.178242 | B9 | B12 | 0.031666 | B9 |

Core numbers:

- **ACTION_RELEVANT = 8/8**; layer spread range **0.118495 – 0.484200**, every condition far above the 0.01 threshold
- ResNet-18 UNIQUE_BEST: CIFAR-100 `layer4`, EuroSAT `layer4`, PathMNIST `layer4`, DermaMNIST `layer4` → **distinct = 1** (required ≥ 2)
- ViT-B/16 UNIQUE_BEST: CIFAR-100 `B12`, EuroSAT `B9`, PathMNIST `B6`, DermaMNIST `B9` → **distinct = 3**
- Aggregation recovery `max_abs = 1.296e-11` (tolerance 1e-8), computed over train and test partitions of all eight conditions
- All values finite

---

## 6. X2 gate-by-gate evaluation

Frozen thresholds were applied exactly as specified; none was adjusted.

**X2-G1 — action-relevant conditions — PASS.**
`action_relevant_count = 8`, required 6. Spread threshold 0.01. The action
(representation layer identity) changes downstream balanced accuracy in every
prespecified structural condition.

**X2-G2 — ResNet-18 UNIQUE_BEST layer breadth — FAIL.**
`distinct_unique_best = ['layer4']`, required 2, margin threshold 0.002. All four
ResNet-18 datasets produced the same unique best action. No arbitrary tie-breaking
was used to manufacture layer breadth; UNIQUE_BEST was assigned only where the best
layer exceeded the second-best by at least 0.002 (observed ResNet margins:
0.018407, 0.018675, 0.027817, 0.154300).

**X2-G3 — ViT-B/16 UNIQUE_BEST block breadth — PASS.**
`distinct_unique_best = ['B12', 'B6', 'B9']`, required 2. ViT block identity is not
dataset-invariant: a shallow block (B6) wins on PathMNIST, mid-depth blocks (B9) on
EuroSAT and DermaMNIST, and the deepest block (B12) on CIFAR-100.

**X2-G4 — aggregation recovery — PASS.**
Worst `max_abs = 1.2960299500264227e-11` over all eight conditions, two partitions
and four layers, i.e. more than three orders of magnitude inside the 1e-8 tolerance.
The one-shot client message reconstructs the centralized class sums.

**X2-G5 — finiteness — PASS.**
No non-finite feature, sum, prototype, score or metric anywhere in the run.

All gates pass ⇒ X2 PASS. Otherwise ⇒ `ACTION_FAMILY_NOT_DECISION_RELEVANT`.
Here G2 failed, so the frozen failure branch applies.

---

## 7. Why G2 failed

G2 does not test whether the action matters. It tests whether the **identity** of the
best action varies across datasets for a fixed encoder — a cross-dataset
action-identity breadth criterion.

For ResNet-18 the winner was `layer4` in all four datasets, with margins from
0.0184 to 0.1543. The mechanism behind this uniformity is visible in the accuracy
ordering, not just the winner: BACC increased monotonically with depth in every
ResNet condition (`layer1 < layer2 < layer3 < layer4`). Deeper ResNet-18 stages
therefore dominated uniformly across CIFAR-100, EuroSAT, PathMNIST and DermaMNIST
in these frozen kernel-feature prototype classifiers.

Consequently the ResNet arm contributed exactly one distinct UNIQUE_BEST layer where
the protocol required at least two. This is a breadth failure of the frozen criterion,
not a failure of the action to be decision-relevant.

---

## 8. Formal CMR-D decision

Applying the frozen decision matrix mechanically:

- `CMR-A` requires X1, X2, X3, X4, X5 all PASS → not satisfied (X2 FAIL)
- `CMR-B` requires X2 PASS → not satisfied
- `CMR-C` requires X2 PASS → not satisfied
- `CMR-D` is required when `X2_FAIL` → **satisfied**

```
CMR_X2_DECISION    = ACTION_FAMILY_NOT_DECISION_RELEVANT
FINAL_CMR_DECISION = CMR-D
PROTOCOL_STOP      = true
X3_STATUS          = NOT_RUN_PROTOCOL_STOP
X4_STATUS          = NOT_RUN_PROTOCOL_STOP
X5_STATUS          = NOT_RUN_PROTOCOL_STOP
```

X3, X4 and X5 were not run and are not authorized after the X2 failure. The GPU
result was not replaced by a CPU formal result, and no dataset, encoder, layer,
threshold or backend was reselected on the basis of any observed outcome.

---

## 9. Scientific interpretation

Recommended statement of the result:

> Representation depth was strongly decision-relevant in all eight prespecified
> conditions, but the second mechanism failed its frozen cross-dataset
> action-identity breadth gate because all four ResNet-18 datasets selected
> `layer4` as the unique best action. This frozen G2 failure triggered the
> prespecified CMR-D stop.

What this means, stated without over-reading:

- The action was **empirically relevant**, but its prespecified **identity breadth
  criterion failed**. It is incorrect to say "cross-mechanism replication failed
  because layer selection was irrelevant" — the opposite is recorded in X2-G1.
- The machine label `ACTION_FAMILY_NOT_DECISION_RELEVANT` is the frozen protocol
  string for the G2 failure branch. It must not be paraphrased as "the
  representation layer was not decision relevant", because 8/8 conditions were
  ACTION_RELEVANT with spreads of 0.118–0.484.
- CMR-D means precisely one thing: **CMR-V1 did not establish the prespecified
  second-mechanism empirical breadth**. Nothing wider follows from it.

---

## 10. What CMR-D does NOT invalidate

- **X1 is not disproved.** X1 remains a valid exact witness: an analytically exact
  kernel-prototype collision in which two distinct representation layers are
  simultaneously optimal in opposite directions. X1's decision stands as
  `X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION`.
- **The original Gaussian-mechanism results of the manuscript are unaffected.**
  CMR-D is scoped to the CMR-V1 replication design; it does not withdraw the
  manuscript's exact result, controlled evidence or natural evidence.
- **The X2 layer-relevance measurements remain valid findings.** That the four
  candidate layers differ substantially in downstream balanced accuracy — and that
  ViT block identity varies across datasets — is a real, reproducible measurement
  under the frozen protocol and CUDA backend.
- **The CUDA migration is a validated engineering asset.** Parity, determinism and
  provenance all pass; the migrated backend may serve future, separately frozen
  studies.
- **No negative claim about layer selection in general is licensed.** CMR-D is a
  statement about one prespecified breadth criterion inside one frozen protocol.

---

## 11. Prohibited post-hoc rescue actions

The following are prohibited within CMR-V1 and were not performed:

- changing G2 or any gate threshold
- relaxing the 0.002 UNIQUE_BEST margin
- dropping ResNet-18, or reporting the ViT arm alone
- replacing the ResNet layer set
- replacing the action with bandwidth selection
- replacing the action with ridge regularization or any other action family
- changing the dataset set, or adding a dataset to satisfy G2
- redefining UNIQUE_BEST
- continuing to X3, X4 or X5 after the X2 failure
- switching the formal backend on the basis of observed results, or mixing CPU and
  CUDA feature banks as formal evidence

Any future attempt to design a new mechanism must be a **new prospectively frozen
study**. It may not be written back as a CMR-V1 rescue or presented as a corrective
re-analysis of this result.

---

## 12. Artifact and provenance inventory

**Protocol and X1**

- `docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md`
- `configs/cross_mechanism_replication_protocol_v1.yaml`
- `docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv`
- `docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json`
- `evidence/cmr_v1/x1/gate_summary.json`
- `evidence/cmr_v1/x1/evidence_manifest.json`

**X2 implementation and tests**

- `src/frontier/cmr_kernel_layer.py` — frozen RFF, message, prototype, X2 gates
- `src/frontier/cmr_feature_extract.py` — frozen layer hooks, L2 normalization, Phase-1 loaders and Dirichlet client partition
- `tools/run_cmr_v1.py` — X1/X2 formal runner with `--device {cpu,cuda}` and per-backend cache roots
- `tests/test_cmr_v1_x1.py`, `tests/test_cmr_v1_protocol.py`, `tests/test_cmr_v1_x2.py` — 61 tests passing
- `tools/cmr_x2_cuda_migration.py` — CPU→CUDA migration harness (env, identity, parity, benchmark, determinism, gate)
- `tools/audit_cmr_v1_x2_cmr_d_consistency.py` — read-only closeout consistency audit
- `tools/build_cmr_v1_final_closeout.py` — this closeout's gate summary builder

**X2 formal evidence (CUDA backend)**

- `configs/cmr_v1_feature_map_manifest.json` — 8 unique Gaussian-RBF maps, m = 256, σ = 1.0, master seed 20261001, float64, per-layer input widths 64/128/256/512/768
- `configs/cmr_v1_feature_bank_manifest.json` — 8 banks, `inference_backend = CUDA`, CPU pre-migration cache excluded
- `runs/cmr_v1/x2/condition_results.csv`
- `runs/cmr_v1/x2/run_summary.json`
- `runs/cmr_v1/x2/failed/INCOMPLETE_RUN.json` — engineering-failure accounting
- `evidence/cmr_v1/x2/gate_summary.json`
- `evidence/cmr_v1/x2/evidence_manifest.json`
- `evidence/cmr_v1/x2/protocol_snapshot.yaml`
- `evidence/cmr_v1/final_gate_summary.json`

**CUDA migration evidence**

- `runs/cmr_v1/x2_cuda_migration/environment_cpu.json`, `environment_cuda.json`
- `runs/cmr_v1/x2_cuda_migration/identity_cpu.json`, `identity_cuda.json`, `weight_identity.json`
- `runs/cmr_v1/x2_cuda_migration/parity_sample_manifest.json`
- `runs/cmr_v1/x2_cuda_migration/cpu_reference/`, `gpu_reference/`, `cpu_reference_summary.json`, `gpu_reference_summary.json`
- `runs/cmr_v1/x2_cuda_migration/parity_report.json`
- `runs/cmr_v1/x2_cuda_migration/throughput_benchmark.json`
- `runs/cmr_v1/x2_cuda_migration/determinism_report.json`
- `runs/cmr_v1/x2_cuda_migration/migration_gate_summary.json`
- `runs/cmr_v1/x2_cuda_migration/engineering_correction_rff_width.json`
- `runs/cmr_v1/x2_cuda_migration/x2_cmr_d_consistency_audit.json`

**Data planes (repository-excluded scratch)**

- `data/cmr_v1/x2/raw/`, `data/cmr_v1/x2/features/` — CPU reference, marked `CPU_PREMIGRATION_REFERENCE_ONLY`
- `data/cmr_v1/x2_cuda/features/` — formal CUDA feature cache
- `data/cmr_v1/x2/rff/` — frozen RFF arrays

### 12.1 Disclosed implementation provenance gap

The frozen artifacts are internally consistent and every reported number was
re-derived read-only from them. One source-level provenance gap is disclosed here
rather than silently repaired:

| Source path | Frozen artifact records | Committed content | Status |
| --- | --- | --- | --- |
| `src/frontier/cmr_kernel_layer.py` | `bfd9cd4184b4945c…` | `bfd9cd4184b4945c…` | matches |
| `tests/test_cmr_v1_x1.py` | `ebf9f921dda42857…` | `ebf9f921dda42857…` | matches |
| `tests/test_cmr_v1_protocol.py` | `21afacda58881365…` | refreshed for the CMR-D closeout stage | stage-updated, recorded |
| `tests/test_cmr_v1_x2.py` | `f825d2d0067612ec…` | refreshed for the CMR-D closeout stage | stage-updated, recorded |
| `tools/run_cmr_v1.py` | `d659eb274b3d089f…` | `bd54354a996055…` | **gap disclosed** |
| `src/frontier/cmr_feature_extract.py` | `ade3cc4582297839…` (437 lines) | `403a7ef13756e9ee…` (23 lines) | **gap disclosed** |

Facts about the gap:

- The two flagged paths were overwritten after the frozen formal run (timestamps
  `2026-10-02 17:19:58` and `2026-10-02 20:47:40`; the frozen run window was
  `2026-10-02 17:17–17:20`). The overwrite happened outside this closeout.
- `evidence/cmr_v1/x2/source_recovery/` (timestamped `20:47–20:50`) is a
  pre-existing recovery attempt, not a closeout artifact. It records
  `original_source_recovered: false` and only a behaviour-level shim from preserved
  Python 3.11 bytecode; there is no source-level recovery.
- The committed file path exists, but its content is not the source recorded by the
  frozen artifacts. **Exact re-derivation of the frozen feature caches from the
  current tree's feature-extraction source is therefore not guaranteed.**
- What this does *not* affect: the read-only consistency audit of the frozen
  artifacts passes; both backends reproduce the identical frozen gate outcome; CPU
  and CUDA agreed exactly on all eight conditions; the frozen RFF arrays, feature
  caches and formal manifests verify against their recorded hashes.
- No rescue was performed: no re-run, no source substitution, no artifact
  rewriting. The gap is recorded here and in
  `evidence/cmr_v1/final_gate_summary.json` under
  `source_provenance_gap_disclosure`.

The two test files were refreshed only to reflect the stage transition (the
CMR-D closeout legitimately produces `evidence/cmr_v1/final_gate_summary.json`,
which the earlier X1/X2 stage guards forbade). The X3, X4 and X5 prohibitions in
those guards are unchanged and remain unconditional. No scientific threshold,
layer, seed, dataset or gate was touched.

---

## 13. Final freeze declaration

CMR-V1 is closed at the X2 hard scientific continuation gate.

```
X1                        PASS / FROZEN
X2                        FAIL (X2-G2) / FROZEN
CMR_X2_DECISION           ACTION_FAMILY_NOT_DECISION_RELEVANT
FINAL_CMR_DECISION        CMR-D
PROTOCOL_STOP             true
X3 / X4 / X5              NOT_RUN_PROTOCOL_STOP
CUDA_MIGRATION_STATUS     PASS
READ_ONLY_CONSISTENCY      PASS
```

The X2 outcomes, gate results and CMR-D decision recorded here are frozen. The
frozen protocol, gate thresholds, dataset set, encoder set, layer set and final
decision matrix are unchanged from the pre-outcome freeze. Any further work on
this question must begin as a new prospectively frozen study with its own protocol
freeze; it may not be presented as a revision, extension or rescue of CMR-V1.
