# one_shot_frontier

Code and frozen reproducibility artifacts for the study:

**Model reconstruction does not guarantee decision identifiability in one-shot federated learning**

## Repository contents

- `src/` — core implementation
- `tools/` — data preparation, auditing and experiment utilities
- `tests/` — automated tests
- `configs/` — frozen experimental protocols and manifests
- `runs/` — compact canonical outputs for reproducibility
- `evidence/` — frozen machine-readable evidence used for manuscript figures and supplementary analyses

Raw third-party datasets, downloaded archives, local feature banks, model weights,
development reports and staging artifacts are not redistributed.

## Data

Benchmark datasets should be obtained from their original providers under their
respective licenses and terms.

## Reproducibility

Experimental thresholds, support rules, pair selection, unblinding and statistical
criteria are defined in the frozen configuration and manifest files included here.

## Reproducible research release

**Scientific scope.** This repository accompanies a manuscript on the separation between
model reconstructability, exact decision identifiability and local decision stability in
one-shot communicated summaries. It carries the frozen protocol chain, the feature-bank,
pair and feature-map manifests, every gate summary, and the machine-readable artifacts
behind the figures and supplementary analyses. It also carries a prospectively frozen
cross-mechanism falsification study (CMR-V1), reported as a supplementary negative
replication.

**Release identity.**

| Field | Value |
| --- | --- |
| **Archival release (cite this one)** | `v1.0.1-nature-submission` |
| Release commit | the `Finalize Zenodo archival metadata` commit on `main`; resolve it with `git rev-list -n 1 v1.0.1-nature-submission` |
| Initial archival candidate | `v1.0-nature-submission` at `2fde82c9fb1bb6567b0bdaf1d63080c11273796d` |
| Historical project tag | `v1.0` at `aa04d80b4f9cb52f36c75a11a940595083f11cbe` (2026-09-19) — **not** a manuscript release |
| CMR-V1 scientific closeout authority | `6239ec45e1175fdd268ce34b2e5e17fc053861d2` |
| Persistent DOI | none yet. A deposit is prepared but no DOI has been registered. Do not cite a DOI for this release until one appears here. |

**Release lineage.** Three tags exist and they are not interchangeable:

| Tag | Role | Scientific content |
| --- | --- | --- |
| `v1.0-nature-submission` | initial archival candidate | frozen |
| `v1.0.1-nature-submission` | metadata-corrected archival release; **supersedes `v1.0-nature-submission` for citation** | **unchanged** |
| `v1.0` | pre-existing project tag, unrelated to the manuscript | different, much earlier state |

`v1.0.1-nature-submission` corrects archival *metadata* only — citation fields, release
description and the release lineage statement. No source file, protocol, manifest, gate
outcome, figure or numerical result differs between `v1.0-nature-submission` and
`v1.0.1-nature-submission`. The earlier tags are retained and are not deleted or moved.

**Environment.** `pyproject.toml` declares the base Python requirement and pinned scientific
dependencies; `requirements.txt`, `requirements-phase1.txt` and `requirements-phase5.txt`
record the phase-specific sets. The CMR-V1 formal run used a CUDA build whose versions,
GPU, driver and determinism flags are recorded in
`runs/cmr_v1/x2_cuda_migration/environment_cuda.json`.

**Main reproduction entry points.**

| Purpose | Entry point |
| --- | --- |
| Phase 1–5 pipeline | `tools/run_phase1.py` … `tools/run_phase5.py` with `configs/phase*_protocol.yaml` |
| Phase-1 feature bank | `tools/prepare_phase1_feature_bank.py` |
| CMR-V1 X1 / X2 formal stages | `tools/run_cmr_v1.py --mode x1`, `--mode feature-map`, `--mode extract`, `--mode x2` |
| CMR-V1 CPU→CUDA migration and parity | `tools/cmr_x2_cuda_migration.py` |
| Read-only consistency and recovery audits | `tools/audit_cmr_v1_x2_cmr_d_consistency.py`, `evidence/cmr_v1/x2/source_recovery/` |
| Automated tests | `python -m pytest tests/` |

**Data are not redistributed.** Raw third-party datasets, downloaded archives, local
feature banks and pretrained model weights are excluded. Dataset versions, official
train/test semantics and split hashes are recorded in `configs/phase1_feature_manifest.json`
and `configs/phase5_tier1_feature_manifest.json`; acquisition notes, including the RxRx1
reconstruction and the iWildCam selective-extraction route, are in the frozen Phase-5
protocol and evidence chain. Obtain each dataset from its original provider under that
provider's terms.

**Pretrained weights.** `ResNet18_Weights.IMAGENET1K_V1` and `ViT_B_16_Weights.IMAGENET1K_V1`
are downloaded by torchvision at run time and are not stored here. Their identities are
recorded by state hash (`265e8731…` and `ec73907d…`) in the Phase-1, Phase-5 and CMR-V1
manifests.

**Known provenance limitations.** Two pre-registration development states are recorded in
the frozen artifacts (`abe7de0ab1747fa0b0206448dafcc1e8b3739a67`, the pair-freezing state,
and `9fb6289e685bc715be775861bb6eace6d402e493`, the formal Tier-1 evaluation state) but
their Git objects are not present in this published export; the freeze control is therefore
carried by the artifact hashes and by the pair manifest that was committed before
unblinding. Full hashes and the verification procedure are in
`docs/phases/cmr_v1/CMR_V1_FINAL_REPORT.md` and the reconstructed Supplementary Section F.

**CMR-V1 status.** CMR-V1 is a **supplementary negative replication**. Its exact witness
passed, all eight real conditions were action-relevant, and the study then stopped at its
frozen cross-dataset action-identity breadth gate; the terminal decision was `CMR-D` with
`protocol_stop = true`, and stages X3–X5 were never run. Its provenance status is
deliberately split: artefact internal consistency **PASS**, CPU–CUDA numerical parity
**PASS**, decision-level determinism **PASS**, and **source-level reproducibility
INCOMPLETE / DISCLOSED**, because the two source files that produced the frozen extraction
run are not present at their recorded hashes. **Nothing in this repository should be read
as claiming complete source-level reproducibility for CMR-V1.** That limitation is confined
to CMR-V1 and does not apply to the Phase 0–5 evidence chain, which carries its own protocol
snapshots and artifact hashes.

## License

MIT. Copyright (c) 2026 Ning Li, Fei Li, Wenyan Hao, Yejin Jin. See `LICENSE`.
Third-party dependencies are not redistributed and remain under their own licences; see
`docs/release/LICENSE_COMPATIBILITY_AUDIT.md`.
