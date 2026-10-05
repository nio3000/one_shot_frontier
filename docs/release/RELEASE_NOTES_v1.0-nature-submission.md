# One-shot frontier — Nature submission research release

Scientific software and frozen reproducibility artifacts accompanying a manuscript on the
separation between model reconstructability, exact decision identifiability and local
decision stability in one-shot communicated summaries.

## Release identity

| Field | Value |
| --- | --- |
| Release tag | `v1.0-nature-submission` |
| Commit | `2fde82c9fb1bb6567b0bdaf1d63080c11273796d` |
| CMR-V1 scientific closeout authority | `6239ec45e1175fdd268ce34b2e5e17fc053861d2` |
| Historical tag (not this release) | `v1.0` at `aa04d80b4f9cb52f36c75a11a940595083f11cbe` |
| Licence | MIT |
| Persistent DOI | **none yet** — a Zenodo deposit is prepared; do not cite a DOI for this release until one is registered |

## What is in this release

- The frozen protocol chain for Phases 0–5, with per-phase protocol snapshots, feature-bank
  and pair manifests, gate summaries and environment manifests.
- The machine-readable artifacts behind the manuscript figures and supplementary analyses.
- A prospectively frozen cross-mechanism falsification study (CMR-V1), including its frozen
  protocol, gate matrix, freeze manifest, closeout report and a read-only source-recovery audit.
- Automated tests, the license compatibility audit, and archival citation metadata
  (`CITATION.cff`, `.zenodo.json`).

## What is not redistributed

Raw third-party datasets, downloaded archives, local feature banks and pretrained model
weights are excluded. Dataset versions, official train/test semantics and split hashes are
recorded in the frozen manifests; obtain each dataset from its original provider under that
provider's terms. Pretrained ResNet-18 and ViT-B/16 weights are downloaded by torchvision at
run time and are recorded by state hash only.

## Frozen results and negative findings

The release reports every prespecified gate outcome, including the failures: the controlled
intervention breadth gate returned 5 of 16 conditions against a required six; the natural-domain
all-dataset breadth gate failed with no qualifying RxRx1 condition; and the third-moment
corroboration gate failed (pooled Spearman −0.0753, blocked-permutation p = 0.6633). Negative
results are retained rather than removed, and a post-hoc power diagnostic bounds what the
third-moment null can and cannot support.

## Known provenance limitation (CMR-V1)

CMR-V1 is a **supplementary negative replication**, not a result used to broaden the
manuscript's claim. Its exact witness passed, all eight real conditions were action-relevant,
and it then stopped at its frozen cross-dataset action-identity breadth gate; the terminal
decision was `CMR-D` with `protocol_stop = true`, and stages X3–X5 were never run.

Its provenance status is stated in four separate parts, and they must not be merged:

- Artefact internal consistency: **PASS**
- CPU–CUDA numerical parity: **PASS** (100% prediction agreement, 0.0 balanced-accuracy difference)
- Decision-level determinism: **PASS**
- Source-level reproducibility: **INCOMPLETE / DISCLOSED** — the two source files that produced
  the frozen extraction run are not present at their recorded hashes. Nothing in this release
  claims complete source-level reproducibility for CMR-V1.

This limitation is confined to CMR-V1. The Phase 0–5 evidence chain carries its own protocol
snapshots and artifact hashes and is unaffected. Two pre-registration development states
(`abe7de0ab1747fa0b0206448dafcc1e8b3739a67` and `9fb6289e685bc715be775861bb6eace6d402e493`)
are recorded inside the frozen artifacts but their Git objects are not present in this export;
the freeze control rests on the artifact hashes and on the pair manifest committed before
unblinding.

## Documentation

- `README.md` — repository layout, environment, reproduction entry points, data and weight provenance
- `docs/release/LICENSE_COMPATIBILITY_AUDIT.md` — licence compatibility audit for this repository
- `docs/phases/cmr_v1/CMR_V1_FINAL_REPORT.md` — CMR-V1 closeout report
- `CITATION.cff`, `.zenodo.json` — archival citation metadata (DOI pending)
