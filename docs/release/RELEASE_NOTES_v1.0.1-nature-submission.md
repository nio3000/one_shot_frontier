# One-shot frontier — Nature submission archival release v1.0.1

**Metadata-only corrective release.** This release supersedes `v1.0-nature-submission`
for archival citation. **The scientific content is unchanged.**

## What changed relative to `v1.0-nature-submission`

Archival metadata only:

- `CITATION.cff`: the unsupported `doi: PENDING` placeholder was removed, since no DOI has
  been registered; the `version` field now reads `v1.0.1-nature-submission`; the citation
  message no longer implies an existing DOI, and the notes now state that none is recorded.
- `.zenodo.json`: inaccurate wording that referred to a DOI "below" was replaced with an
  explicit statement that no DOI is included; the non-schema `prereserve_doi` key was
  removed; a `publication_date` was added; `version` now reads `v1.0.1-nature-submission`.
- `README.md`: the release-lineage table now distinguishes `v1.0`, `v1.0-nature-submission`
  and `v1.0.1-nature-submission`.

No source file, protocol, manifest, gate outcome, figure, table or numerical result differs
between the two releases.

## Release identity

| Field | Value |
| --- | --- |
| Tag | `v1.0.1-nature-submission` (annotated) |
| Role | metadata-corrected archival release; supersedes `v1.0-nature-submission` for citation |
| Superseded release | `v1.0-nature-submission` at `2fde82c9fb1bb6567b0bdaf1d63080c11273796d` (retained, not deleted, not moved) |
| Unrelated earlier tag | `v1.0` at `aa04d80b4f9cb52f36c75a11a940595083f11cbe` |
| CMR-V1 scientific closeout authority | `6239ec45e1175fdd268ce34b2e5e17fc053861d2` |
| Licence | MIT |
| Persistent DOI | **not claimed.** No DOI has been registered for this release. Cite a DOI only after one exists. |

## Contents

Scientific software and frozen reproducibility artifacts for a study of the separation
between model reconstructability, exact decision identifiability and local decision
stability in one-shot communicated summaries:

- the frozen protocol chain for Phases 0–5, with per-phase protocol snapshots;
- feature-bank, pair and feature-map manifests, gate summaries and environment manifests;
- the machine-readable artifacts behind the manuscript figures and supplementary analyses;
- a prospectively frozen cross-mechanism falsification study (CMR-V1), including its frozen
  protocol, gate matrix, freeze manifest, closeout report and a read-only source-recovery audit;
- automated tests, the licence compatibility audit, and archival citation metadata.

## Not redistributed

Raw third-party datasets, downloaded archives, local feature banks and pretrained model
weights are excluded. Dataset versions, official train/test semantics and split hashes are
recorded in the frozen manifests; obtain each dataset from its original provider under that
provider's terms. Pretrained ResNet-18 and ViT-B/16 weights are downloaded by torchvision at
run time and are recorded by state hash only.

## Frozen results and negative findings

Every prespecified gate outcome is reported, including the failures. The controlled
intervention breadth gate returned 5 of 16 conditions against a required six; the
natural-domain all-dataset breadth gate failed with no qualifying RxRx1 condition; and the
third-moment corroboration gate failed (pooled Spearman −0.0753, blocked-permutation
p = 0.6633). Negative results are retained rather than removed, and a post-hoc power
diagnostic bounds what the third-moment null can and cannot support.

## CMR-V1 status

CMR-V1 remains a **supplementary negative replication**, not a result used to broaden the
manuscript's claim. Its exact witness passed, all eight real conditions were action-relevant,
and it then stopped at its frozen cross-dataset action-identity breadth gate; the terminal
decision was `CMR-D` with `protocol_stop = true`, and stages X3–X5 were never run.

Its provenance status is stated in four separate parts, and they must not be merged:

- Artefact internal consistency: **PASS**
- CPU–CUDA numerical parity: **PASS** (100% prediction agreement, 0.0 balanced-accuracy difference)
- Decision-level determinism: **PASS**
- Source-level reproducibility: **INCOMPLETE / DISCLOSED** — the two source files that produced
  the frozen extraction run are not present at their recorded hashes.

**No part of this release claims complete source-level reproducibility for CMR-V1.** That
limitation is confined to CMR-V1; the Phase 0–5 evidence chain carries its own protocol
snapshots and artifact hashes and is unaffected. Two pre-registration development states
(`abe7de0ab1747fa0b0206448dafcc1e8b3739a67` and `9fb6289e685bc715be775861bb6eace6d402e493`)
are recorded inside the frozen artifacts but their Git objects are not present in this
export; the freeze control rests on the artifact hashes and on the pair manifest committed
before unblinding.

## Documentation

- `README.md` — repository layout, environment, reproduction entry points, release lineage, data and weight provenance
- `docs/release/LICENSE_COMPATIBILITY_AUDIT.md` — licence compatibility audit
- `docs/phases/cmr_v1/CMR_V1_FINAL_REPORT.md` — CMR-V1 closeout report
- `CITATION.cff`, `.zenodo.json` — archival citation metadata (no DOI recorded)
