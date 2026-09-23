# NATURE-UPGRADE-N3 Executable Reconstruction — Engineering README

## Status

This overlay reconstructs an executable N3 runner from the frozen historical protocol
`NATURE_UPGRADE_N3_PROTOCOL_V1_0_FROZEN.md`.

This is **not** a blinded pre-outcome experiment. The historical N3 outcome is already known.
The purpose of this overlay is executable-authority reconstruction/reproduction in the public
`one_shot_frontier` repository without changing any frozen scientific rule.

## Scientific rules preserved

- N1 authoritative rows SHA256: `5913d81c66754c9e0b56351796ac338eb3a16f1ebc2a016ed7c237179672ea34`
- PTB-XL metadata SHA256: `cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156`
- age bands: `<40`, `40-59`, `60-74`, `75+`
- support: `N>=1024`, `n_positive>=40`, `n_negative>=40`
- development hard decision: mean probability >= `0.3475392659505208`
- Georgia source BACC: `0.921114990521441`
- primary margin: `0.075`
- cluster bootstrap: 20,000/context; master seed `20260922`
- one-ECG-per-patient selector: SHA256(`20260922|patient_id|record_id`), lexicographic minimum
- exact class-support-matched replay: 20,000; seed `20260922`; ddof=0
- alpha: 0.05
- PASS requires all seven frozen N3 gates.

## Engineering authority sequence

1. Install overlay.
2. Run unit tests and smoke.
3. Git-freeze implementation files.
4. Run `binding-freeze` on the two exact authoritative inputs.
5. Verify exact patient-identity fingerprints.
6. Git-freeze the generated binding manifest.
7. Run formal reproduction once into an empty output directory.
8. Audit formal outputs against the historical frozen N3 authority. Numerical historical
   comparisons are audit references, not post-hoc gates and must not be used to alter the protocol.

## Binding-freeze fail-closed checks

Before a binding manifest can be created, the machine protocol, frozen protocol source, module,
runner and tests must all be tracked, committed and clean. Then the runner verifies:

- both input SHA256 hashes and row counts;
- N1 institution, age-band, probability-mean and hard-decision reconstruction;
- PTB-XL metadata row count and unique ECG/patient counts;
- 100% PTB-XL row-to-patient linkage;
- frozen N3-1 identity fingerprint:
  - bound rows 21,544;
  - unique patients 18,628;
  - repeated patients 2,101;
  - records belonging to repeated patients 5,017;
  - max ECGs/patient 10;
  - cross-age patients 36;
- deterministic binding digest;
- implementation SHA256 values.

No scientific bootstrap/null result is generated in `binding-freeze` mode.

## Formal fail-closed checks

Formal execution requires the implementation authority and generated binding manifest to be
committed and clean, verifies their SHA bindings again, then executes only the frozen N3 analyses.
A non-empty formal output directory is refused rather than overwritten.

## Outputs

Formal mode writes:

- `ptb_record_level_wilson_status.csv`
- `ptb_patient_cluster_bootstrap_status.csv`
- `ptb_one_ecg_per_patient_wilson_status.csv`
- `one_ecg_per_patient_support.csv`
- `one_ecg_per_patient_natural_contexts.csv`
- `one_ecg_per_patient_arm_c_null.csv`
- `cross_age_patient_selection_audit.csv`
- `secondary_selection_rule_sensitivity.json`
- `primary_gate.json`
- `binding_audit.json`
- `protocol_snapshot.yaml`
- `binding_manifest_snapshot.json`
- `N3_SUMMARY.json`

## Provenance note

The frozen protocol specifies the N3 master seed `20260922`. The reconstructed implementation
uses NumPy `default_rng(20260922)` deterministically. Unless the original historical executable is
recovered and source-compared, do not describe the reconstructed null draws as byte-identical to
the historical Monte Carlo stream. Deterministic structural quantities and the frozen PASS/FAIL
semantics remain the primary reproduction checks.
