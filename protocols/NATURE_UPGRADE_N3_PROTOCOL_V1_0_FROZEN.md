# NATURE-UPGRADE-N3 — Patient-Level Dependence Robustness

**Protocol version:** V1.0-FROZEN  
**Stage:** N3  
**Prerequisites:** N1 closed PASS; N2 closed PASS; N3-0 identity audit complete; N3-1 PTB-XL patient binding verified.  
**Scope:** Known and auditable repeated-patient dependence in PTB-XL only.

## 1. Scientific question

Does repeated-patient dependence in PTB-XL materially alter:

1. context-level deployment-risk uncertainty/status within the three frozen eligible PTB-XL age contexts; or
2. the N1 conclusion that natural institution-by-age deployment units retain excess heterogeneity beyond an exact class-support-matched random-partition null?

N3 is a robustness analysis. It does not modify the N1 primary result, N2 interval validation, frozen support thresholds, age bands, prediction threshold, source reference, or model outputs.

## 2. Frozen inputs

### 2.1 N1 bound development rows
- file: `N1_authoritative_development_rows.csv`
- SHA256: `5913d81c66754c9e0b56351796ac338eb3a16f1ebc2a016ed7c237179672ea34`
- rows: 76,959

### 2.2 PTB-XL v1.0.1 metadata
- file: `ptbxl_database.csv`
- SHA256: `cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156`
- rows: 21,837
- unique `ecg_id`: 21,837
- unique `patient_id`: 18,885

### 2.3 Verified N1-bound PTB-XL identity structure
- bound PTB-XL rows: 21,544
- unique patients: 18,628
- patients with >1 bound ECG: 2,101
- records belonging to repeated patients: 5,017
- maximum ECGs per patient: 10
- patients appearing in >1 age band: 36

These values are pre-outcome N3 identity fingerprints.

## 3. Frozen context and prediction rules

Age bands:
- `<40`
- `40-59`
- `60-74`
- `75+`

Support eligibility:
- `N >= 1024`
- `n_positive >= 40`
- `n_negative >= 40`

Frozen development hard decision:
`y_pred = 1(mean(prob_seed42, prob_seed52, prob_seed62) >= 0.3475392659505208)`

Frozen Georgia source-test BACC:
`0.921114990521441`

Primary failure margin:
`delta = 0.075`

No 2/3 majority-vote rule may replace the frozen probability-mean threshold.

## 4. Primary N3 analysis A — PTB-XL patient-cluster bootstrap

Analyze only the three frozen eligible PTB-XL age contexts:
- `40-59`
- `60-74`
- `75+`

Within each age context:
1. define patient as the resampling cluster;
2. sample the context's unique patients with replacement;
3. whenever a patient is sampled, include all that patient's ECGs in that age context;
4. compute BACC for each bootstrap draw;
5. construct a two-sided 95% percentile interval for BACC;
6. convert to a drop interval using the frozen Georgia source BACC;
7. classify at `delta=0.075`:
   - `DEFINITE_FAILURE` if drop lower bound > 0.075
   - `DEFINITE_NONFAILURE` if drop upper bound < 0.075
   - `AMBIGUOUS` otherwise.

Replicates: 20,000 per PTB-XL eligible age context.

The comparison reference is the frozen record-level Wilson status from the same N1 rows.

## 5. Primary N3 analysis B — one-ECG-per-patient N1 replay

Construct a patient-independent PTB-XL sensitivity population by retaining exactly one bound ECG per PTB-XL patient.

### 5.1 Frozen selection rule
For each PTB-XL patient:
- compute SHA256 of the UTF-8 string  
  `20260922|<patient_id>|<record_id>`
- retain the record with the lexicographically smallest hexadecimal hash.

This rule is:
- global across all age bands;
- label-blind;
- prediction-blind;
- deterministic;
- fixed before N3 outcome reveal.

The 36 cross-age patients therefore contribute to only one age context after deduplication.

All non-PTB institutions remain unchanged.

### 5.2 Exact class-support-matched replay
After deduplication:
1. reconstruct the natural eligible institution-by-age contexts using the unchanged support rule;
2. require the same 13 eligible contexts to remain available;
3. compute natural `T_HET` exactly as N1:
   - `R_c = 1 - BACC_c`
   - for institution `j`, `V_j = Var({R_c})` using population variance (`ddof=0`)
   - `T_HET = mean_j(V_j)`;
4. generate 20,000 exact class-support-matched random partitions:
   - operate within each institution;
   - shuffle positive and negative records separately;
   - allocate exactly the observed post-dedup `N`, positive count and negative count to each natural context slot;
5. compute:
   `p_HET = (1 + #{T_HET^(b) >= T_HET^natural}) / (20000 + 1)`.

## 6. Primary N3 PASS/FAIL gate

Return `N3_PASS_PATIENT_DEPENDENCE_ROBUST` only if **all** conditions hold:

1. PTB-XL patient linkage is 100% for all 21,544 bound PTB-XL rows.
2. Identity fingerprints match the frozen N3-1 audit.
3. The one-ECG-per-patient dataset retains all 13 previously eligible natural contexts under the unchanged support rule.
4. PTB-XL cluster-bootstrap status concordance with frozen record-level Wilson status is `3/3`.
5. PTB-XL one-ECG-per-patient Wilson status concordance with frozen record-level Wilson status is `3/3`.
6. There are zero `DEFINITE_FAILURE` ↔ `DEFINITE_NONFAILURE` polarity reversals for both cluster-bootstrap and one-ECG-per-patient comparisons.
7. The one-ECG-per-patient exact class-support-matched replay satisfies:
   - `p_HET <= 0.05`, and
   - `T_HET_natural > class-matched null median`.

If any condition fails, freeze:
`N3_PATIENT_DEPENDENCE_SENSITIVITY_DETECTED`.

A failed N3 does not retroactively alter N1/N2; it requires qualification of record-level independence and patient-aware uncertainty in the manuscript.

## 7. Secondary analyses

Secondary analyses cannot rescue a failed primary gate.

1. Report PTB-XL BACC and drop point-estimate changes before vs after one-ECG-per-patient selection.
2. Report interval-width inflation/deflation from record-level Wilson to patient-cluster bootstrap.
3. Report PTB-XL age-context risk ordering before vs after deduplication.
4. Repeat deterministic one-ECG-per-patient sensitivity using:
   - minimum `ecg_id` per patient;
   - maximum `ecg_id` per patient.
   These are selection-rule sensitivity checks only.
5. Report the 36 cross-age patients and the selected age band under each deterministic rule.
6. CPSC-family remains explicitly marked `patient identity unavailable`; no inferred patient clustering is permitted.
7. Chapman-Shaoxing/Ningbo are not assigned synthetic patient IDs.

## 8. Frozen computation constants

- master seed: `20260922`
- patient selection hash seed: `20260922`
- patient-cluster bootstrap replicates/context: `20,000`
- class-matched replay replicates: `20,000`
- alpha: `0.05`
- Wilson z: `1.959963984540054`
- primary margin: `0.075`
- expected eligible contexts: `13`
- expected eligible PTB contexts: `3`

## 9. Interpretation boundaries

A PASS supports:

> The frozen context-level conclusions were robust to the known and auditable repeated-patient dependence in PTB-XL: patient-cluster uncertainty preserved the three PTB-XL context-status calls, and removing repeated PTB-XL records did not eliminate the excess heterogeneity beyond an exact class-support-matched random-partition null.

A PASS does **not** establish that every CinC source contains one ECG per patient, does not resolve unavailable CPSC-family patient identity, and does not justify treating `record_id` as `patient_id`.

No protocol element may be changed after N3 outcome reveal.
