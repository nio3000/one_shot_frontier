# N3 Historical Reproduction Audit Reference

**Purpose:** post-execution comparison only. These historical values are already known and are not
additional gates. They must never be used to retune the reconstructed N3 executable.

Historical frozen N3 reference:

- decision: `N3_PASS_PATIENT_DEPENDENCE_ROBUST`
- bound PTB-XL rows: 21,544
- unique PTB-XL patients: 18,628
- repeated patients: 2,101
- records from repeated patients: 5,017
- maximum ECGs per patient: 10
- cross-age patients: 36
- one-ECG-per-patient full development rows: 74,043
- eligible contexts after deduplication: 13
- PTB cluster-bootstrap status concordance: 3/3
- PTB one-ECG-per-patient Wilson status concordance: 3/3
- polarity reversals: 0
- one-ECG-per-patient natural T_HET: `0.0015162565763262425`
- class-support-matched null median: `0.00011791079397783861`
- null 95% interval: `[3.423257266200924e-05, 0.00030983555050955597]`
- null exceedances: 0 / 20,000
- Monte Carlo p: `4.999750012499375e-05`

The null-distribution quantiles are historical-comparison values. If deterministic structural
quantities reproduce but the reconstructed Monte Carlo quantiles differ while using the frozen
master seed and unchanged algorithm, investigate RNG-stream provenance; do not alter the
scientific protocol or choose a seed to force numerical equality.
