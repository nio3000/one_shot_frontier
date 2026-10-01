# CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.0

**Protocol ID:** `CMR-V1`  
**Formal name:** Kernel-Prototype Layer-Selection Cross-Mechanism Replication  
**Version:** `1.0.0-FROZEN`  
**Status:** `FROZEN_BEFORE_CMR_OUTCOME_ACCESS`  
**Freeze date:** `2026-10-01`  
**Author freeze confirmation:** confirmed in conversation on 2026-10-01  
**Target repository:** `nio3000/one_shot_frontier`  
**Manuscript baseline:** `main_v2_2_prose_tightened.tex`  
**Manuscript baseline SHA256:** `d4e686850d81117555e99e5f4ca6f01e76b2b4702dc8c2b9c7b45c54c24739ba`  

---

## 1. Purpose

The existing manuscript establishes a separation between model reconstructability and decision identifiability for a one-shot communication object built from class-wise first- and second-order statistics and a downstream Gaussian covariance-complexity action.

`CMR-V1` tests whether the same separation survives a deliberately different mechanism:

\[
\boxed{
\text{finite kernel-feature class means}
\rightarrow
\text{kernel prototype classifiers}
\rightarrow
\text{representation-layer selection}
}
\]

This protocol is a **cross-mechanism replication**, not an extension of the Gaussian-head action grid. The communication object, reconstructed classifier family and downstream action all change.

The protocol is falsifiable. It allows a complete empirical failure and forbids replacing the mechanism, datasets, layers, thresholds, seeds or gates after outcome access.

---

## 2. Scientific question

For a fixed finite kernel feature map \(\phi_\ell\) at representation layer \(\ell\), clients communicate class counts and class-wise kernel-feature sums. The server reconstructs class prototypes and one prototype classifier per candidate layer.

The primary question is:

\[
T_K(P)=T_K(Q)
\quad\text{or}\quad
d_K(T_K(P),T_K(Q))\text{ small}
\]

while

\[
\arg\min_{\ell\in\mathcal L}R_P(\ell)
\cap
\arg\min_{\ell\in\mathcal L}R_Q(\ell)
=
\varnothing?
\]

If yes, the reconstruction--decision separation is not specific to Gaussian covariance-complexity selection.

---

## 3. Claims permitted by each evidence layer

The protocol uses five stages. Each stage has a fixed interpretation.

| Stage | Evidence type | What it may support | What it may not support |
|---|---|---|---|
| X1 | exact synthetic collision | exact non-identifiability for kernel-feature mean communication | natural prevalence |
| X2 | action relevance | layer choice is non-trivial in real frozen representations | non-identifiability |
| X3 | controlled mean-preserving intervention | risk can change while the kernel-prototype message is unchanged | natural breadth |
| X4 | WILDS natural near-summary replication | local decision instability under the second mechanism | exact full-precision non-identifiability |
| X5 | PACS + OfficeHome external ecosystem | cross-ecosystem breadth of the natural instability result | universality across all tasks or summaries |

No later stage may retroactively strengthen an earlier claim beyond this table.

---

## 4. Freeze boundary and manuscript rule

`main_v2_2_prose_tightened.tex` is frozen while `CMR-V1` is running.

Allowed manuscript actions during CMR execution:

- typo-only fixes that do not alter scientific claims;
- bibliography metadata corrections that do not alter the scientific argument;
- no insertion of preliminary CMR outcomes.

Not allowed until the final `CMR-A/B/C/D` decision is frozen:

- changing the main claim to anticipate a desired CMR result;
- changing the title to imply cross-mechanism replication;
- adding provisional CMR numbers;
- deleting prior negative evidence.

The prior manuscript and prior Phase 1--5 evidence remain authoritative for the original mechanism.

---

## 5. Mechanism 2: communication object

### 5.1 Frozen representation layers

Two encoders are used.

#### ResNet-18

Encoder ID:

`resnet18_imagenet1k_v1`

Candidate actions:

\[
\mathcal L_R=
\{\text{layer1},\text{layer2},\text{layer3},\text{layer4}\}.
\]

For each stage, the tensor is taken after the corresponding residual stage, followed by adaptive average pooling to \(1\times1\), flattening and \(L_2\) normalization.

#### ViT-B/16

Encoder ID:

`vit_b_16_imagenet1k_v1`

Candidate actions:

\[
\mathcal L_V=
\{B_3,B_6,B_9,B_{12}\},
\]

where blocks are numbered 1--12.

For each selected block, use the post-block CLS token **before the final encoder LayerNorm**, followed by \(L_2\) normalization.

The exact pretrained weight enum and preprocessing transforms must match the existing Phase-1 encoder authority. The implementation preflight must record model-state and transform hashes before feature extraction.

No layer may be added, removed or substituted after protocol freeze.

---

### 5.2 Frozen random Fourier map

For each `(encoder_id, layer_id)` a fixed Gaussian-RBF random Fourier map is used:

\[
\phi_\ell(u)
=
\sqrt{\frac{2}{m}}
\cos(W_\ell u+b_\ell),
\]

with

\[
m=256,\qquad \sigma=1.0,
\]

and

\[
W_{\ell,ij}\sim\mathcal N(0,\sigma^{-2}),
\qquad
b_{\ell,i}\sim\mathrm{Uniform}(0,2\pi).
\]

Input vectors \(u\) are the \(L_2\)-normalized frozen layer features.

**Master RFF seed:** `20261001`.

Per-layer seed derivation:

1. UTF-8 string  
   `20261001|<encoder_id>|<layer_id>`
2. compute SHA256;
3. take the first 8 bytes, big-endian;
4. reduce modulo \(2^{32}\);
5. initialize `numpy.random.Generator(PCG64(seed))`.

Formal RFF arrays are generated in `float64`. Their SHA256 values are stored in the feature-map manifest. Formal downstream runs use the frozen arrays, not regenerated maps.

No bandwidth tuning, RFF-dimension tuning or seed search is permitted.

---

### 5.3 Client message

For each class \(c\) and layer \(\ell\), a client sends:

\[
n_c,\qquad
s_{\ell c}=\sum_{i:y_i=c}\phi_\ell(x_i).
\]

The complete communication object is

\[
\boxed{
T_K(D)=
\{n_c,s_{\ell c}\}_{c,\ell}.
}
\]

The server aggregates client messages by summation and reconstructs

\[
\mu_{\ell c}
=
\frac{s_{\ell c}}{n_c}.
\]

No covariance matrix, raw feature vector, raw sample, held-out label or evaluation statistic belongs to `T_K`.

---

### 5.4 Reconstructed classifier family

For each candidate layer:

\[
h_\ell(x)=
\arg\max_c
\langle
\phi_\ell(x),
\mu_{\ell c}
\rangle.
\]

Ties in class score use the smallest canonical class ID.

This is a kernel-feature prototype classifier. It is intentionally simpler than a full kernel-space LDA. The protocol is **not** a reproduction of FedKSS; FedKSS is literature motivation for kernel-space one-shot statistics and layer selection, not implementation authority for CMR-V1.

---

## 6. Downstream decision and risk

The action is representation identity:

\[
\boxed{a=\ell\in\mathcal L.}
\]

The primary performance metric is balanced accuracy:

\[
\mathrm{BACC}_P(\ell).
\]

Risk is

\[
R_P(\ell)=1-\mathrm{BACC}_P(\ell).
\]

The optimal action set is

\[
\mathcal O_P=
\arg\min_{\ell\in\mathcal L}R_P(\ell).
\]

Numerical ties use tolerance:

`optimal_set_tolerance = 1.0e-12`.

For two tasks \(P,Q\), deterministic pair minimax regret is

\[
\Gamma_K(P,Q)
=
\min_{\ell\in\mathcal L}
\max
\left[
R_P(\ell)-R_P^\star,\,
R_Q(\ell)-R_Q^\star
\right].
\]

Randomized minimax regret is computed only where explicitly required; deterministic regret is the primary natural-domain conflict statistic.

---

# 7. X1 — Exact collision

## 7.1 Purpose

Demonstrate an exact communication-form collision under class-wise finite kernel-feature means with a downstream layer-selection action.

X1 is analytic and deterministic. It is not an outcome-search experiment.

---

## 7.2 Frozen exact feature map

For X1 only, use the exact two-dimensional Fourier feature map

\[
\phi_F(\theta)=(\cos\theta,\sin\theta).
\]

This keeps every synthetic feature vector on the unit circle and makes the witness an exact finite Fourier-feature construction rather than an unconstrained Euclidean toy.

Let

\[
a=\frac15,
\qquad
q=\sqrt{1-a^2}=\sqrt{\frac{24}{25}}.
\]

Define two class-conditional feature patterns.

### Good pattern

Class 0:

\[
(-a,+q),(-a,-q)
\]

repeated three times.

Class 1:

\[
(+a,+q),(+a,-q)
\]

repeated three times.

Both classes contain six observations.

### Bad pattern

Class 0:

- two copies of \((+a,+q)\),
- two copies of \((+a,-q)\),
- two copies of \((-1,0)\).

Class 1:

- two copies of \((-a,+q)\),
- two copies of \((-a,-q)\),
- two copies of \((+1,0)\).

The class means are identical under both patterns:

\[
\mu_0=(-a,0),
\qquad
\mu_1=(+a,0).
\]

The class counts are also identical.

---

## 7.3 Two tasks

Task A:

- layer 1 = Good pattern;
- layer 2 = Bad pattern.

Task B:

- layer 1 = Bad pattern;
- layer 2 = Good pattern.

Therefore, for both layers and both classes,

\[
T_K(A)=T_K(B).
\]

The reconstructed prototype family is exactly the same.

For the prototype classifier:

\[
\mathrm{BACC}_A(\ell_1)=1,
\qquad
\mathrm{BACC}_A(\ell_2)=\frac13,
\]

and

\[
\mathrm{BACC}_B(\ell_1)=\frac13,
\qquad
\mathrm{BACC}_B(\ell_2)=1.
\]

Hence

\[
\mathcal O_A=\{\ell_1\},
\qquad
\mathcal O_B=\{\ell_2\},
\]

and

\[
\mathcal O_A\cap\mathcal O_B=\varnothing.
\]

Expected exact regrets:

\[
\Gamma^{\mathrm{det}}_K=\frac23,
\qquad
\Gamma^{\mathrm{rand}}_K=\frac13.
\]

---

## 7.4 X1 gate

All conditions must pass:

1. exact class counts match;
2. exact class sums match symbolically;
3. floating reconstruction max absolute difference \(\le10^{-12}\);
4. reconstructed prototype family is identical;
5. optimal sets are disjoint;
6. deterministic regret equals \(2/3\) within \(10^{-12}\);
7. randomized regret equals \(1/3\) within \(10^{-12}\);
8. unit tests pass.

If X1 fails because implementation disagrees with the analytic witness, the project is `IMPLEMENTATION_BLOCKED`. There is no scientific CMR decision until the bug is corrected and the failed artifacts are preserved.

---

# 8. X2 — Action relevance

## 8.1 Purpose

Test whether representation layer is a genuine decision variable in real frozen feature spaces.

X2 is the only hard scientific continuation gate. If it fails, CMR-V1 stops. The action family may not be replaced.

---

## 8.2 Frozen datasets

Use the same four datasets already present in the Phase-1 representation audit:

- CIFAR-100;
- EuroSAT;
- PathMNIST;
- DermaMNIST.

Use the same dataset versions, official train/test partition semantics, pretrained encoder authority and preprocessing conventions as Phase 1.

Structural conditions:

\[
4\ \text{datasets}\times2\ \text{encoders}=8.
\]

Projection dimensions are **not** used in CMR-V1.

---

## 8.3 Federated reconstruction check

The training set is partitioned into 20 clients using the Phase-1 authority:

- Dirichlet alpha = `0.10`;
- partition seed = `20260908`.

Client messages are aggregated to global class sums.

Required aggregation recovery:

\[
\max |s_{\mathrm{aggregated}}-s_{\mathrm{centralized}}|
\le10^{-8}.
\]

Client partition is not a scientific axis; it is an integrity check that the one-shot message reconstructs the same prototypes as centralized aggregation.

---

## 8.4 X2 evaluation

For each structural condition:

1. build the four candidate layer prototype classifiers using training data;
2. evaluate all four on the frozen official test set;
3. record BACC for every layer;
4. compute
   \[
   \Delta_{\mathrm{layer}}
   =
   \max_\ell \mathrm{BACC}_\ell
   -
   \min_\ell \mathrm{BACC}_\ell.
   \]

A condition is `ACTION_RELEVANT` if

\[
\Delta_{\mathrm{layer}}\ge0.01.
\]

For layer-identity breadth, define a `UNIQUE_BEST` layer only when the best layer exceeds the second-best layer by at least

\[
0.002.
\]

---

## 8.5 X2 PASS gate

All conditions below are required:

1. at least `6/8` structural conditions are `ACTION_RELEVANT`;
2. for ResNet-18, at least two different layers appear as `UNIQUE_BEST` across the four datasets;
3. for ViT-B/16, at least two different blocks appear as `UNIQUE_BEST` across the four datasets;
4. all aggregation recovery checks pass;
5. all values are finite;
6. no layer, RFF map or threshold was changed after outcome access.

If X2 fails:

`CMR_X2_DECISION = ACTION_FAMILY_NOT_DECISION_RELEVANT`

and the project stops with final decision `CMR-D`.

No bandwidth selection, regularization selection, alternative layer set or replacement action may be introduced within CMR-V1.

---

# 9. X3 — Mean-preserving controlled intervention

## 9.1 Purpose

Test whether risk and the preferred layer can change while the one-shot kernel-prototype message is held fixed.

X3 is run only after X2 PASS.

X3 outcome does **not** control whether X4 is run. Once X2 passes, X3, X4 and X5 continue according to protocol even if an earlier scientific gate fails.

---

## 9.2 Frozen intervention

For training kernel feature \(z_i=\phi_\ell(x_i)\) in class \(c\), with global training prototype \(\mu_{\ell c}\):

\[
\widetilde z_i
=
2\mu_{\ell c}-z_i.
\]

For a test record of true class \(c\), apply the same deterministic paired transformation around the training prototype:

\[
\widetilde z^{\mathrm{test}}_i
=
2\mu_{\ell c}-z^{\mathrm{test}}_i.
\]

This is a controlled counterfactual distributional intervention. It is not an operational deployment transform.

The training class sum is preserved:

\[
\sum_i \widetilde z_i
=
\sum_i z_i.
\]

Therefore the one-shot message and reconstructed prototype classifier family are unchanged up to numerical tolerance.

---

## 9.3 X3 invariant gates

For every condition:

- class counts identical;
- message max absolute difference \(\le10^{-8}\);
- reconstructed prototype max absolute difference \(\le10^{-8}\);
- same layer action set;
- no raw labels enter the communicated message.

An invariant failure is an implementation failure, not a scientific negative result.

---

## 9.4 X3 conflict rule

For each original/counterfactual pair, calculate \(\Gamma_K\).

A condition is `ROBUST_COUNTERFACTUAL_LAYER_CONFLICT` only if all three hold:

1. \(\Gamma_K\ge0.005\);
2. 95% paired-bootstrap lower bound on \(\Gamma_K\) is at least `0.0025`;
3. original and counterfactual optimal-layer sets are disjoint.

Paired bootstrap:

- 5,000 replicates;
- resample test records within class;
- the same sampled record indices are used for the original and reflected task;
- master bootstrap seed = `20261005`, with deterministic condition-specific seed derivation by SHA256.

---

## 9.5 X3 breadth gate

`X3_PASS` requires:

- at least `4/8` structural conditions qualify;
- qualifying conditions span at least two datasets;
- at least one ResNet-18 condition qualifies;
- at least one ViT-B/16 condition qualifies.

A failure is frozen as:

`COUNTERFACTUAL_EFFECT_PRESENT_BUT_NOT_BROAD`

if at least one condition qualifies, or

`COUNTERFACTUAL_LAYER_CONFLICT_NOT_SUPPORTED`

if none qualify.

Both outcomes continue to X4 after evidence freeze.

---

# 10. X4 — WILDS natural near-summary replication

## 10.1 Purpose

Test natural local decision instability under the second communication mechanism while holding the deployment datasets and support rules as close as possible to the original Phase-5 analysis.

---

## 10.2 Frozen datasets and context rules

Reuse the Phase-5 Tier-1 authority:

| Dataset | Context | Group | Shared-class rule | Minimum fit/eval support |
|---|---|---|---|---|
| Camelyon17-WILDS | hospital | slide | 2 | 40 per class |
| RxRx1-WILDS | cell type | experiment | 32--64 | 2 per class |
| iWildCam-WILDS | location | sequence | 3--64 | 5 per class |

Reuse:

- group split seed `20260910`;
- group fit fraction `0.50`;
- class subset seed `20260910`.

The same raw examples and split semantics are used, but the communication object is recomputed from the new intermediate-layer RFF features.

Structural conditions:

\[
3\ \text{datasets}\times2\ \text{encoders}=6.
\]

---

## 10.3 Pair-specific support matching

For a candidate context pair \(A,B\) and shared class \(c\):

\[
m_c=
\min
\left(
n^{A}_{c,\mathrm{fit}},
n^{B}_{c,\mathrm{fit}}
\right).
\]

Exactly \(m_c\) fit records are retained from each context using deterministic record hashing.

**Support-match master seed:** `20261003`.

Hash key:

`20261003|<dataset>|<encoder>|<context>|<class>|<record_id>`

SHA256 ascending order determines retained records.

This ensures the communicated class counts are equal across a pair. No evaluation label or evaluation risk enters the support-matching operation.

---

## 10.4 Kernel-summary distance

For pair \(A,B\), define

\[
d_K(A,B)
=
\sqrt{
\frac{1}{|\mathcal L||\mathcal C|}
\sum_{\ell,c}
\|
\mu^A_{\ell c}
-
\mu^B_{\ell c}
\|_2^2
}.
\]

All layers use the same RFF dimension and scale.

Within each structural condition, candidate pairs are ranked by \(d_K\) only.

---

## 10.5 Pair selection

Frozen parameters:

- `pair_near_fraction = 0.20`;
- `pair_max_selected = 10`;
- `pair_min_selected = 1`.

Number selected:

\[
n_{\mathrm{select}}
=
\min
\left[
10,\,
\max
\left(
1,\,
\lceil0.20\,n_{\mathrm{eligible}}\rceil
\right)
\right].
\]

Tie-break order:

1. smaller \(d_K\);
2. canonical context-A ID;
3. canonical context-B ID.

The pair manifest must contain:

- dataset and encoder;
- contexts;
- shared class set;
- pair-specific fit record hashes;
- class counts;
- \(d_K\);
- rank;
- eligible-pair count;
- selected-pair count;
- feature-map manifest SHA;
- protocol SHA;
- git HEAD.

The manifest must not contain evaluation BACC, optimal layer, regret or robust-conflict status.

---

## 10.6 Mandatory blackout

Evaluation risk remains blacked out until:

1. feature maps are frozen;
2. fit feature banks are frozen;
3. support matching is complete;
4. pair manifest is generated;
5. pair manifest is committed to Git;
6. a separate unblind-authorization artifact records the committed pair-manifest SHA.

Only then may the risk runner read evaluation labels.

Any pre-authorization evaluation-risk access invalidates X4 as confirmatory evidence.

---

## 10.7 X4 robust conflict

For each frozen pair:

1. reconstruct each context's candidate prototype family from its support-matched fit message;
2. evaluate all candidate layers on that context's evaluation records restricted to the frozen shared classes;
3. compute \(\Gamma_K\);
4. compute a 1,000-replicate record-stratified-within-class bootstrap CI.

A pair is `ROBUST_KERNEL_LAYER_CONFLICT` only if:

- \(\Gamma_K\ge0.005\);
- bootstrap 95% lower bound \(\ge0.0025\);
- optimal-layer sets are disjoint.

A non-gating group-cluster sensitivity analysis may also be reported for WILDS where group support permits it. It cannot replace the primary bootstrap.

---

## 10.8 X4 panel gate

Final panel gate uses 5,000 condition-cluster bootstrap replicates.

`X4_PASS` requires all:

1. robust-pair fraction \(\ge0.15\);
2. condition-cluster 95% CI lower bound \(\ge0.05\);
3. at least `3/6` structural conditions contain at least one robust pair;
4. at least `2/3` datasets contain at least one robust pair;
5. both encoders contain at least one robust pair.

No higher-order diagnostic is required for X4. The failed M3 route from the first mechanism is not recycled as a gate.

---

# 11. X5 — Independent ecosystem external replication

## 11.1 Datasets

Tier-2 external panel:

- PACS;
- OfficeHome.

These datasets are not part of the WILDS ecosystem.

Before feature extraction, freeze a raw-data manifest containing:

- authoritative source;
- archive/file SHA256;
- file count;
- class names;
- domain names;
- record-ID derivation;
- preprocessing version.

No source may be changed after raw-manifest freeze because of downstream results.

---

## 11.2 Domains and support

PACS:

- 4 domains;
- all 7 nominal classes required in an eligible domain pair;
- minimum fit support = 10 per class per domain;
- minimum evaluation support = 10 per class per domain.

OfficeHome:

- 4 domains;
- all 65 nominal classes required in an eligible domain pair;
- minimum fit support = 5 per class per domain;
- minimum evaluation support = 5 per class per domain.

No group metadata is assumed. Within each domain and class, records are split 50/50 into fit/evaluation partitions by deterministic path-hash ordering.

**External split seed:** `20261002`.

Odd sample counts allocate the extra record to fit.

---

## 11.3 X5 pair selection

There are six possible domain pairs per dataset.

Within each dataset × encoder condition:

- require at least 4 eligible domain pairs;
- rank eligible pairs by the same \(d_K\);
- `external_pair_near_fraction = 0.50`;
- select
  \[
  \min(3,\max(1,\lceil0.50n_{\mathrm{eligible}}\rceil)).
  \]

The external pair manifest is frozen and committed before evaluation risk is opened.

Risk unblinding follows the same authorization rule as X4.

---

## 11.4 X5 robust conflict

Use exactly the X4 rule:

- \(\Gamma_K\ge0.005\);
- bootstrap lower 95% bound \(\ge0.0025\);
- disjoint optimal-layer sets.

Use 1,000 record-stratified-within-class bootstrap replicates per pair.

---

## 11.5 X5 gate

`X5_PASS` requires all:

1. pooled robust-pair fraction \(\ge0.15\);
2. PACS contains at least one robust pair;
3. OfficeHome contains at least one robust pair;
4. ResNet-18 contains at least one robust pair;
5. ViT-B/16 contains at least one robust pair.

A Wilson 95% interval for the pooled robust fraction is reported descriptively but is not an X5 gate because the external panel has only four structural conditions.

---

# 12. Final CMR decision

The final decision is mechanical.

## CMR-A — Strong cross-mechanism, cross-ecosystem replication

Requirements:

- X1 PASS;
- X2 PASS;
- X3 PASS;
- X4 PASS;
- X5 PASS.

Permitted manuscript interpretation:

> Reconstruction--decision separation reproduced under a qualitatively different one-shot communication object and downstream action family, with controlled and natural evidence spanning WILDS and a second benchmark ecosystem.

---

## CMR-B — Cross-mechanism replication with external breadth limitation

Requirements:

- X1 PASS;
- X2 PASS;
- X3 PASS;
- X4 PASS;
- X5 FAIL.

Permitted interpretation:

> Cross-mechanism replication supported in controlled and WILDS natural analyses, but independent-ecosystem breadth was not established.

---

## CMR-C — Cross-mechanism evidence present but incomplete

Requirements:

- X1 PASS;
- X2 PASS;
- not CMR-A or CMR-B;
- at least one of X3, X4 or X5 PASS.

Permitted interpretation:

> A second communication/action mechanism shows non-trivial replication evidence, but the prespecified breadth chain is incomplete.

---

## CMR-D — Empirical cross-mechanism replication not supported

Either:

- X2 FAIL, causing protocol stop;

or, after X2 PASS:

- X3 FAIL;
- X4 FAIL;
- X5 FAIL.

Permitted interpretation:

> The exact kernel-feature communication form can exhibit reconstruction--decision separation, but the empirical cross-mechanism replication was not supported under the frozen protocol.

---

## Integrity state outside A/B/C/D

If a required invariant, outcome blackout or implementation authority is violated, use:

`CMR_INTEGRITY_BLOCKED`

This is not a scientific negative result. The invalid run is preserved and excluded from confirmatory evidence.

---

# 13. Prohibited post-outcome changes

After any X2 outcome has been accessed, the following are forbidden within CMR-V1:

- replace representation-layer selection with bandwidth, ridge, covariance or another action;
- add or remove candidate layers;
- change RFF dimension;
- tune kernel bandwidth;
- search RFF seeds;
- relax `0.01`, `0.002`, `0.005`, `0.0025`, `0.15` or `0.05` gates;
- replace PACS or OfficeHome;
- drop RxRx1;
- change WILDS support rules because of results;
- change near-pair fractions;
- use evaluation risk in pair selection;
- select only favorable datasets, encoders, layers, contexts or pairs;
- redefine CMR-A/B/C/D after seeing results.

A reproducibility bug may be corrected only through a versioned implementation amendment that:

1. preserves all previous run artifacts;
2. documents the bug;
3. shows that scientific protocol semantics did not change;
4. reruns all affected stages;
5. never changes a gate because the corrected result is unfavorable.

A scientific protocol change requires a new protocol version, not an amendment to V1.0.

---

# 14. Seeds and frozen constants

| Item | Value |
|---|---:|
| RFF dimension | 256 |
| RBF sigma | 1.0 |
| RFF master seed | 20261001 |
| external split seed | 20261002 |
| WILDS support-match seed | 20261003 |
| reserved intervention seed | 20261004 |
| bootstrap master seed | 20261005 |
| Phase-1 client partition seed | 20260908 |
| Phase-1 client Dirichlet alpha | 0.10 |
| WILDS group split seed | 20260910 |
| WILDS class subset seed | 20260910 |
| primary metric | balanced accuracy |
| X2 action spread | 0.01 |
| X2 unique-winner margin | 0.002 |
| conflict gap | 0.005 |
| conflict CI lower bound | 0.0025 |
| X4/X5 robust fraction | 0.15 |
| X4 cluster-CI lower bound | 0.05 |
| pair bootstrap reps | 1000 |
| X3 paired bootstrap reps | 5000 |
| panel bootstrap reps | 5000 |
| message invariant tolerance | \(10^{-8}\) |
| X1 numeric tolerance | \(10^{-12}\) |

Seeds are authorities, not search starting points.

---

# 15. Required artifact chain

## Protocol freeze

- `docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md`
- `configs/cross_mechanism_replication_protocol_v1.yaml`
- `docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv`
- `docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json`

## X1

- `src/frontier/cmr_kernel_layer.py`
- `tools/run_cmr_v1.py`
- `tests/test_cmr_v1_x1.py`
- `runs/cmr_v1/x1/`
- `evidence/cmr_v1/x1/`

## X2

- `configs/cmr_v1_feature_map_manifest.json`
- `configs/cmr_v1_feature_bank_manifest.json`
- `runs/cmr_v1/x2/`
- `evidence/cmr_v1/x2/`

## X3

- `runs/cmr_v1/x3/`
- `evidence/cmr_v1/x3/`

## X4

- `configs/cmr_v1_wilds_pair_manifest.json`
- `configs/cmr_v1_wilds_unblind_authorization.json`
- `runs/cmr_v1/x4/`
- `evidence/cmr_v1/x4/`

## X5

- `configs/cmr_v1_external_raw_manifest.json`
- `configs/cmr_v1_external_pair_manifest.json`
- `configs/cmr_v1_external_unblind_authorization.json`
- `runs/cmr_v1/x5/`
- `evidence/cmr_v1/x5/`

## Final

- `evidence/cmr_v1/final_gate_summary.json`
- `docs/phases/cmr_v1/CMR_V1_FINAL_REPORT.md`

---

# 16. Stage transitions

The only hard scientific continuation gate is X2.

```text
PROTOCOL FREEZE
    |
    v
X1 exact collision + tests
    |
    v
X2 action relevance
    |
    +-- FAIL --> CMR-D / STOP
    |
    +-- PASS
          |
          v
         X3
          |
          v
   freeze X3 evidence
          |
          v
   build X4 fit-only pair manifest
          |
          v
   commit pair manifest
          |
          v
   authorize X4 risk unblind
          |
          v
         X4
          |
          v
   build X5 raw + fit-only pair manifests
          |
          v
   commit manifests
          |
          v
   authorize X5 risk unblind
          |
          v
         X5
          |
          v
 mechanical CMR-A/B/C/D
```

X3 failure does not stop X4. X4 failure does not stop X5.

---

# 17. Pre-outcome implementation rule

Before X1 implementation begins, the protocol files must be committed.

Before X2 outcome access:

- X1 code and unit tests must pass;
- RFF seed derivation must be tested;
- encoder layer hooks must be tested;
- class-message aggregation must be tested;
- no X2 result may be used to modify the protocol.

Before X4 or X5 unblinding, pair-selection code must be demonstrably outcome-blind.

---

# 18. Literature positioning

CMR-V1 is not justified by inventing a hypothetical layer-selection problem. Kernel-space one-shot federated statistics and adaptive layer selection are already used in the contemporary literature, including FedKSS (Ding et al., *Information Fusion*, 2026, DOI `10.1016/j.inffus.2026.104443`).

This citation is **motivation only**. CMR-V1 deliberately uses a simpler class-mean prototype message rather than reproducing FedKSS's complete classifier. This preserves mechanism independence from the existing Gaussian second-order analysis and keeps the reconstruction claim exact.

---

# 19. Freeze declaration

The scientific choices in this document are frozen as `CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.0`.

The next authorized action is **protocol integration and Git freeze only**.

No X2--X5 outcome experiment is authorized by this document before the protocol commit. After protocol commit, X1 implementation and unit testing are authorized. X2 may begin only after X1 closes with PASS.

