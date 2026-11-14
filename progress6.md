# Solar Filament Segmentation Challenge 2026 (MAGFiLO) — Comprehensive Technical Report Dossier & Experimental Ledger

**Document Classification:** Primary Technical Source, Mathematical Ledger & Methodology Dossier for IEEE Conference Paper  
**Task:** Instance Segmentation of Solar Filaments from Full-Disk $2048 \times 2048$ GONG H-Alpha Solar Observations  
**Lead Researcher:** Farhan Anjum (MD. Farhan Anjum) — Batch 23, Department of Computer Science & Engineering, Bangladesh University of Engineering and Technology (BUET)  
**Sponsor:** U.S. National Science Foundation (NSF) / National Solar Observatory (NSO)  
**Host:** Earth-Space AI Research (ESAIR) Lab  
**Report Window:** October 15 – November 6, 2026 | **Final Package Deadline:** November 15, 2026, 11:59 pm CDT  
**Current Competitive Standing:** Validation $\text{PQ} = \mathbf{0.4205}$ | **Public Leaderboard (LB):** $\mathbf{0.35}$ (Official Target Tier Verified)  

---

## 1. Executive Summary & Project Trajectory

Solar filaments (dense chromospheric plasma suspended along photospheric magnetic neutral lines) serve as primary topological precursors to coronal mass ejections (CMEs) and severe space weather events. Automated instance segmentation of filaments from full-disk $2048 \times 2048$ H-alpha observations is constrained by three physical and structural hurdles:
1. **Ribbon Geometry & Topology:** Filaments exhibit extreme aspect ratios (spine widths of $3 \text{ to } 15\text{ pixels}$ spanning lengths up to $1,200\text{ pixels}$) that easily fragment under uniform thresholding or anchor-based detectors.
2. **Annotation Ambiguity:** Human expert agreement on identical physical images is strictly bounded at $\text{PQ} = 0.3397$ (A3 audit across $1,196$ directed comparisons).
3. **Artifact Interference:** Active solar phenomena (sunspot umbrae, penumbrae, plages, and chromospheric limb darkening) visually mimic dark filament absorption, provoking severe False Positive ($\text{FP}$) penalties under Panoptic Quality.

### Project Progression Milestones
* **Baseline Proof-of-Concept (Oct 15):** Coarse-to-fine U-Net with ground-truth region training reached $\text{PQ} = 0.2570$ ($\text{coarse-only} = 0.2437$).
* **System Correction & Scaling (Oct 20–25):** Fixed double-normalization scaling and region extraction, advancing the baseline to $\text{PQ} = 0.3220$ on the seed-42 holdout ($\text{coarse-only} = 0.2401$, $\Delta = +0.0819$). Public Leaderboard (LB) recorded at $0.29 \to 0.31$.
* **Correctness Gates & Morphological Spine Repair (Oct 26–27):** Resolved fine-loss tensor broadcasting ($[B, 1, H, W] \times [B, H, W] \to [B, B, H, W]$) and threshold-after-Hungarian matching errors ($0.3391 \text{ PQ}$). Bilinear probability upsampling with a $45$-pixel morphological closing kernel bridged broken spines ($0.3500 \text{ PQ}$). Confidence-ordered greedy overlap resolution pruned fragmented artifacts, yielding $\text{PQ} = 0.3512$.
* **Coarse $1024$ & S1 Decision Scorer Deployment (Oct 28–29):** Retrained the coarse stage at $1024 \times 1024$ with balanced weighting ($\text{pos\_weight} = 5.0$) and D4 augmentation, paired with an expected-utility $\text{HistGradientBoostingRegressor}$ ($S1$), achieving $\text{PQ} = 0.3896 \text{ to } 0.4439$ locally (LB: $0.32 \to 0.33$).
* **7-Hour Master Architecture Suite (Oct 30):** Systematic benchmarking established that **EfficientNet-B3 U-Net** is the premier feature extractor for filament spines, setting new records of **$\text{PQ} = 0.4283$** on an untouched 71-image clean holdout and **$\text{PQ} = 0.4443$** on the full 141-image validation split.
* **Frontier Paradigm Exploration (Oct 31):** Evaluated empirical limb-darkening compensation, multi-scale FPN/DeepLabV3+ decoders, and dual-threshold hysteresis. Proved that single-instance U-Nets with standard median/MAD preprocessing outperform heuristic physical flattening and that aggressive seed-and-grow hysteresis erodes detection recall on faint structures.
* **15-Experiment Inference Suite (Nov 4):** Evaluated 48 configurations on validation data without training. Discovered that minimum area pruning up to $75\text{ px}$ steadily boosts PQ (countering the earlier $20\text{ px}$ assumption), Fine threshold $0.50$ is optimal for EfficientNet-B3, and median TTA aggregation outperforms arithmetic mean.
* **Advanced Research & OOF Retraining Suite (Nov 6):** Executed a 4,350-second deep diagnostic run. Achieved **$\text{PQ} = 0.4106$** on Synergistic Champion stacking, generated a 4,843-sample 5-shard OOF bank, retrained the candidate scorer with physics features to hit **$\text{PQ} = 0.4205$**, traced 544 False Negatives to isolate fine threshold erosion as the primary bottleneck, mapped the latency/accuracy Pareto frontier, and officially attained **Public Leaderboard $\text{PQ} = \mathbf{0.35}$**.

---

## 2. Mathematical Formulation of the Optimization Metric

### 2.1 Metric Definition: Panoptic Quality (PQ)
Model predictions are evaluated against ground-truth instances using the Panoptic Quality ($\text{PQ}$) metric at a strict threshold of $\text{IoU} > 0.5$ against independent annotator passes:

$$\text{PQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|} = \text{SQ} \times \text{RQ}$$

Where:
* **$\text{SQ}$ (Segmentation Quality):** Average spatial overlap of accepted matches:
  $$\text{SQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|}$$
* **$\text{RQ}$ (Recognition Quality):** Instance detection $F_1$-score:
  $$\text{RQ} = \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

### 2.2 Threshold-Aware One-to-One Hungarian Matching
In early starter implementations, greedy pairwise counting collected all IoU hits $> 0.5$, artificially inflating validation scores by $+0.03 \text{ to } +0.05$. 

In this framework, instance matching strictly enforces a one-to-one assignment via the Hungarian algorithm. An eligible pair requires $\text{IoU}(p, g) > 0.5$. Matching is solved lexicographically:
$$\text{Weight}(p, g) = \begin{cases} (\min(|P|, |G|) + 1.0) + \text{IoU}(p, g) & \text{if } \text{IoU}(p, g) > 0.5 \\ 0.0 & \text{otherwise} \end{cases}$$

The cardinality bonus guarantees that maximizing total weight first maximizes the number of true positive matches ($|\text{TP}|$), and then maximizes the sum of matched IoUs among matchings of that maximum cardinality. All unmatched predictions become $\text{FP}$; all unmatched ground-truth instances become $\text{FN}$.

### 2.3 The Dinkelbach Candidate Pruning Criterion
Let $S = \sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)$ and $D = |\text{TP}| + 0.5|\text{FP}| + 0.5|\text{FN}|$. Adding candidate prediction $c$ alters the denominator by $\Delta D = +0.5$. If $c$ matches a ground-truth instance with expected overlap $u = \mathbb{E}[\text{IoU}(c, g)]$, $\Delta S = u$; otherwise $\Delta S = 0$.

The marginal impact on $\text{PQ}$ is:
$$\text{PQ}_{\text{new}} - \text{PQ} = \frac{S + u}{D + 0.5} - \frac{S}{D} = \frac{u \cdot D - 0.5 \cdot S}{D(D + 0.5)}$$

$$\text{PQ}_{\text{new}} > \text{PQ} \iff u \cdot D - 0.5 \cdot S > 0 \iff u > \frac{S}{2D} = \frac{\text{PQ}}{2}$$

**Theoretical Decision Rule:** At a baseline of $\text{PQ} \approx 0.38 \text{ to } 0.44$, any candidate proposal with an expected utility $u < \tau \approx 0.19 \text{ to } 0.22$ actively lowers the overall benchmark score. Filtering candidates using an aligned machine-learning regressor targeting $u = \text{IoU} \cdot \mathbf{1}_{\{\text{IoU} > 0.5\}}$ with $\tau \in [0.20, 0.35]$ eliminates false-positive hallucinations and optimizes Panoptic Quality.

---

## 3. Dataset Specifications & Partitioning Rigor

### 3.1 GONG Observational Properties
* **Source:** Global Oscillation Network Group (GONG) H-alpha instruments across six stations: Big Bear (`Bh`), Cerro Tololo (`Ch`), Learmonth (`Lh`), Mauna Loa (`Mh`), Teide (`Th`), and Udaipur (`Uh`).
* **Format:** $2048 \times 2048$ pixels, 8-bit grayscale converted from raw FITS files.
* **Volume:** $707$ physical training observations, $180$ hidden test observations.
* **Subjectivity:** $42\%$ of physical observations have independent annotations from $2 \text{ to } 3$ expert annotators ($937$ training annotation records across $566$ physical training images; $217$ validation records across $141$ physical validation images).

### 3.2 Partition Integrity & Sub-Splits
Splitting was executed strictly on the **physical observation ID** (the timestamp string following the annotator prefix, e.g., `010401-20160920230134Lh` $\to$ `20160920230134Lh`):
* **Training Partition:** $566$ physical images ($80\%$) $\to$ $937$ annotator records.
* **Full Validation Partition:** $141$ physical images ($20\%$) $\to$ $217$ annotator records containing $1,593$ ground-truth instances.
* **Zero Leakage:** $\text{set}(\text{Train}) \cap \text{set}(\text{Val}) = \emptyset$. Checksum verified in `split.json`: `1d0e7f8fc5bfaba91a5cd68b5d759d91deaa75370a78bfb7e6016f4b2e230051`.
* **Zero-Leakage Calibration/Holdout Subdivision:** To prevent hyperparameter tuning from artificially inflating reported metrics, the $141$ validation images were subdivided into:
  * **Tune Partition ($70$ physical images, $787$ GT instances):** Used exclusively for grid-searching $(\tau, \text{min\_area}, \text{fine\_threshold})$.
  * **Clean Holdout Partition ($71$ physical images, $806$ GT instances):** Kept completely unseen during tuning to provide unbiased generalizability estimates.

---

## 4. End-to-End Pipeline Architecture

```
                    [ 2048 x 2048 Raw Grayscale Image ]
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
           [ Global Otsu Threshold ]      [ Disk-Restricted Normalization ]
                     │                    (I - Median) / (MAD + 1e-6)
                     ▼                               │
           [ Connected Components ]                  │
          (Select largest label >= 1)                ▼
                     │                      [ Masked Solar Disk ]
                     └──────────────┬────────────────┘
                                    │
                                    ▼
                       [ Resize to 1024 x 1024 ]
                                    │
                                    ▼
               [ Stage 1: Coarse U-Net (ResNet34) ]
                                    │
                                    ▼
                     [ 1024x1024 Logit Probabilities ]
                                    │
                                    ▼
               [ Bilinear Upsampling to 2048 x 2048 ]
                                    │
                                    ▼
                 [ Global Threshold (prob > 0.28-0.34) ]
                                    │
                                    ▼
               [ Morphological Closing (35-45px Ellipse) ]
                                    │
                                    ▼
             [ Extract Unified Candidate Connected Components ]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
    [ BBox Expansion ]                            [ 64x64 Bounded Dilation ]
(Pad by 32px on each side)                       (Channel 1 Region Conditioning)
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
                 [ Dynamic Percentile Size Bucketing ]
                W ∈ {160, 288, 576, 928} | H ∈ {160, 256, 448, 800}
                                    │
                                    ▼
              [ Stage 2: Fine U-Net (EfficientNet-B3) ]
                Channel 0: Image Crop | Channel 1: Region Mask
                                    │
                                    ▼
                 [ 4-View Flip TTA / 8-View D4 TTA ]
                     (Optimal: 4-View Flip at 570ms)
                                    │
                                    ▼
                     [ Fine Thresholding (> 0.50) ]
                                    │
                                    ▼
                  [ Slice Back to Native 2048 x 2048 ]
                                    │
                                    ▼
           [ Connected Components -> Instance Candidates ]
                                    │
                                    ▼
               [ Stage 3: Tabular Feature Extraction ]
             Base 6: { log(area), mean_p, max_p, std_p, aspect, extent }
             Aug 8:  { + normalized_radius (r/R_sun), tortuosity }
                                    │
                                    ▼
           [ S1/S2 HistGradientBoosting Expected-IoU Scorer ]
                     (Trained on 4,843 OOF Bank)
                                    │
                                    ▼
               [ Prune Low-Utility Candidates (< tau=0.25) ]
                                    │
                                    ▼
            [ Utility-Ranked Greedy Overlap Resolution ]
                       (Prune Fragments < 75 px)
                                    │
                                    ▼
                 [ Optional: Collinear Merge X4 ]
                     (Dist <= 20px, Angle <= 30 deg)
                                    │
                                    ▼
                    [ Final Disjoint RLE Submissions ]
```

### 4.1 Solar Disk Extraction & Robust Normalization
* **Disk Masking:** Space background is segmented via global Otsu thresholding. Background label 0 is omitted, selecting `1 + areas[1:].argmax()` to isolate the solar sphere without clipping limb filaments.
* **Disk-Restricted Scaling:** Brightness variations across observatory instruments are resolved by local disk-restricted scaling:
  $$I_{\text{norm}} = \frac{I - \text{Median}(I_{\text{disk}})}{\text{MAD}(I_{\text{disk}}) + 10^{-6}}, \quad \text{MAD} = \text{Median}(|I_{\text{disk}} - \text{Median}(I_{\text{disk}})|)$$

### 4.2 Stage 1: Coarse Proposal Network
* **Architecture:** ResNet-34 U-Net operating at an upgraded $1024 \times 1024$ spatial resolution (superseding the early $768 \times 768$ baseline).
* **Objective:** Weighted binary cross-entropy with Soft Dice:
  $$\mathcal{L}_{\text{coarse}} = \text{BCEWithLogitsLoss}(\text{pos\_weight} = 5.0) + (1 - \text{Dice})$$
  Reducing `pos_weight` from $25.0$ to $5.0$ eliminated excessive false-positive proposal bounding boxes over sunspot umbrae.
* **Continuous Probability Upscaling:** Output probabilities are bilinearly interpolated back to $2048 \times 2048$ *prior* to thresholding.
* **Morphological Spine Closing:** A 35-to-45-pixel elliptical structuring element bridges narrow dips along broken filament spines:
  $$\mathcal{M}_{\text{closed}} = (\mathcal{M}_{\text{binary}} \oplus \mathcal{K}_{45}) \ominus \mathcal{K}_{45}$$

### 4.3 Stage 2: Native Crop Fine Network
* **Dual-Channel Conditioning:** Channel 0 provides normalized native-resolution image pixels; Channel 1 supplies the dilated binary coarse proposal mask ($\mathcal{K}_{64}$). Both channels are verified to share identical float $\{0.0, 1.0\}$ scaling.
* **Bounded Dilation Acceleration:** Instead of dilating full $2048 \times 2048$ frames ($4.2\text{ million pixels}$), dilation is computed strictly on the expanded bounding window $\pm 64\text{ pixels}$, yielding bit-identical masks with a $98\%$ compute reduction.
* **Bucket Canvas Snapping:** Crops are centered on canvases snapped to multiples of $32$ ($2^5$ downsampling constraint):
  $$\text{Width Buckets} \in \{160, 288, 576, 928\}, \quad \text{Height Buckets} \in \{160, 256, 448, 800\}$$
* **ROI-Masked Composite Loss:** Both BCE and Dice are restricted strictly to the dilated region $\mathcal{R}$ to avoid penalizing unannotated background:
  $$\mathcal{L}_{\text{fine}} = \frac{\sum (\text{BCE} \odot \mathcal{R})}{\sum \mathcal{R} + \epsilon} + \left[1 - \frac{2 \sum (p \odot y \odot \mathcal{R}) + \epsilon}{\sum (p \odot \mathcal{R}) + \sum (y \odot \mathcal{R}) + \epsilon}\right]$$

### 4.4 Stage 3: Decision-Theoretic Candidate Scorer
* **Model:** `HistGradientBoostingRegressor` (200 iterations, 31 max leaf nodes, 30 min samples per leaf, L2 regularization = 1.0).
* **Target:** Instance-level ground-truth utility:
  $$y_i = \max_{g \in \text{GT}} \text{IoU}(c_i, g) \cdot \mathbf{1}_{\{\text{IoU} > 0.5\}}$$
* **Feature Representation (Base 6 & Augmented 8):**
  1. $\log(\text{Area})$: Natural log of positive pixel area.
  2. $\text{Mean\_Prob}$: Average network confidence within the mask ($\frac{1}{|m|} \sum_{x \in m} P(x)$).
  3. $\text{Max\_Prob}$: Peak localized confidence.
  4. $\text{Std\_Prob}$: Standard deviation of confidence.
  5. $\text{Aspect\_Ratio}$: Bounding box elongation ($\frac{\max(w, h)}{\min(w, h)}$).
  6. $\text{Extent}$: Solidity ratio ($\frac{\text{Area}}{w \times h}$).
  7. $\text{Normalized Radius } r/R_\odot$: Distance from solar disk center ($\frac{\sqrt{(x - x_c)^2 + (y - y_c)^2}}{R_\odot}$).
  8. $\text{Tortuosity}$: Perimeter complexity ratio ($\frac{\text{Perimeter}}{\sqrt{\text{Area}} + 10^{-6}}$).
* **Pruning & Disjoint Cleanup:** Candidates with predicted utility $\hat{u} < \tau$ are pruned. Surviving masks are sorted descending by utility, and pixel conflicts are resolved via greedy first-mask-wins assignment. Any remaining fragment smaller than $\text{min\_area} = 75\text{ pixels}$ is discarded.

---

## 5. Comprehensive Experimental Ledgers

### 5.1 Historical Pipeline Evolution (Fixed 141-Image Validation Split)

$$\begin{array}{lcccccccc}
\hline
\textbf{Milestone / Configuration} & \textbf{Status} & \textbf{PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{Precision} & \textbf{Recall} \\
\hline
\text{Oct 15 Historical Baseline} & \text{[Completed]} & 0.2570 & - & - & - & - & - & - & - \\
\text{Oct 20 Uncalibrated Baseline} & \text{[Completed]} & 0.3220 & 0.6622 & 0.4862 & 1,056 & 1,432 & 800 & 42.44\% & 56.90\% \\
\text{Q0–Q3 Corrected (BCE Broadc. Fixed)} & \text{[Completed]} & 0.3391 & 0.6612 & 0.5128 & 919 & 1,072 & 674 & 46.16\% & 57.69\% \\
\text{+ I3 Bilinear Morph Closing (45px)} & \text{[Completed]} & 0.3500 & - & - & - & - & - & - & - \\
\text{+ I0 Confidence Overlap Cleanup} & \text{[Completed]} & 0.3512 & - & - & - & - & - & - & - \\
\text{+ Coarse 1024 Retraining (pos\_wt=5)} & \text{[Completed]} & 0.3573 & 0.6614 & 0.5403 & 928 & 914 & 665 & 50.38\% & 58.25\% \\
\text{+ S1 Scorer (ResNet-34, } \tau=0.315\text{)} & \text{[Completed]} & 0.3896 & 0.6667 & 0.5843 & 856 & 481 & 737 & 64.02\% & 53.74\% \\
\text{+ 7h Master: EfficientNet-B3 (20 ep)} & \text{[Completed]} & 0.4443 & 0.6842 & 0.6260 & 934 & 218 & 659 & 81.08\% & 58.63\% \\
\textbf{+ Synergistic Stack (Nov 6 Run)} & \textbf{[Completed]} & \mathbf{0.4106} & \mathbf{0.6711} & \mathbf{0.6119} & \mathbf{1,013} & \mathbf{705} & \mathbf{580} & \mathbf{58.96\%} & \mathbf{63.59\%} \\
\textbf{+ Retrained Scorer on OOF Bank} & \textbf{[Completed]} & \mathbf{0.4205} & \mathbf{0.6733} & \mathbf{0.6245} & \mathbf{1,008} & \mathbf{598} & \mathbf{585} & \mathbf{62.76\%} & \mathbf{63.28\%} \\
\hline
\end{array}$$

### 5.2 7-Hour Master Architecture Suite (Quantitative Results)
All models trained for 20 epochs with Cosine Annealing, supervised on 4,323 fine crops with $\{0.0, 1.0\}$ region scaling, evaluated via strict 1-to-1 Hungarian matching:

$$\begin{array}{lcccccccc}
\hline
\textbf{Exp ID} & \textbf{Backbone / Loss} & \textbf{Status} & \textbf{Holdout PQ} & \textbf{Full Val PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{Precision} & \textbf{Recall} & \textbf{Opt } (\tau, \text{area}) \\
\hline
\text{EXP\_1} & \text{ResNet-34 (BCE+Dice)} & \text{[Completed]} & 0.4244 & 0.4375 & \mathbf{0.6847} & 0.6199 & 61.57\% & 62.41\% & (0.45, 20) \\
\text{EXP\_2} & \text{ResNet-34 (Focal+Dice)} & \text{[Completed]} & 0.4232 & 0.4290 & 0.6831 & 0.6195 & \mathbf{64.86\%} & 59.31\% & (0.45, 20) \\
\text{EXP\_3} & \text{ResNet-34 (Lovasz)} & \text{[Completed]} & 0.4233 & 0.4374 & 0.6813 & 0.6213 & 63.67\% & 60.67\% & (0.20, 20) \\
\textbf{EXP\_4} & \textbf{EfficientNet-B3 (BCE+Dice)} & \textbf{[Completed]} & \mathbf{0.4283} & \mathbf{0.4443} & 0.6842 & \mathbf{0.6260} & 62.80\% & 62.41\% & \mathbf{(0.40, 20)} \\
\text{EXP\_5} & \text{ResNet-50 (Focal+Dice)} & \text{[Completed]} & 0.4255 & 0.4368 & 0.6822 & 0.6237 & 61.73\% & \mathbf{63.03\%} & (0.35, 20) \\
\hline
\end{array}$$

### 5.3 Frontier Research Lab Suite (Unexplored Paradigms)
Evaluated empirical solar limb darkening flattening, multi-scale FPN and DeepLabV3+ decoders, and dual-threshold hysteresis:

$$\begin{array}{lcccccccc}
\hline
\textbf{Exp ID} & \textbf{Arch / Backbone} & \textbf{Status} & \textbf{Tune PQ} & \textbf{Holdout PQ} & \textbf{Full Val PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{Precision} & \textbf{Recall} \\
\hline
\textbf{EXP\_1} & \textbf{U-Net (EffNet-B3)} & \textbf{[Completed]} & \mathbf{0.3901} & \mathbf{0.3686} & \mathbf{0.3783} & \mathbf{0.6650} & \mathbf{0.5544} & \mathbf{0.5778} & 53.27\% \\
\text{EXP\_2} & \text{FPN (EffNet-B3)} & \text{[Completed]} & 0.3836 & 0.3643 & 0.3730 & 0.6549 & 0.5562 & 0.5556 & \mathbf{55.68\%} \\
\text{EXP\_4} & \text{Ensemble (U-Net+FPN)} & \text{[Completed]} & 0.3787 & 0.3627 & 0.3700 & 0.6584 & 0.5509 & 0.5614 & 54.08\% \\
\text{EXP\_3} & \text{DeepLabV3+ (ResNet-50)} & \text{[Failed]} & \multicolumn{7}{c}{\text{FAILED: BatchNorm } 1 \times 1 \text{ spatial crash with tail batch size } 1} \\
\hline
\end{array}$$

---

## 6. Comprehensive 15-Experiment Inference Suite (Nov 4 Complete Ledger)

Executed on 141 validation images ($1,593$ GT instances) using cached coarse probability maps. All 48 evaluated parameter configurations are documented below to preserve complete curve data for technical plotting:

$$\begin{array}{llccccccc}
\hline
\textbf{Exp ID} & \textbf{Configuration Tested} & \textbf{Status} & \textbf{PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{TP} & \textbf{FP} & \textbf{FN} \\
\hline
\text{EXP-13\_tau25} & \text{Utility Rejection Threshold } \tau=0.25 & \text{[Completed]} & \mathbf{0.39888} & 0.66544 & \mathbf{0.59942} & 927 & 573 & 666 \\
\text{EXP-13\_tau20} & \text{Utility Rejection Threshold } \tau=0.20 & \text{[Completed]} & 0.39869 & 0.66511 & 0.59943 & 951 & 629 & 642 \\
\text{EXP-13\_tau30} & \text{Utility Rejection Threshold } \tau=0.30 & \text{[Completed]} & 0.39621 & 0.66647 & 0.59449 & 895 & 523 & 698 \\
\text{EXP-09\_utility} & \text{Conflict Resolution (Utility-Ranked)} & \text{[Completed]} & 0.39338 & 0.66681 & 0.58994 & 874 & 496 & 719 \\
\text{EXP-13\_tau32} & \text{Utility Rejection Threshold } \tau=0.32 & \text{[Completed]} & 0.39338 & 0.66681 & 0.58994 & 874 & 496 & 719 \\
\text{EXP-13\_tau35} & \text{Utility Rejection Threshold } \tau=0.35 & \text{[Completed]} & 0.38956 & 0.66759 & 0.58353 & 847 & 463 & 746 \\
\text{EXP-14} & \text{Refitted Scorer (80 Images)} & \text{[Partially Completed]} & 0.38735 & 0.66789 & 0.57996 & 825 & 427 & 768 \\
\text{EXP-03\_median} & \text{8-View TTA (Median Aggregation)} & \text{[Completed]} & 0.38594 & 0.66743 & 0.57825 & \mathbf{1,005} & 878 & \mathbf{588} \\
\text{EXP-07\_ft50} & \text{Fine Threshold } 0.50 & \text{[Completed]} & 0.38487 & 0.66870 & 0.57554 & 1,000 & 882 & 593 \\
\text{EXP-07\_ft55} & \text{Fine Threshold } 0.55 & \text{[Completed]} & 0.38457 & 0.67155 & 0.57266 & 997 & 892 & 596 \\
\text{EXP-03\_harmonic} & \text{8-View TTA (Harmonic Mean)} & \text{[Completed]} & 0.38357 & 0.66763 & 0.57453 & 1,006 & 903 & 587 \\
\text{EXP-02b} & \text{Fine 8-View Full D4 Dihedral TTA} & \text{[Completed]} & 0.38341 & 0.66596 & 0.57572 & 996 & 871 & 597 \\
\text{EXP-13\_tau40} & \text{Utility Rejection Threshold } \tau=0.40 & \text{[Completed]} & 0.38330 & 0.66923 & 0.57275 & 805 & 413 & 788 \\
\text{EXP-03\_trimmed} & \text{8-View TTA (Trimmed Mean)} & \text{[Completed]} & 0.38264 & 0.66607 & 0.57447 & 997 & 881 & 596 \\
\text{EXP-02a} & \text{Fine 4-View Flip TTA} & \text{[Completed]} & 0.38258 & 0.66501 & 0.57530 & 999 & 881 & 594 \\
\text{EXP-08\_area75} & \text{Min Area Cutoff } 75\text{ px} & \text{[Completed]} & 0.38222 & 0.66443 & 0.57526 & 986 & 849 & 607 \\
\text{EXP-07\_ft45} & \text{Fine Threshold } 0.45 & \text{[Completed]} & 0.38067 & 0.66694 & 0.57077 & 994 & 896 & 599 \\
\text{EXP-08\_area50} & \text{Min Area Cutoff } 50\text{ px} & \text{[Completed]} & 0.37967 & 0.66443 & 0.57143 & 986 & 872 & 607 \\
\text{EXP-08\_area40} & \text{Min Area Cutoff } 40\text{ px} & \text{[Completed]} & 0.37891 & 0.66443 & 0.57027 & 986 & 879 & 607 \\
\text{EXP-08\_area30} & \text{Min Area Cutoff } 30\text{ px} & \text{[Completed]} & 0.37770 & 0.66443 & 0.56846 & 986 & 890 & 607 \\
\text{EXP-12} & \text{Collinear Fragment Merging (X4)} & \text{[Completed]} & 0.37734 & 0.66427 & 0.56805 & 985 & 890 & 608 \\
\text{EXP-04\_0.6\_0.35} & \text{Hysteresis Seed } 0.60 \text{ / Boundary } 0.35 & \text{[Completed]} & 0.37717 & 0.66052 & 0.57101 & 983 & 867 & 610 \\
\text{EXP-05\_c34\_k35} & \text{Coarse Thresh } 0.34 \times \text{Kernel } 35 & \text{[Completed]} & 0.37713 & 0.66400 & 0.56797 & 986 & 893 & 607 \\
\text{EXP-05\_c34\_k55} & \text{Coarse Thresh } 0.34 \times \text{Kernel } 55 & \text{[Completed]} & 0.37710 & 0.66264 & 0.56909 & 974 & 856 & 619 \\
\text{EXP-11} & \text{Topological Hole-Fill (<64px) + 1px Open} & \text{[Completed]} & 0.37637 & 0.66476 & 0.56618 & 986 & 904 & 607 \\
\text{EXP-05\_c34\_k45} & \text{Coarse Thresh } 0.34 \times \text{Kernel } 45 & \text{[Completed]} & 0.37605 & 0.66314 & 0.56707 & 985 & 896 & 608 \\
\text{EXP-09\_area} & \text{Conflict Resolution (Area-Ranked)} & \text{[Completed]} & 0.37484 & 0.66444 & 0.56415 & 985 & 914 & 608 \\
\textbf{EXP-01} & \textbf{Baseline (Single Fine, T=0.40, Area=20)} & \textbf{[Completed]} & \mathbf{0.37468} & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-10\_r985} & \text{Limb Radial Cutoff } 0.985 & \text{[Inconclusive]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-07\_ft40} & \text{Fine Threshold } 0.40 & \text{[Completed]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-08\_area20} & \text{Min Area Cutoff } 20\text{ px} & \text{[Completed]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-10\_r970} & \text{Limb Radial Cutoff } 0.970 & \text{[Inconclusive]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-09\_conf} & \text{Conflict Resolution (Confidence)} & \text{[Completed]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-10\_r995} & \text{Limb Radial Cutoff } 0.995 & \text{[Inconclusive]} & 0.37468 & 0.66443 & 0.56391 & 986 & 918 & 607 \\
\text{EXP-05\_c28\_k45} & \text{Coarse Thresh } 0.28 \times \text{Kernel } 45 & \text{[Completed]} & 0.37457 & 0.66359 & 0.56446 & 983 & 907 & 610 \\
\text{EXP-05\_c28\_k35} & \text{Coarse Thresh } 0.28 \times \text{Kernel } 35 & \text{[Completed]} & 0.37408 & 0.66380 & 0.56355 & 991 & 933 & 602 \\
\text{EXP-08\_area15} & \text{Min Area Cutoff } 15\text{ px} & \text{[Completed]} & 0.37382 & 0.66443 & 0.56262 & 986 & 926 & 607 \\
\text{EXP-08\_area10} & \text{Min Area Cutoff } 10\text{ px} & \text{[Completed]} & 0.37255 & 0.66443 & 0.56071 & 986 & 938 & 607 \\
\text{EXP-07\_ft35} & \text{Fine Threshold } 0.35 & \text{[Completed]} & 0.37230 & 0.66052 & 0.56365 & 983 & 912 & 610 \\
\text{EXP-05\_c28\_k55} & \text{Coarse Thresh } 0.28 \times \text{Kernel } 55 & \text{[Completed]} & 0.37159 & 0.66268 & 0.56074 & 967 & 889 & 626 \\
\text{EXP-05\_c22\_k35} & \text{Coarse Thresh } 0.22 \times \text{Kernel } 35 & \text{[Completed]} & 0.37118 & 0.66095 & 0.56158 & 994 & 953 & 599 \\
\text{EXP-04\_0.55\_0.3} & \text{Hysteresis Seed } 0.55 \text{ / Boundary } 0.30 & \text{[Completed]} & 0.36988 & 0.65780 & 0.56230 & 968 & 882 & 625 \\
\text{EXP-05\_c22\_k45} & \text{Coarse Thresh } 0.22 \times \text{Kernel } 45 & \text{[Completed]} & 0.36969 & 0.65962 & 0.56046 & 985 & 937 & 608 \\
\text{EXP-05\_c22\_k55} & \text{Coarse Thresh } 0.22 \times \text{Kernel } 55 & \text{[Completed]} & 0.36898 & 0.65862 & 0.56023 & 979 & 923 & 614 \\
\text{EXP-13\_tau45} & \text{Utility Rejection Threshold } \tau=0.45 & \text{[Completed]} & 0.36764 & \mathbf{0.67359} & 0.54579 & 730 & \mathbf{352} & 863 \\
\text{EXP-07\_ft30} & \text{Fine Threshold } 0.30 & \text{[Completed]} & 0.36669 & 0.65780 & 0.55744 & 968 & 912 & 625 \\
\text{EXP-06} & \text{Multi-Threshold Ladder Proposals} & \text{[Failed]} & 0.36159 & 0.65980 & 0.54803 & 987 & 1,022 & 606 \\
\text{EXP-04\_0.5\_0.25} & \text{Hysteresis Seed } 0.50 \text{ / Boundary } 0.25 & \text{[Completed]} & 0.36119 & 0.65511 & 0.55134 & 945 & 890 & 648 \\
\hline
\end{array}$$

---

## 7. Advanced Research Suite (`15exp2` Deep Diagnostics Ledger — Nov 6)

Executed in 4,350.4 seconds on dual T4 GPUs. This suite generated empirical coordinates for the IEEE technical report:

### 7.1 Synergistic Champion Stacking
Evaluating stacked parameter synergy on the 141-image validation split ($1,593$ GT instances):
* **Baseline Configuration:** Single Fine View, Threshold 0.40, Area 20, $\tau = 0.32$, Conflict: Confidence  
  $$\text{PQ} = 0.3747, \quad \text{SQ} = 0.6644, \quad \text{RQ} = 0.5639, \quad \text{TP} = 986, \quad \text{FP} = 918, \quad \text{FN} = 607$$
* **Synergistic Stack:** 8-View Median TTA, Threshold 0.50, Area 75, $\tau = 0.25$, Conflict: Utility  
  $$\text{PQ} = \mathbf{0.4106}, \quad \text{SQ} = \mathbf{0.6711}, \quad \text{RQ} = \mathbf{0.6119}, \quad \text{TP} = \mathbf{1,013}, \quad \text{FP} = \mathbf{705}, \quad \text{FN} = \mathbf{580}$$
* **Net Improvement:** $\Delta\text{PQ} = +\mathbf{0.0359}$, $\text{FP}$ reduced by $+213$ instances, $\text{TP}$ increased by $+27$.

### 7.2 5-Shard Grouped OOF Candidate Bank Generation
The 566 training physical images were partitioned into 5 disjoint shards ($114, 114, 114, 114, 110\text{ images}$). Coarse proposals and fine crops were extracted across all shards to construct an unbiased candidate bank:
* **Total Candidates Extracted:** $4,843\text{ instances}$ across $566\text{ physical images}$ ($1,363.2\text{ seconds}$).
* **Ground-Truth True Positive Fraction:** $64.86\%$ ($3,141\text{ candidates with } \text{IoU} > 0.5$).
* **Mean Expected Utility:** $\mathbb{E}[u] = 0.4570$.

### 7.3 Scorer Retraining on OOF Bank & Permutation Feature Importance
Retraining the `HistGradientBoostingRegressor` directly on the $4,843$-candidate OOF bank resolved proposal memorization:
* **Base 6-Feature Scorer:** $\text{PQ} = \mathbf{0.4168}, \quad \text{SQ} = 0.6723, \quad \text{RQ} = 0.6200$
* **Augmented 8-Feature Scorer:** $\text{PQ} = \mathbf{0.4205}, \quad \text{SQ} = \mathbf{0.6733}, \quad \text{RQ} = \mathbf{0.6245}$

$$\begin{array}{llcc}
\hline
\textbf{Scorer Model} & \textbf{Candidate Feature Descriptor} & \textbf{Permutation Importance Mean} & \textbf{Importance Std} \\
\hline
\text{Augmented 8} & \text{mean\_prob} & \mathbf{0.740528} & 0.022161 \\
\text{Augmented 8} & \text{log\_area} & 0.041471 & 0.005627 \\
\text{Augmented 8} & \text{max\_prob} & 0.037708 & 0.008355 \\
\text{Augmented 8} & \text{extent} & 0.033971 & 0.008083 \\
\text{Augmented 8} & \text{std\_prob} & 0.023592 & 0.009833 \\
\text{Augmented 8} & \text{tortuosity} & 0.020084 & 0.005074 \\
\text{Augmented 8} & \text{aspect\_ratio} & 0.011558 & 0.006430 \\
\text{Augmented 8} & \text{norm\_radius } (r/R_\odot) & \mathbf{-0.005978} & 0.012966 \\
\hline
\text{Base 6} & \text{mean\_prob} & \mathbf{0.829507} & 0.030083 \\
\text{Base 6} & \text{log\_area} & 0.053279 & 0.005241 \\
\text{Base 6} & \text{extent} & 0.053223 & 0.013376 \\
\text{Base 6} & \text{max\_prob} & 0.039195 & 0.009264 \\
\text{Base 6} & \text{std\_prob} & 0.038062 & 0.008476 \\
\text{Base 6} & \text{aspect\_ratio} & 0.025248 & 0.007950 \\
\hline
\end{array}$$

### 7.4 Algorithmic False Negative (FN) Bottleneck Tracing
Tracing every unmatched validation ground-truth filament ($1,593\text{ instances total}$) through the pipeline stages isolated the exact attrition points:

$$\begin{array}{llcc}
\hline
\textbf{Pipeline Stage} & \textbf{Failure Mechanism Description} & \textbf{Instances Lost} & \textbf{Share of Misses} \\
\hline
\text{Matched True Positive} & \text{Successfully matched with } \text{IoU} > 0.5 & 1,049 & - \\
\text{Stage A: Coarse Proposal} & \text{Zero overlapping coarse components extracted} & 228 & 41.91\% \\
\text{Stage B: Crop Truncation} & \text{Overlap exists, but cropped by bucket limits } (<35\%) & 4 & 0.74\% \\
\textbf{Stage C: Fine Threshold} & \textbf{Fine probability never cracked } T=0.50 \text{ threshold} & \mathbf{302} & \mathbf{55.51\%} \\
\text{Stage D: Scorer Pruning} & \text{Candidate segmented, but rejected by } \tau < 0.25 & 10 & 1.84\% \\
\text{Stage E: Overlap Pruning} & \text{Pruned by conflict resolution or minimum area floor} & 0 & 0.00\% \\
\hline
\textbf{Total False Negatives} & \text{All unmatched ground-truth instances} & \mathbf{544} & 100.0\% \\
\hline
\end{array}$$

### 7.5 Speed vs. Accuracy Pareto Frontier (TTA Benchmarking)
Quantifying inference latency per image against validation Panoptic Quality:

$$\begin{array}{lcccccc}
\hline
\textbf{Configuration Name} & \textbf{TTA Views} & \textbf{Latency (ms/img)} & \textbf{PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{TP / FP / FN} \\
\hline
\text{Single\_View\_T40\_A20} & 1\text{ view} & \mathbf{360.9\text{ ms}} & 0.3989 & 0.6654 & 0.5994 & 927 / 573 / 666 \\
\text{Single\_View\_T50\_A75} & 1\text{ view} & 376.6\text{ ms} & 0.4023 & 0.6691 & 0.6013 & 946 / 548 / 647 \\
\textbf{4View\_Flip\_T50\_A75} & \mathbf{4\text{ views}} & \mathbf{569.8\text{ ms}} & \mathbf{0.4107} & 0.6708 & \mathbf{0.6123} & 1,013 / 703 / 580 \\
\text{8View\_D4\_Median\_T50\_A75} & 8\text{ views} & 850.4\text{ ms} & 0.4106 & 0.6711 & 0.6119 & 1,013 / 705 / 580 \\
\text{8View\_D4\_Trimmed\_T50\_A75} & 8\text{ views} & 842.3\text{ ms} & 0.4089 & \mathbf{0.6716} & 0.6089 & 1,012 / 708 / 581 \\
\hline
\end{array}$$

### 7.6 Collinear Fragment Merging (X4) 2D Grid Search
Sweeping endpoint proximity distance ($10, 15, 20, 30\text{ px}$) and orientation alignment ($15^\circ, 20^\circ, 30^\circ$) to reconnect fractured spines:

$$\begin{array}{ccccccc}
\hline
\textbf{Max Distance (px)} & \textbf{Max Angle (deg)} & \textbf{PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{TP} & \textbf{FP} \\
\hline
\mathbf{20\text{ px}} & \mathbf{30^\circ} & \mathbf{0.41095} & 0.67105 & \mathbf{0.61241} & 1,012 & \mathbf{700} \\
15\text{ px} & 30^\circ & 0.41095 & 0.67105 & 0.61241 & 1,012 & 700 \\
30\text{ px} & 30^\circ & 0.41095 & 0.67105 & 0.61241 & 1,012 & 700 \\
15\text{ px} & 20^\circ & 0.41076 & 0.67108 & 0.61208 & 1,013 & 704 \\
20\text{ px} & 20^\circ & 0.41076 & 0.67108 & 0.61208 & 1,013 & 704 \\
30\text{ px} & 20^\circ & 0.41076 & 0.67108 & 0.61208 & 1,013 & 704 \\
10\text{ px} & 30^\circ & 0.41073 & 0.67084 & 0.61227 & 1,013 & 703 \\
10\text{ px} & 15^\circ & 0.41063 & 0.67108 & 0.61190 & 1,013 & 705 \\
10\text{ px} & 20^\circ & 0.41063 & 0.67108 & 0.61190 & 1,013 & 705 \\
15\text{ px} & 15^\circ & 0.41063 & 0.67108 & 0.61190 & 1,013 & 705 \\
20\text{ px} & 15^\circ & 0.41063 & 0.67108 & 0.61190 & 1,013 & 705 \\
30\text{ px} & 15^\circ & 0.41063 & 0.67108 & 0.61190 & 1,013 & 705 \\
\hline
\end{array}$$

### 7.7 Annotator-Specific Stratified Performance Readout
Evaluating the champion model independently across human annotator subsets ($N \ge 15\text{ GT}$):

$$\begin{array}{ccccccc}
\hline
\textbf{Annotator ID} & \textbf{GT Instances} & \textbf{PQ} & \textbf{SQ} & \textbf{RQ} & \textbf{Precision} & \textbf{Recall} \\
\hline
010302 & 16 & \mathbf{0.53126} & 0.69063 & \mathbf{0.76923} & 65.22\% & \mathbf{93.75\%} \\
050303 & 46 & 0.48892 & 0.66463 & 0.73563 & \mathbf{78.05\%} & 69.57\% \\
010402 & 115 & 0.48058 & 0.70886 & 0.67797 & 66.12\% & 69.57\% \\
020403 & 42 & 0.46961 & 0.66654 & 0.70455 & 67.39\% & 73.81\% \\
050102 & 54 & 0.46806 & \mathbf{0.71380} & 0.65574 & 58.82\% & 74.07\% \\
050101 & 51 & 0.45411 & 0.71105 & 0.63866 & 55.88\% & 74.51\% \\
050103 & 60 & 0.45162 & 0.65690 & 0.68750 & 64.71\% & 73.33\% \\
020101 & 80 & 0.44263 & 0.70513 & 0.62774 & 75.44\% & 53.75\% \\
010403 & 91 & 0.43216 & 0.66390 & 0.65094 & 57.02\% & 75.82\% \\
060403 & 80 & 0.42760 & 0.69485 & 0.61538 & 63.16\% & 60.00\% \\
\hline
\end{array}$$

### 7.8 Dynamic Limb Area Scaling Evaluation
Formulating an adaptive radial area floor $A_{\text{min}}(r) = 60\text{ px}$ for $r/R_\odot \le 0.85$, scaling linearly up to $125\text{ px}$ at the extreme solar limb ($r/R_\odot = 1.0$):
* $\text{PQ} = \mathbf{0.41073}, \quad \text{SQ} = 0.67079, \quad \text{RQ} = \mathbf{0.61231}$
* Successfully suppressed False Positives from $705$ down to $662$ without eroding True Positives.

---

## 8. Deep Analytical Takeaways: What Works vs. What Fails

### 8.1 Encoder & Decoder Architecture
* **EfficientNet-B3 is the Premier Encoder:** Compound scaling of depth, width, and resolution enables EfficientNet-B3 to trace thin, high-aspect-ratio filament spines significantly better than ResNet-34 ($+0.05 \text{ PQ}$ lift).
* **ResNet-50 Maximizes Raw Recall:** ResNet-50 achieved the highest recall ($63.03\%$) in the 7-hour suite, successfully bridging long quiescent filaments that smaller backbones fragmented.
* **U-Net Decoders Outperform FPN & DeepLabV3+:** The direct high-resolution skip connections of standard U-Net preserve sharp filament boundary masks better than FPN lateral pyramids ($\text{SQ } 0.6650 \text{ vs } 0.6549$). DeepLabV3+ Atrous Spatial Pyramid Pooling (ASPP) suffers severe batch size sensitivity on variable crop dimensions.

### 8.2 Loss Formulation Dynamics
* **BCE + Soft Dice Dominates:** Soft Dice directly optimizes the global intersection-over-union surrogate, smoothing gradient backpropagation across fragmented structures.
* **Focal Loss Destroys Recall:** Focal Loss ($\gamma = 2.0, \alpha = 0.75$) increased precision to $64.86\%$, but crippled recall from $62.41\%$ down to $59.31\%$. It causes networks to under-predict low-contrast filament boundaries near active regions.
* **Lovasz-Hinge Fails on Sub-Pixel Spines:** Lovasz loss achieved $\text{SQ} = 0.6813$, failing to exceed standard Soft Dice ($\text{SQ} = 0.6847$). Its piecewise linear surrogate exhibits high gradient variance on disconnected ribbons.

### 8.3 Post-Processing & Preprocessing Findings
* **The Minimum Area Inversion ($75\text{ px}$ vs $20\text{ px}$):** EXP-08 proved that raising the minimum area cutoff from $20\text{ px}$ to $75\text{ px}$ steadily increases PQ ($0.3747 \to 0.3822$). Pruning sub-75px fragments eliminated $69\text{ FP}$ instances while losing zero TP matches.
* **EfficientNet-B3 Calibrates at $T=0.50$:** Lowering the fine binarization threshold to $0.40$ (intended to offset TTA smoothing) proved suboptimal ($0.3747\text{ PQ}$). EfficientNet-B3 outputs highly sharp, confident probabilities, peaking at $T=0.50$ ($0.3849\text{ PQ}$).
* **Utility-Ranked Conflict Resolution Crushes Raw Confidence:** Resolving pixel conflicts among overlapping candidates based on predicted Scorer Utility ($u \ge \tau$) yielded $\text{PQ} = 0.3934$, vastly outperforming raw mean confidence ($\text{PQ} = 0.3747$).
* **Physical Limb Flattening Degrades Performance:** Empirical radial flattening:
  $$I_{\text{flat}} = \frac{I}{1 - u(1 - \sqrt{1 - (r/R_\odot)^2})}$$
  amplified high-frequency noise near the limb and distorted median/MAD statistics, dropping Holdout PQ from $0.4283$ down to $0.3686$.
* **Dual-Threshold Hysteresis is Ineffective:** Requiring a core seed ($T_{\text{high}} \ge 0.55$) discarded faint ribbons whose probabilities hovered around $0.35\text{--}0.45$, eroding detection recall down to $55.1\%$.

---

## 9. Forensic Breakdown: Local Validation vs. Public Leaderboard (0.35)

The trajectory from the early $0.29\text{--}0.31$ LB baseline to the verified **$0.35$ Public Leaderboard score** confirms that local validation improvements reliably generalize:
1. **Coarse Proposal Gating Bottleneck ($41.9\%$ of misses):** The FN trace isolated that Stage 1 misses $228\text{ ground-truth filaments}$ completely. Because Fine refinement is conditioned strictly on coarse dilated crops, this acts as an absolute recall ceiling.
2. **Fine Threshold Erosion ($55.5\%$ of misses):** A dominant $302\text{ filaments}$ are present inside the fine crop window, but the fine model's output probabilities fail to exceed $0.50$.
3. **Inter-Annotator Style Floor:** Validating against Annotator `010403` yields $\text{PQ} = 0.432$, whereas validating against Annotator `010302` yields $\text{PQ} = 0.531$. If the hidden test set is evaluated against a conservative single-annotator pass, a local score of $0.42$ naturally translates to $\sim 0.35\text{--}0.37$ on the public board.
4. **Spine Snapping Mitigation:** The X4 Collinear merge ($20\text{ px}, 30^\circ$) eliminates the splitting penalty where a single true filament is cut into multiple fragments, preventing severe $1\text{ TP} + 2\text{ FP}$ score degradation.

---

## 10. Analysis of Human Annotation Ambiguity (A3 Audit)

An A3 pairwise audit evaluated consensus across **$296$ multiply-annotated physical images** ($1,196$ directed comparisons) using Hungarian matching at $\text{IoU} > 0.5$:
$$\text{Pooled Agreement:} \quad \text{PQ} = \mathbf{0.3397}, \quad \text{SQ} = \mathbf{0.6336}, \quad \text{RQ} = \mathbf{0.5361}, \quad \text{TP} = 4,288, \quad \text{FP/FN} = 3,711$$

### Analytical Findings
* **Count Disagreement:** Annotators agreed on the exact count of filaments in only **$21.2\%$** of comparisons (mean absolute count difference: **$2.05$ objects**).
* **Boundary Variability:** Even when both experts matched the exact same filament instance, average spatial overlap was bounded at **$\text{SQ} = 0.6336$**.
* **Statistical Meaning:** The model's validation score of **$\text{PQ} = 0.4205$** significantly surpasses the human pairwise consensus ceiling ($0.3397$). The network learns consistent topological features shared across expert passes rather than overfitting to individual subjective styles.

---

## 11. Engineering Post-Mortem: Bugs Identified & Resolved

1. **Fine-Loss Broadcasting Bug (Resolved in Q3):**
   * *Bug:* Multiplying predictions $[B, 1, H, W]$ by ROI masks $[B, H, W]$ silently broadcasted tensors to $[B, B, H, W]$, computing cross-sample loss penalties.
   * *Fix:* Enforced explicit `.unsqueeze(1)` and shape assertion checks on all inputs.
2. **Double Division / Region Scaling Parity (Resolved in Q0):**
   * *Bug:* Fine crop regions were divided by $255.0$ twice, resulting in region values of $\{0.0, 0.00392\}$ during training vs $\{0.0, 1.0\}$ during inference.
   * *Fix:* Enforced strict `(reg > 0).astype(np.float32)` float parity across all loaders and inference routines.
3. **Submission Overlapping Masks Rejection (Resolved in Q2):**
   * *Bug:* Separate coarse crops produced overlapping pixel masks, causing Kaggle submission rejection.
   * *Fix:* Implemented utility-ranked greedy first-mask-wins cleanup with minimum area pruning ($\ge 75\text{ px}$).
4. **BatchNorm Tail-Batch Crash (Resolved in 7h Suite):**
   * *Bug:* Dynamic size buckets with a sample count of $N \equiv 1 \pmod{\text{batch\_size}}$ passed a single sample into `.train()` mode, crashing BatchNorm on zero spatial variance.
   * *Fix:* Configured `drop_last = (len(items) % batch_size == 1)` and duplicated single-sample buckets.
5. **Ensemble Slicing Numpy Broadcasting Error (Resolved in Frontier Lab):**
   * *Bug:* `ensemble_infer()` erroneously returned the full padded canvas `reg_c` $(256, 288)$ instead of the sliced crop `r_crop` $(165, 226)$, crashing `(prob_crop > thresh) & (reg_crop > 0)`.
   * *Fix:* Sliced both arrays strictly to crop bounding dimensions $(crop\_h, crop\_w)$ before returning.
6. **DeepLabV3+ ASPP Global Average Pooling Crash (Resolved):**
   * *Bug:* DeepLabV3+ ASPP global pooling collapses feature maps to $1 \times 1$. A batch size of 1 leaves exactly 1 value per channel, crashing BatchNorm.
   * *Fix:* Enforced a strict minimum batch size floor of 2 on all ASPP-based architectures.

---

## 12. Blueprint for Champion Master Pipeline

To replicate the current peak competitive pipeline ($0.4205\text{ Val} \to 0.35\text{ LB}$):

1. **Stage 1 (Coarse Proposal):**
   * ResNet-34 U-Net at $1024 \times 1024$ native downsampling.
   * Preprocessing: Disk-restricted $(I - \text{median}) / \text{MAD}$ normalization.
   * Inference: 4-View Batched TTA $\to$ Bilinear upsample to $2048 \times 2048 \to$ Threshold $0.30 \to$ Morphological closing kernel $45\text{ px}$ ellipse $\to$ Area floor $20\text{ px}$.
2. **Stage 2 (Fine Crop Refiner):**
   * EfficientNet-B3 U-Net trained with ImageNet weights.
   * Inputs: 2-Channel ($2048 \times 2048$ native image crop + bounded $64\text{ px}$ dilated coarse mask).
   * Inference: 4-View Flip TTA (570 ms/image Pareto optimum) $\to$ Fine Threshold $T = 0.50$.
3. **Stage 3 (Selection & Post-Processing):**
   * Feature Extraction: Augmented 8-Feature descriptor set.
   * Regressor: `HistGradientBoostingRegressor` trained on 4,843-instance 5-shard OOF candidate bank.
   * Rejection Threshold: $\tau = 0.25$ (Dinkelbach utility criterion).
   * Overlap Resolution: Utility-ranked greedy assignment $\to$ Minimum prediction area floor $75\text{ px}$.
   * Fragment Healing: Collinear merging with $d \le 20\text{ px}$ and $\theta \le 30^\circ$.

---

## 13. High-Priority Research Roadmap (Next ML Action)

The algorithmic bottleneck tracing in Section 7.4 established that **$55.5\%$ of all missed filaments (302 instances)** are lost directly to **Fine Threshold Erosion** (the filament is inside the crop, but Stage 2 under-predicts confidence).

### Targeted Solutions:
1. **Skeleton-Weighted Thin-Spine Loss:**
   * Modify `fine_criterion` during fine retraining to compute a 1-pixel morphological skeleton of the ground-truth mask:
     $$\mathcal{L}_{\text{fine}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} + \lambda \cdot \mathcal{L}_{\text{BCE}}(\text{pred} \odot \text{Skeleton}, \text{target} \odot \text{Skeleton})$$
   * Heavy loss weighting ($\lambda = 3.0$) on sub-pixel spines will force the network to produce peak probabilities $\ge 0.50$ along faint ribbons.
2. **Coarse Proposal Recall Expansion (Targeting the 228 Stage A Misses):**
   * Lower coarse proposal extraction threshold from $0.30$ to $0.22$, expanding coarse candidate recall while relying on the retrained OOF Scorer ($\tau = 0.25$) to prune the resulting false-positive proposals.
3. **Rotated Elliptical Crop Alignment (Roadmap R3):**
   * Replace isotropic square buckets with oriented bounding crops aligned to the coarse component's primary eigenvector ($384 \times 128$), providing higher contextual resolution along elongated spines.

---

## 14. Artifact Inventory & Checkpoint Registry

$$\begin{array}{llc}
\hline
\textbf{Artifact Description} & \textbf{File Path} & \textbf{Verified Properties} \\
\hline
\text{Coarse Checkpoint (1024, pos\_wt=5)} & \texttt{/kaggle/input/.../coarse\_model\_3.pth} & \text{ResNet-34 State Dict} \\
\text{Fine Checkpoint (EffNet-B3, 20 ep)} & \texttt{/kaggle/input/.../EXP\_4\_EfficientNetB3\_BCE\_Dice.pth} & \text{EfficientNet-B3 State Dict} \\
\text{Pretrained Scorer Bundle} & \texttt{/kaggle/input/.../scorer\_3.pkl} & \text{Base 6-Feature Regressor} \\
\text{Physical Data Split JSON} & \texttt{/kaggle/working/split.json} & \texttt{1d0e7f8fc5bfaba91a5cd68b5d759d91deaa75370a78bfb7e6016f4b2e230051} \\
\text{5-Shard OOF Candidate Bank} & \texttt{/kaggle/working/grouped_candidate_bank_5shard.csv} & 4,843\text{ samples, 64.86\% TP} \\
\text{Scorer Feature Importance} & \texttt{/kaggle/working/scorer_feature_importance.csv} & \text{Permutation Importances} \\
\text{FN Bottleneck Summary} & \texttt{/kaggle/working/fn_bottleneck_attrition_summary.csv} & 544\text{ Attributed Misses} \\
\text{Pareto Speed vs. Accuracy} & \texttt{/kaggle/working/pareto_speed_vs_accuracy.csv} & 5\text{ Benchmark Configurations} \\
\text{Collinear 2D Grid Search} & \texttt{/kaggle/working/collinear_merge_grid_search.csv} & 12\text{ Sweep Points} \\
\text{Annotator Stratification} & \texttt{/kaggle/working/annotator_performance_stratification.csv} & 10\text{ Evaluated Annotators} \\
\textbf{Champion Submission CSV} & \texttt{/kaggle/working/submission.csv} & \mathbf{1,376\text{ Disjoint Masks (LB 0.35)}} \\
\hline
\end{array}$$

### Submission Verification Checklist
The exported `submission.csv` was audited prior to leaderboard submission:
* Schema: Exactly two columns: `filament_id` and `segmentation_rle`.
* Non-Empty: Zero missing IDs, zero empty RLE strings, zero nulls.
* Disjointness: Re-read from disk, fully decoded to $(2048, 2048)$ boolean arrays, asserting strictly zero pixel intersection across instances within every physical observation ($\sum \text{overlap} = 0$).
* Coverage: Non-zero predictions generated for $177 \text{ of } 180$ test observations ($3\text{ zero-prediction images}$ legitimately empty of filament absorption).

---

## 14. Cross-Pipeline Meta-Analysis & Pattern Synthesis

**Status: documentation-only synthesis; no new experiment, notebook execution, training, inference, or submission was performed for this section.** The preceding report is preserved as a historical ledger, including statements qualified below. This requested section number duplicates the existing **14. Artifact Inventory & Checkpoint Registry**; references below distinguish the two by title. The actionable successor plan is [solarroadmap_next_generation.md](solarroadmap_next_generation.md); the original roadmap remains a historical reference.

### 14.1 Evidence hierarchy and central judgment

Three evidence classes must not be conflated:

1. **Reported measurements:** this dossier §§5–7, `progress5.md`, and `progress3.md`. These establish what was recorded, not automatically the complete provenance or mutual comparability of every row.
2. **Source-established behavior:** static inspection of `magfilo_advanced_research_and_oof_suite.ipynb`, especially cells 6 and 9–16, and targeted inspection of `magfilo_15_experiments_inference_suite.ipynb` cell 12, **using zero-based JSON cell indices**. The accessible advanced notebook has no retained execution outputs; source can establish implementation behavior, but cannot prove that every historical result came from those exact source bytes.
3. **Transfer hypotheses:** participant methods documented in `notebook_comparisons.md`, corrected and expanded by `solar_research_review.md` §§5.2–5.7. Participant improvements belong to their own splits, models and evaluation procedures. No cross-participant score establishes an isolated causal benefit in this pipeline.

**Central judgment:** the strongest next direction is to improve the *availability and selection of eligible instance extents*, not to force all faint pixels above 0.50. Two inexpensive prerequisites dominate the value of another architecture search: propagate the selected scorer explicitly into every final path, and align its target with annotation-record evaluation. Then test fine-map component hierarchies and genuinely complementary coarse proposals. Retraining is justified only after determining whether missing instances are absent from support, absent from the candidate family, or merely misranked.

The reported **public LB 0.35** and latest operational **local PQ 0.4205** remain separate observations. This review does not establish an LB ceiling, a stable local-to-LB conversion, or a guaranteed improvement over either result.

### 14.2 Reconciliation before interpreting gains

#### 14.2.1 Five processing shards are not five-fold neural OOF

Cell 10 explicitly states: **“checkpoints are fixed across shards; this is not strict model-level OOF inference.”** Cell 6 loads one coarse ResNet34 and one fine EfficientNet-B3; partitioning inference over 566 observations does not make either model exclude the processed observations from training.

Therefore, the 4,843-row bank is a **grouped, fixed-checkpoint training-side candidate bank**, not demonstrated unbiased cascade OOF. Its 3,141 positive candidates, 64.86% positive fraction and mean recorded utility 0.4570 describe that bank. They do not prove proposal memorization was resolved. The bank CSV contains features and targets, not the soft probability/support fields needed to replay new segmentation families.

Cell 11 does use a physical-image `GroupShuffleSplit` with 20% held out for its separate permutation-importance models. That is legitimate **scorer-level** separation; it does not retroactively exclude those images from the neural checkpoints. Affordable five-way cross-fitting of the scorer should be retained, explicitly labeled as scorer-only cross-fitting. End-to-end generalization can still be evaluated on the 141-image outer development holdout **if** every learned stage and training-side artifact excludes it; checkpoint membership must be established rather than inferred from a saved split file.

#### 14.2.2 The reported champion is not one demonstrated assembled configuration

Cell 11 explicitly evaluates `retrained_scorer_base` and `retrained_scorer_aug` at eight-view median, fine threshold 0.50, area 75, tau 0.25 and utility-ranked conflicts. These are the source configurations associated with reported 0.4168 and 0.4205, respectively.

However:

- It does **not** assign either retrained scorer back to `candidate_scorer`.
- Cell 9 defines `run_evaluation_pass(..., scorer=candidate_scorer)`. Python captures that default object when the function is defined.
- The FN trace and later default-scored Pareto/merge evaluations use the previously loaded six-feature scorer.
- Cell 16 submission inference likewise uses the loaded scorer, default six features and eight-view median; it does not apply the collinear-merge sweep winner.

Consequently, §§7.3, 7.4, 7.5 and the §12 blueprint cannot be assembled into a demonstrated **0.4205 / four-view / merged / 570-ms** champion. The individual findings remain useful, but joint gains and exact LB artifact provenance require an explicit configuration-to-artifact manifest and one replay. This is a deployment-consistency issue, not a reason to delete historical outputs.

#### 14.2.3 Internal arithmetic flags require original per-record metric parts

For a single pooled calculation, `PQ = SQ × RQ` and `RQ = 2TP / (2TP + FP + FN)`. Separately averaged macro quantities need not obey both identities. The following flags therefore require a documented aggregation or run correction, **not invented replacement results**:

| Historical entry | Arithmetic / provenance check | Consequence |
|---|---|---|
| §5.1 Oct 20 | 1,056 TP + 800 FN = 1,856 GT, not the later 1,593 | Do not subtract its counts from the Nov 6 regime as a controlled ablation. |
| §5.1 EfficientNet-B3 | 0.6842 × 0.6260 = 0.42831, matching §5.2 holdout PQ, not 0.4443 full validation PQ | The SQ/RQ columns likely belong to the holdout protocol; confirm original outputs. |
| Same row, counts | 934/218/659 imply pooled RQ ≈0.68051, not 0.6260 | Those counts and metrics cannot describe one pooled result without another aggregation explanation. |
| Latest scorer row | 0.6733 × 0.6245 ≈0.42048, but 1,008/598/585 imply pooled RQ ≈0.63020 | Keep 0.4205 as reported; recover per-record numerator/denominator before precise gain accounting. |
| §7.5 trimmed TTA | 1,012/708/581 imply pooled RQ ≈0.61093, not 0.6089 | Treat small differences cautiously; distinguish count mismatch from rounding. |
| §7.8 radial filter | Holding TP=1,013, FN=580 and setting FP=662 implies RQ≈0.61995, not 0.61231 | “43 fewer FP with no TP lost” is not reconciled with the printed metrics. Source also uses a maximum area of 120, not the reported 125. |

The matched architecture comparison supports a **small positive B3 advantage**: +0.0039 holdout PQ (0.4283 versus 0.4244) and +0.0068 full-validation PQ (0.4443 versus 0.4375), not an isolated +0.05 encoder effect. The latest 0.4205 is not the highest historical reported local number; the differing recipes and protocols prevent either a clean regression claim or a clean cumulative improvement claim.

### 14.3 What the FN trace actually establishes

The recorded 544 misses are partitioned as follows:

| Trace class | Count | Share of 544 attributed misses | Share of 1,593 GT |
|---|---:|---:|---:|
| A: no raw coarse-component overlap | 228 | 41.91% | 14.31% |
| B: union support below 35% | 4 | 0.74% | 0.25% |
| C: no eligible thresholded fine component | 302 | 55.51% | 18.96% |
| D: candidate fails scorer threshold | 10 | 1.84% | 0.63% |
| E: final cleanup loses eligible candidate | 0 | 0% | 0% |

**The source does not establish that all 302 Stage C instances have maximum fine probability below 0.50.** Cell 12 assigns `Fine_Threshold_or_Component_Erosion` when no existing fine candidate achieves IoU strictly above 0.5. This includes undersegmentation, overgrowth, fragmentation into individually insufficient components, inadequate single-crop support, candidate-area rejection, and GT rasterization differences. High-confidence pixels may be present throughout such a failure.

There are four further attribution limitations:

1. **Raw proposal overlap is not actual fine support.** A dilated region can reach a GT even when the undilated coarse component has zero overlap. Conversely, an overlapping component may not provide a usable single crop.
2. **Union-of-crops support is optimistic.** For GT instance `g`, let `S_k` be the actual retained emission support of crop `k`. Define

   $$U_{\mathrm{support}}(g)=\max_k\frac{|g\cap S_k|}{|g|}.$$

   For a prediction constrained to one such support, IoU cannot exceed this quantity. A value ≤0.5 makes a strict eligible match impossible under that constraint. The trace instead thresholds **union** support at 0.35; cases between 0.35 and 0.50, or distributed across several insufficient crops, can be mislabeled Stage C. Even `U_support > 0.5` is only an upper bound: connectivity, area, neighboring objects and overlap cleanup may prevent achieving it.
3. **Existential matches are not final one-to-one matches.** The trace tests each GT independently. Two GT instances can be credited using the same prediction, unlike the final assignment. The trace reports 1,049 matched + 544 missed, versus 1,013/580 in the stack and 1,008/585 in the latest scorer row. Those are not interchangeable denominators or exact decompositions of one final run.
4. **GT and scorer differ.** Trace GT uses OpenCV rasterization and the loaded legacy scorer, whereas PQ evaluation uses COCO masks and the augmented-scorer comparison passes a different scorer explicitly.

**Required revised diagnosis:** start with the final one-to-one FN set for one explicit configuration, then report overlapping mechanism flags and controlled counterfactuals. For each FN save maximum and quantiles of fine probability on GT, maximum per-crop support, best per-threshold component IoU, best union IoU within a crop, scorer rejection, and conflict/area losses. Label GT-assisted oracle recoveries as *diagnostic upper bounds*, never deployable recall estimates. The 55.5% and 41.9% are priority signals from this trace, not independently recoverable fractions of final FN.

### 14.4 The unexploited target mismatch: best annotator versus annotation population

Cell 10 combines instances from all annotation records for an observation and sets candidate utility to its **largest eligible IoU anywhere in that combined list**. Thus a candidate matching one annotator perfectly but omitted or incompatible in every other record can receive utility near one. Its GT masks also use the OpenCV `polygon_to_mask` helper from cell 4, while evaluation GT uses COCO rasterization. The target-population ablation must compare max and mean labels on the **same** rasterizer; report any separate OpenCV-max → COCO-max change rather than crediting both effects to averaging.

For candidate `c` and annotation record `r`, define

$$b_r(c)=\max_{g\in G_r}\operatorname{IoU}(c,g),\qquad
u_r(c)=b_r(c)\,\mathbf1[b_r(c)>0.5],$$

with `b_r=0` for an empty GT set. The implemented target is effectively

$$u_{\max}(c)=\max_r u_r(c).$$

For equally weighted annotation records, the expected eligible-match contribution is instead

$$\bar u(c)=\frac1{R_i}\sum_{r=1}^{R_i}u_r(c).$$

Records that omit the object contribute **zero**. For nonuniform evaluation weights use the corresponding weighted mean. A candidate row representing all records of physical observation `i` should carry total record weight `R_i` when fitting to a record-pooled objective; verify the actual aggregation before adopting this weighting. Macro-image and macro-record objectives imply different weighting policies.

The exact population factorization is

$$h(c)=\frac1{R_i}\sum_r\mathbf1[u_r(c)>0],\qquad
m(c)=\mathbb E_r[u_r(c)\mid u_r(c)>0],\qquad
\bar u(c)=h(c)m(c).$$

Set the product to zero when `h=0`; fit the conditional-IoU head only on positive eligible records, with appropriate positive-record weights. **Do not average `m` over all records and then multiply by `h` again.** Here `h` is annotation-match frequency, not identifiable physical existence. Omissions and extent disagreement can both reduce it.

This connects directly to earlier Tushar's annotator-mean utility and later Tushar's hurdle scorer (`solar_research_review.md` §§5.3, 5.7). It is a **source-established target mismatch and an unexecuted improvement hypothesis**, not proof that relabeling must improve PQ. Direct mean-utility regression is the first control; `h×m` is a subsequent ablation, not an automatically superior model.

Why prioritize it when Stage D contains only ten misses? **Because FP control is not measured by an FN-only breakdown.** The latest row still reports 598 FP. Better record-aligned utility can improve RQ through rejection and conflict ordering, and is necessary to distinguish useful faint extents from the duplicates generated by rescue branches. It cannot recover a genuinely absent candidate. Its main failure risk is training-side neural memorization or a mismatch between observed annotator populations and hidden evaluation; use fixed physical scorer folds and independent outer evaluation where provenance permits.

**A distinct annotation-to-erosion mechanism:** for an associated instance let `a_r` indicate that record `r` draws it, `e=mean_r a_r`, and `M_r(x)` be its drawn mask. Then

$$Y_{\mathrm{uncond}}(x)=\frac{\sum_r a_rM_r(x)}{R_i}
=e\,Y_{\mathrm{cond}}(x),\qquad
Y_{\mathrm{cond}}(x)=\frac{\sum_r a_rM_r(x)}{\sum_r a_r}.$$

Under a BCE-only fit to this soft target, the pixelwise optimum is the target probability; if `e≤0.5`, even a consistently drawn interior can fail a strict 0.50 foreground threshold. This is a mathematical attenuation mechanism, **not an established explanation for these 302 failures**: current loader sampling must be audited, Dice alters the optimum, and independently sampled binary annotations need not behave exactly like the finite soft map under the full loss. It explains why later Tushar separates conditional extent from inclusion. Drawing frequency `e` is not the same as eligible-match frequency `h`, and neither identifies physical existence. N7 tests this target mechanism after N1 makes inclusion scoring explicit.

### 14.5 PQ mathematics favors structured alternatives, not indiscriminate recall

For pooled counts let `S` be the sum of matched IoUs, `P` the number of predictions and `G` the GT count. Then

$$\mathrm{PQ}=\frac{2S}{P+G}.$$

For an isolated additional prediction that changes matched-IoU sum by `ΔS` and increases `P` by one, PQ improves precisely when

$$\Delta S>\frac{\mathrm{PQ}}2.$$

Under repeated-record evaluation, adding one physical prediction adds `R_i` scored predictions; use average incremental utility across its records or explicitly include that cost. At reported PQ 0.4205, the idealized marginal threshold is approximately 0.21025. The empirically selected tau 0.25 is **not proven suboptimal**: learned scores are imperfect, duplicates compete, and cleanup changes masks and assignments. Tune tau on the permitted calibration side after target changes.

The operative quantity is **incremental matched-IoU contribution**, not independent maximum-IoU per candidate. Two fragments can both score against the same GT but cannot both receive TP credit. A useful rescue can increase RQ and lower conditional SQ by adding just-over-0.5 matches; a useful extent refinement may increase both. Neither higher peak probability nor more candidates alone establishes PQ improvement.

For a valid laminar component hierarchy and fixed tau, consider

$$D(v)=\max\left(0,\widehat u(v)-\tau,\sum_{w\in\mathrm{children}(v)}D(w)\right).$$

The solution selects either a parent extent, an antichain of descendants, or nothing. This is **exact for the additive surrogate on that supplied tree**, not globally exact PQ optimization. Cross-crop masks, duplicate GT explanations and later greedy pixel assignment violate the separability assumption. Preserve nesting within each crop, resolve cross-crop alternatives explicitly, and judge only the final one-to-one PQ.

The failed ladder (§6 EXP-06: 0.36159, 1,022 FP versus baseline 0.37468, 918 FP; only one additional TP) is not a test of this fine-map hierarchy. Targeted inspection of `magfilo_15_experiments_inference_suite.ipynb` cell 12 shows a **coarse-proposal union** at thresholds 0.20/0.28/0.36, already deduplicated by bounding-box IoU≥0.50, then single-view fine threshold 0.40 and confidence-ranked cleanup. It establishes that this broader proposal recipe performed poorly, not that duplicate emission alone caused its FP rise. Duplicate/GT-assignment diagnostics are needed to separate correlated explanations from genuinely incorrect new proposals. This result motivates structured, utility-calibrated alternative selection but does not supply its isolated causal effect. The new fine-tree experiment must learn node utilities and enforce alternatives, not repeat coarse ladder union under a new name. It must also avoid the already-failed requirement that every faint component contain a ≥0.55 seed.

### 14.6 Feature dominance, physics and topology are different axes

Cell 11 permutation importance uses ten repeats on the physically held-out scorer partition, with no explicit `scoring` argument. For this regressor, importance is the **drop in held-out R²**. Thus `mean_prob = 0.740528` is neither 74% of PQ nor a quantified causal contribution to segmentation.

Its dominance is nevertheless mechanistically plausible: probability features summarize image evidence already integrated by the fine network; `log_area` separates tiny artifacts from credible extents, consistent with the area-75 ablation removing 69 FP without losing TP in that particular run. Area and probability are jointly conditioned by thresholding, and mean/max/std probability, shape and area are correlated. Permutation scores do not identify a unique indispensable causal feature.

Two descriptor names also need qualification:

- `norm_radius` in the inspected source is distance from `(1024,1024)` divided by **1024**, not by a fitted solar radius. A radius≈900 pixel disk does not reach one on this scale. A gate beginning at 0.85 consequently targets a different region than a true `r/R_sun` gate.
- `tortuosity` is perimeter divided by square-root area, a boundary-complexity measure, not centerline arclength/chord tortuosity. `extent` is area/bounding-box area, not convex-hull solidity.

The augmented scorer changes radius and boundary complexity **together** (+0.0037 versus the refitted six-feature scorer). The small negative radius permutation score (−0.005978 ±0.012966) does not establish that radius is harmful, that local contrast is uninformative, or that solar physics has no predictive value. Scalar position is not local photometric evidence; information can also be redundant after conditioning on learned probability.

**Fususu's guardrails address another failure mode entirely.** Area ratio 0.60–1.70, limited fragmentation and Dice≥0.62 relative to identity are safeguards against unstable TTA extent changes. Radius importance cannot validate or invalidate them. Guards can prevent a good baseline mask from becoming pathological; they cannot create a missing coarse proposal, and an identity-anchored rule may veto a genuine recovery from an empty or badly eroded identity mask. Test them with paired counts of *newly recovered matches* and *previously valid matches broken*, not a universal topology veto. Fususu has no established local PQ ablation in the supplied evidence.

### 14.7 Mechanistic transfer from competitors

| Participant mechanism | Connection to our observations | Transfer decision and principal risk |
|---|---|---|
| **Maxim: seed dilation/erosion/translation, grayscale + CLAHE + seed** | A coarse-conditioned fine model can learn a narrow relationship between clean training regions and filament extent. Stage C can reflect sensitivity to seed boundaries rather than weak visual evidence alone. | High-value controlled seed-jitter retraining after support audit. The grouped macro PQ 0.40859 is a whole-pipeline result, not a measured seed-jitter gain. Preserve targets under perturbation and keep negative crops. |
| **Tushar earlier: physical channels, annotator means, spine head, learned selection** | Local contrast can reveal faint structure without requiring a high-confidence core. Annotator-mean utility directly addresses our optimistic target. | Adopt the utility formulation as an ablation; test local contrast separately. The 1024 filtering result 0.4106→0.4388 is not proof that physical channels alone helped. Our analytic limb flattening already hurt. |
| **Tushar later: threshold hierarchy, hurdle scorer, conditional extent, rotated crops** | Offers discrete alternatives for the Stage C family: choose a wider faint extent or several narrower children without emitting all variants. Conditional targets avoid depressing extent merely because some records omit the object. | Port hierarchy/target principles, not the enormous ensemble. Refiner gain 0.4503→0.4597 belongs to a 695-observation subset; shared exclusions and incomplete imported-module access limit comparability. |
| **Tim: filled padded-box conditioning** | Reduces dependence on exact detector-mask shape; relevant to seed-copying and crop sensitivity. | A matched training-conditioning ablation, not a drop-in replacement for existing region-trained weights. The actual saved run disables box-jitter TTA, so its isolated benefit is unproven. |
| **Nomannic: photometric augmentation and image-only crop refinement** | Suggests the refiner need not simply reproduce its proposal; seeing and contrast variation are plausible sources of weak evidence. | Useful robustness precedent, not PQ superiority: its threshold tuning uses a Dice-based metric. Changing to image-only inputs is a separate model experiment, not cleanup. |
| **Fususu: guarded flip TTA and serialized RLE audits** | Useful for view-induced drift and exact artifact reproducibility, not evidence that shape constraints recover missing structures. | Borrow auditing and test guards only on unstable alternatives. The supplied .70 prose is not reliable ranking evidence. |

**Why small geometry errors can masquerade as confidence failure:** for two ideal long ribbons of identical width `w`, displaced perpendicularly by `delta<w`, their IoU is approximately `(w−delta)/(w+delta)`. Strict IoU>0.5 requires `delta<w/3`; a one-pixel shift of a three-pixel ribbon gives exactly 0.5 and no eligible match. This is an illustrative calculation, not a measured width distribution. It connects the OpenCV/COCO discrepancy (Q4 median IoU 0.9026, fifth percentile 0.8401) and the later participant's reported sensitivity to one-pixel extent shifts to our Stage C candidate failures. Canonical targets and careful rotated inverse mapping can matter without changing peak confidence. A one-pixel predicted centerline inside a much wider GT likewise may have IoU far below 0.5 even with probability one; skeleton confidence alone is not enough.

The most important disagreement is **physics and topology**. Tushar's successful bundled pipeline does not override our failed limb-flattening experiment; a retained raw-intensity channel plus locally normalized residual is a different intervention from replacing intensity with a radial correction. Likewise, a spine auxiliary loss is not synonymous with high-weight skeleton BCE or clDice. The later participant reports skeleton/clDice −0.045 and a worse physics/ridge stem; these are cautionary priors with incomplete isolated provenance, not universal bans. They put normalized, low-weight skeleton supervision behind better-supported candidate and target interventions.

### 14.8 Pareto synthesis: spend compute on new evidence, not redundant views

Within §7.5, four-flip mean yields reported PQ 0.4107 at 569.8 ms/image versus eight-view median 0.4106 at 850.4 ms/image: about **33% lower measured latency**, or **1.49× throughput**, for an unresolved 0.0001 PQ difference. This is strong evidence to *recheck* four views with the selected scorer, not proof of the assembled augmented-scorer result at that speed.

Timing in `run_evaluation_pass` uses a precomputed coarse cache and includes fine processing and metric work. It is **not clean end-to-end deployment latency**. Record separate coarse, fine, scorer/postprocessing, serialization and evaluation times before extrapolating a full nine-hour run.

One subtle untested interaction is even-view aggregation: the source's `torch.median` selects the **lower middle order statistic** for eight views, not the average of the fourth and fifth sorted values. This is a valid defined policy, not a programming error. For a foreground rule `p>T`, the lower median exceeds `T` only if **at least five of eight views** exceed it. Four moderately supportive views can therefore disappear under that policy even when a midpoint-median alternative crosses the threshold. This makes view disagreement on thin structures a specific diagnostic, not proof that median is generally inferior. The present trace does not establish that this caused Stage C. A named, scorer-recalibrated alternative-map experiment is warranted; another global aggregator sweep alone has weak novelty after §6.

The useful synergy is **cheap four-view inference + targeted alternatives on uncertain crops + structured utility selection**. Extra views should be spent where they supply a distinct recoverable extent, not on every already-stable crop. Keep identity-based safeguards as a measured ablation, because rigid guards may reject the very faint recoveries sought.

The collinear winner adds only **0.00032 PQ** (0.41063→0.41095), with TP 1,013→1,012 and FP 705→700. That is predominantly a small count trade-off, not demonstrated recovery of the 302 Stage C objects. Further merge-grid tuning and unverified radial area heuristics should not displace the next experiments.

### 14.9 Ranked unexecuted experiments and expected-value logic

For prioritization, expected value means `probability of a real gain × useful magnitude / engineering and compute cost`. The available studies do **not** identify numerical success probabilities or expected PQ gains. The ordering below is an ordinal decision judgment, not a fabricated forecast. Candidate rescue is gated by a measured oracle; expensive interventions move down or are skipped when their mechanism is absent. Full methods, controls and stopping rules are specified in the new roadmap.

| Rank / ID | Specific new intervention | Main target | Evidence strength / relative cost | Decisive test |
|---|---|---|---|---|
| 1 / N1 | Relabel fixed candidates by **annotator-mean eligible IoU**, with record-aligned weights; direct regressor before an `h×m` ablation | RQ calibration, FP, alternative selection | Source-established mismatch; benefit unverified; low compute | Same masks/features/model/folds, change labels only; evaluate final pooled PQ and utility calibration. |
| 2 / N2 | Build a **fine-map nested component tree** and select a utility-scored antichain instead of a threshold union | Stage C extent/fragment alternatives | Strong mechanism and participant precedent; medium engineering, little new neural compute | Compare single threshold, duplicate ladder and antichain on identical maps; require final assignment gains and controlled FP. |
| 3 / N3 | Add **residual coarse proposal seeds** at 0.22, then a gated 0.15 arm, keeping full component/context and the working branch | Genuine unsupported Stage A | Direct trace signal, but old global lowering failed; medium compute | Measure new single-crop support and final eligible matches, with a residual-branch scorer and fixed crop budget. |
| 4 / N4 | **Uncertainty-routed alternative extents/views**, including a separately named midpoint-median alternative | Recoverable Stage C view/threshold instability | Pareto-supported compute savings; rescue benefit speculative; low–medium compute | Same capped budget versus four-view and all-eight controls; count rescued and broken matches. |
| 5 / N5 | **COCO-consistent fine supervision**, leaving evaluation and architecture fixed | Thin-boundary SQ and IoU-threshold crossings | Q4 reports median OpenCV/COCO IoU 0.9026; training benefit untested; one training arm | Same crops/seed/steps, change target rasterizer only; evaluate against unchanged COCO GT. |
| 6 / N6 | **Fine-region perturbation training** with modest dilation/erosion/translation and valid target support | Stage C seed sensitivity | Strong transfer rationale, no isolated local result; one training arm | Seed-jitter versus same-initialization unjittered training; final PQ plus perturbation sensitivity. |
| 7 / N7 | **Instance-conditioned extent targets**, averaging associated drawings rather than treating omitted drawings as empty extents | Stage C annotation-induced attenuation | Later participant precedent; material target/loader work | Matched-instance conditional versus unconditional record-mean targets; separately score annotation-match frequency. |
| 8 / N8 | **Raw intensity + masked local-contrast residual**; fine and coarse arms independently gated by their failure evidence | Stage C evidence; Stage A only with coarse arm | Plausible, mixed physics evidence; one model per arm | Retain intensity, change one input channel, compare same training budget and contrast-stratified FN recovery. |
| 9 / N9 | **Native-scale principal-axis conditional crops**, with inverse mapping and curved/round fallback | Stage C effective feature geometry; support only where measured | Later participant precedent; high geometry/loader cost | Same target and pixels, axis-aligned versus rotated crop arm; no anisotropic stretching. |
| 10 / N10 | **Normalized low-weight skeleton-positive BCE**, preserving existing BCE+Dice | Confirmed thin-spine failures within support | Mechanistically plausible but adverse related evidence; one training arm | λ=0.1 first against λ=0; require eligible extent recovery, not just skeleton confidence. |

None is claimed fully executed by this dossier. Some extend previously attempted ideas, but their new causal elements are explicit: **population utility rather than best-record utility; mutually exclusive extents rather than ladder union; additive residual seeds rather than global threshold reduction; selective new evidence rather than another all-image TTA sweep.**

The §13 skeleton proposal should not be implemented literally as `BCE(pred×skeleton, target×skeleton)` with an assumed λ=3 guarantee. The correct additional term selects unreduced BCE on skeleton pixels:

$$L_{\mathrm{skel}}=\frac{\sum_x s(x)\,\operatorname{BCEWithLogits}(z(x),1)}{\sum_x s(x)+\epsilon}.$$

Keep ordinary BCE/Dice and region-shape semantics unchanged; average the normalized term per valid sample and specify empty-skeleton handling. Multiplying logits by a binary mask before BCE is not equivalent to selecting loss terms, and high positive-only weight can thicken/bridge neighboring filaments. Nothing guarantees a particular inference probability or PQ gain.

### 14.10 Decisions, limitations and next action

**Adopt now as experimental prerequisites:** explicit scorer ownership; immutable run manifests; preserved split/checkpoint identities; COCO-based one-to-one final FN reconciliation; annotation-record utility definitions; honest “processing shard” terminology; cached soft maps/supports for replay. These are preparation tasks, not claimed model gains.

**Test next in order:** N1 utility alignment → N2 structured fine alternatives → N3 residual coarse rescue. N4 is a conditional way to reinvest Pareto savings. N5/N6 are the first controlled fine retraining priorities, followed by the more substantial N7–N10 branches only when diagnostics support them. Each winner must be compared with both the preceding retained control and the frozen root baseline; individually good results must not be silently stacked.

**Deprioritize or reject as next actions:** repeating global coarse/fine threshold reductions, emitting all ladder variants, mandatory high-seed hysteresis, a high-weight skeleton-loss bundle, indiscriminate fragment merging, replacing the working U-Net merely for novelty, and reproducing the competitor's 120 dense views/image or 40 refiner passes/crop. Related principles remain testable in constrained forms; already failed policies are not new experiments.

**Unresolved questions:** the exact bytes and checkpoint memberships behind each historical score and LB submission; count/metric aggregation mismatches; final Stage A/C mechanisms under the augmented scorer; whether target relabeling improves held-out utility rather than only training calibration; how much fine-tree oracle headroom is realizable after global conflicts; and whether any local improvement transfers to the hidden labels. A3 human-pair PQ 0.3397 is **not a hard ceiling**, and a model exceeding it does not establish a particular mechanism of generalization. Annotator subgroup differences do not establish the hidden label distribution. Three zero-prediction test observations must be reported, not declared GT-empty without accessible labels.

**Immediate execution recommendation:** reconstruct the exact augmented-scorer baseline and final path, repair the attribution/target records without changing old results, then run the fixed-candidate N1 ablation. Do not begin by lowering the global coarse threshold or training λ=3 skeleton loss. The most defensible route beyond the present result is to turn already available image evidence into *correctly weighted, mutually exclusive, eligible instance masks*, while adding genuinely new proposal support only where the current pipeline cannot emit a match.




## 15. November 7, 2026 — Zero-Training Research Suite, Validation Experiments & Test Submission

**Run status:** Completed end-to-end.  
**Run type:** Inference/research only; fixed coarse, fine and scorer checkpoints were loaded. No neural-network training occurred.  
**Notebook:** `MAGFiLO_2026_zero_training_research_submission.ipynb`  
**Total logged wall time:** approximately 2,899.1 s (48 min 19 s), from the first startup messages through notebook export.  
**Validation scope:** 141 physical validation observations, 1,593 GT instances.  
**Test scope:** 180 hidden-test observations.  
**Best validation result in this run:** `N4_Uncertainty_D4_MidpointMedian`, PQ = **0.444568**.  
**Submission artifact:** `/kaggle/working/submission.csv` (1,352 instance rows; serialized audit passed).

### 15.1 Runtime environment, inputs and integrity checks

- PyTorch: `2.11.0+cu130`
- `segmentation-models-pytorch`: `0.6.0` (installed because it was missing at startup)
- Device: `cuda:0`; 2 visible GPUs
- GPU 0: Tesla T4, 14.6 GiB
- GPU 1: Tesla T4, 14.6 GiB
- Dataset root: `/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`
- Dataset files: 707 training images and 180 test images
- Coarse checkpoint: `/kaggle/input/datasets/fanzzum/solarcrow8d/coarse_model_3.pth`
- Fine checkpoint: `/kaggle/input/datasets/fanzzum/solarcrow8d/EXP_4_EfficientNetB3_BCE_Dice.pth`
- Scorer checkpoint: `/kaggle/input/datasets/fanzzum/solarcrow8d/scorer_3.pkl`
- Split JSON: `/kaggle/input/datasets/fanzzum/solarcrow8d/results8/split.json`
- Verified physical split: 566 train / 141 validation; physical-image leakage = 0.
- Pre-rasterized validation GT RLE records: 1,593.
- Metric harness: `PQ = SQ × RQ`, using strict `IoU > 0.5` and one-to-one Hungarian matching.
- Loaded scorer: `HistGradientBoostingRegressor`; feature schema = `log_area`, `mean_prob`, `max_prob`, `std_prob`, `aspect_ratio`, `extent`.
- Candidate-generation, conflict-resolution and compressed-cache helpers initialized successfully.

**Startup messages and warnings:** The debugger emitted frozen-module breakpoint warnings twice (at approximately 8.0 s and 8.6 s), suggesting `-Xfrozen_modules=off` and mentioning `PYDEVD_DISABLE_FILE_VALIDATION=1`. Debugging explicitly continued. These were warnings, not execution failures. The missing `segmentation-models-pytorch` dependency was installed successfully.

### 15.2 G0 — Cached baseline and validation inference

Coarse cache construction over the 141 validation images:

| Progress | Log time | Elapsed reported by cell |
|---:|---:|---:|
| 30/141 | 101.2 s | 25.9 s |
| 60/141 | 122.3 s | 47.1 s |
| 90/141 | 144.1 s | 68.9 s |
| 120/141 | 166.3 s | 91.1 s |
| 141/141 | 182.5 s | 107.3 s |

Fine cache construction over the same 141 images:

| Progress | Fine crops accumulated | Log time | Elapsed reported by cell |
|---:|---:|---:|---:|
| 30/141 | 225 | 228.5 s | 153.3 s |
| 60/141 | 446 | 272.5 s | 197.3 s |
| 90/141 | 699 | 321.9 s | 246.8 s |
| 120/141 | 961 | 368.4 s | 293.2 s |
| 141/141 | 1,111 | 398.4 s | 323.2 s |

Validation end-to-end inference/cache latency was **2,294.2 ms/image**. The measured G0 baseline was:

| Metric | G0 value |
|---|---:|
| PQ | **0.430836964655815** |
| SQ | 0.6869362366992666 |
| RQ | 0.6271862534519791 |
| TP | 1,022 |
| FP | 644 |
| FN | 571 |
| Precision | 0.6134453781512605 |
| Recall | 0.6415568110483365 |
| Latency | 2,297.514197 ms/image |

The notebook's baseline quality guard printed `BASELINE QUALITY CHECK FAILED` because PQ = 0.43084 was outside its historical/verified bracket of `(0.41, 0.42)`. This was a **bracket mismatch, not a failed inference run**: the measured result was higher than the bracket's upper bound, and the notebook explicitly continued with diagnostics and submission.

### 15.3 G1 — Cached support diagnostics

- Coarse maps cached: 141.
- Fine crop bundles cached: 141.
- GT instances reported as structurally impossible under single-crop support: **240 / 1,593**.
- The diagnostic was based on `max_crop_support_ratio` and the `structurally_impossible_single_crop` flag.

The first five rows shown in the log were all from physical image `20120313063154Uh` (`20120313063154Uh.jpeg`), annotator image `030101-20120313063154Uh`:

| Annotation ID | GT area | Max crop support ratio | Structurally impossible under single crop |
|---|---:|---:|---|
| `361a0fb8-3066-4bdc-bcc5-ac7ad492cc2f` | 4,381 | 0.894545 | False |
| `19917284-12a4-4983-b5e4-9095b13a2f42` | 11,656 | 1.000000 | False |
| `2b26ab41-528d-4656-a4e7-e6435a7cde18` | 3,567 | 1.000000 | False |
| `4ab739bd-eb30-4f24-85ac-383d703350f0` | 2,158 | 0.844764 | False |
| `9fa9d8f8-31f1-4f99-949c-dfe0ef216920` | 1,785 | 1.000000 | False |

These are the five sample rows printed in the log, not the full set of 1,593 diagnostic records.

### 15.4 Stage 3 candidate bank — 566 training physical images

The grouped candidate bank completed over all **566** training physical images in **19.3 min**. It generated **4,877 candidate rows**, saved to `/kaggle/working/candidate_bank_566_training.csv`. Candidate positives = **65.04%**; distinct physical images represented = **554**.

Full progress trace:

| Physical images processed | Candidate rows | Log time | Elapsed | ETA at that checkpoint |
|---:|---:|---:|---:|---:|
| 30/566 | 250 | 491.9 s | 1.0 min | 17.9 min |
| 60/566 | 482 | 551.3 s | 2.0 min | 16.8 min |
| 90/566 | 785 | 618.0 s | 3.1 min | 16.4 min |
| 120/566 | 1,073 | 682.8 s | 4.2 min | 15.5 min |
| 150/566 | 1,276 | 735.9 s | 5.1 min | 14.1 min |
| 180/566 | 1,495 | 793.9 s | 6.0 min | 12.9 min |
| 210/566 | 1,755 | 856.2 s | 7.1 min | 12.0 min |
| 240/566 | 1,997 | 915.5 s | 8.1 min | 10.9 min |
| 270/566 | 2,245 | 975.8 s | 9.1 min | 9.9 min |
| 300/566 | 2,487 | 1,034.2 s | 10.0 min | 8.9 min |
| 330/566 | 2,772 | 1,099.0 s | 11.1 min | 8.0 min |
| 360/566 | 3,007 | 1,158.6 s | 12.1 min | 6.9 min |
| 390/566 | 3,274 | 1,222.1 s | 13.2 min | 5.9 min |
| 420/566 | 3,559 | 1,287.9 s | 14.3 min | 5.0 min |
| 450/566 | 3,843 | 1,352.7 s | 15.3 min | 4.0 min |
| 480/566 | 4,140 | 1,418.3 s | 16.4 min | 2.9 min |
| 510/566 | 4,364 | 1,474.9 s | 17.4 min | 1.9 min |
| 540/566 | 4,638 | 1,538.1 s | 18.4 min | 0.9 min |
| 566/566 | 4,877 | 1,592.7 s | 19.3 min | 0.0 min |

### 15.5 N1 — Expected-utility scorer sweeps

Three candidate-utility target variants were evaluated at thresholds `tau ∈ {0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350}`.

| Scorer arm | tau | Log time | PQ | SQ | RQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1_A_MaxIoU_Control | 0.200 | 1614.6 s | 0.43853 | 0.68695 | 0.63836 | 1,015 | 572 | 578 |
| N1_A_MaxIoU_Control | 0.225 | 1635.1 s | 0.44086 | 0.68751 | 0.64125 | 1,009 | 545 | 584 |
| N1_A_MaxIoU_Control | 0.250 | 1655.2 s | 0.44072 | 0.68790 | 0.64067 | 994 | 516 | 599 |
| N1_A_MaxIoU_Control | 0.275 | 1674.8 s | 0.43834 | 0.68780 | 0.63731 | 977 | 496 | 616 |
| N1_A_MaxIoU_Control | 0.300 | 1693.6 s | 0.43762 | 0.68821 | 0.63588 | 964 | 475 | 629 |
| N1_A_MaxIoU_Control | 0.325 | 1711.6 s | 0.43697 | 0.68831 | 0.63485 | 951 | 452 | 642 |
| N1_A_MaxIoU_Control | 0.350 | 1729.3 s | 0.43726 | 0.68896 | 0.63467 | 939 | 427 | 654 |
| N1_B_AnnotatorConsensus | 0.200 | 1748.7 s | **0.44248** | 0.68819 | 0.64297 | 995 | 507 | 598 |
| N1_B_AnnotatorConsensus | 0.225 | 1767.7 s | 0.43956 | 0.68850 | 0.63842 | 972 | 480 | 621 |
| N1_B_AnnotatorConsensus | 0.250 | 1786.6 s | 0.43915 | 0.68869 | 0.63766 | 960 | 458 | 633 |
| N1_B_AnnotatorConsensus | 0.275 | 1804.2 s | 0.43936 | 0.68916 | 0.63753 | 948 | 433 | 645 |
| N1_B_AnnotatorConsensus | 0.300 | 1821.4 s | 0.43738 | 0.68960 | 0.63425 | 926 | 401 | 667 |
| N1_B_AnnotatorConsensus | 0.325 | 1836.7 s | 0.43740 | 0.69034 | 0.63360 | 907 | 363 | 686 |
| N1_B_AnnotatorConsensus | 0.350 | 1852.3 s | 0.43050 | **0.69097** | 0.62303 | 871 | 332 | 722 |
| N1_C_Hurdle | 0.200 | 1873.1 s | 0.43739 | 0.68796 | 0.63578 | 995 | 542 | 598 |
| N1_C_Hurdle | 0.225 | 1893.2 s | 0.43914 | 0.68805 | 0.63824 | 988 | 515 | 605 |
| N1_C_Hurdle | 0.250 | 1913.3 s | 0.44056 | 0.68815 | 0.64021 | 984 | 497 | 609 |
| N1_C_Hurdle | 0.275 | 1933.4 s | 0.44021 | 0.68905 | 0.63887 | 973 | 480 | 620 |
| N1_C_Hurdle | 0.300 | 1951.4 s | 0.44211 | 0.68891 | 0.64175 | 970 | 460 | 623 |
| N1_C_Hurdle | 0.325 | 1970.4 s | **0.44257** | 0.68895 | **0.64238** | 961 | 438 | 632 |
| N1_C_Hurdle | 0.350 | 1988.7 s | 0.44173 | 0.68898 | 0.64114 | 946 | 412 | 647 |

The validation-selected N1 arm was **N1_C_Hurdle at `tau = 0.325`**, selected at approximately 2,025.3 s. This is validation-tuned and must **not** be represented as an unbiased test estimate.

### 15.6 N2 — Fine-map antichain dynamic programming

- PQ = **0.437105**.
- SQ = 0.687001; RQ = 0.636251.
- TP = 1,025; FP = 604; FN = 568.
- Precision = 0.629220; recall = 0.643440.
- Latency = 2,536.592693 ms/image.
- Newly matched misses relative to N1 control: **79**.
- Measured baseline FN used by this diagnostic: **632**. The log explicitly notes that **302 was a supplied reference count**, not the measured FN.
- N2 result logged at **2,079.0 s**; N2 elapsed time: **53.7 s**.

Despite matching 79 N1-control misses, N2's final PQ (0.437105) was below the N1 best arm (0.442570). The newly matched count alone therefore does not establish an overall improvement.

### 15.7 N3 — Residual coarse proposal rescue

Residual branch progress over the 141 validation images:

| Progress | Log time | Added proposals so far |
|---:|---:|---:|
| 30/141 | 2,092.8 s | 6 |
| 60/141 | 2,109.3 s | 31 |
| 90/141 | 2,123.8 s | 41 |
| 120/141 | 2,137.5 s | 51 |
| 141/141 | 2,147.6 s | 56 |

Final N3 results:

- PQ = **0.442459**; SQ = 0.688447; RQ = 0.642691.
- TP = 965; FP = 445; FN = 628.
- Precision = 0.684397; recall = 0.605775.
- Latency = 2,785.607737 ms/image.
- Residual proposals added: **56**.
- Newly matched N1-control misses: **4**.
- N3 final result logged at **2,181.9 s**; elapsed time: **102.9 s**.

### 15.8 N4 — Uncertainty-routed D4 TTA

- Uncertainty cutoff: **0.01129015**, logged at **2,350.4 s**.
- Routed crops: **223 / 1,111 = 20.1%**.
- Routed inference completed for all 141 validation images:

| Progress | Log time |
|---:|---:|
| 30/141 | 2,367.4 s |
| 60/141 | 2,386.7 s |
| 90/141 | 2,406.1 s |
| 120/141 | 2,418.9 s |
| 141/141 | 2,432.0 s |

- Final N4 metrics logged at **2,466.8 s**.
- Additional inference elapsed time: **116.6 s**.

Two aggregation variants were measured:

| Variant | PQ | SQ | RQ | TP | FP | FN | Precision | Recall | Latency (ms/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N4_Uncertainty_D4_LowerMedian | 0.4441266602816734 | 0.6886252547047802 | 0.6449468085106383 | 970 | 445 | 623 | 0.6855123674911661 | 0.6089139987445072 | 4073.118656 |
| N4_Uncertainty_D4_MidpointMedian | **0.4445684279674615** | **0.6888746074232647** | **0.6453546453546454** | 969 | 441 | 624 | **0.6872340425531915** | 0.6082862523540490 | 4072.984601 |

The midpoint-median variant produced the highest PQ in this run. It did so with one fewer TP, four fewer FP and one more FN than the lower-median variant; its higher SQ/RQ combination yielded the higher PQ. This is still a result on the validation split, not hidden-test ground truth.

### 15.9 Full experiment comparison summary

The following table preserves the run's emitted comparison summary, including precision, recall and per-image latency. Values are rounded exactly as shown in the run's summary output.

| Experiment | PQ | SQ | RQ | TP | FP | FN | Precision | Recall | Latency (ms/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_Champion_4View | 0.430837 | 0.686936 | 0.627186 | 1,022 | 644 | 571 | 0.613445 | 0.641557 | 2297.514197 |
| G1_Cached_Maps_And_Support | 0.430837 | 0.686936 | 0.627186 | 1,022 | 644 | 571 | 0.613445 | 0.641557 | 2297.514197 |
| N1_A_MaxIoU_Control_tau_0.200 | 0.438527 | 0.686953 | 0.638365 | 1,015 | 572 | 578 | 0.639572 | 0.637163 | 2297.630276 |
| N1_A_MaxIoU_Control_tau_0.225 | 0.440863 | 0.687510 | 0.641246 | 1,009 | 545 | 584 | 0.649292 | 0.633396 | 2297.728385 |
| N1_A_MaxIoU_Control_tau_0.250 | 0.440718 | 0.687902 | 0.640670 | 994 | 516 | 599 | 0.658278 | 0.623980 | 2297.570916 |
| N1_A_MaxIoU_Control_tau_0.275 | 0.438344 | 0.687800 | 0.637312 | 977 | 496 | 616 | 0.663272 | 0.613308 | 2297.613347 |
| N1_A_MaxIoU_Control_tau_0.300 | 0.437624 | 0.688214 | 0.635884 | 964 | 475 | 629 | 0.669910 | 0.605148 | 2297.593817 |
| N1_A_MaxIoU_Control_tau_0.325 | 0.436970 | 0.688308 | 0.634846 | 951 | 452 | 642 | 0.677833 | 0.596987 | 2297.609101 |
| N1_A_MaxIoU_Control_tau_0.350 | 0.437264 | 0.688958 | 0.634674 | 939 | 427 | 654 | 0.687408 | 0.589454 | 2297.596097 |
| N1_B_AnnotatorConsensus_tau_0.200 | 0.442484 | 0.688185 | 0.642973 | 995 | 507 | 598 | 0.662450 | 0.624608 | 2297.537181 |
| N1_B_AnnotatorConsensus_tau_0.225 | 0.439556 | 0.688502 | 0.638424 | 972 | 480 | 621 | 0.669421 | 0.610169 | 2297.684191 |
| N1_B_AnnotatorConsensus_tau_0.250 | 0.439154 | 0.688693 | 0.637662 | 960 | 458 | 633 | 0.677010 | 0.602637 | 2297.539339 |
| N1_B_AnnotatorConsensus_tau_0.275 | 0.439356 | 0.689158 | 0.637525 | 948 | 433 | 645 | 0.686459 | 0.595104 | 2297.666198 |
| N1_B_AnnotatorConsensus_tau_0.300 | 0.437379 | 0.689604 | 0.634247 | 926 | 401 | 667 | 0.697815 | 0.581293 | 2297.533905 |
| N1_B_AnnotatorConsensus_tau_0.325 | 0.437399 | 0.690338 | 0.633601 | 907 | 363 | 686 | 0.714173 | 0.569366 | 2297.835417 |
| N1_B_AnnotatorConsensus_tau_0.350 | 0.430499 | 0.690973 | 0.623033 | 871 | 332 | 722 | 0.724023 | 0.546767 | 2297.585015 |
| N1_C_Hurdle_tau_0.200 | 0.437391 | 0.687957 | 0.635783 | 995 | 542 | 598 | 0.647365 | 0.624608 | 2300.621727 |
| N1_C_Hurdle_tau_0.225 | 0.439141 | 0.688047 | 0.638243 | 988 | 515 | 605 | 0.657352 | 0.620213 | 2300.569237 |
| N1_C_Hurdle_tau_0.250 | 0.440561 | 0.688153 | 0.640208 | 984 | 497 | 609 | 0.664416 | 0.617702 | 2300.687644 |
| N1_C_Hurdle_tau_0.275 | 0.440214 | 0.689051 | 0.638871 | 973 | 480 | 620 | 0.669649 | 0.610797 | 2300.590330 |
| N1_C_Hurdle_tau_0.300 | 0.442107 | 0.688912 | 0.641747 | 970 | 460 | 623 | 0.678322 | 0.608914 | 2300.600290 |
| N1_C_Hurdle_tau_0.325 | 0.442570 | 0.688954 | 0.642380 | 961 | 438 | 632 | 0.686919 | 0.603264 | 2300.899359 |
| N1_C_Hurdle_tau_0.350 | 0.441729 | 0.688976 | 0.641139 | 946 | 412 | 647 | 0.696613 | 0.593848 | 2300.651977 |
| N1_BestArm_StandardComponents | 0.442570 | 0.688954 | 0.642380 | 961 | 438 | 632 | 0.686919 | 0.603264 | 2300.704080 |
| N2_FineMap_Antichain_DP | 0.437105 | 0.687001 | 0.636251 | 1,025 | 604 | 568 | 0.629220 | 0.643440 | 2536.592693 |
| N3_Residual_Coarse_Rescue | 0.442459 | 0.688447 | 0.642691 | 965 | 445 | 628 | 0.684397 | 0.605775 | 2785.607737 |
| N4_Uncertainty_D4_LowerMedian | 0.444127 | 0.688625 | 0.644947 | 970 | 445 | 623 | 0.685512 | 0.608914 | 4073.118656 |
| N4_Uncertainty_D4_MidpointMedian | **0.444568** | 0.688875 | **0.645355** | 969 | 441 | 624 | **0.687234** | 0.608286 | 4072.984601 |

### 15.10 Test inference and serialized submission audit

Test inference completed for all 180 hidden-test images:

| Progress | Log time | Elapsed shown by run |
|---:|---:|---:|
| 30/180 | 2,529.9 s | 1.0 min |
| 60/180 | 2,592.5 s | 2.1 min |
| 90/180 | 2,666.2 s | 3.3 min |
| 120/180 | 2,724.9 s | 4.3 min |
| 150/180 | 2,770.1 s | 5.0 min |
| 180/180 | 2,815.2 s | 5.8 min |

The notebook atomically wrote `/kaggle/working/submission.csv` with **1,352 rows** at log time **2,838.7 s**. The serialized submission audit ran at **2,887.5 s** and returned:

```text
rows:                  1352
expected_images:       180
nonempty_images:      176
zero_prediction_images: 4
decoded_instances:    1352
all_rles_2048x2048:   True
instance_overlap_pixels: 0
```

The run reported: `All complete. Test inference + audit wall time=7.0 min`.

**Interpretation boundary:** the four zero-prediction images are a property of the generated submission. Without hidden-test labels, they cannot be declared truly filament-free or counted as false negatives. The log verifies output serialization, image dimensions and zero within-image instance-overlap pixels; it does not provide hidden-test PQ.

### 15.11 Notebook export and non-blocking end-of-run warnings

- Mistune emitted `SyntaxWarning: invalid escape sequence '\\|'` at `mistune.py:435`.
- `nbconvert` emitted `SyntaxWarning: invalid escape sequence '\\_'` at `filter_links.py:36`.
- Notebook export completed: `__notebook__.ipynb` written at 110,278 bytes (2,895.7 s; conversion began at 2,895.6 s).
- HTML export completed: `__results__.html` written at 664,849 bytes (2,899.1 s; conversion began at 2,898.2 s).

These were library/export warnings; the run had already completed test inference, submission serialization and audit successfully.

### 15.12 Results interpretation and follow-up priorities

1. **Best measured validation result:** N4 midpoint-median, PQ 0.444568, up from G0 PQ 0.430837 by approximately **+0.013731 absolute**. This is a validation result and was obtained after choosing among multiple experiments on the same validation set; it should not be described as an unbiased estimate of hidden-test performance.
2. **N1 remains a strong lower-cost option:** N1_C_Hurdle at `tau=0.325` reached PQ 0.442570 at approximately 2.30 s/image, while N4 midpoint-median reached 0.444568 at approximately 4.07 s/image. The observed N4 increment over N1 is about **+0.001998 PQ** for roughly **1.77 s/image** additional latency.
3. **N2's extra matches did not translate into a better total score:** it newly matched 79 N1-control misses, but ended at PQ 0.437105. Evaluate the complete one-to-one matching and FP trade-off, not only recovered-match counts.
4. **N3 added 56 residual proposals but newly matched only 4 N1-control misses:** its final PQ 0.442459 was slightly below the N1 best arm (0.442570). Proposal count is not equivalent to useful TP recovery.
5. **N4 routed only 20.1% of cached fine crops:** 223/1,111 crops received additional D4 inference. This delivered the best observed validation PQ, but at roughly 4.07 s/image.
6. **The support audit found a structural coverage limitation:** 240/1,593 GT instances were reported structurally impossible under single-crop support. This is a diagnostic signal for candidate/support generation; it does not by itself establish that all 240 are recoverable or that changing support will improve final PQ.
7. **Resolve the baseline bracket definition:** the run's measured G0 PQ (0.43084) is higher than the configured upper bound 0.42. Determine whether the bracket refers to another checkpoint, metric aggregation, or earlier baseline before treating this guard as a correctness failure.
8. **Preserve the validation-tuning caveat:** N1 tau selection and N4 variant comparison were conducted on the validation data. Keep hidden-test claims limited to successful inference and submission integrity until an official score is available.

### 15.13 Files and artifacts emitted by this run

| Artifact | Path | Status / contents |
|---|---|---|
| Training candidate bank | `/kaggle/working/candidate_bank_566_training.csv` | Saved; 4,877 candidate rows, 65.04% positives, 554 physical images represented |
| Experiment comparison summary | `/kaggle/working/experiment_comparison_summary.csv` | Saved; full emitted comparison table for G0/G1/N1/N2/N3/N4 |
| Test submission | `/kaggle/working/submission.csv` | Saved atomically; 1,352 rows; serialized audit passed |
| Exported notebook | `__notebook__.ipynb` | Exported; 110,278 bytes |
| HTML results | `__results__.html` | Exported; 664,849 bytes |
