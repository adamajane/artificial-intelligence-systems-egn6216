# First Model Exploration — Submission

**Commit:** `24edd1c` (pushed October 26, 2026)

All reported numbers come from the executed `training_experiments.ipynb` (run on **October 26, 2026, 22:39–22:42 EDT**) and the run records in `experiments/results/`.

---

## 1. Notebook

**Notebook:** [training_experiments.ipynb](https://github.com/adamajane/artificial-intelligence-systems-egn6216/blob/24edd1c/training_experiments.ipynb)

**Commit:** `24edd1c`

## 2. Stage Explored First

The **per-region flare classifier (proposal C5)** was explored first.

It is the **only learned stage**; the deterministic stages (sampling, filters, attribution, labelling, split) were verified by the data step's known-value checks.

## 3. Exploration Notes

### Model Selection and Experimental Setup

**Selected candidates:** ImageNet-pretrained **ResNet-18** and **DeiT-Tiny**, fine-tuned on 128×128 Br cutouts with an identical recipe.

**Objective:** Compare a convolutional neural network (CNN) with a vision transformer (ViT).

### Initial Results

The initial evaluation used **630 validation cases from 2015**, consisting of **210 flaring cases across 167 regions**.

- **ResNet-18:** TSS **0.648** (95% HARP-bootstrap CI: 0.520–0.744)
- **DeiT-Tiny:** TSS **0.626** (95% HARP-bootstrap CI: 0.510–0.712)

The difference (**−0.021, CI −0.110 to +0.072**) is within seed-to-seed variation:

- **ResNet-18:** 0.631 ± 0.016 (3 seeds)
- **DeiT-Tiny:** 0.627 ± 0.025 (3 seeds)

**Conclusion:** The architectural comparison is **inconclusive at this data size**.

### Observed Failure Modes

Errors are concentrated at the **class boundary**:

- **False positives:** The most confident false positives are large regions that produced **C-class flares**, with two possibly mislabelled cases.
- **False negatives:** The most confident false negatives are small regions with flares at **M1.0–M1.5**.
- **Calibration:** Predicted probabilities are also uncalibrated, with a **prevalence-corrected Brier skill score of −4.1**.

### Targeted Change and Effect

**Change:** Excluded the **63 flagged (possibly mislabelled) negatives** from ResNet-18 training.

**Rationale:** Neither model overfit, and DeiT did not clearly trail ResNet-18, satisfying the pre-set rule for this intervention.

**Observed effect:**

- **TSS:** 0.648 → 0.633
- **Paired 95% CI for the change:** −0.068 to +0.035
- **False alarm rate:** 0.190 → 0.167
- **Recall:** 0.838 → 0.800

**Conclusion:** No clear improvement or deterioration in TSS. The change produced **fewer false alarms at the expense of lower recall**.

### Runtime and Resource Constraints

Both architectures satisfy the deployment constraints:

- **ResNet-18:** 28.5 ms per 15-region timestep on a laptop CPU; model size **44.76 MB**.
- **DeiT-Tiny:** 20.0 ms per 15-region timestep on a laptop CPU; model size **21.66 MB**.
- **Model size target:** < 50 MB.
- **Training speed:** Approximately **1.1 seconds per epoch**.

### Main Limitation

**JSOC export throughput (~1,240 cutouts/hour)** limited this experiment to **630 training and 630 validation images**.

Validation negatives were subsampled:

- **TSS:** Unaffected by the altered class prevalence.
- **Precision and PR-AUC:** Prevalence-corrected to account for subsampling.

### Next Action

Complete the **overnight subset download**, consisting of:

- Training positives
- **5,000 training negatives**
- **Full 2015 validation set**

**Estimated download time:** Approximately **9.6 hours** at the observed throughput.

Afterward, **rerun experiments A/B/C on the full validation set with 3 seeds**.

---

## 4. Experiment Results

| Run   | Model / Configuration                   | Change from Baseline                     | Val TSS (95% CI)        | ROC-AUC | PR-AUC\* |
| ----- | --------------------------------------- | ---------------------------------------- | ----------------------- | ------- | -------- |
| **A** | ResNet-18, 128×128, 15 epochs, seed 42  | Baseline                                 | **0.648** (0.520–0.744) | 0.878   | 0.289    |
| **B** | DeiT-Tiny/16 @128, same recipe, seed 42 | Architecture only                        | **0.626** (0.510–0.712) | 0.873   | 0.246    |
| **C** | ResNet-18, same recipe, seed 42         | Flagged negatives excluded from training | **0.633** (0.492–0.731) | 0.865   | 0.245    |

### Model Configurations

**Run A — ResNet-18 (Baseline)**

- Model: `timm/resnet18.a1_in1k`
- Development cases: **630 (210 positive)**
- Training: **1.14 s/epoch** on MPS
- CPU inference: **1.9 ms/region**
- Model size: **44.76 MB**

**Run B — DeiT-Tiny**

- Model: `timm/deit_tiny_patch16_224.fb_in1k`
- Development cases: **630 (210 positive)**
- Training: **1.08 s/epoch** on MPS
- CPU inference: **1.33 ms/region**
- Model size: **21.66 MB**

**Run C — ResNet-18 (Flagged Negatives Excluded)**

- Model: ResNet-18, same recipe as Run A
- Change: Excluded `has_unattributed_mx` negatives from training (**567 training cases**)
- Development cases: **630 (210 positive)**
- Training: **1.02 s/epoch** on MPS
- CPU inference and model size: **Same as Run A**

### Evaluation Notes

**PR-AUC\*** is prevalence-corrected, with validation negatives weighted to the natural 2015 count.

**Optimal thresholds** (maximum TSS on validation):

- **A:** 0.498
- **B:** 0.108
- **C:** 0.518

**CPU inference measurements:** Milliseconds per region at batch size 15 (one timestep), using the median of 200 measurements.
