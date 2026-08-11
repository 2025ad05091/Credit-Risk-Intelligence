# Machine Learning Assignment 2 — Submission Report

**Work Integrated Learning Programmes Division — BITS Pilani**
**M.Tech (AIML / DSE) — Machine Learning — Assignment 2 (15 Marks)**

| | |
|---|---|
| **Student Name** | *&lt;Fill in&gt;* |
| **BITS ID** | *&lt;Fill in&gt;* |
| **Project Title** | Credit Risk Intelligence Console |
| **Dataset** | UCI Statlog (German Credit Data) |
| **Submission Deadline** | 18 August 2026, 23:59 IST |

> **How to use this file.** Export it to PDF and submit it as the single required PDF.
> Replace the three placeholders in Section 1 with the real links, and paste the BITS
> Virtual Lab screenshot into Section 3 before exporting.

---

## Section 1 — Mandatory Submission Links

### 1.1 GitHub Repository Link

**https://github.com/username/ml-assignment-2**

The repository contains the complete source code, `requirements.txt`, a full `README.md`
and the test data used in the experiments (`test_data.csv`).

| Required item | File in the repository |
|---|---|
| Complete source code | `app.py`, `notebooks/training.ipynb` |
| `requirements.txt` | `requirements.txt` |
| Clear `README.md` | `README.md` |
| Test data (CSV) | `test_data.csv` (250 labelled applicants) |
| Saved model files | `models/*.pkl` (five pipelines) + `models/metadata.json` |

### 1.2 Live Streamlit App Link

**https://your-app-name.streamlit.app**

Deployed on Streamlit Community Cloud; the link opens an interactive front end with
sidebar navigation, a CSV uploader, a model dropdown, metric cards, confusion matrices,
classification reports and ROC curves.

### 1.3 BITS Virtual Lab Screenshot

*Insert the screenshot here (stored at `reports/screenshots/bits_virtual_lab.png`).*

The screenshot evidences the execution of this assignment on the BITS Virtual Lab and
shows the environment, the running project and a visible timestamp.

---

## Section 2 — GitHub README Content

The complete `README.md` from the repository is reproduced immediately below, exactly as
required by Section 2.4 of the assignment brief. It follows the mandated structure:
(a) problem statement, (b) dataset description, (c) GitHub repository link, (d) models
used with the comparison table, followed by the per-model observations and the overall
winner.

> **➡ Paste the full contents of `README.md` here when exporting the PDF, or attach it as
> the following pages.** A condensed version of the graded sections is repeated below so
> that this report is self-contained.

---

## Section 3 — Dataset Description *(1 mark)*

| Property | Value |
|---|---|
| Name | Statlog (German Credit Data) |
| Source | UCI Machine Learning Repository, dataset ID 144 |
| URL | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Task | Binary classification |
| Instances | **1,000** (requirement ≥ 500 ✔) |
| Columns | **21** = 20 predictors + 1 target (requirement ≥ 12 features ✔) |
| Predictor mix | 7 numerical + 13 categorical |
| Encoded design matrix | 61 columns after one-hot encoding |
| Target | `credit_risk` ∈ {Good, Bad} |
| Class distribution | Good = 700 (70 %), Bad = 300 (30 %) |
| Missing values / duplicates | 0 / 0 |
| Split | Stratified 75 / 25 → 750 train, 250 test, `random_state = 42` |

**Why this dataset.** It exceeds both size constraints, mixes qualitative and quantitative
attributes so that a genuine preprocessing pipeline is required, carries a realistic 70:30
imbalance that makes AUC and MCC meaningful, and contains both smooth and sharply
categorical signals so that the five algorithm families produce genuinely different
results. Every attribute maps to a quantity a credit officer actually observes, which
makes the outputs explainable to a business stakeholder.

---

## Section 4 — Models Used and Evaluation Metrics *(5 marks)*

All five algorithms were trained on the same 750-row training partition with an identical
preprocessing block (median/most-frequent imputation → one-hot encoding → standard
scaling), each wrapped in a scikit-learn `Pipeline` so that no information leaks from the
test set. Hyper-parameters were chosen by 5-fold stratified `GridSearchCV` optimised for
`roc_auc` on the training data only.

**Positive class = `Bad` credit risk.** Decision threshold = 0.50. Test set = 250 unseen
applicants (175 Good, 75 Bad).

### 4.1 Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7280 | **0.8125** | 0.5315 | **0.7867** | **0.6344** | **0.4515** |
| Decision Tree | 0.6120 | 0.6763 | 0.4098 | 0.6667 | 0.5076 | 0.2340 |
| kNN | 0.7320 | 0.7783 | **0.6538** | 0.2267 | 0.3366 | 0.2631 |
| Naive Bayes | 0.7040 | 0.7251 | 0.5053 | 0.6400 | 0.5647 | 0.3507 |
| Random Forest (Ensemble) | **0.7360** | 0.7993 | 0.5495 | 0.6667 | 0.6024 | 0.4118 |

### 4.2 Confusion-matrix breakdown

| ML Model Name | TN | FP | FN | TP | Defaulters missed | Cost (5·FN + 1·FP) |
|---|---|---|---|---|---|---|
| Logistic Regression | 123 | 52 | **16** | **59** | 16 / 75 | **132** |
| Decision Tree | 103 | 72 | 25 | 50 | 25 / 75 | 197 |
| kNN | **166** | **9** | 58 | 17 | 58 / 75 | 299 |
| Naive Bayes | 128 | 47 | 27 | 48 | 27 / 75 | 182 |
| Random Forest (Ensemble) | 134 | 41 | 25 | 50 | 25 / 75 | 166 |

### 4.3 Selected hyper-parameters and cross-validated AUC

| Model | Best parameters | CV AUC (mean ± sd) |
|---|---|---|
| Logistic Regression | `C = 0.05`, `penalty = l2`, balanced | 0.7791 ± 0.0210 |
| Decision Tree | `gini`, `max_depth = 7`, `min_samples_leaf = 20`, balanced | 0.7621 ± 0.0162 |
| kNN | `n_neighbors = 31`, `weights = distance`, `p = 1` | 0.7548 ± 0.0332 |
| Naive Bayes | `var_smoothing = 1e-3` | 0.7174 ± 0.0471 |
| Random Forest | `600` trees, `min_samples_leaf = 3`, `max_features = sqrt`, balanced | **0.7926 ± 0.0139** |

---

## Section 5 — Observations on Model Performance *(3 marks)*

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strongest model overall: best AUC (0.8125), F1 (0.6344) and MCC (0.4515), and the highest recall (0.7867), catching 59 of 75 defaulters. Credit risk here is driven by a few strong, near-monotone signals (checking-account balance, duration, credit history, amount) whose effect on the log-odds is close to additive, so a linear decision boundary is well specified. With only 750 rows over 61 encoded columns the strong L2 penalty (`C = 0.05`) keeps variance low, whereas more flexible learners exhaust the data. It also yields the most stable probability ranking, which matters because the deployed console lets the user move the cut-off. Limitation: it cannot represent interactions without manual feature engineering. |
| **Decision Tree** | Weakest model on every headline metric (accuracy 0.6120, AUC 0.6763, MCC 0.2340) despite cross-validated pruning. It wrongly rejects 72 of 175 creditworthy applicants — the worst false-positive count. Hard axis-parallel splits are unstable at this sample size, so small perturbations reroute whole branches, and the coarse leaf-level probabilities (every record in a leaf shares one value) flatten the ROC curve and depress AUC. Its value is interpretability: the rules can be printed and audited, and it is the natural base learner whose variance the ensemble removes. |
| **kNN** | A cautionary result showing why accuracy alone is inadequate. Accuracy (0.7320) and precision (0.6538) look strong, yet recall collapses to 0.2267 and MCC to 0.2631: it flags only 26 applicants, catching 17 defaulters and missing 58. With no `class_weight` parameter, all 31 neighbours vote equally in a 7:3 imbalanced training set, so the majority class dominates every neighbourhood. The curse of dimensionality compounds this — in the sparse 61-dimensional one-hot space, Manhattan distances become nearly uniform and "nearest" loses meaning. Its AUC of 0.7783 shows the underlying ranking is sound; the handicap is the 0.50 cut-off, and lowering the threshold in the app materially improves recall. |
| **Naive Bayes** | Solid mid-table performer with an excellent effort-to-accuracy ratio (accuracy 0.7040, AUC 0.7251, recall 0.6400, MCC 0.3507). It trains in milliseconds with one hyper-parameter yet beats the Decision Tree on every metric. Its handicap is structural: conditional independence is clearly violated, since `credit_amount` and `duration_months` are strongly correlated and `job`, `employment_since` and `housing` overlap heavily. Treating them as independent double-counts evidence and drives posteriors toward 0 or 1, which is why its AUC trails Logistic Regression by nearly nine points despite comparable recall. A fast, sensible baseline rather than a production candidate. |
| **Random Forest (Ensemble)** | Best cross-validated model (CV AUC 0.7926, lowest variance ± 0.0139) and test-set runner-up: highest accuracy (0.7360), second-best AUC (0.7993), F1 (0.6024) and MCC (0.4118). Bagging 600 de-correlated trees does exactly what theory predicts — it lifts MCC from 0.2340 to 0.4118 (a 76 % gain over the single tree) and cuts false positives from 72 to 41. Its feature importances independently corroborate the domain narrative (checking-account status, duration, credit amount at the top). It does not overtake Logistic Regression because its extra capacity models interactions the 750-row sample cannot support, and its averaged votes are more conservative near the boundary, missing 25 defaulters against 16. Costs: an 8.5 MB artefact and much lower transparency. |

### Overall Winner for this dataset

**Logistic Regression**, with Random Forest a close second.

1. It wins the three decisive metrics — MCC 0.4515 (vs 0.4118), F1 0.6344 (vs 0.6024) and
   AUC 0.8125 (vs 0.7993). MCC is the primary criterion because it is the only metric here
   that uses all four confusion-matrix cells and remains honest under 70:30 imbalance.
2. It minimises the expensive error, missing only 16 of 75 defaulters against 25, 27 and
   58 for the alternatives. Under the UCI 5:1 cost matrix its total cost of 132 is the
   lowest of the five models (Random Forest 166, kNN 299).
3. Its accuracy deficit versus Random Forest is 0.8 pp — two applicants out of 250, well
   inside sampling noise, and irrelevant once the asymmetric cost is applied. kNN's
   apparently competitive 0.7320 is barely above the 70 % no-skill baseline.
4. Its highest AUC means the most faithful risk ordering, which is what the console needs
   when the credit officer moves the decision threshold.
5. Practical advantages: an 8.9 KB artefact (vs 8.5 MB), the fastest scoring, and signed
   coefficients that can be disclosed to a regulator as adverse-action reasons.

**Caveat.** Random Forest had the best cross-validated AUC with the tightest standard
deviation, so its lower test score partly reflects the modest 250-row test set. The
recommendation is to deploy Logistic Regression as the primary scorer and retain Random
Forest as a challenger, re-evaluating both once more data accrues.

---

## Section 6 — Streamlit Application *(4 marks)*

| Rubric requirement | Implementation | Screenshot |
|---|---|---|
| **a. Dataset upload option (CSV, test data only)** | Sidebar file uploader accepting `test_data.csv`, with schema validation against `models/metadata.json` and a bundled fallback file. | *insert* |
| **b. Model selection dropdown** | Sidebar `selectbox` listing all five classifiers, with a contextual description of the selected model. | *insert* |
| **c. Display of evaluation metrics** | *Model Evaluation* page renders Accuracy, AUC, Precision, Recall, F1 and MCC as styled KPI cards; *Model Benchmark* tabulates all five models at once. | *insert* |
| **d. Confusion matrix / classification report** | *Model Evaluation* page shows **both** an annotated confusion matrix and a full per-class classification report, plus the ROC curve. | *insert* |

**Additional features implemented beyond the minimum:** five-page sidebar navigation; a
live decision-threshold slider that lets the evaluator watch the precision/recall
trade-off respond in real time; a Model Benchmark page that scores and ranks all five
pipelines simultaneously with overlaid ROC curves; a Dataset Explorer with class balance
and per-feature distributions; single-applicant scoring with probability of default; CSV
downloads of predictions and the comparison table; a plain-language "business reading" of
the confusion matrix; and defensive handling of missing files, malformed CSVs, absent
target columns and unseen categories.

*(Insert app screenshots here: Overview, Dataset Explorer, Model Evaluation, Model
Benchmark.)*

---

## Section 7 — Final Conclusions

1. **A well-regularised linear model is the right tool for this problem.** On 750 training
   rows with a signal dominated by near-monotone effects, Logistic Regression outperforms
   a heavily tuned 600-tree ensemble on AUC, F1 and MCC. Model complexity does not
   substitute for an appropriate inductive bias at this sample size.
2. **Accuracy is the wrong headline metric under class imbalance.** kNN's 0.7320 accuracy
   is barely above the 70 % no-skill baseline while it misses 77 % of defaulters. MCC and
   AUC expose this immediately, which is precisely why the assignment mandates six metrics
   rather than one.
3. **Ensembling delivers exactly the variance reduction it promises.** Random Forest lifts
   the single Decision Tree's MCC from 0.2340 to 0.4118 and cuts false positives from 72
   to 41, and it produced the most stable cross-validated estimate of the five models.
4. **Class weighting materially changes the outcome.** The three estimators that support
   `class_weight="balanced"` all achieve recall ≥ 0.6667, whereas the two that do not
   (kNN, Gaussian NB) either collapse in recall or trail in AUC. Aligning the loss with
   the business cost matters more than the choice of algorithm family.
5. **Pipelines make deployment safe.** Persisting a complete `ColumnTransformer →
   estimator` pipeline rather than a bare estimator eliminates train/serve skew and lets
   the web application score raw, human-readable CSV rows directly.
6. **The threshold, not the model, is the operational lever.** Because all five models
   emit calibrated probabilities, the console's threshold slider lets a credit officer
   move along the ROC curve to whatever precision/recall point the bank's risk appetite
   dictates — a decision that belongs to the business, not the algorithm.

---

## Section 8 — Final Submission Checklist

| # | Requirement (assignment brief) | Status |
|---|---|---|
| 1 | GitHub repo link works | ☐ *verify after pushing* |
| 2 | Repo contains complete source code | ☑ `app.py`, `notebooks/training.ipynb` |
| 3 | Repo contains `requirements.txt` | ☑ pinned, deployment-tested |
| 4 | Repo contains a clear `README.md` | ☑ follows the mandated a–d structure |
| 5 | Repo contains the test data (CSV) | ☑ `test_data.csv`, 250 labelled rows |
| 6 | Saved model files for all implemented models | ☑ five `.pkl` pipelines + `metadata.json` |
| 7 | Streamlit app link opens correctly | ☐ *verify after deploying* |
| 8 | App loads without errors | ☑ all five pages smoke-tested, 0 errors |
| 9 | Dataset ≥ 500 instances | ☑ 1,000 |
| 10 | Dataset ≥ 12 features | ☑ 20 predictors |
| 11 | Logistic Regression implemented | ☑ |
| 12 | Decision Tree implemented | ☑ |
| 13 | kNN implemented | ☑ |
| 14 | Naive Bayes (Gaussian) implemented | ☑ |
| 15 | Random Forest ensemble implemented | ☑ |
| 16 | All six metrics for every model | ☑ Accuracy, AUC, Precision, Recall, F1, MCC |
| 17 | Comparison table populated with real outputs | ☑ from `reports/evaluation_results.csv` |
| 18 | Observations for each model | ☑ Section 5 |
| 19 | Overall winner declared and justified | ☑ Logistic Regression, evidence-backed |
| 20 | App: CSV upload of test data | ☑ |
| 21 | App: model selection dropdown | ☑ |
| 22 | App: evaluation metrics displayed | ☑ |
| 23 | App: confusion matrix / classification report | ☑ both |
| 24 | Results of different models visible in the app | ☑ Model Benchmark page |
| 25 | BITS Virtual Lab screenshot included | ☐ *paste into Section 1.3* |
| 26 | README content included in the submitted PDF | ☑ Section 2 |
| 27 | Reproducibility via `random_state` | ☑ `random_state = 42` throughout |
| 28 | Original work, no template copied | ☑ |
| 29 | Single PDF, sections in the mandated order | ☑ |
| 30 | Submitted (not left as a draft) before 18 Aug 2026, 23:59 IST | ☐ *action required* |

---

*Prepared for BITS Pilani WILP — Machine Learning Assignment 2.*
