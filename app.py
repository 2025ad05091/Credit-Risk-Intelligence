"""Credit Risk Intelligence Console - Streamlit front-end.

BITS Pilani WILP | M.Tech (AIML / DSE) | Machine Learning Assignment 2

An interactive web application that scores the UCI Statlog (German Credit) test
set with five pre-trained scikit-learn pipelines and reports the six evaluation
metrics required by the assignment.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Application constants
# --------------------------------------------------------------------------- #

APP_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = APP_ROOT / "models"
REPORT_DIR = APP_ROOT / "reports"
DEFAULT_TEST_FILE = APP_ROOT / "test_data.csv"

MODEL_CATALOGUE: dict[str, dict[str, str]] = {
    "Logistic Regression": {
        "file": "logistic_regression.pkl",
        "family": "Linear / parametric",
        "note": "L2-regularised linear model with balanced class weights.",
    },
    "Decision Tree": {
        "file": "decision_tree.pkl",
        "family": "Rule based / non-parametric",
        "note": "Depth-limited CART tree, fully interpretable decision rules.",
    },
    "KNN": {
        "file": "knn.pkl",
        "family": "Instance based / lazy learner",
        "note": "Distance-weighted neighbourhood vote on scaled features.",
    },
    "Naive Bayes": {
        "file": "naive_bayes.pkl",
        "family": "Probabilistic / generative",
        "note": "Gaussian likelihoods with a conditional-independence assumption.",
    },
    "Random Forest": {
        "file": "random_forest.pkl",
        "family": "Bagging ensemble",
        "note": "600 de-correlated trees aggregated by soft voting.",
    },
}

METRIC_HELP = {
    "Accuracy": "Share of applicants classified correctly.",
    "AUC": "Ranking quality of the risk score, independent of the cut-off.",
    "Precision": "Of the applicants flagged Bad, how many truly defaulted.",
    "Recall": "Of the truly Bad applicants, how many were caught.",
    "F1": "Harmonic mean of Precision and Recall.",
    "MCC": "Balanced correlation coefficient, robust to class imbalance.",
}

st.set_page_config(
    page_title="Credit Risk Intelligence Console",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bespoke styling so the console does not look like a stock Streamlit template.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      .cr-banner {
          background: linear-gradient(120deg, #10243e 0%, #1f4e79 55%, #2e7d8f 100%);
          padding: 1.5rem 1.9rem; border-radius: 14px; color: #f4f8fb;
          box-shadow: 0 6px 18px rgba(16,36,62,0.28); margin-bottom: 1.4rem;
      }
      .cr-banner h1 {margin: 0; font-size: 2.0rem; letter-spacing: 0.4px;}
      .cr-banner p {margin: 0.45rem 0 0 0; font-size: 0.98rem; opacity: 0.92;}
      .cr-chip {
          display: inline-block; background: rgba(255,255,255,0.16);
          border-radius: 999px; padding: 0.18rem 0.8rem; font-size: 0.78rem;
          margin-right: 0.45rem; margin-top: 0.65rem;
      }
      .cr-card {
          background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #1f4e79;
          border-radius: 11px; padding: 0.85rem 1rem; height: 100%;
          box-shadow: 0 2px 6px rgba(15,23,42,0.06);
      }
      .cr-card .label {font-size: 0.76rem; text-transform: uppercase;
          letter-spacing: 0.7px; color: #64748b; font-weight: 600;}
      .cr-card .value {font-size: 1.72rem; font-weight: 700; color: #10243e;}
      .cr-card .foot {font-size: 0.72rem; color: #94a3b8;}
      .cr-note {background:#f1f5f9; border-radius:9px; padding:0.75rem 1rem;
          font-size:0.88rem; border-left:4px solid #2e7d8f;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Return the training metadata written by the notebook."""
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading trained pipelines ...")
def load_models() -> dict[str, object]:
    """Deserialise every available pipeline from the models directory."""
    loaded: dict[str, object] = {}
    for name, spec in MODEL_CATALOGUE.items():
        path = MODEL_DIR / spec["file"]
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                st.warning(f"Could not load '{name}': {error}")
    return loaded


@st.cache_data(show_spinner=False)
def load_default_test_data() -> pd.DataFrame | None:
    """Load the bundled held-out test set, if it is present."""
    if DEFAULT_TEST_FILE.exists():
        return pd.read_csv(DEFAULT_TEST_FILE)
    return None


@st.cache_data(show_spinner=False)
def load_training_scores() -> pd.DataFrame | None:
    """Load the notebook comparison table used on the benchmark page."""
    path = REPORT_DIR / "evaluation_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


META = load_metadata()
TARGET_COLUMN = META.get("target_column", "credit_risk")
POSITIVE_LABEL = META.get("positive_label", "Bad")
NEGATIVE_LABEL = META.get("negative_label", "Good")
FEATURE_ORDER = META.get("feature_order", [])


# --------------------------------------------------------------------------- #
# Domain helpers
# --------------------------------------------------------------------------- #


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate predictors from the ground-truth column when it is present."""
    if TARGET_COLUMN in frame.columns:
        labels = frame[TARGET_COLUMN]
        binary = (labels.astype(str).str.strip() == POSITIVE_LABEL).astype(int)
        return frame.drop(columns=[TARGET_COLUMN]), binary
    return frame.copy(), None


def validate_schema(features: pd.DataFrame) -> list[str]:
    """Return the list of predictor columns that the uploaded file is missing."""
    if not FEATURE_ORDER:
        return []
    return [column for column in FEATURE_ORDER if column not in features.columns]


def score_model(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and positive-class probabilities."""
    ordered = features[FEATURE_ORDER] if FEATURE_ORDER else features
    predictions = model.predict(ordered)
    probabilities = model.predict_proba(ordered)[:, 1]
    return predictions, probabilities


def compute_metrics(truth: pd.Series, predictions: np.ndarray, scores: np.ndarray) -> dict:
    """Compute the six evaluation metrics mandated by the assignment."""
    return {
        "Accuracy": accuracy_score(truth, predictions),
        "AUC": roc_auc_score(truth, scores),
        "Precision": precision_score(truth, predictions, zero_division=0),
        "Recall": recall_score(truth, predictions, zero_division=0),
        "F1": f1_score(truth, predictions, zero_division=0),
        "MCC": matthews_corrcoef(truth, predictions),
    }


def metric_cards(metrics: dict[str, float]) -> None:
    """Render the metric dictionary as a row of styled KPI cards."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.markdown(
            f"""
            <div class="cr-card">
              <div class="label">{label}</div>
              <div class="value">{value:.4f}</div>
              <div class="foot">{METRIC_HELP.get(label, "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_confusion(truth: pd.Series, predictions: np.ndarray, title: str):
    """Draw an annotated confusion matrix."""
    matrix = confusion_matrix(truth, predictions)
    figure, axis = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def plot_roc(truth: pd.Series, scores: np.ndarray, label: str):
    """Draw the ROC curve of a single model."""
    fpr, tpr, _ = roc_curve(truth, scores)
    figure, axis = plt.subplots(figsize=(4.4, 3.6))
    axis.plot(fpr, tpr, color="#1f4e79", linewidth=2,
              label=f"{label} (AUC = {roc_auc_score(truth, scores):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Random")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC curve", fontsize=10)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a data frame for the Streamlit download button."""
    buffer = BytesIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar - navigation and data source
# --------------------------------------------------------------------------- #

MODELS = load_models()

st.sidebar.title("Credit Risk Console")
st.sidebar.caption("BITS Pilani WILP | ML Assignment 2")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Dataset Explorer", "Model Evaluation", "Model Benchmark", "Single Applicant"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Test data source")
uploaded = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the held-out test set. The bundled file is used if nothing is uploaded.",
)

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        source_label = f"Uploaded file: {uploaded.name}"
    except Exception as error:  # noqa: BLE001 - user supplied file
        st.sidebar.error(f"Could not parse the CSV: {error}")
        data = load_default_test_data()
        source_label = "Bundled test_data.csv (upload failed)"
else:
    data = load_default_test_data()
    source_label = "Bundled test_data.csv"

if data is None:
    st.error(
        "No test data available. Upload a CSV in the sidebar or place `test_data.csv` "
        "next to `app.py`."
    )
    st.stop()

if not MODELS:
    st.error("No trained models found in `models/`. Run `notebooks/training.ipynb` first.")
    st.stop()

st.sidebar.success(f"{source_label}\n\n{data.shape[0]} rows x {data.shape[1]} columns")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Model selection")
selected_model_name = st.sidebar.selectbox("Choose a classifier", list(MODELS.keys()))
st.sidebar.caption(MODEL_CATALOGUE[selected_model_name]["note"])

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold on P(Bad)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Lower the threshold to catch more defaulters at the cost of more false alarms.",
)

X_input, y_true = split_features_target(data)
missing_columns = validate_schema(X_input)
if missing_columns:
    st.error(
        "The uploaded file is missing required predictor columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #

if page == "Overview":
    st.markdown(
        f"""
        <div class="cr-banner">
          <h1>Credit Risk Intelligence Console</h1>
          <p>Comparing five supervised classifiers on the UCI Statlog (German Credit) data
          to decide whether a loan applicant is a <b>Good</b> or a <b>Bad</b> credit risk.</p>
          <span class="cr-chip">{len(MODELS)} trained pipelines</span>
          <span class="cr-chip">{data.shape[0]} test applicants</span>
          <span class="cr-chip">{len(FEATURE_ORDER) or data.shape[1]} predictors</span>
          <span class="cr-chip">Positive class = {POSITIVE_LABEL} risk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Business problem")
        st.write(
            "A retail bank must decide which loan applications to approve. Approving an "
            "applicant who later defaults destroys capital, while rejecting a "
            "creditworthy applicant only forgoes interest income. The UCI cost matrix "
            "weighs the first error five times heavier than the second, so the console "
            "treats **Bad credit risk as the positive class** and emphasises Recall, "
            "AUC and MCC rather than raw Accuracy."
        )
        st.subheader("How to use this console")
        st.markdown(
            """
            1. **Dataset Explorer** - inspect the uploaded test file and its class balance.
            2. **Model Evaluation** - pick a classifier and read all six metrics, the
               confusion matrix, the classification report and the ROC curve.
            3. **Model Benchmark** - score every classifier at once and rank them.
            4. **Single Applicant** - score one record and read the probability of default.
            """
        )
    with right:
        st.subheader("Model catalogue")
        catalogue = pd.DataFrame(
            [
                {"Model": name, "Family": spec["family"], "Loaded": name in MODELS}
                for name, spec in MODEL_CATALOGUE.items()
            ]
        )
        st.dataframe(catalogue, hide_index=True, width="stretch")
        if META:
            st.markdown(
                f"""
                <div class="cr-note">
                <b>Training provenance</b><br>
                Dataset: {META.get('dataset', 'n/a')}<br>
                Train / test rows: {META.get('n_train_rows', '?')} / {META.get('n_test_rows', '?')}<br>
                random_state: {META.get('random_state', '?')} &nbsp;|&nbsp;
                scikit-learn: {META.get('sklearn_version', '?')}
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# Page: Dataset Explorer
# --------------------------------------------------------------------------- #

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.caption(source_label)

    top = st.columns(4)
    top[0].metric("Rows", f"{data.shape[0]}")
    top[1].metric("Columns", f"{data.shape[1]}")
    top[2].metric("Missing cells", f"{int(data.isna().sum().sum())}")
    top[3].metric(
        "Bad-risk rate",
        f"{y_true.mean():.1%}" if y_true is not None else "n/a",
    )

    st.subheader("Preview")
    st.dataframe(data.head(25), width="stretch")

    explorer_left, explorer_right = st.columns(2)
    with explorer_left:
        st.subheader("Class balance")
        if y_true is not None:
            counts = data[TARGET_COLUMN].value_counts()
            figure, axis = plt.subplots(figsize=(4.4, 3.4))
            sns.barplot(
                x=counts.index,
                y=counts.values,
                hue=counts.index,
                palette="Blues_r",
                legend=False,
                ax=axis,
            )
            axis.set_ylabel("Applicants")
            axis.set_xlabel("")
            axis.set_title("Distribution of credit risk", fontsize=10)
            figure.tight_layout()
            st.pyplot(figure)
        else:
            st.info("The uploaded file has no target column, so metrics cannot be computed.")

    with explorer_right:
        st.subheader("Feature distribution")
        numeric_columns = X_input.select_dtypes(include=np.number).columns.tolist()
        object_columns = X_input.select_dtypes(exclude=np.number).columns.tolist()
        chosen = st.selectbox("Select a feature", numeric_columns + object_columns)
        figure, axis = plt.subplots(figsize=(4.4, 3.4))
        if chosen in numeric_columns:
            sns.histplot(data=data, x=chosen, hue=TARGET_COLUMN if y_true is not None else None,
                         bins=25, element="step", ax=axis)
        else:
            order = data[chosen].value_counts().index
            sns.countplot(data=data, y=chosen, order=order, ax=axis, color="#2e7d8f")
        axis.set_title(chosen, fontsize=10)
        figure.tight_layout()
        st.pyplot(figure)

    with st.expander("Descriptive statistics of numerical attributes"):
        st.dataframe(X_input.describe().T.round(2), width="stretch")

# --------------------------------------------------------------------------- #
# Page: Model Evaluation
# --------------------------------------------------------------------------- #

elif page == "Model Evaluation":
    st.header(f"Model Evaluation - {selected_model_name}")
    st.caption(
        f"{MODEL_CATALOGUE[selected_model_name]['family']} | "
        f"decision threshold on P({POSITIVE_LABEL}) = {threshold:.2f}"
    )

    model = MODELS[selected_model_name]
    try:
        default_predictions, probabilities = score_model(model, X_input)
    except Exception as error:  # noqa: BLE001 - user supplied file
        st.error(f"Scoring failed. Check that the CSV schema matches the training data.\n\n{error}")
        st.stop()

    predictions = (probabilities >= threshold).astype(int)

    if y_true is None:
        st.warning(
            f"Column `{TARGET_COLUMN}` is absent, so only predictions are shown. "
            "Upload a labelled test file to see the evaluation metrics."
        )
    else:
        metrics = compute_metrics(y_true, predictions, probabilities)
        metric_cards(metrics)
        st.markdown("")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.subheader("Confusion matrix")
            st.pyplot(plot_confusion(y_true, predictions, selected_model_name))
        with chart_right:
            st.subheader("ROC curve")
            st.pyplot(plot_roc(y_true, probabilities, selected_model_name))

        st.subheader("Classification report")
        report = classification_report(
            y_true,
            predictions,
            target_names=[f"{NEGATIVE_LABEL} risk", f"{POSITIVE_LABEL} risk"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

        matrix = confusion_matrix(y_true, predictions)
        true_neg, false_pos, false_neg, true_pos = matrix.ravel()
        st.markdown(
            f"""
            <div class="cr-note">
            <b>Business reading:</b> {true_pos} defaulters were correctly flagged and
            {false_neg} slipped through (costly misses), while {false_pos} creditworthy
            applicants were wrongly rejected out of {true_neg + false_pos} good applicants.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Scored records")
    scored = data.copy()
    scored["predicted_risk"] = np.where(predictions == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    scored[f"probability_{POSITIVE_LABEL.lower()}"] = probabilities.round(4)
    st.dataframe(scored.head(30), width="stretch")
    st.download_button(
        "Download full predictions (CSV)",
        data=to_csv_bytes(scored),
        file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Page: Model Benchmark
# --------------------------------------------------------------------------- #

elif page == "Model Benchmark":
    st.header("Model Benchmark")
    st.caption("Every loaded pipeline scored on the same test data at a 0.50 threshold.")

    if y_true is None:
        st.warning(
            f"The benchmark needs the ground-truth column `{TARGET_COLUMN}`. "
            "Upload the labelled test file to compare models."
        )
        st.stop()

    rows, curves = [], {}
    for name, model in MODELS.items():
        try:
            predictions, probabilities = score_model(model, X_input)
        except Exception as error:  # noqa: BLE001
            st.warning(f"Skipped '{name}': {error}")
            continue
        rows.append({"ML Model": name, **compute_metrics(y_true, predictions, probabilities)})
        curves[name] = probabilities

    comparison = pd.DataFrame(rows).set_index("ML Model").round(4)
    st.subheader("Comparison table")
    st.dataframe(
        comparison.style.background_gradient(cmap="Greens", axis=0).format("{:.4f}"),
        width="stretch",
    )

    winner_mcc = comparison["MCC"].idxmax()
    winner_auc = comparison["AUC"].idxmax()
    summary = st.columns(3)
    summary[0].metric("Best MCC", winner_mcc, f"{comparison.loc[winner_mcc, 'MCC']:.4f}")
    summary[1].metric("Best AUC", winner_auc, f"{comparison.loc[winner_auc, 'AUC']:.4f}")
    summary[2].metric(
        "Best Recall",
        comparison["Recall"].idxmax(),
        f"{comparison['Recall'].max():.4f}",
    )

    plot_left, plot_right = st.columns(2)
    with plot_left:
        st.subheader("Metrics side by side")
        melted = comparison.reset_index().melt(
            id_vars="ML Model", var_name="Metric", value_name="Score"
        )
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        sns.barplot(data=melted, x="Metric", y="Score", hue="ML Model", ax=axis)
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
        figure.tight_layout()
        st.pyplot(figure)

    with plot_right:
        st.subheader("ROC curves")
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for name, scores in curves.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            axis.plot(fpr, tpr, linewidth=1.6,
                      label=f"{name} ({roc_auc_score(y_true, scores):.3f})")
        axis.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        st.pyplot(figure)

    st.download_button(
        "Download comparison table (CSV)",
        data=to_csv_bytes(comparison.reset_index()),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    reference = load_training_scores()
    if reference is not None:
        with st.expander("Reference results recorded in the training notebook"):
            st.dataframe(reference, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Page: Single Applicant
# --------------------------------------------------------------------------- #

else:
    st.header("Single Applicant Scoring")
    st.caption(f"Scored with {selected_model_name} at a threshold of {threshold:.2f}.")

    index = st.number_input(
        "Select a row from the loaded test data",
        min_value=0,
        max_value=len(X_input) - 1,
        value=0,
        step=1,
    )
    record = X_input.iloc[[int(index)]]

    st.subheader("Applicant profile")
    # Transposing mixes ints and strings in one column, which pyarrow cannot
    # serialise; cast to str so the table renders without an Arrow fallback.
    profile = record.T.reset_index()
    profile.columns = ["Attribute", "Value"]
    profile["Value"] = profile["Value"].astype(str)
    left, right = st.columns([1.2, 1])
    left.dataframe(profile, hide_index=True, width="stretch", height=430)

    model = MODELS[selected_model_name]
    probability = float(score_model(model, record)[1][0])
    decision = POSITIVE_LABEL if probability >= threshold else NEGATIVE_LABEL

    with right:
        st.markdown(
            f"""
            <div class="cr-card">
              <div class="label">Predicted class</div>
              <div class="value">{decision} risk</div>
              <div class="foot">P({POSITIVE_LABEL}) = {probability:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))
        if y_true is not None:
            actual = POSITIVE_LABEL if y_true.iloc[int(index)] == 1 else NEGATIVE_LABEL
            if actual == decision:
                st.success(f"Ground truth: {actual} risk - prediction is correct.")
            else:
                st.error(f"Ground truth: {actual} risk - prediction is incorrect.")
        st.markdown(
            f"""
            <div class="cr-note">
            A probability above the {threshold:.2f} cut-off routes the application to
            manual underwriting. Lowering the cut-off in the sidebar increases the share
            of defaulters detected at the price of more false alarms.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: UCI Statlog (German Credit). Built with scikit-learn pipelines and Streamlit."
)
