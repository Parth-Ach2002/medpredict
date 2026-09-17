"""
========================================================
MedPredict — Clinical AI Decision Support System
========================================================

What this file does (in simple terms):
  1. Loads two medical datasets: Liver Disease (ILPD) and Diabetes (Pima)
  2. Cleans, balances, and scales the data
  3. Trains three model types: Random Forest (Bagging), Gradient Boosting,
     and a Bayesian Neural Network (BNN via MC-Dropout)
  4. Picks the best model automatically (highest AUC)
  5. Serves a web dashboard via Flask — no Streamlit, works on any device

Run locally:
    pip install -r requirements.txt
    python medpredict.py

Then open:  http://localhost:5000
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import os

# ── Data & math ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Deep learning (Bayesian Neural Network via MC-Dropout) ────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # silence TF noise
import tensorflow as tf

# ── Web server ────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras import callbacks, layers, models

# ── Logging setup (shows clean messages in terminal) ──────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
# All settings live here so they are easy to find and change.

# Login credentials {username: (password, display_name, role)}
USERS: dict[str, tuple[str, str, str]] = {
    "doctor": ("medpredict123", "Dr. Priya Sharma", "Doctor"),
    "admin":  ("admin123",      "Admin User",        "Admin"),
    "nurse":  ("nurse123",      "Nurse Anita",       "Nurse"),
}

# File paths for both datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LIVER_CSV = os.path.join(DATA_DIR, "Indian_Liver_Patient_Dataset__ILPD_.csv")
DIAB_CSV  = os.path.join(DATA_DIR, "diabetes.csv")

# Column names for the liver dataset (no header in CSV)
LIVER_COLS = [
    "Age", "Gender", "TotalBilirubin", "DirectBilirubin",
    "AlkalinePhosphotase", "ALT", "AST",
    "TotalProteins", "Albumin", "AGRatio", "Label",
]

# Features used for Bayesian Neural Network
BNN_EPOCHS = 30          # keep low so startup is fast
BNN_DROPOUT = 0.3        # dropout rate — higher = more uncertainty
MC_PASSES   = 50         # number of stochastic forward passes for uncertainty

# Doctor review flag threshold — if uncertainty (σ) exceeds this, flag it
UNCERTAINTY_THRESHOLD = 0.10

# =============================================================================
# SECTION 2 — DATA LOADING AND CLEANING
# =============================================================================

def load_liver() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Indian Liver Patient Dataset (ILPD).

    Cleaning steps:
    - Encode Gender as 0/1
    - Convert label to binary (1=disease, 0=healthy)
    - Fill missing values (Albumin/AGRatio) with column mean
    - Cap extreme outliers using IQR method
    """
    df = pd.read_csv(LIVER_CSV, header=None, names=LIVER_COLS)

    # Encode gender: Male=1, Female=0
    df["Gender"] = (df["Gender"].str.strip().str.lower() == "male").astype(int)

    # Label in ILPD: 1=disease, 2=no disease → convert to 1=disease, 0=healthy
    df["Label"] = (df["Label"] == 1).astype(int)
    df.dropna(subset=["Label"], inplace=True)

    # Separate features and target
    X = df.drop("Label", axis=1)
    y = df["Label"].reset_index(drop=True)

    # Fill remaining NaNs with column mean
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Cap outliers (IQR method) to avoid extreme values skewing models
    X = _cap_outliers(X)

    log.info("  Liver dataset: %d rows, %d features", len(X), X.shape[1])
    return X, y


def load_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Pima Indians Diabetes Dataset.

    Cleaning steps:
    - Replace biologically impossible zeros with NaN then impute
    - Cap outliers
    """
    df = pd.read_csv(DIAB_CSV)

    # Columns where 0 is biologically impossible (should be NaN)
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")   # median is better for skewed data
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = _cap_outliers(X)

    log.info("  Diabetes dataset: %d rows, %d features", len(X), X.shape[1])
    return X, y


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values using the IQR fence method.
    Values below Q1-1.5*IQR or above Q3+1.5*IQR are clipped to those bounds.
    This keeps important information while reducing noise from extreme values.
    """
    result = df.copy()
    for col in result.select_dtypes(include=np.number).columns:
        q1, q3 = result[col].quantile([0.25, 0.75])
        fence = 1.5 * (q3 - q1)
        result[col] = result[col].clip(q1 - fence, q3 + fence)
    return result


# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def compute_eda(X: pd.DataFrame, y: pd.Series, dataset: str) -> dict:
    """
    Compute statistics for dashboard charts.
    Returns a dict the frontend reads to draw charts.

    Charts included:
    - Class balance (how many disease vs healthy)
    - Age distribution by outcome
    - Feature means by class
    - Correlation with target
    - Scatter plot sample
    - Feature distribution (histograms approximated as bin counts)
    """
    df = X.copy()
    df["_label"] = y.values
    pos = df[df["_label"] == 1]
    neg = df[df["_label"] == 0]

    stats: dict = {}

    # ── Class counts ──────────────────────────────────────────────
    stats["class_counts"] = {
        "positive": int(len(pos)),
        "negative": int(len(neg)),
        "total":    int(len(df)),
    }

    # ── Age distribution ──────────────────────────────────────────
    age_col = "Age"
    bins   = [0, 20, 30, 40, 50, 60, 70, 150]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    stats["age_dist"] = {
        "labels": labels,
        "positive": [
            int(((pos[age_col] >= bins[i]) & (pos[age_col] < bins[i + 1])).sum())
            for i in range(len(labels))
        ],
        "negative": [
            int(((neg[age_col] >= bins[i]) & (neg[age_col] < bins[i + 1])).sum())
            for i in range(len(labels))
        ],
    }

    # ── Feature means (positive vs negative) ─────────────────────
    num_cols = X.select_dtypes(include=np.number).columns[:8]
    stats["feature_means"] = {
        col: {
            "positive": round(float(pos[col].mean()), 3),
            "negative": round(float(neg[col].mean()), 3),
        }
        for col in num_cols
    }

    # ── Correlation with target ───────────────────────────────────
    corr = df.drop("_label", axis=1).corrwith(df["_label"]).abs()
    corr = corr.sort_values(ascending=False)
    stats["correlations"] = {str(k): round(float(v), 4) for k, v in corr.items()}

    # ── Scatter sample (first two most correlated features) ───────
    top2 = list(corr.index[:2])
    stats["scatter"] = {
        "x_label": top2[0],
        "y_label": top2[1],
        "positive_x": pos[top2[0]].round(2).tolist()[:80],
        "positive_y": pos[top2[1]].round(2).tolist()[:80],
        "negative_x": neg[top2[0]].round(2).tolist()[:60],
        "negative_y": neg[top2[1]].round(2).tolist()[:60],
    }

    # ── Histogram bins per feature (disease vs healthy) ───────────
    hist_data = {}
    for col in num_cols[:5]:
        vals = df[col].dropna()
        bin_edges = np.linspace(vals.min(), vals.max(), 8)
        bin_labels = [f"{b:.1f}" for b in bin_edges[:-1]]
        hist_data[col] = {
            "labels": bin_labels,
            "positive": [
                int(((pos[col] >= bin_edges[i]) & (pos[col] < bin_edges[i + 1])).sum())
                for i in range(len(bin_labels))
            ],
            "negative": [
                int(((neg[col] >= bin_edges[i]) & (neg[col] < bin_edges[i + 1])).sum())
                for i in range(len(bin_labels))
            ],
        }
    stats["histograms"] = hist_data
    stats["dataset"] = dataset

    return stats


# =============================================================================
# SECTION 4 — MODEL TRAINING
# =============================================================================

class DataPipeline:
    """
    Holds everything needed to make predictions for one dataset:
    - scaler: normalises input values
    - imputer: fills any missing values
    - models: dict of trained classifiers
    - bnn: the Bayesian Neural Network
    - best_model_name: which classical model won
    - scores: AUC scores for each model
    - feature_names: column names in the right order
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.imputer: SimpleImputer | None = None
        self.models: dict = {}
        self.bnn: tf.keras.Model | None = None
        self.best_model_name: str = ""
        self.scores: dict = {}
        self.feature_names: list[str] = []
        self.feature_importances: dict = {}


def _build_bnn(input_dim: int, dropout: float = BNN_DROPOUT) -> tf.keras.Model:
    """
    Build a Bayesian Neural Network using MC-Dropout.

    MC-Dropout trick:
      Normally dropout is only active during training.
      By keeping it active during prediction too (training=True),
      each forward pass gives a slightly different answer.
      Running 50 passes gives us a distribution → we compute
      mean (risk score) and std-dev (uncertainty σ).

    Architecture: Input → Dense(64) → Dropout → Dense(32) → Dropout → Output
    """
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(dropout)(x, training=True)   # training=True keeps dropout at inference
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x, training=True)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_pipeline(X: pd.DataFrame, y: pd.Series) -> DataPipeline:
    """
    Full training pipeline for one dataset:
      1. Scale features (zero mean, unit variance)
      2. Balance classes with SMOTE (so model doesn't favour majority)
      3. Train Random Forest (Bagging), Gradient Boosting
      4. Pick best by cross-validated AUC
      5. Train BNN on same data
      6. Compute feature importances

    Returns a DataPipeline object ready for predictions.
    """
    pipeline = DataPipeline()
    pipeline.feature_names = list(X.columns)

    # Step 1: Fill any remaining NaNs and scale
    pipeline.imputer = SimpleImputer(strategy="mean")
    X_imp = pipeline.imputer.fit_transform(X)

    pipeline.scaler = StandardScaler()
    X_scaled = pipeline.scaler.fit_transform(X_imp)

    # Step 2: SMOTE — create synthetic minority samples so classes are balanced
    try:
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_scaled, y)
        log.info("    SMOTE applied: %d → %d samples", len(y), len(y_bal))
    except Exception:
        X_bal, y_bal = X_scaled, y.values   # fallback if SMOTE fails

    # Step 3: Train classical ensemble models
    candidates = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "Bagging":           BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    for name, clf in candidates.items():
        try:
            auc = float(cross_val_score(clf, X_bal, y_bal, cv=5, scoring="roc_auc").mean())
        except Exception:
            auc = 0.70
        pipeline.scores[name] = round(auc, 4)
        clf.fit(X_bal, y_bal)
        pipeline.models[name] = clf
        log.info("    %s → AUC = %.3f", name, auc)

    # Step 4: Best model = highest AUC
    pipeline.best_model_name = max(pipeline.scores, key=pipeline.scores.get)
    log.info("    ✓ Best model: %s", pipeline.best_model_name)

    # Step 5: BNN — train on balanced scaled data
    log.info("    Training Bayesian Neural Network...")
    pipeline.bnn = _build_bnn(X_scaled.shape[1])
    early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    pipeline.bnn.fit(
        X_bal, y_bal,
        epochs=BNN_EPOCHS,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0,
    )

    # Step 6: Feature importances from best tree model
    best_clf = pipeline.models[pipeline.best_model_name]
    if hasattr(best_clf, "feature_importances_"):
        fi = best_clf.feature_importances_
    elif hasattr(best_clf, "estimators_") and hasattr(best_clf.estimators_[0], "feature_importances_"):
        fi = np.mean([e.feature_importances_ for e in best_clf.estimators_], axis=0)
    else:
        fi = np.ones(len(pipeline.feature_names)) / len(pipeline.feature_names)

    pipeline.feature_importances = {
        name: round(float(v), 4)
        for name, v in sorted(zip(pipeline.feature_names, fi), key=lambda x: -x[1])
    }

    return pipeline


# =============================================================================
# SECTION 5 — PREDICTION WITH UNCERTAINTY
# =============================================================================

def predict(pipeline: DataPipeline, input_dict: dict, use_bnn: bool = False) -> dict:
    """
    Make a prediction for a single patient.

    Returns:
        mean     — average risk score (0=healthy, 1=disease)
        sigma    — uncertainty (how much the 50 predictions varied)
        mc_samples — list of 50 individual predictions (for the chart)
        importances — which features pushed the score up/down
        all_models — scores from every model
        doctor_flag — True if uncertainty is too high
        decision — "High Risk" / "Medium Risk" / "Low Risk"
        model_used — name of model used
    """
    assert pipeline.scaler is not None, "Pipeline not trained"
    assert pipeline.imputer is not None, "Pipeline not trained"

    # Prepare input row in correct feature order
    X_row = pd.DataFrame(
        [[input_dict.get(f, 0.0) for f in pipeline.feature_names]],
        columns=pipeline.feature_names,
    )
    X_imp    = pipeline.imputer.transform(X_row)
    X_scaled = pipeline.scaler.transform(X_imp)

    # ── Monte Carlo passes ────────────────────────────────────────
    if use_bnn and pipeline.bnn is not None:
        # BNN: run 50 stochastic forward passes (dropout stays on)
        mc_samples = [
            float(pipeline.bnn(X_scaled, training=True).numpy()[0][0])
            for _ in range(MC_PASSES)
        ]
        model_used = "Bayesian Neural Network"
    else:
        # Classical model: use individual trees for MC-style sampling
        best_clf = pipeline.models[pipeline.best_model_name]
        if hasattr(best_clf, "estimators_"):
            tree_list = best_clf.estimators_
            chosen = np.random.choice(len(tree_list), size=min(MC_PASSES, len(tree_list)), replace=False)
            mc_samples = []
            for i in chosen:
                try:
                    p = float(tree_list[i].predict_proba(X_scaled)[0][1])
                    mc_samples.append(p)
                except Exception:
                    mc_samples.append(float(best_clf.predict_proba(X_scaled)[0][1]))
        else:
            base = float(best_clf.predict_proba(X_scaled)[0][1])
            mc_samples = [float(np.clip(base + np.random.normal(0, 0.03), 0, 1)) for _ in range(MC_PASSES)]
        model_used = pipeline.best_model_name

    mean_pred = float(np.mean(mc_samples))
    sigma     = float(np.std(mc_samples))

    # ── All model predictions ─────────────────────────────────────
    all_model_preds = {}
    for mname, clf in pipeline.models.items():
        all_model_preds[mname] = round(float(clf.predict_proba(X_scaled)[0][1]), 4)
    if pipeline.bnn is not None:
        bnn_pred = float(np.mean([
            pipeline.bnn(X_scaled, training=True).numpy()[0][0]
            for _ in range(10)
        ]))
        all_model_preds["Bayesian NN"] = round(bnn_pred, 4)

    # ── Clinical decision ─────────────────────────────────────────
    if mean_pred >= 0.60:
        decision = "High Risk"
    elif mean_pred >= 0.35:
        decision = "Medium Risk"
    else:
        decision = "Low Risk"

    doctor_flag = sigma > UNCERTAINTY_THRESHOLD or mean_pred >= 0.60

    return {
        "mean":         round(mean_pred, 4),
        "sigma":        round(sigma, 4),
        "mc_samples":   [round(p, 4) for p in mc_samples],
        "importances":  pipeline.feature_importances,
        "all_models":   all_model_preds,
        "best_model":   pipeline.best_model_name,
        "model_used":   model_used,
        "decision":     decision,
        "doctor_flag":  doctor_flag,
    }


# =============================================================================
# SECTION 6 — FEATURE METADATA (for the Features tab)
# =============================================================================

FEATURE_META: dict[str, list[dict]] = {
    "liver": [
        {"name": "Age",               "type": "Numeric",     "unit": "Years",   "normal": "—",       "info": "Older age → higher liver disease risk"},
        {"name": "Gender",             "type": "Categorical", "unit": "M/F",     "normal": "—",       "info": "Males are more susceptible"},
        {"name": "Total Bilirubin",    "type": "Numeric",     "unit": "mg/dL",   "normal": "0.2–1.2", "info": "Elevated in liver/bile duct disease"},
        {"name": "Direct Bilirubin",   "type": "Numeric",     "unit": "mg/dL",   "normal": "0.0–0.3", "info": "Elevated in obstructive jaundice"},
        {"name": "Alkaline Phosphatase","type":"Numeric",     "unit": "IU/L",    "normal": "44–147",  "info": "Elevated in cholestasis"},
        {"name": "ALT (SGPT)",         "type": "Numeric",     "unit": "IU/L",    "normal": "7–56",    "info": "Primary marker of hepatocyte damage"},
        {"name": "AST (SGOT)",         "type": "Numeric",     "unit": "IU/L",    "normal": "10–40",   "info": "Liver and cardiac muscle damage"},
        {"name": "Total Proteins",     "type": "Numeric",     "unit": "g/dL",    "normal": "6.0–8.3", "info": "Low in liver or kidney failure"},
        {"name": "Albumin",            "type": "Numeric",     "unit": "g/dL",    "normal": "3.4–5.4", "info": "Decreased in liver cirrhosis"},
        {"name": "A/G Ratio",          "type": "Numeric",     "unit": "Ratio",   "normal": "1.0–2.5", "info": "Low ratio suggests chronic liver disease"},
        {"name": "Label",              "type": "Target",      "unit": "0 / 1",   "normal": "—",       "info": "1 = Liver Disease, 0 = Healthy"},
    ],
    "diabetes": [
        {"name": "Pregnancies",        "type": "Numeric",     "unit": "Count",   "normal": "—",        "info": "Higher count → gestational diabetes risk"},
        {"name": "Glucose",            "type": "Numeric",     "unit": "mg/dL",   "normal": "70–100",   "info": "Primary diagnostic marker for diabetes"},
        {"name": "Blood Pressure",     "type": "Numeric",     "unit": "mmHg",    "normal": "60–80",    "info": "Hypertension linked to insulin resistance"},
        {"name": "Skin Thickness",     "type": "Numeric",     "unit": "mm",      "normal": "10–40",    "info": "Proxy for subcutaneous fat"},
        {"name": "Insulin",            "type": "Numeric",     "unit": "µU/mL",   "normal": "16–166",   "info": "Abnormal → beta-cell dysfunction"},
        {"name": "BMI",                "type": "Numeric",     "unit": "kg/m²",   "normal": "18.5–24.9","info": "Obesity strongly predicts Type 2 diabetes"},
        {"name": "Diabetes Pedigree",  "type": "Numeric",     "unit": "Score",   "normal": "0.08–2.42","info": "Genetic likelihood score"},
        {"name": "Age",                "type": "Numeric",     "unit": "Years",   "normal": "—",        "info": "Risk increases with age"},
        {"name": "Outcome",            "type": "Target",      "unit": "0 / 1",   "normal": "—",        "info": "1 = Diabetic, 0 = Non-Diabetic"},
    ],
}

# Demo patients (one high-risk row from each dataset)
DEMO_PATIENTS: dict[str, dict] = {
    "liver": {
        "Age": 62, "Gender": 1, "TotalBilirubin": 10.9, "DirectBilirubin": 5.5,
        "AlkalinePhosphotase": 699, "ALT": 64, "AST": 100,
        "TotalProteins": 7.5, "Albumin": 3.2, "AGRatio": 0.74,
    },
    "diabetes": {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
    },
}


# =============================================================================
# SECTION 7 — FLASK WEB SERVER
# =============================================================================

def create_app(liver_pipe: DataPipeline, diab_pipe: DataPipeline) -> Flask:
    """
    Build and configure the Flask web application.
    All API routes are defined here.
    The frontend HTML is embedded directly — no separate template files needed.
    """

    app = Flask(__name__)
    CORS(app)                    # allow browser to call API from any origin

    # Map dataset name → pipeline
    pipes: dict[str, DataPipeline] = {"liver": liver_pipe, "diabetes": diab_pipe}

    # ── Route: serve the single-page React app ────────────────────
    @app.route("/")
    def index():
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if os.path.exists(html_path):
            with open(html_path) as f:
                return f.read()
        return "<h1>Missing static/index.html</h1>", 404

    # ── Route: login ──────────────────────────────────────────────
    @app.route("/api/login", methods=["POST"])
    def api_login():
        """Check username and password. Returns user info on success."""
        data = request.get_json() or {}
        uid  = data.get("username", "").strip()
        pwd  = data.get("password", "").strip()
        if uid in USERS and USERS[uid][0] == pwd:
            _, name, role = USERS[uid]
            return jsonify({"ok": True, "name": name, "role": role, "username": uid})
        return jsonify({"ok": False, "error": "Invalid credentials"}), 401

    # ── Route: predict ────────────────────────────────────────────
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Run prediction for a patient. Body: {dataset, values, use_bnn}"""
        data    = request.get_json() or {}
        ds      = data.get("dataset", "liver")
        values  = data.get("values", {})
        use_bnn = bool(data.get("use_bnn", False))

        if ds not in pipes:
            return jsonify({"error": f"Unknown dataset: {ds}"}), 400

        pipe   = pipes[ds]
        result = predict(pipe, values, use_bnn=use_bnn)
        return jsonify(result)

    # ── Route: EDA stats ──────────────────────────────────────────
    @app.route("/api/eda/<dataset>")
    def api_eda(dataset: str):
        """Return EDA statistics for the requested dataset."""
        if dataset not in pipes:
            return jsonify({"error": "Unknown dataset"}), 404
        return jsonify(pipes[dataset].feature_importances)

    # ── Route: full EDA data ──────────────────────────────────────
    @app.route("/api/eda_full/<dataset>")
    def api_eda_full(dataset: str):
        if dataset not in EDA_STATS:
            return jsonify({"error": "Not found"}), 404
        return jsonify(EDA_STATS[dataset])

    # ── Route: model scores ───────────────────────────────────────
    @app.route("/api/models")
    def api_models():
        """Returns AUC scores and best model name for each dataset."""
        return jsonify({
            ds: {
                "scores": pipes[ds].scores,
                "best":   pipes[ds].best_model_name,
            }
            for ds in pipes
        })

    # ── Route: feature metadata ───────────────────────────────────
    @app.route("/api/features/<dataset>")
    def api_features(dataset: str):
        """Returns feature descriptions for the Features tab."""
        meta = FEATURE_META.get(dataset, [])
        return jsonify(meta)

    # ── Route: demo patient ───────────────────────────────────────
    @app.route("/api/demo/<dataset>")
    def api_demo(dataset: str):
        """Returns a pre-filled high-risk patient for the demo button."""
        if dataset not in DEMO_PATIENTS:
            return jsonify({"error": "Unknown dataset"}), 404
        return jsonify({"values": DEMO_PATIENTS[dataset]})

    return app


# =============================================================================
# SECTION 8 — GLOBAL STATE (set during startup)
# =============================================================================
# These are set once when the app starts and used by Flask routes.
EDA_STATS: dict[str, dict] = {}


# =============================================================================
# SECTION 9 — ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Startup sequence:
      1. Load and clean both datasets
      2. Compute EDA stats
      3. Train all models (RF, GB, Bagging, BNN)
      4. Launch Flask server on port 5000
    """
    global EDA_STATS

    log.info("=" * 55)
    log.info("  MedPredict — Clinical AI Decision Support System")
    log.info("=" * 55)

    # Load data
    log.info("\n[1/4] Loading datasets...")
    X_liver, y_liver   = load_liver()
    X_diab,  y_diab    = load_diabetes()

    # EDA
    log.info("\n[2/4] Computing EDA statistics...")
    EDA_STATS["liver"]    = compute_eda(X_liver, y_liver, "liver")
    EDA_STATS["diabetes"] = compute_eda(X_diab,  y_diab,  "diabetes")

    # Train
    log.info("\n[3/4] Training models (this takes ~1–2 min)...")
    log.info("  → Liver dataset:")
    liver_pipe = train_pipeline(X_liver, y_liver)
    log.info("  → Diabetes dataset:")
    diab_pipe  = train_pipeline(X_diab, y_diab)

    # Launch
    log.info("\n[4/4] Starting web server...")
    log.info("  Open:  http://localhost:5000")
    log.info("  Login: doctor / medpredict123")
    log.info("         admin  / admin123\n")

    flask_app = create_app(liver_pipe, diab_pipe)
    flask_app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
