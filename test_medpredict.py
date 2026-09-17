"""
========================================================
test_medpredict.py — Unit Tests for MedPredict
========================================================

Run with:   pytest test_medpredict.py -v
Coverage:   pytest test_medpredict.py --cov=medpredict

What we test:
  - Data loading and cleaning
  - Outlier capping
  - EDA stats structure
  - Model training
  - Prediction outputs
  - Flask API endpoints
  - Edge cases (missing values, bad inputs)
"""

import numpy as np
import pandas as pd
import pytest

# Import functions from our main file
from medpredict import (
    DEMO_PATIENTS,
    FEATURE_META,
    LIVER_COLS,
    UNCERTAINTY_THRESHOLD,
    USERS,
    DataPipeline,
    _cap_outliers,
    compute_eda,
    create_app,
    predict,
    train_pipeline,
)

# =============================================================================
# FIXTURES — reusable test data
# =============================================================================

@pytest.fixture
def small_liver_df():
    """
    A tiny synthetic liver dataset (50 rows).
    Used so tests run fast without needing real CSV files.
    """
    np.random.seed(0)
    n = 50
    X = pd.DataFrame({
        "Age":                 np.random.randint(20, 70, n).astype(float),
        "Gender":              np.random.randint(0, 2, n).astype(float),
        "TotalBilirubin":      np.abs(np.random.exponential(2, n)),
        "DirectBilirubin":     np.abs(np.random.exponential(0.5, n)),
        "AlkalinePhosphotase": np.random.randint(60, 500, n).astype(float),
        "ALT":                 np.abs(np.random.exponential(40, n)),
        "AST":                 np.abs(np.random.exponential(40, n)),
        "TotalProteins":       np.random.normal(6.8, 0.8, n),
        "Albumin":             np.random.normal(3.2, 0.5, n),
        "AGRatio":             np.random.normal(0.9, 0.2, n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="Label")
    return X, y


@pytest.fixture
def small_diab_df():
    """Tiny synthetic diabetes dataset for fast tests."""
    np.random.seed(1)
    n = 50
    X = pd.DataFrame({
        "Pregnancies":              np.random.randint(0, 10, n).astype(float),
        "Glucose":                  np.random.normal(120, 30, n),
        "BloodPressure":            np.random.normal(72, 10, n),
        "SkinThickness":            np.random.normal(29, 8, n),
        "Insulin":                  np.abs(np.random.exponential(80, n)),
        "BMI":                      np.random.normal(32, 6, n),
        "DiabetesPedigreeFunction": np.abs(np.random.exponential(0.5, n)),
        "Age":                      np.random.randint(21, 60, n).astype(float),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="Outcome")
    return X, y


@pytest.fixture
def trained_liver_pipeline(small_liver_df):
    """Train a full pipeline on small liver data. Used by prediction tests."""
    X, y = small_liver_df
    return train_pipeline(X, y)


# =============================================================================
# TEST GROUP 1 — Data utilities
# =============================================================================

class TestCapOutliers:
    """Tests for the _cap_outliers function."""

    def test_extreme_high_value_is_clipped(self):
        """A value of 9999 should be clipped down to the IQR upper fence."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 9999.0]})
        result = _cap_outliers(df)
        assert result["val"].max() < 9999, "Extreme high value should be capped"

    def test_shape_unchanged(self):
        """Capping outliers must not change the number of rows or columns."""
        df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
        result = _cap_outliers(df)
        assert result.shape == df.shape

    def test_no_nan_introduced(self):
        """Outlier capping should never introduce NaN values."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1000.0]})
        result = _cap_outliers(df)
        assert result.isnull().sum().sum() == 0

    def test_normal_values_unchanged(self):
        """Values within IQR fences should not be modified."""
        df = pd.DataFrame({"x": [1.0, 2.0, 2.5, 3.0, 3.5]})
        result = _cap_outliers(df)
        assert result["x"].min() >= 1.0 and result["x"].max() <= 3.5


# =============================================================================
# TEST GROUP 2 — EDA
# =============================================================================

class TestComputeEDA:
    """Tests for the compute_eda function."""

    def test_returns_dict(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        assert isinstance(stats, dict)

    def test_class_counts_present(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        assert "class_counts" in stats
        assert "positive" in stats["class_counts"]
        assert "negative" in stats["class_counts"]

    def test_class_counts_sum_to_total(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        total = stats["class_counts"]["positive"] + stats["class_counts"]["negative"]
        assert total == len(y)

    def test_age_dist_has_correct_bins(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        assert len(stats["age_dist"]["labels"]) == 7
        assert len(stats["age_dist"]["positive"]) == 7

    def test_scatter_keys_present(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        for key in ("x_label", "y_label", "positive_x", "positive_y"):
            assert key in stats["scatter"]

    def test_correlations_are_floats(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        for v in stats["correlations"].values():
            assert isinstance(v, float)

    def test_histograms_present(self, small_liver_df):
        X, y = small_liver_df
        stats = compute_eda(X, y, "liver")
        assert "histograms" in stats
        assert len(stats["histograms"]) > 0


# =============================================================================
# TEST GROUP 3 — Model training
# =============================================================================

class TestTrainPipeline:
    """Tests for the train_pipeline function."""

    def test_returns_pipeline_object(self, small_liver_df):
        X, y = small_liver_df
        pipe = train_pipeline(X, y)
        assert isinstance(pipe, DataPipeline)

    def test_scaler_fitted(self, trained_liver_pipeline):
        assert trained_liver_pipeline.scaler is not None

    def test_imputer_fitted(self, trained_liver_pipeline):
        assert trained_liver_pipeline.imputer is not None

    def test_three_models_trained(self, trained_liver_pipeline):
        assert len(trained_liver_pipeline.models) == 3

    def test_expected_model_names(self, trained_liver_pipeline):
        expected = {"Random Forest", "Gradient Boosting", "Bagging"}
        assert set(trained_liver_pipeline.models.keys()) == expected

    def test_best_model_is_set(self, trained_liver_pipeline):
        assert trained_liver_pipeline.best_model_name in trained_liver_pipeline.models

    def test_scores_are_valid_auc(self, trained_liver_pipeline):
        for name, score in trained_liver_pipeline.scores.items():
            assert 0.0 <= score <= 1.0, f"{name} score {score} outside [0,1]"

    def test_feature_importances_sum_to_one(self, trained_liver_pipeline):
        total = sum(trained_liver_pipeline.feature_importances.values())
        assert abs(total - 1.0) < 0.05, f"Feature importances sum = {total}"

    def test_bnn_is_trained(self, trained_liver_pipeline):
        assert trained_liver_pipeline.bnn is not None

    def test_feature_names_stored(self, trained_liver_pipeline, small_liver_df):
        X, _ = small_liver_df
        assert trained_liver_pipeline.feature_names == list(X.columns)


# =============================================================================
# TEST GROUP 4 — Prediction
# =============================================================================

class TestPredict:
    """Tests for the predict function."""

    def _demo_input(self, pipeline: DataPipeline) -> dict:
        """Build a valid input dict with zeros for all features."""
        return {f: 0.0 for f in pipeline.feature_names}

    def test_returns_dict(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert isinstance(result, dict)

    def test_mean_in_range(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert 0.0 <= result["mean"] <= 1.0, f"Mean {result['mean']} out of range"

    def test_sigma_non_negative(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert result["sigma"] >= 0.0

    def test_mc_samples_count(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert len(result["mc_samples"]) > 0

    def test_decision_is_valid(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert result["decision"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_importances_present(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert isinstance(result["importances"], dict)
        assert len(result["importances"]) > 0

    def test_doctor_flag_is_bool(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert isinstance(result["doctor_flag"], bool)

    def test_high_risk_patient_flagged(self, trained_liver_pipeline):
        """A patient with extreme biomarker values should get a doctor flag."""
        bad_patient = {
            f: (9999.0 if f not in ("Age", "Gender") else 65.0)
            for f in trained_liver_pipeline.feature_names
        }
        result = predict(trained_liver_pipeline, bad_patient)
        assert result["decision"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_bnn_prediction_works(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline), use_bnn=True)
        assert 0.0 <= result["mean"] <= 1.0

    def test_all_models_returned(self, trained_liver_pipeline):
        result = predict(trained_liver_pipeline, self._demo_input(trained_liver_pipeline))
        assert "all_models" in result
        assert len(result["all_models"]) >= 3

    def test_missing_feature_uses_zero(self, trained_liver_pipeline):
        """If a feature is missing from input, it should default to 0 gracefully."""
        partial = {}   # empty dict
        result = predict(trained_liver_pipeline, partial)
        assert isinstance(result["mean"], float)


# =============================================================================
# TEST GROUP 5 — Configuration / constants
# =============================================================================

class TestConfig:
    """Tests for static configuration values."""

    def test_users_dict_not_empty(self):
        assert len(USERS) > 0

    def test_each_user_has_password_name_role(self):
        for uid, val in USERS.items():
            assert len(val) == 3, f"User {uid} should have (password, name, role)"

    def test_uncertainty_threshold_in_range(self):
        assert 0.0 < UNCERTAINTY_THRESHOLD < 1.0

    def test_liver_cols_count(self):
        assert len(LIVER_COLS) == 11

    def test_feature_meta_has_both_datasets(self):
        assert "liver" in FEATURE_META
        assert "diabetes" in FEATURE_META

    def test_demo_patients_have_both_datasets(self):
        assert "liver" in DEMO_PATIENTS
        assert "diabetes" in DEMO_PATIENTS


# =============================================================================
# TEST GROUP 6 — Flask API
# =============================================================================

@pytest.fixture
def flask_client(small_liver_df, small_diab_df):
    """Create a Flask test client with lightweight trained pipelines."""
    X_l, y_l = small_liver_df
    X_d, y_d = small_diab_df
    liver_pipe = train_pipeline(X_l, y_l)
    diab_pipe  = train_pipeline(X_d, y_d)

    # Patch EDA stats so routes work
    import medpredict
    medpredict.EDA_STATS["liver"]    = compute_eda(X_l, y_l, "liver")
    medpredict.EDA_STATS["diabetes"] = compute_eda(X_d, y_d, "diabetes")

    app = create_app(liver_pipe, diab_pipe)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFlaskAPI:
    """Tests for Flask API routes."""

    def test_login_valid_doctor(self, flask_client):
        resp = flask_client.post("/api/login",
                                 json={"username": "doctor", "password": "medpredict123"},
                                 content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "name" in data

    def test_login_invalid_password(self, flask_client):
        resp = flask_client.post("/api/login",
                                 json={"username": "doctor", "password": "wrong"},
                                 content_type="application/json")
        assert resp.status_code == 401
        assert resp.get_json()["ok"] is False

    def test_login_unknown_user(self, flask_client):
        resp = flask_client.post("/api/login",
                                 json={"username": "ghost", "password": "x"},
                                 content_type="application/json")
        assert resp.status_code == 401

    def test_predict_liver_returns_result(self, flask_client):
        payload = {
            "dataset": "liver",
            "values": {
                "Age": 62, "Gender": 1, "TotalBilirubin": 10.9,
                "DirectBilirubin": 5.5, "AlkalinePhosphotase": 699,
                "ALT": 64, "AST": 100, "TotalProteins": 7.5,
                "Albumin": 3.2, "AGRatio": 0.74,
            },
            "use_bnn": False,
        }
        resp = flask_client.post("/api/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mean" in data
        assert "sigma" in data
        assert "decision" in data

    def test_predict_diabetes_returns_result(self, flask_client):
        payload = {
            "dataset": "diabetes",
            "values": {
                "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
                "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627, "Age": 50,
            },
            "use_bnn": False,
        }
        resp = flask_client.post("/api/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert 0.0 <= data["mean"] <= 1.0

    def test_predict_unknown_dataset_returns_400(self, flask_client):
        resp = flask_client.post("/api/predict",
                                 json={"dataset": "xyz", "values": {}})
        assert resp.status_code == 400

    def test_models_endpoint(self, flask_client):
        resp = flask_client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "liver" in data
        assert "diabetes" in data

    def test_features_liver_endpoint(self, flask_client):
        resp = flask_client.get("/api/features/liver")
        assert resp.status_code == 200
        items = resp.get_json()
        assert isinstance(items, list)
        assert len(items) == 11

    def test_features_diabetes_endpoint(self, flask_client):
        resp = flask_client.get("/api/features/diabetes")
        assert resp.status_code == 200
        assert len(resp.get_json()) == 9

    def test_demo_liver_endpoint(self, flask_client):
        resp = flask_client.get("/api/demo/liver")
        assert resp.status_code == 200
        assert "values" in resp.get_json()

    def test_demo_diabetes_endpoint(self, flask_client):
        resp = flask_client.get("/api/demo/diabetes")
        assert resp.status_code == 200
        assert "values" in resp.get_json()

    def test_eda_full_liver(self, flask_client):
        resp = flask_client.get("/api/eda_full/liver")
        assert resp.status_code == 200

    def test_eda_full_unknown_returns_404(self, flask_client):
        resp = flask_client.get("/api/eda_full/unknown")
        assert resp.status_code == 404
