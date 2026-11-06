"""Train, evaluate, and save a churn classification pipeline."""
import json
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import FEATURES, make_dataset

NUMERIC = ["tenure_months", "monthly_charges", "support_tickets"]
CATEGORICAL = ["contract_type", "autopay"]

def pipeline(estimator):
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("preprocess", prep), ("classifier", estimator)])

def train(output_dir="models"):
    df = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df["churn"], test_size=0.2, stratify=df["churn"], random_state=42)
    candidates = {
        "logistic_regression": pipeline(LogisticRegression(max_iter=1000, random_state=42)),
        "random_forest": pipeline(RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        scores[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        }
    best_name = max(scores, key=lambda name: scores[name]["roc_auc"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[best_name], out / "churn_pipeline.joblib")
    report = {"dataset": "synthetic", "n_samples": len(df), "test_samples": len(y_test), "selected_model": best_name, "metrics": scores}
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
