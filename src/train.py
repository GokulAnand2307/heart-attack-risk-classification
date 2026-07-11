"""Train reproducible binary classifiers for the heart-attack risk dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    blood_pressure = data.pop("Blood Pressure").str.split("/", expand=True).astype(float)
    data["Systolic BP"], data["Diastolic BP"] = blood_pressure[0], blood_pressure[1]
    data = data.drop(columns=["Patient ID"])
    target = data.pop("Heart Attack Risk").astype(int)
    return data, target


def transformer(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns
    categorical = features.columns.difference(numeric)
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    features, target = prepare(args.data)
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=.2, random_state=42, stratify=target)
    prep = transformer(features)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1),
    }
    results = []
    fitted = {}
    for name, model in models.items():
        pipeline = Pipeline([("prepare", prep), ("model", model)])
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(test_x)
        probability = pipeline.predict_proba(test_x)[:, 1]
        fitted[name] = pipeline
        results.append({"Model": name, "Accuracy": accuracy_score(test_y, prediction), "Precision": precision_score(test_y, prediction), "Recall": recall_score(test_y, prediction), "F1": f1_score(test_y, prediction), "ROC-AUC": roc_auc_score(test_y, probability)})
        fig, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(test_y, prediction, cmap="Blues", ax=axis)
        axis.set_title(name)
        fig.tight_layout()
        fig.savefig(args.output / f"confusion_matrix_{name.lower().replace(' ', '_')}.svg")
        plt.close(fig)

    metrics = pd.DataFrame(results).set_index("Model")
    metrics.to_csv(args.output / "model_metrics.csv")
    axis = metrics.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    axis.set_title("Heart-attack risk classification metrics")
    axis.set_ylabel("Score")
    axis.figure.tight_layout()
    axis.figure.savefig(args.output / "model_comparison.svg")
    plt.close(axis.figure)

    forest = fitted["Random Forest"]
    names = forest.named_steps["prepare"].get_feature_names_out()
    importance = pd.Series(forest.named_steps["model"].feature_importances_, index=names).nlargest(15).sort_values()
    fig, axis = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=axis, color="#2979ff")
    axis.set_title("Random Forest — top feature importance")
    fig.tight_layout()
    fig.savefig(args.output / "feature_importance.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

