"""
train.py
End-to-end training script for a disease prediction system.

What it does:
- Loads a CSV dataset
- Cleans column names and values
- Removes duplicates and handles missing values
- Detects the target column
- Encodes the target with LabelEncoder
- Splits data using stratified sampling
- Applies preprocessing (imputation + scaling/encoding)
- Handles class imbalance with SMOTE
- Trains a RandomForestClassifier
- Evaluates the model
- Runs Stratified K-Fold cross-validation
- Saves trained artifacts for the Streamlit app
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


MISSING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "?",
    "-",
    "--",
    "unknown",
}


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def normalize_string_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in MISSING_TOKENS:
            return np.nan
        return cleaned.lower()
    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(normalize_string_value)
            # remove repeated spaces inside strings
            df[col] = df[col].astype("object")
            mask = df[col].notna()
            df.loc[mask, col] = (
                df.loc[mask, col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            )

    return df


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "disease",
        "prognosis",
        "target",
        "label",
        "diagnosis",
        "condition",
        "class",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Fall back to the last column if no obvious target exists
    return df.columns[-1]


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_pipeline), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline(categorical_pipeline), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


def make_smote(y_train: pd.Series) -> SMOTE | None:
    class_counts = Counter(y_train)
    if len(class_counts) < 2:
        return None

    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        return None

    k_neighbors = min(5, min_class_count - 1)
    return SMOTE(random_state=42, k_neighbors=k_neighbors)


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.csv",
        help="Path to the CSV dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where trained files will be saved",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place the Kaggle CSV there or pass --data path."
        )

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    print(df.head(3))
    print(df.info())

    df = clean_dataframe(df)
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = detect_target_column(df)
    print(f"Detected target column: {target_col}")

    # Drop rows where the target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(str).str.strip().str.lower()

    # Clean the feature matrix further
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map(normalize_string_value)
            mask = X[col].notna()
            X.loc[mask, col] = (
                X.loc[mask, col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            )

    print(f"Features shape: {X.shape}")
    print(f"Target classes: {y.nunique()}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = make_smote(pd.Series(y_train))
    if smote is not None:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
        print(f"Applied SMOTE. Balanced training set shape: {X_train_balanced.shape}")
    else:
        X_train_balanced, y_train_balanced = X_train_processed, y_train
        print("SMOTE skipped because the minority class was too small.")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print("\n=== Test Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(report_text)

    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(y_test, y_pred, label_encoder.classes_, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Cross-validation with preprocessing + SMOTE + model inside each fold
    class_counts = Counter(y_encoded)
    min_class_count = min(class_counts.values())
    cv_splits = max(2, min(5, min_class_count))

    cv_smote = make_smote(pd.Series(y_encoded))
    cv_pipeline_steps = [("preprocessor", preprocessor)]
    if cv_smote is not None:
        cv_pipeline_steps.append(("smote", cv_smote))
    cv_pipeline_steps.append(
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        )
    )
    cv_pipeline = ImbPipeline(cv_pipeline_steps)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cv_pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    mean_f1 = float(np.mean(cv_scores))
    std_f1 = float(np.std(cv_scores))

    print("\n=== Cross Validation ===")
    print(f"F1-macro scores: {np.round(cv_scores, 4)}")
    print(f"Mean F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")

    # Save final artifacts
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")

    metadata = {
        "target_column": target_col,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "classes": label_encoder.classes_.tolist(),
        "accuracy": float(accuracy),
        "precision_per_class": {
            k: float(v["precision"]) for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "recall_per_class": {
            k: float(v["recall"]) for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "f1_per_class": {
            k: float(v["f1-score"]) for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_f1_cv_mean": mean_f1,
        "macro_f1_cv_std": std_f1,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(f"- {output_dir / 'model.pkl'}")
    print(f"- {output_dir / 'label_encoder.pkl'}")
    print(f"- {output_dir / 'preprocessor.pkl'}")
    print(f"- {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
