"""Text classification utilities: XGBoost and SVM with TF-IDF.

Usage (example):
    python src/text_classification.py --data data/processed/labeled_comments_Chinese.csv --text_col comment --label_col label --out_dir output/models --language chinese
    python src/text_classification.py --data data/processed/labeled_comments_English.csv --text_col comment --label_col label --out_dir output/models --language english
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

from utils import ensure_dir

def load_data(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and {label_col} must exist in {path}")
    return df[[text_col, label_col]].dropna()


def prepare_tfidf(texts, max_features: int = 20000, ngram_range=(1, 2)) -> Tuple[TfidfVectorizer, any]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    return vec, X


def train_xgboost(X, y, **kwargs) -> XGBClassifier:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
    model.fit(X, y)
    return model


def train_svm(X, y, **kwargs) -> SVC:
    model = SVC(**kwargs)
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    report["accuracy"] = acc
    return report


def save_artifacts(out_dir: str, name: str, model, vectorizer=None):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, model_path)
    if vectorizer is not None:
        joblib.dump(vectorizer, os.path.join(out_dir, f"{name}_vectorizer.joblib"))
    return model_path


def run_pipeline(
    data_path: str,
    text_col: str,
    label_col: str,
    out_dir: str,
    language: str = "english",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_data(data_path, text_col=text_col, label_col=label_col)
    texts = df[text_col].astype(str)
    labels = df[label_col]

    vec, X = prepare_tfidf(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Train XGBoost (set multi-class objective if needed)
    unique_labels = pd.Series(y_train).unique()
    xgb_kwargs = {}
    if len(unique_labels) > 2:
        xgb_kwargs.update({"objective": "multi:softprob", "num_class": int(len(unique_labels))})
    xgb = train_xgboost(X_train, y_train, **xgb_kwargs)
    xgb_report = evaluate(xgb, X_test, y_test)

    # Plot and save confusion matrix for XGBoost with higher contrast
    preds_xgb = xgb.predict(X_test)
    cm_xgb = confusion_matrix(y_test, preds_xgb)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
    fig_x, ax_x = plt.subplots(figsize=(6, 6))
    disp_xgb.plot(ax=ax_x, cmap=plt.cm.hot, colorbar=False)
    # increase contrast by setting color limits to the data range
    if ax_x.images:
        im = ax_x.images[0]
        im.set_clim(0, cm_xgb.max() if cm_xgb.max() > 0 else 1)
        cbar = fig_x.colorbar(im, ax=ax_x)
    ax_x.set_title(f"XGBoost Confusion Matrix ({language})")
    plt.tight_layout()
    xgb_cm_path = os.path.join(out_dir, f"{language}_xgboost_confusion_matrix.png")
    ensure_dir(xgb_cm_path)
    fig_x.savefig(xgb_cm_path, dpi=300)
    plt.close(fig_x)

    # Train multiple SVMs with different kernels and plot confusion matrices
    svm_kernels = ["linear", "rbf", "poly"]
    svm_reports = {}
    for kernel in svm_kernels:
        svm_model = train_svm(X_train, y_train, kernel=kernel, probability=True)
        report = evaluate(svm_model, X_test, y_test)
        svm_reports[kernel] = report

        # Save model and vectorizer per kernel
        save_artifacts(out_dir, f"{language}_svm_{kernel}_model", svm_model, vectorizer=vec)

        # Plot confusion matrix with higher contrast (similar to XGBoost)
        preds = svm_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.hot, colorbar=False)
        # increase contrast by setting color limits to the data range
        if ax.images:
            im = ax.images[0]
            im.set_clim(0, cm.max() if cm.max() > 0 else 1)
            cbar = fig.colorbar(im, ax=ax)
        ax.set_title(f"SVM ({kernel}) Confusion Matrix ({language})")
        plt.tight_layout()
        cm_path = os.path.join(out_dir, f"{language}_svm_{kernel}_confusion_matrix.png")
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)

    # Save xgboost and its vectorizer
    save_artifacts(out_dir, f"{language}_xgboost_model", xgb, vectorizer=vec)

    results = {"xgboost": xgb_report, "svm": svm_reports}

    # Save final report (serializing numpy types) to JSON in out_dir
    report = {
        "language": language,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }

    def _to_native(o):
        if isinstance(o, dict):
            return {str(k): _to_native(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_native(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        return o

    serializable = _to_native(report)
    report_path = os.path.join(out_dir, f"training_report_{language}.json")
    ensure_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)

    return results


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data", required=True, help="Path to CSV with text and label columns")
    p.add_argument("-t", "--text_col", default="text", help="Name of the text column")
    p.add_argument("-l", "--label_col", default="label", help="Name of the label column")
    p.add_argument("-o", "--out_dir", default="output/models", help="Output directory for models and artifacts")
    p.add_argument("--language", default="english", choices=["english", "chinese"], help="Language of the text data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_pipeline(args.data, args.text_col, args.label_col, args.out_dir, args.language)
    print("Training results:")
    for k, v in results.items():
        print(f"---- {k} ----{args.language}----")
        if k == "svm":
            for kernel, report in v.items():
                print(f"  Kernel: {kernel}")
                print(f"    Accuracy: {report.get('accuracy')}")
                for cls, metrics in report.items():
                    if cls in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    print(f"    {cls}: precision={metrics.get('precision')}, recall={metrics.get('recall')}, f1={metrics.get('f1-score')}")
        else:
            print(f"  Accuracy: {v.get('accuracy')}")
            for cls, metrics in v.items():
                if cls in ("accuracy", "macro avg", "weighted avg"):
                    continue
                print(f"  {cls}: precision={metrics.get('precision')}, recall={metrics.get('recall')}, f1={metrics.get('f1-score')}")
