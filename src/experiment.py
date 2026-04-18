from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from src.models import (
    evaluate_predictions,
    train_bigru_attention_model,
    train_doc2vec_model,
    train_lda_model,
    train_lda_model_tfidf,
    train_nb_model,
    train_svm_model,
    train_svm_model_bow,
    threshold_topk,
    train_hierarchical_bigru_attention_model,
)


def prepare_train_test(df, test_size=0.1, random_state=42):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tag_list"])
    X_flat = df["processed_text"].tolist()
    X_hier = df["processed_sentences"].tolist()

    X_flat_train, X_flat_test, X_hier_train, X_hier_test, y_train, y_test = train_test_split(
        X_flat, X_hier, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_flat_train, X_flat_test, X_hier_train, X_hier_test, y_train, y_test, mlb


def _time_call(fn):
    """Run fn(), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _train_all(
    X_train: list[str],
    X_test: list[str],
    y_train: np.ndarray,
    n_topics: int,
    glove_path: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict]]:
    """Shared runner: trains every model, records timings and NN histories."""
    raw: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    histories: dict[str, dict] = {}

    jobs = [
        ("NB", lambda: train_nb_model(X_train, y_train, X_test)),
        ("SVM", lambda: train_svm_model(X_train, y_train, X_test)),
        ("SVM_BOW", lambda: train_svm_model_bow(X_train, y_train, X_test)),
        ("LDA", lambda: train_lda_model(X_train, y_train, X_test, n_topics=n_topics)),
        ("LDA_TFIDF", lambda: train_lda_model_tfidf(X_train, y_train, X_test, n_topics=n_topics)),
        ("Paragraph Vector", lambda: train_doc2vec_model(X_train, y_train, X_test)),
        ("Bi-GRU+Att", lambda: train_bigru_attention_model(X_train, y_train, X_test, glove_path=glove_path)),
        ("HAN_BiGRU_Att", lambda: train_hierarchical_bigru_attention_model(X_train, y_train, X_test, glove_path=glove_path)),
    ]

    for name, fn in jobs:
        result, elapsed = _time_call(fn)
        timings[name] = elapsed
        if isinstance(result, tuple):
            proba, history = result
            histories[name] = history
            raw[name] = proba
        else:
            raw[name] = result

    return raw, timings, histories


def _predict(raw: dict[str, np.ndarray], threshold: float, topk: int | None) -> dict[str, np.ndarray]:
    preds = {}
    for name, proba in raw.items():
        if topk is not None:
            preds[name] = threshold_topk(proba, k=topk)
        else:
            preds[name] = (proba >= threshold).astype(int)
    return preds


def _build_metrics_df(y_test: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_name, y_pred in preds.items():
        result = evaluate_predictions(model_name, y_test, y_pred)
        rows.append(asdict(result))
    metrics = pd.DataFrame(rows)[["name", "micro_recall", "micro_precision", "micro_f1"]]
    return metrics.sort_values(by="micro_f1", ascending=False).reset_index(drop=True)


def run_all_models(
    X_train: list[str],
    X_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_topics: int,
    glove_path: str | None = "data/glove.6B.300d.txt",
    threshold: float = 0.5,
    topk: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, dict]]:
    """Train every model on flat inputs.

    Returns
    -------
    metrics : pd.DataFrame
        Sorted micro-metrics per model.
    preds : dict[str, np.ndarray]
        Binarized predictions per model (after threshold / top-k).
    raw : dict[str, np.ndarray]
        Raw probabilities per model (useful for threshold / top-K plots).
    timings : dict[str, float]
        Wall-clock training time per model, in seconds.
    histories : dict[str, dict]
        Keras training histories for NN models (empty for sklearn/gensim ones).
    """
    raw, timings, histories = _train_all(X_train, X_test, y_train, n_topics, glove_path)
    preds = _predict(raw, threshold=threshold, topk=topk)
    metrics = _build_metrics_df(y_test, preds)
    return metrics, preds, raw, timings, histories


def run_all_models_hierarchical(
    X_hier_train: list[str],
    X_hier_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_topics: int,
    glove_path: str | None = "data/glove.6B.300d.txt",
    threshold: float = 0.5,
    topk: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, dict]]:
    """Same as `run_all_models` but feeds the sentence-segmented inputs."""
    raw, timings, histories = _train_all(X_hier_train, X_hier_test, y_train, n_topics, glove_path)
    preds = _predict(raw, threshold=threshold, topk=topk)
    metrics = _build_metrics_df(y_test, preds)
    return metrics, preds, raw, timings, histories
