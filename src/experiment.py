from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from src.models import (
    evaluate_predictions,
    train_han_model,
    train_doc2vec_model,
    train_lda_model,
    train_nb_model,
    train_svm_model,
)

# Paper protocol: 10 LDA topics, 5 runs averaged over random 90/10 splits
LDA_N_TOPICS_PAPER = 10
N_RUNS_PAPER = 5
TEST_SIZE_PAPER = 0.1


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tag_list"])
    text_col = "processed_text" if "processed_text" in df.columns else "text"
    X = df[text_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test, mlb


def run_experiments_k_times(
    df: pd.DataFrame,
    n_runs: int = N_RUNS_PAPER,
    test_size: float = TEST_SIZE_PAPER,
    n_lda_topics: int = LDA_N_TOPICS_PAPER,
    glove_path: str | None = "data/glove.6B.300d.txt",
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all models n_runs times with independent random 90/10 splits.

    Paper protocol (Hassan et al., RecSys 2018):
        - 5 runs, each with a different random seed
        - 90 % train / 10 % test
        - LDA K = n_tags (10 for the top-10 setup)
        - Results reported as mean over the 5 runs

    Returns:
        agg_df     — one row per model with mean ± std columns
        all_runs_df — raw per-run results (includes 'run' and 'seed' columns)
    """
    text_col = "processed_text" if "processed_text" in df.columns else "text"
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(df["tag_list"])
    X_all = df[text_col].tolist()

    all_rows: list[pd.DataFrame] = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"  Run {run_idx + 1}/{n_runs}  (seed={seed})", flush=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=seed, shuffle=True
        )
        run_metrics, _ = run_all_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_topics=n_lda_topics,
            glove_path=glove_path,
        )
        run_metrics = run_metrics.copy()
        run_metrics["run"] = run_idx + 1
        run_metrics["seed"] = seed
        all_rows.append(run_metrics)

    all_runs_df = pd.concat(all_rows, ignore_index=True)

    agg = (
        all_runs_df.groupby("name")[["micro_recall", "micro_precision", "micro_f1"]]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg_df = (
        agg.reset_index()
        .sort_values("micro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    return agg_df, all_runs_df


def run_all_models(
    X_train: list[str],
    X_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_topics: int,
    glove_path: str | None = "data/glove.6B.300d.txt",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    preds = {}
    preds["NB"] = train_nb_model(X_train, y_train, X_test)
    preds["SVM"] = train_svm_model(X_train, y_train, X_test)
    preds["LDA"] = train_lda_model(X_train, y_train, X_test, n_topics=n_topics)
    preds["Paragraph Vector"] = train_doc2vec_model(X_train, y_train, X_test)
    preds["Bi-GRU+Att"] = train_han_model(X_train, y_train, X_test, glove_path=glove_path)

    rows = []
    for model_name, y_pred in preds.items():
        result = evaluate_predictions(model_name, y_test, y_pred)
        rows.append(asdict(result))

    metrics = pd.DataFrame(rows)[["name", "micro_recall", "micro_precision", "micro_f1"]]
    metrics = metrics.sort_values(by="micro_f1", ascending=False).reset_index(drop=True)
    return metrics, preds
