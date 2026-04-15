from __future__ import annotations

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
    train_nb_model,
    train_svm_model,
)


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
    preds["Bi-GRU+Att"] = train_bigru_attention_model(X_train, y_train, X_test, glove_path=glove_path)

    rows = []
    for model_name, y_pred in preds.items():
        result = evaluate_predictions(model_name, y_test, y_pred)
        rows.append(asdict(result))

    metrics = pd.DataFrame(rows)[["name", "micro_recall", "micro_precision", "micro_f1"]]
    metrics = metrics.sort_values(by="micro_f1", ascending=False).reset_index(drop=True)
    return metrics, preds
