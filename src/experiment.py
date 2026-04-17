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
    train_lda_model_tfidf,
    train_nb_model,
    train_svm_model,
    train_svm_model_bow,
    threshold_topk,
    train_hierarchical_bigru_attention_model
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
    


def run_all_models(
    X_train: list[str],
    X_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_topics: int,
    glove_path: str | None = "data/glove.6B.300d.txt",
    threshold: float = 0.5,
    topk: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    raw = {}
    raw["NB"] = train_nb_model(X_train, y_train, X_test)
    raw["SVM"] = train_svm_model(X_train, y_train, X_test)
    raw["SVM_BOW"] = train_svm_model_bow(X_train, y_train, X_test)
    raw["LDA"] = train_lda_model(X_train, y_train, X_test, n_topics=n_topics)
    raw["LDA_TFIDF"] = train_lda_model_tfidf(X_train, y_train, X_test, n_topics=n_topics)
    raw["Paragraph Vector"] = train_doc2vec_model(X_train, y_train, X_test)
    raw["Bi-GRU+Att"] = train_bigru_attention_model(X_train, y_train, X_test, glove_path=glove_path)
    raw["HAN_BiGRU_Att"] = train_hierarchical_bigru_attention_model(X_train, y_train, X_test, glove_path=glove_path)
    
    preds = {}
    for name, proba in raw.items():
        if topk is not None:
            preds[name] = threshold_topk(proba, k=topk)
        else:
            preds[name] = (proba >= threshold).astype(int)
    
    rows = []
    for model_name, y_pred in preds.items():
        result = evaluate_predictions(model_name, y_test, y_pred)
        rows.append(asdict(result))

    metrics = pd.DataFrame(rows)[["name", "micro_recall", "micro_precision", "micro_f1"]]
    metrics = metrics.sort_values(by="micro_f1", ascending=False).reset_index(drop=True)
    return metrics, preds


def run_all_models_hierarchical(
    X_hier_train: list[str],
    X_hier_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_topics: int,
    glove_path: str | None = "data/glove.6B.300d.txt",
    threshold: float = 0.5,
    topk: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    raw = {}
    raw["NB"] = train_nb_model(X_hier_train, y_train, X_hier_test)
    raw["SVM"] = train_svm_model(X_hier_train, y_train, X_hier_test)
    raw["SVM_BOW"] = train_svm_model_bow(X_hier_train, y_train, X_hier_test)
    raw["LDA"] = train_lda_model(X_hier_train, y_train, X_hier_test, n_topics=n_topics)
    raw["LDA_TFIDF"] = train_lda_model_tfidf(X_hier_train, y_train, X_hier_test, n_topics=n_topics)
    raw["Paragraph Vector"] = train_doc2vec_model(X_hier_train, y_train, X_hier_test)
    raw["Bi-GRU+Att"] = train_bigru_attention_model(X_hier_train, y_train, X_hier_test, glove_path=glove_path)
    raw["HAN_BiGRU_Att"] = train_hierarchical_bigru_attention_model(X_hier_train, y_train, X_hier_test, glove_path=glove_path)
    
    preds = {}
    for name, proba in raw.items():
        if topk is not None:
            preds[name] = threshold_topk(proba, k=topk)
        else:
            preds[name] = (proba >= threshold).astype(int)
    
    rows = []
    for model_name, y_pred in preds.items():
        result = evaluate_predictions(model_name, y_test, y_pred)
        rows.append(asdict(result))

    metrics = pd.DataFrame(rows)[["name", "micro_recall", "micro_precision", "micro_f1"]]
    metrics = metrics.sort_values(by="micro_f1", ascending=False).reset_index(drop=True)
    return metrics, preds