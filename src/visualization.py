from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


sns.set_theme(style="whitegrid")


def _extract_tag_lists(df: pd.DataFrame, sep: str = "|") -> list[list[str]]:
    """Return per-document list of tags, handling both ``tag_list`` (list[str])
    and the raw ``tags`` (sep-joined string) columns."""
    if "tag_list" in df.columns:
        return [list(ts) if ts is not None else [] for ts in df["tag_list"]]
    if "tags" in df.columns:
        return [
            [t.strip().lower() for t in str(raw).split(sep) if t.strip()]
            for raw in df["tags"].fillna("")
        ]
    raise KeyError(
        "DataFrame must contain either a 'tag_list' (list[str]) or 'tags' "
        "(sep-joined string) column."
    )


def plot_tag_distribution(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    tag_lists = _extract_tag_lists(df)
    tags = [tag for ts in tag_lists for tag in ts]
    counts = pd.Series(tags).value_counts().head(top_n)

    print(
        f"[Fig] Top Tag Distribution — fréquence des {top_n} tags les plus courants "
        f"dans le corpus (plus la barre est longue, plus le tag apparaît souvent)."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title("Top Tag Distribution")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Tag")
    fig.tight_layout()
    return fig


def plot_text_length_distributions(df: pd.DataFrame) -> plt.Figure:
    print(
        "[Fig] Text Length Distributions — trois histogrammes côte à côte : "
        "longueur des titres (mots), longueur des résumés (mots), et nombre de tags "
        "par document. Sert à caler les hyperparamètres (max_len, max_words)."
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(df["title_words"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Title Length (words)")
    sns.histplot(df["abstract_words"], bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Abstract Length (words)")
    sns.histplot(df["num_tags"], bins=15, kde=False, ax=axes[2], color="#C44E52")
    axes[2].set_title("Tags per Document")
    fig.tight_layout()
    return fig


def plot_model_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    best = metrics_df.sort_values("micro_f1", ascending=False).iloc[0]
    print(
        "[Fig] Model Comparison — barres groupées de micro-precision / micro-recall / "
        f"micro-F1 pour chaque modèle. Meilleur modèle : {best['name']} "
        f"(F1={best['micro_f1']:.3f})."
    )
    melt_df = metrics_df.melt(id_vars=["name"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melt_df, x="name", y="value", hue="metric", ax=ax)
    ax.set_title("Model Comparison (Micro metrics)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_article_vs_current(metrics_df: pd.DataFrame) -> plt.Figure:
    print(
        "[Fig] Article vs Current — compare nos scores (micro P/R/F1) à ceux rapportés "
        "dans Wang et al. pour les mêmes modèles. Permet d'évaluer la reproduction."
    )
    article = pd.DataFrame(
        [
            {"name": "NB", "micro_recall": 0.03, "micro_precision": 0.21, "micro_f1": 0.05},
            {"name": "SVM", "micro_recall": 0.05, "micro_precision": 0.20, "micro_f1": 0.09},
            {"name": "LDA", "micro_recall": 0.16, "micro_precision": 0.20, "micro_f1": 0.17},
            {"name": "Paragraph Vector", "micro_recall": 0.24, "micro_precision": 0.22, "micro_f1": 0.22},
            {"name": "Bi-GRU+Att", "micro_recall": 0.44, "micro_precision": 0.20, "micro_f1": 0.28},
        ]
    )

    merged = metrics_df.merge(article, on="name", suffixes=("_current", "_article"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for i, metric in enumerate(["micro_recall", "micro_precision", "micro_f1"]):
        plot_df = merged[["name", f"{metric}_current", f"{metric}_article"]].melt(
            id_vars=["name"],
            var_name="source",
            value_name="value",
        )
        sns.barplot(data=plot_df, x="name", y="value", hue="source", ax=axes[i])
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_xlabel("Model")
        axes[i].tick_params(axis="x", rotation=25)

    axes[0].set_ylabel("Score")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_tag_cooccurrence_heatmap(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    print(
        f"[Fig] Tag Co-occurrence (Jaccard) — heatmap symétrique des {top_n} tags les "
        "plus fréquents. Case claire = les deux tags apparaissent souvent ensemble. "
        "Utile pour expliquer les confusions inter-tags en multi-label."
    )
    tag_lists = _extract_tag_lists(df)
    tag_counts = pd.Series([t for ts in tag_lists for t in ts]).value_counts()
    top_tags = tag_counts.head(top_n).index.tolist()
    mlb = MultiLabelBinarizer(classes=top_tags)
    Y = mlb.fit_transform([[t for t in ts if t in top_tags] for ts in tag_lists])
    inter = Y.T @ Y
    counts = Y.sum(axis=0)
    union = counts[:, None] + counts[None, :] - inter
    jaccard = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
    np.fill_diagonal(jaccard, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(jaccard, index=top_tags, columns=top_tags),
        cmap="magma", annot=False, ax=ax, cbar_kws={"label": "Jaccard"},
    )
    ax.set_title(f"Tag Co-occurrence (Jaccard, top {top_n})")
    fig.tight_layout()
    return fig


def plot_tag_frequency_pareto(df: pd.DataFrame, cutoff_k: int | None = None) -> plt.Figure:
    tag_lists = _extract_tag_lists(df)
    counts = pd.Series([t for ts in tag_lists for t in ts]).value_counts()
    cum = counts.cumsum() / counts.sum()
    n_total = len(counts)
    msg = (
        f"[Fig] Pareto des tags — part cumulée des occurrences pour les tags triés "
        f"par fréquence (axe X log). Corpus : {n_total} tags uniques."
    )
    if cutoff_k and cutoff_k <= n_total:
        msg += (
            f" À la coupure top-{cutoff_k}, on couvre {cum.iloc[cutoff_k - 1] * 100:.1f}% "
            "des occurrences totales."
        )
    print(msg)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(counts) + 1), cum.values, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Tag rank (log)")
    ax.set_ylabel("Cumulative share of tag occurrences")
    ax.set_title("Pareto distribution of tags")
    if cutoff_k:
        ax.axvline(cutoff_k, color="red", linestyle="--", label=f"top-{cutoff_k}")
        ax.axhline(cum.iloc[cutoff_k - 1], color="red", linestyle=":", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    return fig

def plot_preprocessing_impact(df: pd.DataFrame) -> plt.Figure:
    raw = df["text"].str.split().str.len()
    clean = df["processed_text"].str.split().str.len()
    reduction = 1.0 - (clean.sum() / max(raw.sum(), 1))
    print(
        "[Fig] Impact du prétraitement NLTK — superpose la distribution du nombre de "
        f"tokens par document AVANT (bleu) et APRÈS (rouge) nettoyage "
        f"(stopwords + lemmatization). Réduction globale ≈ {reduction * 100:.1f}% "
        "du volume de tokens."
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(raw, bins=40, color="#4C72B0", label="raw", alpha=0.55, ax=ax)
    sns.histplot(clean, bins=40, color="#C44E52", label="processed", alpha=0.55, ax=ax)
    ax.set_title(f"Tokens per document (median raw={raw.median():.0f}, processed={clean.median():.0f})")
    ax.set_xlabel("Tokens")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sentences_per_doc(df: pd.DataFrame) -> plt.Figure:
    n_sents = df["processed_sentences"].str.split(" . ").str.len()
    pct_over_10 = (n_sents > 10).mean() * 100
    print(
        f"[Fig] Phrases par document — histogramme du nombre de phrases après "
        f"segmentation. Médiane = {n_sents.median():.0f}. La barre rouge marque le "
        f"plafond max_sentences=10 utilisé par HAN ; {pct_over_10:.1f}% des documents "
        "sont tronqués à cette valeur."
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(n_sents, bins=range(1, int(n_sents.max()) + 2), color="#55A868", ax=ax)
    ax.axvline(10, color="red", linestyle="--", label="max_sentences=10 (HAN)")
    ax.set_title(f"Sentences per doc (median={n_sents.median():.0f})")
    ax.set_xlabel("# sentences"); ax.legend()
    fig.tight_layout()
    return fig


def plot_training_history(history: Mapping[str, Iterable[float]], title: str = "Training curves") -> plt.Figure:
    """Loss train/val per epoch for a Keras model."""
    train_loss = list(history.get("loss", []))
    val_loss = list(history.get("val_loss", []))
    msg = f"[Fig] Courbes d'apprentissage ({title}) — évolution de la loss par époque"
    if train_loss:
        msg += f". Train loss finale = {train_loss[-1]:.4f}"
    if val_loss:
        msg += f", val loss finale = {val_loss[-1]:.4f}"
    msg += (
        ". Si la val loss remonte alors que la train continue de baisser → "
        "signe d'overfitting ; early stopping (patience=5) s'en charge."
    )
    print(msg)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "loss" in history:
        ax.plot(list(history["loss"]), label="train loss", color="#4C72B0", marker="o")
    if "val_loss" in history:
        ax.plot(list(history["val_loss"]), label="val loss", color="#C44E52", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_time(timings: Mapping[str, float]) -> plt.Figure:
    """Wall-clock training time per model (log scale)."""
    s = pd.Series(dict(timings)).sort_values()
    fastest, slowest = s.index[0], s.index[-1]
    ratio = s.iloc[-1] / max(s.iloc[0], 1e-9)
    print(
        "[Fig] Temps d'entraînement par modèle (échelle log) — "
        f"plus rapide : {fastest} ({s.iloc[0]:.1f}s), plus lent : {slowest} "
        f"({s.iloc[-1]:.1f}s). Ratio lent/rapide ≈ {ratio:.0f}×."
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=s.values, y=s.index, ax=ax, palette="crest")
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s, log scale)")
    ax.set_ylabel("Model")
    ax.set_title("Training time per model")
    for i, v in enumerate(s.values):
        ax.text(v, i, f"  {v:.1f}s", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_per_tag_f1(
    y_true: np.ndarray,
    preds: Mapping[str, np.ndarray],
    tag_names: list[str],
) -> plt.Figure:
    """Per-tag F1 heatmap across models (sorted by mean F1)."""
    rows = []
    for model_name, y_pred in preds.items():
        per_tag = f1_score(y_true, y_pred, average=None, zero_division=0)
        for tag, f1 in zip(tag_names, per_tag):
            rows.append({"model": model_name, "tag": tag, "f1": float(f1)})
    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="tag", columns="model", values="f1")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    best_tag = pivot.mean(axis=1).idxmax()
    worst_tag = pivot.mean(axis=1).idxmin()
    print(
        "[Fig] F1 par tag et par modèle — heatmap (tags en lignes triés par F1 moyen, "
        "modèles en colonnes). Case claire = bien appris, case sombre = mal appris. "
        f"Tag le mieux prédit : '{best_tag}' — tag le plus difficile : '{worst_tag}'."
    )
    fig, ax = plt.subplots(figsize=(1.1 * max(6, len(pivot.columns)), max(4, 0.35 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-tag F1 per model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Tag")
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    raw: Mapping[str, np.ndarray],
    thresholds: np.ndarray | None = None,
) -> plt.Figure:
    """Micro-F1 as a function of the binarization threshold, per model."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    print(
        "[Fig] Sensibilité au seuil — micro-F1 en fonction du seuil de binarisation "
        f"(balayage de {thresholds[0]:.2f} à {thresholds[-1]:.2f}). Le maximum de "
        "chaque courbe indique le seuil optimal pour le modèle correspondant ; si le "
        "pic est loin de 0.5, le 0.5 par défaut est sous-optimal."
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, proba in raw.items():
        f1s = [
            f1_score(y_true, (proba >= t).astype(int), average="micro", zero_division=0)
            for t in thresholds
        ]
        ax.plot(thresholds, f1s, marker="o", label=name)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Micro-F1")
    ax.set_title("F1 vs threshold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_precision_recall_scatter(metrics_df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of micro-precision vs micro-recall with iso-F1 contours."""
    best = metrics_df.sort_values("micro_f1", ascending=False).iloc[0]
    print(
        "[Fig] Precision vs Recall (iso-F1) — chaque point est un modèle, positionné "
        "selon son micro-recall (X) et sa micro-precision (Y). Les courbes grises "
        "pointillées sont les iso-F1 ; un modèle au-dessus d'une courbe dépasse ce F1. "
        f"Point le plus au nord-est = meilleur compromis ({best['name']}, "
        f"F1={best['micro_f1']:.3f})."
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(metrics_df["micro_recall"], metrics_df["micro_precision"], s=90, color="#4C72B0", zorder=3)
    for _, r in metrics_df.iterrows():
        ax.annotate(
            r["name"],
            (r["micro_recall"], r["micro_precision"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    for f in [0.1, 0.2, 0.3, 0.4, 0.5]:
        x = np.linspace(0.001, 1.0, 200)
        denom = 2.0 * x - f
        y = np.where(denom > 0, (f * x) / denom, np.nan)
        mask = (y > 0) & (y <= 1.0)
        ax.plot(x[mask], y[mask], color="grey", alpha=0.3, linestyle="--", linewidth=1)
        try:
            x_label = 0.95
            y_label = (f * x_label) / (2 * x_label - f)
            if 0 < y_label <= 1:
                ax.text(x_label, y_label, f"F1={f}", color="grey", fontsize=8, alpha=0.7)
        except ZeroDivisionError:
            pass
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Micro recall")
    ax.set_ylabel("Micro precision")
    ax.set_title("Precision vs Recall (iso-F1 curves)")
    fig.tight_layout()
    return fig


def plot_topk_sensitivity(
    y_true: np.ndarray,
    raw: Mapping[str, np.ndarray],
    k_values: Iterable[int] = range(1, 6),
) -> plt.Figure:
    """Micro-F1 as a function of K (top-K decision rule)."""
    from src.models import threshold_topk

    ks = list(k_values)
    print(
        "[Fig] Sensibilité au top-K — micro-F1 quand on force la prédiction des K tags "
        f"les plus probables par document (K de {ks[0]} à {ks[-1]}). Permet de choisir "
        "K quand on veut un nombre fixe de recommandations plutôt qu'un seuil de "
        "probabilité."
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, proba in raw.items():
        f1s = [
            f1_score(y_true, threshold_topk(proba, k), average="micro", zero_division=0)
            for k in ks
        ]
        ax.plot(ks, f1s, marker="o", label=name)
    ax.set_xticks(ks)
    ax.set_xlabel("K (top-K tags predicted)")
    ax.set_ylabel("Micro-F1")
    ax.set_title("F1 vs K (top-K decision rule)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    return fig