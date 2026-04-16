from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_tag_distribution(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    tags = [tag for tags_list in df["tag_list"] for tag in tags_list]
    counts = pd.Series(tags).value_counts().head(top_n)
    # height = top_n/2 if top_n < 20 else 10


    fig, ax = plt.subplots(figsize=(10, 30))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title("Top Tag Distribution")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Tag")
    fig.tight_layout()
    return fig


def plot_text_length_distributions(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(df["title_words"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Title Length (words)")
    # Focus the abstract histogram on the main mass instead of a sparse long tail.
    abstract_view_max = min(1000, float(df["abstract_words"].quantile(0.99)))
    abstract_view_max = max(50, abstract_view_max)
    abstract_words_main = df.loc[df["abstract_words"] <= abstract_view_max, "abstract_words"]
    sns.histplot(abstract_words_main, bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Abstract Length (words)")
    axes[1].set_xlim(0, abstract_view_max)
    sns.histplot(df["num_tags"], bins=15, kde=False, ax=axes[2], color="#C44E52")
    axes[2].set_title("Tags per Document")
    fig.tight_layout()
    return fig


def plot_model_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
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


# ---------------------------------------------------------------------------
# Multi-run (mean ± std) variants — used after run_experiments_k_times
# ---------------------------------------------------------------------------

_METRICS = ["micro_recall", "micro_precision", "micro_f1"]
_PALETTE = ["#4C72B0", "#55A868", "#C44E52"]


def plot_model_metrics_agg(agg_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart with std error bars from aggregated multi-run results.

    agg_df must have columns: name, micro_recall_mean/std, micro_precision_mean/std, micro_f1_mean/std.
    """
    names = agg_df["name"].tolist()
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(_METRICS):
        means = agg_df[f"{metric}_mean"].to_numpy()
        stds = agg_df[f"{metric}_std"].to_numpy()
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric.replace("_", " ").title(),
            color=_PALETTE[i],
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title("Model Comparison — Mean ± Std (5 runs, paper protocol)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_article_vs_current_agg(agg_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side comparison of paper Table 2 vs. current aggregated results.

    agg_df must have columns: name, micro_recall_mean, micro_precision_mean, micro_f1_mean.
    """
    article = pd.DataFrame(
        [
            {"name": "NB", "micro_recall": 0.03, "micro_precision": 0.21, "micro_f1": 0.05},
            {"name": "SVM", "micro_recall": 0.05, "micro_precision": 0.20, "micro_f1": 0.09},
            {"name": "LDA", "micro_recall": 0.16, "micro_precision": 0.20, "micro_f1": 0.17},
            {"name": "Paragraph Vector", "micro_recall": 0.24, "micro_precision": 0.22, "micro_f1": 0.22},
            {"name": "Bi-GRU+Att", "micro_recall": 0.44, "micro_precision": 0.20, "micro_f1": 0.28},
        ]
    )

    current = agg_df[["name", "micro_recall_mean", "micro_precision_mean", "micro_f1_mean"]].rename(
        columns={
            "micro_recall_mean": "micro_recall",
            "micro_precision_mean": "micro_precision",
            "micro_f1_mean": "micro_f1",
        }
    )

    merged = current.merge(article, on="name", suffixes=("_current", "_article"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for i, metric in enumerate(_METRICS):
        plot_df = merged[["name", f"{metric}_current", f"{metric}_article"]].melt(
            id_vars=["name"], var_name="source", value_name="value"
        )
        sns.barplot(data=plot_df, x="name", y="value", hue="source", ax=axes[i])
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_xlabel("Model")
        axes[i].tick_params(axis="x", rotation=25)

    axes[0].set_ylabel("Score (mean over 5 runs)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig
