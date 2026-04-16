from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_tag_distribution(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    tags = [tag for tags_list in df["tag_list"] for tag in tags_list]
    counts = pd.Series(tags).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
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
    sns.histplot(df["abstract_words"], bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Abstract Length (words)")
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
