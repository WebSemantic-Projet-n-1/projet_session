from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import keep_top_k_tags, load_simple_dataset, preprocess_text_nltk, text_length_stats
from src.experiment import prepare_train_test, run_all_models
from src.visualization import (
    plot_article_vs_current,
    plot_model_metrics,
    plot_tag_distribution,
    plot_text_length_distributions,
)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "citeulike_top10.csv"
    glove_path = PROJECT_ROOT / "data" / "glove.6B.300d.txt"
    output_dir = PROJECT_ROOT / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_simple_dataset(data_path)
    df = keep_top_k_tags(df, top_k=10)
    df = preprocess_text_nltk(df)
    df = text_length_stats(df)

    fig = plot_tag_distribution(df, top_n=10)
    fig.savefig(output_dir / "tag_distribution.png", dpi=150)
    plt.close(fig)

    fig = plot_text_length_distributions(df)
    fig.savefig(output_dir / "text_lengths.png", dpi=150)
    plt.close(fig)

    X_train, X_test, y_train, y_test, _ = prepare_train_test(df, test_size=0.1, random_state=42)
    metrics, _ = run_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
        n_topics=10,
        glove_path=str(glove_path),
    )
    metrics.to_csv(PROJECT_ROOT / "data" / "metrics_results.csv", index=False)

    fig = plot_model_metrics(metrics)
    fig.savefig(output_dir / "model_metrics.png", dpi=150)
    plt.close(fig)

    fig = plot_article_vs_current(metrics)
    fig.savefig(output_dir / "article_vs_current.png", dpi=150)
    plt.close(fig)

    print(metrics)


if __name__ == "__main__":
    main()
