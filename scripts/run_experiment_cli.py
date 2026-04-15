from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import keep_top_k_tags, load_simple_dataset, preprocess_text_nltk, text_length_stats
from src.experiment import prepare_train_test, run_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tag recommendation experiment and print metrics.")
    parser.add_argument(
        "--data-path",
        default="data/citeulike_top10.csv",
        help="CSV dataset path with columns: title, abstract, tags",
    )
    parser.add_argument(
        "--glove-path",
        default="data/glove.6B.300d.txt",
        help="Optional GloVe txt path (300d).",
    )
    parser.add_argument("--top-k-tags", type=int, default=10, help="Keep top-k frequent tags.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = PROJECT_ROOT / args.data_path
    glove_path = PROJECT_ROOT / args.glove_path

    df = load_simple_dataset(data_path)
    df = keep_top_k_tags(df, top_k=args.top_k_tags)
    df = preprocess_text_nltk(df)
    df = text_length_stats(df)

    X_train, X_test, y_train, y_test, _ = prepare_train_test(
        df,
        test_size=args.test_size,
        random_state=args.seed,
    )
    metrics, _ = run_all_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_topics=args.top_k_tags,
        glove_path=str(glove_path),
    )

    print("\n=== Experiment Results (Micro Metrics) ===")
    print(metrics.to_string(index=False))
    best = metrics.sort_values("micro_f1", ascending=False).iloc[0]
    print(f"\nBest model: {best['name']} | Micro-F1: {best['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
