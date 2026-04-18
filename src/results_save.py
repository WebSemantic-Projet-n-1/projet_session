from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

def save_metrics_enriched(
    metrics_df: pd.DataFrame,
    *,
    dataset_path: str = "data/citeulike-a",
    top_k_tags: int = 10,
    test_size: float = 0.1,
    random_state: int = 42,
    repeats: int = 1,
    threshold: float = 0.5,
    text_column: str = "processed_text",
    n_docs: int | None = None,
    n_unique_tags: int | None = None,
    avg_tags_per_doc: float | None = None,
    glove_path: str | None = "data/glove.6B.300d.txt",
    model_params: dict | None = None,
    predictions: dict[str, np.ndarray] | None = None,
    y_test: np.ndarray | None = None,
    notes: str = "",
):
    run_id = uuid.uuid4().hex[:12]
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    out = metrics_df.copy()

    # Global experiment metadata
    out["run_id"] = run_id
    out["timestamp_utc"] = ts
    out["dataset_path"] = dataset_path
    out["top_k_tags"] = top_k_tags
    out["test_size"] = test_size
    out["random_state"] = random_state
    out["repeats"] = repeats
    out["threshold"] = threshold
    out["text_column"] = text_column
    out["n_docs"] = n_docs
    out["n_unique_tags"] = n_unique_tags
    out["avg_tags_per_doc"] = avg_tags_per_doc
    out["glove_path"] = glove_path
    out["notes"] = notes

    # Per-model params JSON
    model_params = model_params or {}
    out["model_params_json"] = out["name"].map(
        lambda m: json.dumps(model_params.get(m, {}), ensure_ascii=False, sort_keys=True)
    )

    # Optional prediction diagnostics
    if predictions is not None:
        out["pred_pos_rate"] = out["name"].map(
            lambda m: float(np.mean(predictions[m])) if m in predictions else np.nan
        )
        out["pred_avg_labels_per_doc"] = out["name"].map(
            lambda m: float(np.sum(predictions[m], axis=1).mean()) if m in predictions else np.nan
        )
    else:
        out["pred_pos_rate"] = np.nan
        out["pred_avg_labels_per_doc"] = np.nan

    if y_test is not None:
        out["y_test_pos_rate"] = float(np.mean(y_test))
        out["y_test_avg_labels_per_doc"] = float(np.sum(y_test, axis=1).mean())
    else:
        out["y_test_pos_rate"] = np.nan
        out["y_test_avg_labels_per_doc"] = np.nan

    # Append to history file
    output_csv = f"data/metrics_results_history_{top_k_tags}.csv"
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        prev = pd.read_csv(output_path)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(output_path, index=False)
    return output_path, run_id

