from __future__ import annotations

from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


REQUIRED_COLUMNS = ["title", "abstract", "tags"]


def load_simple_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a simple CSV dataset with `title`, `abstract`, and `tags` columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Create it with columns: title, abstract, tags."
        )

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["text"] = (df["title"] + ". " + df["abstract"]).str.strip()
    return df


def ensure_nltk_resources(allow_download: bool = True) -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    missing: list[str] = []
    for path_key, download_key in resources:
        try:
            nltk.data.find(path_key)
        except LookupError:
            if allow_download:
                nltk.download(download_key, quiet=True)
            try:
                nltk.data.find(path_key)
            except LookupError:
                missing.append(download_key)
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Missing NLTK resources: "
            f"{missing_list}. Install them with: "
            f"python -m nltk.downloader {missing_list}"
        )


def preprocess_text_nltk(df: pd.DataFrame, allow_download: bool = True) -> pd.DataFrame:
    """Apply article-like preprocessing: tokenize, remove stopwords, lemmatize."""
    ensure_nltk_resources(allow_download=allow_download)
    sw = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in sw and tok.strip()]
        return " ".join(tokens)

    out = df.copy()
    out["processed_text"] = out["text"].apply(clean)
    return out


def split_tags(tag_string: str, sep: str = "|") -> list[str]:
    return [t.strip().lower() for t in tag_string.split(sep) if t.strip()]


def keep_top_k_tags(df: pd.DataFrame, top_k: int = 10, sep: str = "|") -> pd.DataFrame:
    """Keep only top-k frequent tags and rows having at least one."""
    all_tags: list[str] = []
    for tag_string in df["tags"].tolist():
        all_tags.extend(split_tags(tag_string, sep=sep))

    top_tags = (
        pd.Series(all_tags)
        .value_counts()
        .head(top_k)
        .index.tolist()
    )
    top_tags_set = set(top_tags)

    def filter_tags(tag_string: str) -> list[str]:
        return [t for t in split_tags(tag_string, sep=sep) if t in top_tags_set]

    df = df.copy()
    df["tag_list"] = df["tags"].apply(filter_tags)
    df = df[df["tag_list"].str.len() > 0].reset_index(drop=True)
    return df


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple text length features for EDA."""
    out = df.copy()
    out["title_words"] = out["title"].str.split().str.len()
    out["abstract_words"] = out["abstract"].str.split().str.len()
    text_col = "processed_text" if "processed_text" in out.columns else "text"
    out["text_words"] = out[text_col].str.split().str.len()
    out["num_tags"] = out["tag_list"].apply(len)
    return out

