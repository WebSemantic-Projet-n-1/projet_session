from __future__ import annotations

from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


RAW_DATASET_COLUMNS = ["doc.id", "title", "citeulike.id", "raw.title", "raw.abstract"]


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """Read CSV with a latin-1 fallback for noisy legacy encodings."""
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def _read_lines_with_fallback(file_path: Path) -> list[str]:
    """Read text lines with a latin-1 fallback for legacy dataset files."""
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1").splitlines()


def _load_citeulike_tags(dataset_dir: Path, expected_rows: int) -> list[str]:
    """Build pipe-separated tag strings from item-tag.dat and tags.dat files."""
    tags_path = dataset_dir / "tags.dat"
    item_tag_path = dataset_dir / "item-tag.dat"

    if not tags_path.exists() or not item_tag_path.exists():
        raise ValueError(
            "Detected CiteULike raw schema but missing tag files. "
            "Expected `tags.dat` and `item-tag.dat` in the same directory as `raw-data.csv`."
        )

    tags_vocab = _read_lines_with_fallback(tags_path)
    item_tag_lines = _read_lines_with_fallback(item_tag_path)

    if len(item_tag_lines) < expected_rows:
        raise ValueError(
            f"`item-tag.dat` has fewer rows ({len(item_tag_lines)}) than dataset rows ({expected_rows})."
        )

    def build_tag_string(line: str) -> str:
        tokens = line.strip().split()
        if not tokens:
            return ""
        tag_ids: list[int] = []
        for tok in tokens[1:]:
            if tok.isdigit():
                tag_ids.append(int(tok))
        resolved_tags = [str(tags_vocab[tag_id]) for tag_id in tag_ids if 0 <= tag_id < len(tags_vocab)]
        return "|".join(resolved_tags)

    return [build_tag_string(line) for line in item_tag_lines[:expected_rows]]


def _normalize_raw_citeulike_schema(df_raw: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Map CiteULike raw-data schema to the internal training schema."""
    df = df_raw.copy()
    df["title"] = df["raw.title"].fillna(df["title"]).fillna("").astype(str)
    df["abstract"] = df["raw.abstract"].fillna("").astype(str)
    df["tags"] = _load_citeulike_tags(csv_path.parent, expected_rows=len(df))
    return df[["title", "abstract", "tags"]]


def load_simple_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load CiteULike raw-data CSV and normalize to title/abstract/tags."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Expected the CiteULike raw dataset file `raw-data.csv`."
        )

    df = _read_csv_with_fallback(csv_path)
    if all(col in df.columns for col in RAW_DATASET_COLUMNS):
        normalized = _normalize_raw_citeulike_schema(df, csv_path)
    else:
        missing_raw = [col for col in RAW_DATASET_COLUMNS if col not in df.columns]
        raise ValueError(
            "Unsupported dataset schema. "
            f"Missing raw schema columns: {missing_raw}."
        )

    normalized["title"] = normalized["title"].fillna("").astype(str)
    normalized["abstract"] = normalized["abstract"].fillna("").astype(str)
    normalized["tags"] = normalized["tags"].fillna("").astype(str)
    normalized["text"] = (normalized["title"] + ". " + normalized["abstract"]).str.strip()
    return normalized


def ensure_nltk_resources(allow_download: bool = True) -> None:
    path_resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    missing: list[str] = []

    # 1) Check classique pour tokenizer/stopwords
    for path_key, download_key in path_resources:
        try:
            nltk.data.find(path_key)
        except LookupError:
            if allow_download:
                nltk.download(download_key, quiet=True)
            try:
                nltk.data.find(path_key)
            except LookupError:
                missing.append(download_key)

    # 2) Check wordnet via API (plus fiable que nltk.data.find dans certains conteneurs)
    def check_wordnet() -> bool:
        try:
            _ = wn.synsets("dog")
            return True
        except LookupError:
            return False
    if not check_wordnet():
        if allow_download:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        if not check_wordnet():
            missing.extend(["wordnet", "omw-1.4"])
    if missing:
        uniq = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing NLTK resources: "
            f"{uniq}. Install them with: "
            f"python -m nltk.downloader {' '.join(sorted(set(missing)))}"
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

