from __future__ import annotations

from pathlib import Path
import re
import glob

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize




def load_citeulike_a_dataset(csv_path: str | Path) -> pd.DataFrame:
    dat_files_list = glob.glob(str(csv_path / "*.dat"))

    item_tag_df = None
    tags_df = None
    raw_data_df = None

    for file in dat_files_list:
        print(file)
        df = pd.read_csv(file, header=None)
        if "item-tag" in file:
            item_tag_df = df
        if "tags" in file:
            tags_df = df

    csv_files_list = glob.glob(str(csv_path / "*.csv"))
    for file in csv_files_list:
        df = pd.read_csv(file, encoding="latin1")
        if "raw-data" in file:
            raw_data_df = df

    if item_tag_df is not None:
        row = item_tag_df.shape[0]

    tags_lookup = tags_df.iloc[:, 0].tolist()

    rows: list[dict[str, str]] = []
    for i in range(row):
        parts = item_tag_df.iloc[i, 0].split()
        if int(parts[0]) != 0:
            tags_bag: list[str] = []
            for tags_id in parts[1:]:
                tag = tags_lookup[int(tags_id)]
                if pd.notna(tag):
                    tags_bag.append(str(tag).strip())
            rows.append({
                "title": raw_data_df.iloc[i, 1],
                "abstract": raw_data_df.iloc[i, 4],
                "tags": "|".join(tags_bag),
            })

    df = pd.DataFrame(rows, columns=["title", "abstract", "tags"])

    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["text"] = (df["title"] + ". " + df["abstract"]).str.strip()

    return df



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

