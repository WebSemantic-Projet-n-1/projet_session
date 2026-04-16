from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass
class ModelResult:
    name: str
    micro_recall: float
    micro_precision: float
    micro_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    return ModelResult(
        name=name,
        micro_recall=recall_score(y_true, y_pred, average="micro", zero_division=0),
        micro_precision=precision_score(y_true, y_pred, average="micro", zero_division=0),
        micro_f1=f1_score(y_true, y_pred, average="micro", zero_division=0),
        y_true=y_true,
        y_pred=y_pred,
    )


def train_nb_model(X_train: list[str], y_train: np.ndarray, X_test: list[str]) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("bow", CountVectorizer(max_features=8000, ngram_range=(1, 1))),
            ("clf", OneVsRestClassifier(MultinomialNB())),
        ]
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)
    return (proba >= 0.5).astype(int)


def train_svm_model(X_train: list[str], y_train: np.ndarray, X_test: list[str]) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=8000, stop_words="english")),
            ("clf", OneVsRestClassifier(LinearSVC(random_state=42))),
        ]
    )
    pipeline.fit(X_train, y_train)
    decision = pipeline.decision_function(X_test)
    return (decision >= 0.0).astype(int)


def train_lda_model(X_train: list[str], y_train: np.ndarray, X_test: list[str], n_topics: int) -> np.ndarray:
    train_tokens = [text.split() for text in X_train]
    test_tokens = [text.split() for text in X_test]

    dictionary = corpora.Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=8000)
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]
    test_corpus = [dictionary.doc2bow(tokens) for tokens in test_tokens]
    if len(dictionary) == 0 or all(len(bow) == 0 for bow in train_corpus):
        # Fallback avoids runtime failure on tiny/over-filtered corpora.
        return train_nb_model(X_train, y_train, X_test)

    lda = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        iterations=100,
    )

    def to_topic_vectors(corpus: list[list[tuple[int, int]]]) -> np.ndarray:
        rows = []
        for bow in corpus:
            topic_dist = lda.get_document_topics(bow, minimum_probability=0.0)
            rows.append([weight for _, weight in topic_dist])
        return np.asarray(rows, dtype=np.float32)

    X_train_topic = to_topic_vectors(train_corpus)
    X_test_topic = to_topic_vectors(test_corpus)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_topic, y_train)
    proba = clf.predict_proba(X_test_topic)
    return (proba >= 0.5).astype(int)


def train_doc2vec_model(X_train: list[str], y_train: np.ndarray, X_test: list[str]) -> np.ndarray:
    tagged_docs = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X_train)]
    doc2vec = Doc2Vec(
        vector_size=200,
        window=3,
        min_count=2,
        workers=1,
        epochs=20,
        seed=42,
    )
    doc2vec.build_vocab(tagged_docs)
    doc2vec.train(tagged_docs, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    X_train_vec = np.vstack([doc2vec.infer_vector(text.split()) for text in X_train])
    X_test_vec = np.vstack([doc2vec.infer_vector(text.split()) for text in X_test])

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_vec, y_train)
    proba = clf.predict_proba(X_test_vec)
    return (proba >= 0.5).astype(int)


class TemporalAttention(layers.Layer):
    """Simple attention pooling over sequence axis."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.score = layers.Dense(1, activation="tanh")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        scores = self.score(inputs)  # (batch, time, 1)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


def build_embedding_matrix(
    tokenizer: Tokenizer,
    vocab_size: int,
    glove_path: str | Path | None,
    embedding_dim: int = 300,
) -> tuple[np.ndarray, bool]:
    # Random fallback keeps the experiment runnable if GloVe is missing.
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.05, size=(vocab_size, embedding_dim)).astype(np.float32)
    if glove_path is None:
        return matrix, False

    glove_path = Path(glove_path)
    if not glove_path.exists():
        return matrix, False

    found_vectors = 0
    with glove_path.open("r", encoding="utf-8") as glove_file:
        for line in glove_file:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")
            if vector.shape[0] != embedding_dim:
                continue
            index = tokenizer.word_index.get(word)
            if index is not None and index < vocab_size:
                matrix[index] = vector
                found_vectors += 1
    return matrix, found_vectors > 0


def build_bigru_attention_model(
    vocab_size: int,
    num_labels: int,
    max_len: int,
    embedding_matrix: np.ndarray,
    embeddings_trainable: bool,
) -> tf.keras.Model:
    input_ids = layers.Input(shape=(max_len,), name="tokens")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=embeddings_trainable,
    )(input_ids)
    x = layers.Bidirectional(layers.GRU(25, return_sequences=True))(x)
    x = TemporalAttention()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu")(x)
    output = layers.Dense(num_labels, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_ids, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="binary_crossentropy",
    )
    return model


def train_bigru_attention_model(
    X_train: list[str],
    y_train: np.ndarray,
    X_test: list[str],
    max_words: int = 15000,
    max_len: int = 300,
    glove_path: str | Path | None = "data/glove.6B.300d.txt",
    epochs: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    embedding_matrix, has_pretrained_vectors = build_embedding_matrix(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        glove_path=glove_path,
        embedding_dim=300,
    )

    model = build_bigru_attention_model(
        vocab_size=vocab_size,
        num_labels=y_train.shape[1],
        max_len=max_len,
        embedding_matrix=embedding_matrix,
        embeddings_trainable=not has_pretrained_vectors,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
    )
    proba = model.predict(X_test_pad, verbose=0)
    return (proba >= 0.5).astype(int)
