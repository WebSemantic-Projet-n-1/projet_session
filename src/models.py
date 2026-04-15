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


class AdditiveAttention(layers.Layer):
    """Additive attention with learnable context vector (Yang et al. 2016 HAN).

    For each position t:
        u_t     = tanh(W * h_t + b)           # project hidden state
        alpha_t = softmax(u_t @ u_context)    # score against context vector
        output  = sum_t(alpha_t * h_t)        # weighted sum of hidden states
    """

    def __init__(self, attention_dim: int = 100, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape: Any) -> None:
        hidden_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="W", shape=(hidden_dim, self.attention_dim), initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="b", shape=(self.attention_dim,), initializer="zeros"
        )
        self.u_context = self.add_weight(
            name="u_context", shape=(self.attention_dim,), initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch, time, hidden_dim)
        u = tf.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        # u: (batch, time, attention_dim)
        scores = tf.reduce_sum(u * self.u_context, axis=-1, keepdims=True)
        # scores: (batch, time, 1)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)
        # output: (batch, hidden_dim)

    def get_config(self) -> dict:
        config = super().get_config()
        config["attention_dim"] = self.attention_dim
        return config


class TemporalAttention(layers.Layer):
    """Simple attention pooling (legacy flat model). Use AdditiveAttention for HAN."""

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


def _texts_to_han_tensor(
    texts: list[str],
    tokenizer: Tokenizer,
    NK: int,
    NT: int,
) -> np.ndarray:
    """Build (n_docs, NK, NT) integer tensor from flat preprocessed texts.

    Each document is chunked into NT-word blocks (simulated sentences).
    Up to NK blocks are kept; shorter documents and blocks are zero-padded.
    """
    result = np.zeros((len(texts), NK, NT), dtype=np.int32)
    for i, text in enumerate(texts):
        words = text.split()
        if not words:
            continue
        chunks = [words[j: j + NT] for j in range(0, len(words), NT)][:NK]
        for k, chunk in enumerate(chunks):
            seq = tokenizer.texts_to_sequences([" ".join(chunk)])[0]
            seq = seq[:NT]
            result[i, k, : len(seq)] = seq
    return result


def build_han_model(
    vocab_size: int,
    num_labels: int,
    NK: int,
    NT: int,
    embedding_matrix: np.ndarray,
    embeddings_trainable: bool,
    word_gru_units: int = 25,
    sent_gru_units: int = 5,
    dense_units: int = 50,
    dropout_rate: float = 0.2,
    attention_dim: int = 100,
) -> tf.keras.Model:
    """Hierarchical Attention Network (Hassan et al., RecSys 2018).

    Two-level architecture:
        Word encoder  : Embedding(GloVe 300d) -> BiGRU(word_gru_units)
                        -> AdditiveAttention -> sentence vector
        Sent encoder  : BiGRU(sent_gru_units) -> AdditiveAttention -> doc vector
        Classification: Dropout -> Dense(dense_units, relu) -> Dropout
                        -> Dense(n_tags, sigmoid)

    Input shape : (batch, NK, NT)  e.g. (batch, 10, 50)
    Output shape: (batch, num_labels)
    """
    # --- Word-level encoder (one sentence of NT tokens) ---
    sentence_input = layers.Input(shape=(NT,), name="sentence_tokens")
    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=embeddings_trainable,
        name="word_embedding",
    )(sentence_input)
    h_word = layers.Bidirectional(
        layers.GRU(word_gru_units, return_sequences=True), name="word_bigru"
    )(emb)
    # h_word: (batch, NT, 2 * word_gru_units)
    s_word = AdditiveAttention(attention_dim=attention_dim, name="word_attention")(h_word)
    # s_word: (batch, 2 * word_gru_units)  — one vector per sentence
    word_encoder = tf.keras.Model(sentence_input, s_word, name="word_encoder")

    # --- Document-level encoder ---
    doc_input = layers.Input(shape=(NK, NT), name="document")
    # Apply the word encoder to each of the NK sentences independently
    sent_vecs = layers.TimeDistributed(word_encoder, name="sentence_encoding")(doc_input)
    # sent_vecs: (batch, NK, 2 * word_gru_units)
    h_sent = layers.Bidirectional(
        layers.GRU(sent_gru_units, return_sequences=True), name="sent_bigru"
    )(sent_vecs)
    # h_sent: (batch, NK, 2 * sent_gru_units)
    doc_vec = AdditiveAttention(attention_dim=attention_dim, name="sent_attention")(h_sent)
    # doc_vec: (batch, 2 * sent_gru_units)

    # --- Classification head ---
    x = layers.Dropout(dropout_rate)(doc_vec)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(num_labels, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=doc_input, outputs=output, name="HAN_BiGRU")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="binary_crossentropy",
    )
    return model


def train_han_model(
    X_train: list[str],
    y_train: np.ndarray,
    X_test: list[str],
    max_words: int = 15000,
    NK: int = 10,
    NT: int = 50,
    glove_path: str | Path | None = "data/glove.6B.300d.txt",
    epochs: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    """Train the hierarchical attention model (paper reproduction).

    Flat lemmatized strings are chunked into (NK=10, NT=50) tensors to feed
    the two-level bi-GRU + attention architecture (Hassan et al., RecSys 2018).
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train)

    X_train_3d = _texts_to_han_tensor(X_train, tokenizer, NK, NT)
    X_test_3d = _texts_to_han_tensor(X_test, tokenizer, NK, NT)

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    embedding_matrix, has_pretrained = build_embedding_matrix(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        glove_path=glove_path,
        embedding_dim=300,
    )

    model = build_han_model(
        vocab_size=vocab_size,
        num_labels=y_train.shape[1],
        NK=NK,
        NT=NT,
        embedding_matrix=embedding_matrix,
        embeddings_trainable=not has_pretrained,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        X_train_3d,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
    )
    proba = model.predict(X_test_3d, verbose=0)
    return (proba >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Legacy flat bi-GRU + single attention (kept for comparison / backward compat)
# ---------------------------------------------------------------------------

def build_bigru_attention_model(
    vocab_size: int,
    num_labels: int,
    max_len: int,
    embedding_matrix: np.ndarray,
    embeddings_trainable: bool,
) -> tf.keras.Model:
    """Flat (non-hierarchical) bi-GRU + attention. Legacy — use build_han_model."""
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
    """Flat bi-GRU + attention (legacy). Use train_han_model for paper reproduction."""
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
