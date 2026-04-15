# projet_session

Reproduction simplifiée de l'article **Semantic-based Tag Recommendation in Scientific Bookmarking Systems**.

Source article: [https://dl.acm.org/doi/pdf/10.1145/3240323.3240409](https://dl.acm.org/doi/pdf/10.1145/3240323.3240409)

## Stack (alignée article)

- Python
- Keras + TensorFlow backend
- NLTK (tokenization, stopwords, lemmatization)
- scikit-learn (NB, SVM, classification multi-label)
- gensim (LDA, Paragraph Vector)

## Structure

- `notebooks/tag_reco_experiment.ipynb`: notebook principal.
- `src/data.py`: chargement/préparation des données.
- `src/models.py`: entraînement des modèles.
- `src/experiment.py`: orchestration des expériences.
- `src/visualization.py`: **toutes les visualisations** (sorties du notebook).
- `data/citeulike-a/`: dataset CiteULike-A (depuis GitHub).
- `data/citeulike_top10.csv`: jeu de données CSV simple (exemple).
- `scripts/run_experiment_cli.py`: exécution hors notebook, sortie des métriques dans le terminal.

## Format de données attendu (simple)

Un seul CSV avec colonnes:

- `title`
- `abstract`
- `tags` (séparées par `|`, ex: `nlp|deep learning|attention`)

Le notebook importe les données en **une cellule** via `DATA_PATH`.

## Installer le dataset réel (CiteULike-A)

Dataset source: [js05212/citeulike-a](https://github.com/js05212/citeulike-a).

Depuis la racine du projet:

```bash
git clone https://github.com/js05212/citeulike-a.git data/citeulike-a
```

Fichiers principaux installés:

- `data/citeulike-a/raw-data.csv`
- `data/citeulike-a/users.dat`
- `data/citeulike-a/item-tag.dat`
- `data/citeulike-a/tags.dat`
- `data/citeulike-a/mult.dat`
- `data/citeulike-a/vocabulary.dat`
- `data/citeulike-a/citations.dat`

Remarque: le pipeline actuel lit un CSV simple `title, abstract, tags` (ex: `data/citeulike_top10.csv`). Le dataset `citeulike-a` est bien installé pour la suite des travaux et l'intégration complète dans le pipeline.

## GloVe (optionnel mais recommandé)

Le modèle Bi-GRU+Attention cherche le fichier:

- `data/glove.6B.300d.txt`

S'il est absent, le notebook reste exécutable avec un embedding aléatoire (fallback simple). Fichier téléchargeable ici https://nlp.stanford.edu/projects/glove/

## Lancer avec Docker (NVIDIA TensorFlow base)

Prérequis:

- Docker + plugin compose
- NVIDIA Container Toolkit (si GPU)

Commandes:

```bash
docker compose build
docker compose up
```

Puis ouvrir Jupyter Lab: `http://localhost:8888`.

## Lancer en local avec venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Exécution hors notebook (terminal)

```bash
python scripts/run_experiment_cli.py
```

Options utiles:

```bash
python scripts/run_experiment_cli.py --data-path data/citeulike_top10.csv --glove-path data/glove.6B.300d.txt --top-k-tags 10 --test-size 0.1 --seed 42
```

## Workflow notebook

1. Régler `DATA_PATH` dans la cellule de paramètres.
2. Optionnel: régler `GLOVE_PATH` vers `data/glove.6B.300d.txt`.
3. Exécuter les cellules dans l'ordre.
4. La dernière cellule compile les métriques et affiche:
   - comparaison des modèles courants
   - comparaison avec les résultats de l'article.

## Logigramme de l'expérience

```mermaid
flowchart TD
    A[Démarrage] --> B[Charger CSV<br/>title, abstract, tags]
    B --> C[Filtrer top-10 tags]
    C --> D[Prétraitement NLTK<br/>tokenize + stopwords + lemmatize]
    D --> E[EDA<br/>distributions tags et longueurs]
    E --> F[Split train/test 90/10]
    F --> G[Entraîner modèles]
    G --> G1[NB unigram]
    G --> G2[SVM TF-IDF]
    G --> G3[LDA gensim]
    G --> G4[Paragraph Vector]
    G --> G5[Bi-GRU + Attention<br/>Keras/TensorFlow]
    G5 --> H[GloVe 300d si disponible]
    H --> I[Évaluation<br/>Micro-Recall / Precision / F1]
    I --> J[Compilation des résultats]
    J --> K[Visualisations comparatives]
    K --> L[Comparaison avec résultats article]
```
