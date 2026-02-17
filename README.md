# 20news-nlp-pipeline

A modular NLP pipeline for the 20 Newsgroups dataset, implementing sparse classification, dense embedding classification, and hierarchical clustering with LLM-based labeling.

## Overview
This project explores three key NLP approaches:
1.  **Sparse Classification (TF-IDF)**: Traditional baseline using lexical features.
2.  **Dense Classification (Embeddings)**: Modern approach using semantic embeddings from `sentence-transformers`.
3.  **Topic Modeling**: Hierarchical clustering combined with generative labeling to discover and name topics.

## Dataset
- **Source**: `sklearn.datasets.fetch_20newsgroups`
- **Subset**: `all` (train + test split disregarded in favor of a new stratified split).
- **Size**: ~18,000 documents across 20 topics.
- **Split**: 80% Training, 20% Testing (Stratified).

## Setup

1.  **Create Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Copy `.env.example` to `.env` and set your `LLM_API_KEY` (optional, falls back to keyword extraction if missing).
    ```bash
    cp .env.example .env
    ```

## Usage

### Part 1: Sparse Classification
Trains MultinomialNB, LogisticRegression, LinearSVC, and RandomForest using Count and TF-IDF vectors.
```bash
python -m scripts.run_part1_sparse
```
**Outputs**:
- `outputs/metrics_part1.csv`: Performance metrics.
- `outputs/confusion_part1.png`: Confusion matrix of the best model.
- `outputs/top_confusions_part1.csv`: Top 10 confused class pairs.

### Part 2: Embedding Classification
Generates embeddings using `all-MiniLM-L6-v2` and trains GaussianNB, LogisticRegression, LinearSVC, and RandomForest.
```bash
python -m scripts.run_part2_embeddings
```
**Outputs**:
- `outputs/metrics_part2.csv`: Performance metrics.
- `outputs/confusion_part2.png`: Confusion matrix.
- `outputs/top_confusions_part2.csv`: Top confusions.

### Part 3: Clustering & Topic Tree
Performs K-Means clustering (using Elbow method), sub-clustering, and automatic labeling.
```bash
python -m scripts.run_part3_clustering
```
**Outputs**:
- `outputs/elbow_kmeans.png`: Elbow plot for K selection.
- `outputs/clusters/`: JSON files containing cluster info and representatives.
- `outputs/topic_tree.txt`: Hierarchical text representation of topics.

## Modeling Comparison
- **Sparse Models**: Typically perform excellent on this dataset (Accuracy > 85%) because the 20 Newsgroups classes are often defined by specific jargon (e.g., "windows", "god", "team") which TF-IDF captures perfectly.
- **Dense Models**: While semantically rich, short-vector embeddings without fine-tuning may sometimes underperform huge sparse feature spaces on pure keyword-based classification tasks, though they generalize better to unseen vocabulary.

## Clustering Logic
1.  **Embeddings**: Generated via `all-MiniLM-L6-v2`.
2.  **Elbow Method**: Used to determine optimal K (Range 2-9).
3.  **Labeling**: Representative documents (closest to centroid) are sent to an LLM (or fallback keyword extractor) to generate a "Topic: Label" string.
4.  **Hierarchy**: Large clusters are recursively sub-clustered to find granular topics.
