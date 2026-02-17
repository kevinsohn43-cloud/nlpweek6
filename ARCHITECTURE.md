# Architecture

## Data Flow
1.  **Ingestion** (`src/data.py`): Fetches raw data from sklearn, performs rigorous cleaning (stripping headers/footers), and applies a reproducible stratified split.
2.  **Transformation**:
    -   *Sparse*: Text -> Count/TF-IDF Vectorizer -> Model
    -   *Dense*: Text -> SentenceTransformer -> Embeddings -> Model
3.  **Modeling**: Scikit-learn estimators are trained on transformed data.
4.  **Evaluation** (`src/evaluate.py`): Predictions are scored against ground truth test labels.
5.  **Clustering** (`src/clustering.py`): Unsupervised learning on embeddings to find structure.

## Module Responsibilities
-   `src/data.py`: Data loading and splitting. Pure function, no state.
-   `src/models_sparse.py`: Factory functions for sklearn Pipelines. Prevents leakage by bundling vectorizer and classifier.
-   `src/models_dense.py`: Factory functions for classifiers compatible with dense input.
-   `src/embeddings.py`: Wrapper for `SentenceTransformer` handling batching and normalization.
-   `src/evaluate.py`: Centralized metric calculation and visualization plotting.
-   `src/clustering.py`: K-Means logic, Elbow visualization, and representative sample extraction.
-   `src/labeling_llm.py`: Logic for generating labels from text samples (LLM/Fallback).
-   `src/tree.py`: Data structure for building and printing the topic hierarchy.

## Design Choices

### Pipeline & Data Leakage
We use `sklearn.pipeline.Pipeline` for Part 1 to ensure that vectorizers are fit *only* on the training data during `fit()` and then applied to test data during `predict()`. This prevents information from the test set (vocabulary frequency) from leaking into the training process.

### Random State
`random_state=42` is enforced globally across `train_test_split`, `KMeans`, and all classifiers. This ensures that every run produces identical splits, initializations, and results, essential for academic reproducibility.

### Why Embeddings for Clustering?
While TF-IDF is great for classification, it produces high-dimensional sparse vectors (vocab size > 100k) which are poor for distance-based clustering algorithms like K-Means (curse of dimensionality). Dense embeddings (384 dimensions) capture semantic proximity, meaning "car" and "automobile" are close in space, allowing clusters to form around *meanings* rather than just shared words.
