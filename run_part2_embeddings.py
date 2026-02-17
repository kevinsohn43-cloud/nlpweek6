"""
Script to run Part 2: Embeddings.
"""
import os
import pandas as pd
from src.data import load_data
from src.embeddings import get_embeddings
from src.models_dense import create_dense_model
from src.evaluate import evaluate_model, save_confusion_matrix, save_top_confusions

def main():
    # 1. Load Data
    X_train_text, X_test_text, y_train, y_test, class_names = load_data()
    
    # 2. Generate Embeddings
    # Requirement: Encode train and test separately
    print("\n--- Generating Embeddings ---")
    X_train_emb = get_embeddings(X_train_text, normalize=True)
    X_test_emb = get_embeddings(X_test_text, normalize=True)
    
    # 3. Define models
    models = ['gnb', 'lr', 'svm', 'rf']
    # Note: MultinomialNB is not appropriate for dense embeddings because they contain negative values
    # and MNB expects counts/frequencies. GaussianNB is used instead.
    
    results = []
    best_f1 = -1
    best_model_name = ""
    best_clf = None
    best_y_pred = None
    
    # 4. Train and Evaluate
    print("\n--- Training Models ---")
    for model_type in models:
        print(f"Training {model_type}...")
        clf = create_dense_model(model_type)
        clf.fit(X_train_emb, y_train)
        
        # We can re-use evaluate_model if we pass the classifier instead of pipeline
        # evaluate_model expects an object with .predict()
        acc, f1, y_pred = evaluate_model(clf, X_test_emb, y_test)
        print(f"-> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
        
        results.append({
            'Model': model_type,
            'Accuracy': acc,
            'Macro-F1': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_type
            best_clf = clf
            best_y_pred = y_pred

    # 5. Save Outputs
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/metrics_part2.csv", index=False)
    print(f"\nMetrics saved to outputs/metrics_part2.csv")
    
    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    save_confusion_matrix(y_test, best_y_pred, class_names, "outputs/confusion_part2.png")
    save_top_confusions(y_test, best_y_pred, class_names, "outputs/top_confusions_part2.csv")
    
    # 6. Comparison Summary
    print("\n--- Summary ---")
    print("Typically, sparse models (TF-IDF) perform very well on 20 Newsgroups because the classes")
    print("are distinguished by specific keywords (e.g., 'graphics', 'hockey', 'crypt').")
    print("Embeddings are better for semantic clustering as they capture meaning beyond keyword overlap,")
    print("but for classification on this specific dataset, simple lexical features are often sufficient/superior.")

if __name__ == "__main__":
    main()
