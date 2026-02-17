"""
Script to run Part 1: Sparse Models.
"""
import os
import pandas as pd
from src.data import load_data
from src.models_sparse import create_sparse_pipeline
from src.evaluate import evaluate_model, save_confusion_matrix, save_top_confusions

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test, class_names = load_data()
    
    # 2. Define models to train
    models = ['nb', 'lr', 'svm', 'rf']
    vectorizers = ['count', 'tfidf']
    
    results = []
    best_f1 = -1
    best_model_name = ""
    best_pipeline = None
    best_y_pred = None
    
    # 3. Train and Evaluate
    for model_type in models:
        for vect_type in vectorizers:
            name = f"{model_type}_{vect_type}"
            print(f"Training {name}...")
            
            pipeline = create_sparse_pipeline(model_type, vect_type)
            pipeline.fit(X_train, y_train)
            
            acc, f1, y_pred = evaluate_model(pipeline, X_test, y_test)
            print(f"-> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
            
            results.append({
                'Model': model_type,
                'Vectorizer': vect_type,
                'Accuracy': acc,
                'Macro-F1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_pipeline = pipeline
                best_y_pred = y_pred

    # 4. Save Metrics
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/metrics_part1.csv", index=False)
    print(f"\nMetrics saved to outputs/metrics_part1.csv")
    
    # 5. Save Analysis for Best Model
    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    save_confusion_matrix(y_test, best_y_pred, class_names, "outputs/confusion_part1.png")
    save_top_confusions(y_test, best_y_pred, class_names, "outputs/top_confusions_part1.csv")

if __name__ == "__main__":
    main()
