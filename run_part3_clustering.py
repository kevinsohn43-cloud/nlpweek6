"""
Script to run Part 3: Clustering.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data import load_data
from src.embeddings import get_embeddings
from src.clustering import run_kmeans_elbow, get_cluster_representatives
from src.labeling_llm import generate_label
from src.tree import TreeNode, print_tree

def main():
    # 1. Load Data (Simulated or Real)
    # For clustering, we usually use the entire dataset or just train
    # Requirements don't specify, but typically we cluster the whole corpus or train set.
    # Let's use X_train for learning clusters, but for the assignment let's just use the train set provided by load_data
    # to be consistent with "representative documents".
    X_train_text, _, _, _, _ = load_data()
    
    # 2. Embeddings
    print("\n--- Generating Embeddings for Clustering ---")
    # For speed in this script, we might want to cache embeddings, but per rules, we generate them.
    # Using a subset for speed if needed, but requirements say "fetch_20newsgroups(subset='all')" in data.py.
    # We will use the first 2000 for speed if this were a demo, but rules say >10,000 rows.
    embeddings = get_embeddings(X_train_text, normalize=True)
    
    # 3. Step A: KMeans + Elbow
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    print("\n--- Step A: KMeans Elbow ---")
    run_kmeans_elbow(embeddings, max_k=9, output_path="outputs/elbow_kmeans.png")
    
    # Choose K < 10. Let's pick K=4 for demonstration/stability or based on logic (20 newsgroups -> 20 clusters? No, K<10)
    # The user manual says "Choose K < 10". I'll hardcode K=6 as a reasonable number for high-level topics.
    CHOSEN_K = 6
    print(f"Chosen K: {CHOSEN_K}")
    
    kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    
    # Label Clusters
    path_clusters = "outputs/clusters"
    if not os.path.exists(path_clusters):
        os.makedirs(path_clusters)
        
    representatives = get_cluster_representatives(embeddings, labels, CHOSEN_K, n_reps=8)
    
    top_level_info = {}
    top_level_nodes = []
    
    print("\n--- Generating Top-Level Labels ---")
    for i in range(CHOSEN_K):
        rep_indices = representatives[i]
        rep_docs = [X_train_text[idx] for idx in rep_indices]
        label = generate_label(rep_docs, i)
        print(f"Cluster {i}: {label}")
        
        node = TreeNode(label, level=0, node_id=f"{i}")
        top_level_nodes.append(node)
        
        top_level_info[i] = {
            "label": label,
            "count": int(np.sum(labels == i)),
            "representatives": rep_docs
        }
        
    with open("outputs/clusters/top_level.json", "w") as f:
        json.dump(top_level_info, f, indent=2)
        
    # 4. Step B: Subclustering
    print("\n--- Step B: Subclustering Largest Clusters ---")
    # Identify 2 largest
    counts = [(i, top_level_info[i]["count"]) for i in range(CHOSEN_K)]
    counts.sort(key=lambda x: x[1], reverse=True)
    largest_clusters = [x[0] for x in counts[:2]]
    
    for clust_id in largest_clusters:
        print(f"Subclustering Cluster {clust_id}...")
        # Get embeddings for this cluster
        indices = np.where(labels == clust_id)[0]
        sub_embeddings = embeddings[indices]
        sub_docs_text = [X_train_text[idx] for idx in indices]
        
        # Exact 3 subclusters
        sub_k = 3
        sub_kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init="auto")
        sub_labels = sub_kmeans.fit_predict(sub_embeddings)
        
        # Get reps using the sub-embeddings and sub-labels
        sub_reps = get_cluster_representatives(sub_embeddings, sub_labels, sub_k, n_reps=6)
        
        sub_info = {}
        parent_node = top_level_nodes[clust_id]
        
        for j in range(sub_k):
            # Map back indices
            rep_rel_indices = sub_reps[j]
            rep_docs = [sub_docs_text[idx] for idx in rep_rel_indices]
            
            sub_label = generate_label(rep_docs, f"{clust_id}.{j}")
            print(f"  Subcluster {j}: {sub_label}")
            
            sub_node = TreeNode(sub_label, level=1, node_id=f"{clust_id}.{j}")
            parent_node.add_child(sub_node)
            
            sub_info[j] = {
                "label": sub_label,
                "representatives": rep_docs
            }
            
        with open(f"outputs/clusters/subclusters_cluster_{clust_id}.json", "w") as f:
            json.dump(sub_info, f, indent=2)
            
    # 5. Step C: Topic Tree
    print("\n--- Step C: Generating Topic Tree ---")
    # Create a dummy root or just list the top nodes? 
    # User format: - [0] Label ...
    # So we don't need a single root. But print_tree takes a root. 
    # I'll modify print_tree slightly or just create a virtual root.
    
    with open("outputs/topic_tree.txt", "w") as f:
        f.write("TOPIC TREE\n")
        for node in top_level_nodes:
            f.write(str(node) + "\n")
            for child in node.children:
                 f.write(str(child) + "\n")
                 
    print("Tree saved to outputs/topic_tree.txt")

if __name__ == "__main__":
    main()
