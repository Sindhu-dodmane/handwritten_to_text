import argparse
import time
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

def load_features(features_path):
    """Load features from .npz file"""
    data = np.load(features_path)
    return data['x'], data['y']

def benchmark_classifiers(X_train, y_train, X_val, y_val):
    """Benchmark different classifiers on the same features"""
    results = {}
    
    classifiers = {
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, metric='cosine', n_jobs=-1),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7, metric='cosine', n_jobs=-1),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1),
        'SVM (linear)': SVC(kernel='linear', random_state=42),
        'SVM (rbf)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Train
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = clf.predict(X_val)
        pred_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'train_time': train_time,
            'pred_time': pred_time,
            'total_time': train_time + pred_time
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Pred time: {pred_time:.2f}s")
        print()
    
    return results

def benchmark_embedding_dimensions(train_path, val_path, embedding_dims=[64, 128, 256, 512]):
    """Benchmark different embedding dimensions"""
    print("Note: This requires retraining the CNN with different embedding dimensions")
    print("For now, we'll use the available features and simulate different dimensions")
    
    X_train, y_train = load_features(train_path)
    X_val, y_val = load_features(val_path)
    
    results = {}
    current_dim = X_train.shape[1]
    
    for dim in embedding_dims:
        if dim > current_dim:
            print(f"Skipping dim={dim} (current features have {current_dim} dimensions)")
            continue
            
        # Truncate features to simulate smaller dimensions
        X_train_trunc = X_train[:, :dim]
        X_val_trunc = X_val[:, :dim]
        
        print(f"Benchmarking with {dim} dimensions...")
        
        # Use KNN for fair comparison
        clf = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        clf.fit(X_train_trunc, y_train)
        y_pred = clf.predict(X_val_trunc)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        
        results[dim] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
    
    return results

def benchmark_k_values(X_train, y_train, X_val, y_val, k_values=[1, 3, 5, 7, 10, 15, 20]):
    """Benchmark different k values for KNN"""
    results = {}
    
    for k in k_values:
        print(f"Testing KNN with k={k}...")
        
        clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        
        results[k] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
    
    return results

def save_results(results, save_path):
    """Save benchmark results to file"""
    with open(save_path, 'w') as f:
        f.write("Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark different classifiers and parameters")
    parser.add_argument("--train", type=str, required=True, help="Path to train features .npz")
    parser.add_argument("--val", type=str, required=True, help="Path to val features .npz")
    parser.add_argument("--benchmark", type=str, choices=['classifiers', 'dimensions', 'k_values', 'all'], 
                       default='all', help="Type of benchmark to run")
    parser.add_argument("--save_dir", type=str, default="benchmarks", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load features
    X_train, y_train = load_features(args.train)
    X_val, y_val = load_features(args.val)
    
    print(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print()
    
    # Create save directory
    Path(args.save_dir).mkdir(exist_ok=True)
    
    if args.benchmark in ['classifiers', 'all']:
        print("Benchmarking different classifiers...")
        print("=" * 50)
        classifier_results = benchmark_classifiers(X_train, y_train, X_val, y_val)
        save_results(classifier_results, f"{args.save_dir}/classifier_benchmark.txt")
        print()
    
    if args.benchmark in ['k_values', 'all']:
        print("Benchmarking different k values for KNN...")
        print("=" * 50)
        k_results = benchmark_k_values(X_train, y_train, X_val, y_val)
        save_results(k_results, f"{args.save_dir}/k_values_benchmark.txt")
        print()
    
    if args.benchmark in ['dimensions', 'all']:
        print("Benchmarking different embedding dimensions...")
        print("=" * 50)
        dim_results = benchmark_embedding_dimensions(args.train, args.val)
        save_results(dim_results, f"{args.save_dir}/dimensions_benchmark.txt")
        print()

if __name__ == "__main__":
    main()

