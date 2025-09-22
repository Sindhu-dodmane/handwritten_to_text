import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_features(features_path):
    """Load features from .npz file"""
    data = np.load(features_path)
    return data['x'], data['y']

def evaluate_knn(train_path, val_path, k=5, metric='cosine', weights='distance'):
    """Evaluate KNN classifier"""
    print("Loading features...")
    X_train, y_train = load_features(train_path)
    X_val, y_val = load_features(val_path)
    
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val: {X_val.shape[0]} samples")
    
    # Train KNN
    print(f"Training KNN with k={k}, metric={metric}, weights={weights}")
    knn = KNeighborsClassifier(
        n_neighbors=k, 
        metric=metric, 
        weights=weights, 
        n_jobs=-1
    )
    knn.fit(X_train, y_train)
    
    # Predictions
    y_pred = knn.predict(X_val)
    y_proba = knn.predict_proba(X_val)
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return y_val, y_pred, y_proba, knn

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_embeddings(train_path, val_path, save_dir="analysis"):
    """Analyze embedding quality"""
    X_train, y_train = load_features(train_path)
    X_val, y_val = load_features(val_path)
    
    # Calculate class centers
    unique_classes = np.unique(y_train)
    class_centers = {}
    for cls in unique_classes:
        mask = y_train == cls
        class_centers[cls] = np.mean(X_train[mask], axis=0)
    
    # Calculate intra-class and inter-class distances
    intra_distances = []
    inter_distances = []
    
    for cls in unique_classes:
        # Intra-class distances
        mask = y_train == cls
        class_embeddings = X_train[mask]
        if len(class_embeddings) > 1:
            center = class_centers[cls]
            distances = np.linalg.norm(class_embeddings - center, axis=1)
            intra_distances.extend(distances)
        
        # Inter-class distances
        for other_cls in unique_classes:
            if other_cls != cls:
                other_center = class_centers[other_cls]
                distance = np.linalg.norm(class_centers[cls] - other_center)
                inter_distances.append(distance)
    
    print(f"Average intra-class distance: {np.mean(intra_distances):.4f}")
    print(f"Average inter-class distance: {np.mean(inter_distances):.4f}")
    print(f"Separation ratio: {np.mean(inter_distances) / np.mean(intra_distances):.4f}")
    
    # Save analysis
    Path(save_dir).mkdir(exist_ok=True)
    np.savez(f"{save_dir}/embedding_analysis.npz",
             intra_distances=intra_distances,
             inter_distances=inter_distances,
             class_centers=class_centers)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN+KNN model")
    parser.add_argument("--train", type=str, required=True, help="Path to train features .npz")
    parser.add_argument("--val", type=str, required=True, help="Path to val features .npz")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric")
    parser.add_argument("--weights", type=str, default="distance", help="Weight function")
    parser.add_argument("--plot_cm", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--analyze", action="store_true", help="Analyze embedding quality")
    parser.add_argument("--save_dir", type=str, default="analysis", help="Directory to save analysis")
    
    args = parser.parse_args()
    
    # Evaluate KNN
    y_true, y_pred, y_proba, knn = evaluate_knn(
        args.train, args.val, args.k, args.metric, args.weights
    )
    
    if args.analyze:
        print("\nAnalyzing embedding quality...")
        analyze_embeddings(args.train, args.val, args.save_dir)
    
    if args.plot_cm:
        print("\nPlotting confusion matrix...")
        plot_confusion_matrix(y_true, y_pred, save_path=f"{args.save_dir}/confusion_matrix.png")

if __name__ == "__main__":
    main()

