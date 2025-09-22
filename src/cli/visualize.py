import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import seaborn as sns

def load_features(features_path):
    """Load features from .npz file"""
    data = np.load(features_path)
    return data['x'], data['y']

def plot_tsne(X, y, n_samples=2000, perplexity=30, save_path=None):
    """Plot t-SNE visualization of embeddings"""
    print(f"Computing t-SNE for {min(n_samples, len(X))} samples...")
    
    # Subsample for visualization
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f't-SNE Visualization of Kannada Character Embeddings (n={len(X_sub)})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca(X, y, n_components=2, n_samples=2000, save_path=None):
    """Plot PCA visualization of embeddings"""
    print(f"Computing PCA for {min(n_samples, len(X))} samples...")
    
    # Subsample for visualization
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
    
    # Compute PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_sub)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sub, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'PCA Visualization of Kannada Character Embeddings (n={len(X_sub)})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(y, class_names=None, save_path=None):
    """Plot class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(unique)), counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    if class_names:
        plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_embedding_histogram(X, save_path=None):
    """Plot histogram of embedding values"""
    plt.figure(figsize=(12, 6))
    plt.hist(X.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Embedding Values')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings and dataset")
    parser.add_argument("--features", type=str, required=True, help="Path to features .npz file")
    parser.add_argument("--method", type=str, choices=['tsne', 'pca', 'distribution', 'histogram', 'all'], 
                       default='all', help="Visualization method")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples for t-SNE/PCA")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Load features
    X, y = load_features(args.features)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Create save directory
    Path(args.save_dir).mkdir(exist_ok=True)
    
    if args.method in ['tsne', 'all']:
        plot_tsne(X, y, args.n_samples, args.perplexity, 
                 f"{args.save_dir}/tsne_visualization.png")
    
    if args.method in ['pca', 'all']:
        plot_pca(X, y, n_samples=args.n_samples, 
                save_path=f"{args.save_dir}/pca_visualization.png")
    
    if args.method in ['distribution', 'all']:
        plot_class_distribution(y, save_path=f"{args.save_dir}/class_distribution.png")
    
    if args.method in ['histogram', 'all']:
        plot_embedding_histogram(X, save_path=f"{args.save_dir}/embedding_histogram.png")

if __name__ == "__main__":
    main()

