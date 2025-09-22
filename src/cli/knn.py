import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"])
    p.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"])
    args = p.parse_args()

    tr = np.load(args.train)
    va = np.load(args.val)
    x_tr, y_tr = tr["x"], tr["y"]
    x_va, y_va = va["x"], va["y"]

    knn = KNeighborsClassifier(n_neighbors=args.k, metric=args.metric, weights=args.weights, n_jobs=-1)
    knn.fit(x_tr, y_tr)
    pred = knn.predict(x_va)
    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, average="macro")
    print(f"KNN k={args.k} metric={args.metric} acc={acc:.4f} macroF1={f1:.4f}")
    print(classification_report(y_va, pred))


if __name__ == "__main__":
    main()


