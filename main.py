import time
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Load Melanoma Dataset from folder structure


DATASET_PATH = "melanoma_cancer_dataset_grayscale"
IMAGE_SIZE = (64, 64)

CLASS_MAP = {
    "benign": 0,
    "malignant": 1,
}


def load_melanoma_dataset(dataset_path, image_size=IMAGE_SIZE):
    X = []
    y = []

    for class_name, label in CLASS_MAP.items():
        images_dir = os.path.realpath(__file__)[:-len(os.path.basename(__file__))] + os.path.join(dataset_path, class_name)
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Directory not found: {images_dir}")

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]

        print(f"Loading {len(image_files)} images from class '{class_name}'...")

        for fname in image_files:
            fpath = os.path.join(images_dir, fname)
            try:
                img = Image.open(fpath).convert("L")          
                img = img.resize(image_size, Image.LANCZOS)   
                X.append(np.array(img).flatten())             
                y.append(label)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    return np.array(X, dtype=np.float32), np.array(y)


X, y = load_melanoma_dataset(DATASET_PATH)
print(f"\nTotal samples loaded: {X.shape[0]}")
print(f"Feature vector size: {X.shape[1]}")
print(f"Class distribution: benign={np.sum(y==0)}, malignant={np.sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples    : {X_test.shape[0]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def evaluate_model(model, Xtr, ytr, Xte, yte, name="Model"):
    start = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start

    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)

    if name != "skip":
        print(f"\n{name}")
        print(f"  Training Time : {train_time:.4f} s")
        print(f"  Accuracy      : {acc:.4f}")
        if isinstance(model, SVC):
            print(f"  Support Vectors: {model.n_support_.sum()}")
        print("-" * 50)

    if name == "skip":
        if isinstance(model, SVC):
            return train_time, acc, model.n_support_.sum()
        return train_time, acc

    return train_time


print("\n===== Applying PCA (100 components) =====")
pca_100 = PCA(n_components=100)
Xtr_pca_100 = pca_100.fit_transform(X_train)
Xte_pca_100 = pca_100.transform(X_test)

print("\n===== FULL TRAINING SET – 100 Eigenvectors =====")

full_times = {}

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    full_times[f"KNN k={k}"] = evaluate_model(
        knn, Xtr_pca_100, y_train, Xte_pca_100, y_test,
        name=f"KNN (k={k})"
    )

dt = DecisionTreeClassifier(random_state=42)
full_times["DT"] = evaluate_model(
    dt, Xtr_pca_100, y_train, Xte_pca_100, y_test,
    name="Decision Tree"
)

svm = SVC(kernel="rbf", gamma="scale")
full_times["SVM"] = evaluate_model(
    svm, Xtr_pca_100, y_train, Xte_pca_100, y_test,
    name="SVM"
)

def dbscan_reduce(X, y, eps=15, min_samples=5):
    X_reduced, y_reduced = [], []

    for label in np.unique(y):
        Xc = X[y == label]
        yc = y[y == label]

        db = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = db.fit_predict(Xc)

        mask = cluster_labels == -1
        X_reduced.append(Xc[mask])
        y_reduced.append(yc[mask])

    return np.vstack(X_reduced), np.hstack(y_reduced)


print("\n===== DBSCAN Numerosity Reduction =====")
X_train_red, y_train_red = dbscan_reduce(Xtr_pca_100, y_train)
print(f"Original training size : {Xtr_pca_100.shape[0]}")
print(f"Reduced training size  : {X_train_red.shape[0]}")
reduction_pct = (1 - X_train_red.shape[0] / Xtr_pca_100.shape[0]) * 100
print(f"Reduction              : {reduction_pct:.1f}%")
print("-" * 50)

print("\n===== REDUCED TRAINING SET – 100 Eigenvectors =====")

speedup_results = {}

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    red_time = evaluate_model(
        knn, X_train_red, y_train_red, Xte_pca_100, y_test,
        name=f"KNN (k={k}) – REDUCED"
    )
    speedup_results[f"KNN k={k}"] = full_times[f"KNN k={k}"] / red_time if red_time > 0 else float("inf")

dt = DecisionTreeClassifier(random_state=42)
red_time = evaluate_model(
    dt, X_train_red, y_train_red, Xte_pca_100, y_test,
    name="Decision Tree – REDUCED"
)
speedup_results["DT"] = full_times["DT"] / red_time if red_time > 0 else float("inf")

svm = SVC(kernel="rbf", gamma="scale")
red_time = evaluate_model(
    svm, X_train_red, y_train_red, Xte_pca_100, y_test,
    name="SVM – REDUCED"
)
speedup_results["SVM"] = full_times["SVM"] / red_time if red_time > 0 else float("inf")

print("\n===== SPEEDUP SUMMARY =====")
for model_name, sp in speedup_results.items():
    print(f"  {model_name}: {sp:.2f}x speedup")
print("-" * 50)

print("\n===== DIMENSIONALITY SWEEP (10 → 100, step 10) =====")

results = []

for dim in range(10, 101, 10):
    pca = PCA(n_components=dim)
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)

    X_train_red_d, y_train_red_d = dbscan_reduce(Xtr_pca, y_train)

    row = {"Dimensions": dim}

    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        t, acc = evaluate_model(knn, X_train_red_d, y_train_red_d, Xte_pca, y_test, name="skip")
        sp = full_times[f"KNN k={k}"] / t if t > 0 else float("inf")
        row[f"KNN k={k} Acc"]     = round(acc, 4)
        row[f"KNN k={k} Time"]    = round(t, 4)
        row[f"KNN k={k} Speedup"] = round(sp, 2)

    dt = DecisionTreeClassifier(random_state=42)
    t, acc = evaluate_model(dt, X_train_red_d, y_train_red_d, Xte_pca, y_test, name="skip")
    sp = full_times["DT"] / t if t > 0 else float("inf")
    row["DT Acc"]     = round(acc, 4)
    row["DT Time"]    = round(t, 4)
    row["DT Speedup"] = round(sp, 2)

    svm = SVC(kernel="rbf", gamma="scale")
    t, acc, svecs = evaluate_model(svm, X_train_red_d, y_train_red_d, Xte_pca, y_test, name="skip")
    sp = full_times["SVM"] / t if t > 0 else float("inf")
    row["SVM Acc"]        = round(acc, 4)
    row["SVM Time"]       = round(t, 4)
    row["SVM Speedup"]    = round(sp, 2)
    row["Support Vectors"] = svecs

    results.append(row)
    print(f"  dim={dim:3d} done")

print("\n")
print(tabulate(results, headers="keys", tablefmt="grid"))

print("\n===== BEST CONFIGURATIONS =====")

for k in [3, 5, 7]:
    key = f"KNN k={k} Acc"
    best = max(results, key=lambda r: r[key])
    print(f"Best KNN (k={k}): dim={best['Dimensions']}, acc={best[key]:.4f}")

best_dt = max(results, key=lambda r: r["DT Acc"])
print(f"Best Decision Tree: dim={best_dt['Dimensions']}, acc={best_dt['DT Acc']:.4f}")

best_svm = max(results, key=lambda r: r["SVM Acc"])
print(f"Best SVM          : dim={best_svm['Dimensions']}, acc={best_svm['SVM Acc']:.4f}, "
      f"support vectors={best_svm['Support Vectors']}")
