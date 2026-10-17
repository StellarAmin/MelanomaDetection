import time
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1) LOAD DATA
# ============================================================
print("\n================ LOADING USPS DATASET ================\n")

X_train, y_train = load_svmlight_file("usps_training")
X_test, y_test = load_svmlight_file("usps.t")

X_train = X_train.toarray()
X_test = X_test.toarray()

print("Training Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])
print("Feature Dimension Before PCA:", X_train.shape[1])

# ============================================================
# 2) FEATURE SCALING (VERY IMPORTANT FOR PCA & SVM)
# ============================================================
print("\n================ APPLYING STANDARD SCALING ================\n")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 3) APPLY PCA (100 EIGEN VECTORS)
# ============================================================
print("\n================ APPLYING PCA (100 COMPONENTS) ================\n")

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Feature Dimension After PCA:", X_train_pca.shape[1])

# ============================================================
# 4) CLASSIFIERS ON ORIGINAL DATA
# ============================================================
print("\n================ CLASSIFIERS BEFORE DBSCAN ================\n")

original_times = {}
original_acc = {}

# ---------- KNN ----------
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)

    start = time.time()
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    train_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)

    original_times[f"KNN{k}"] = train_time
    original_acc[f"KNN{k}"] = acc

    print(f"KNN (k={k})")
    print("Training Time:", round(train_time, 4), "sec")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("----------------------------------------")

# ---------- DECISION TREE ----------
dt = DecisionTreeClassifier(random_state=42)

start = time.time()
dt.fit(X_train_pca, y_train)
y_pred = dt.predict(X_test_pca)
dt_time = time.time() - start

dt_acc = accuracy_score(y_test, y_pred)

original_times["DT"] = dt_time
original_acc["DT"] = dt_acc

print("\nDecision Tree")
print("Training Time:", round(dt_time, 4), "sec")
print("Accuracy:", round(dt_acc * 100, 2), "%")

# ---------- SVM ----------
svm = SVC(kernel="rbf", gamma="scale", random_state=42)

start = time.time()
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
svm_time = time.time() - start

svm_acc = accuracy_score(y_test, y_pred)

original_times["SVM"] = svm_time
original_acc["SVM"] = svm_acc

print("\nSupport Vector Machine")
print("Training Time:", round(svm_time, 4), "sec")
print("Accuracy:", round(svm_acc * 100, 2), "%")
print("Number of Support Vectors:", svm.n_support_.sum())

# ============================================================
# 5) DBSCAN BORDER EXTRACTION
# ============================================================
print("\n================ APPLYING DBSCAN FOR DATA REDUCTION ================\n")

X_reduced = []
y_reduced = []

for digit in range(10):
    X_digit = X_train_pca[y_train == digit]

    if X_digit.shape[0] == 0:
        print(f"Skipping digit {digit} (no samples found)")
        continue

    db = DBSCAN(eps=3, min_samples=10)
    labels = db.fit_predict(X_digit)

    core_mask = np.zeros(len(labels), dtype=bool)
    core_mask[db.core_sample_indices_] = True

    border = X_digit[~core_mask]  # ✅ Proper border extraction

    if border.shape[0] == 0:
        border = X_digit[:50]  # Fallback if no border points

    X_reduced.append(border)
    y_reduced.append(np.full(len(border), digit))

X_reduced = np.vstack(X_reduced)
y_reduced = np.concatenate(y_reduced)

original_size = X_train_pca.shape[0]
reduced_size = X_reduced.shape[0]

print("Original Training Size:", original_size)
print("Reduced Training Size :", reduced_size)
print("Reduction Percentage :", round((1 - reduced_size / original_size) * 100, 2), "%")


# ============================================================
# 6) RETRAIN KNN AFTER DBSCAN & SPEED-UP
# ============================================================
print("\n================ SPEED-UP AFTER DBSCAN (KNN k=3) ================\n")

knn = KNeighborsClassifier(n_neighbors=3)

start = time.time()
knn.fit(X_reduced, y_reduced)
knn.predict(X_test_pca)
reduced_time = time.time() - start

speed_up = original_times["KNN3"] / reduced_time

print("Original Training Time:", round(original_times["KNN3"], 4), "sec")
print("Reduced Training Time :", round(reduced_time, 4), "sec")
print("Speed-Up Factor       :", round(speed_up, 2), "x")

# ============================================================
# 7) DIFFERENT PCA DIMENSIONS WITH SVM
# ============================================================
print("\n================ SVM VS PCA DIMENSIONS ================\n")

best_acc = 0
best_dim = 0

for n in range(10, 110, 10):
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_train_pca, y_train)

    acc = svm.score(X_test_pca, y_test)

    print("Eigen Vectors:", n, " -> Accuracy:", round(acc * 100, 2), "%")

    if acc > best_acc:
        best_acc = acc
        best_dim = n

print("\n=========== BEST PCA + SVM RESULT ===========")
print("Best Eigen Vectors:", best_dim)
print("Best Accuracy     :", round(best_acc * 100, 2), "%")

# ============================================================
# 8) FINAL TEXTUAL SUMMARY
# ============================================================
print("\n================ FINAL TEXTUAL ANALYSIS ================\n")

print("1) Best Accuracy Before DBSCAN:")
for model in original_acc:
    print(model, "->", round(original_acc[model] * 100, 2), "%")

print("\n2) Dataset Reduction Using DBSCAN:")
print("Original Size:", original_size)
print("Reduced Size :", reduced_size)

print("\n3) Speed-Up Achieved Using Reduced Dataset:")
print("Speed-Up Factor:", round(speed_up, 2), "times faster")
