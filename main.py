import ipdb
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # 1. 下载数据集
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # 2. 数据预处理
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. 设置不同降维维度
    dimensions = [2, 5, 10, 20, 40, 60, 80, 100, 150, 200]
    accuracies = []

    for dim in dimensions:
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Dim: {dim}, Accuracy: {acc:.4f}")

    # 4. 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(dimensions, accuracies, marker="o")
    plt.xlabel("Number of Dimensions after PCA")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs Dimensionality")
    plt.grid(True)
    plt.show()
