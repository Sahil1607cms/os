import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

df = pd.read_csv("wine.csv")   

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size: ", X_test.shape)

plt.figure(figsize=(12, 8))
sns.heatmap(X.corr())
plt.title("Correlation Heatmap of Wine Dataset Features")
plt.show()

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=500),
    "Decision Tree (max_depth=4)":
        DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = []

classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weight ed")
    auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")

    results.append([name, accuracy, precision, recall, f1, auc])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"]
)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison: Logistic Regression vs Decision Tree")
plt.ylim(0, 1)
plt.show()
