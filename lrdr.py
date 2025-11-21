import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

df = pd.read_csv("wine.csv")  
print(df.head())

X = df.drop("target", axis=1)  
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Testing set size :", X_test.shape[0])

plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr())
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = []
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)

lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_precision = precision_score(y_test, lr_preds, average="weighted")
lr_recall = recall_score(y_test, lr_preds, average="weighted")
lr_f1 = f1_score(y_test, lr_preds, average="weighted")
lr_auc = roc_auc_score(y_test_bin, lr_probs, multi_class="ovr", average="weighted")

results.append(["Logistic Regression",
                lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc])

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

dt_preds = dt.predict(X_test)
dt_probs = dt.predict_proba(X_test)

dt_accuracy = accuracy_score(y_test, dt_preds)
dt_precision = precision_score(y_test, dt_preds, average="weighted")
dt_recall = recall_score(y_test, dt_preds, average="weighted")
dt_f1 = f1_score(y_test, dt_preds, average="weighted")
dt_auc = roc_auc_score(y_test_bin, dt_probs, multi_class="ovr", average="weighted")

results.append(["Decision Tree (max_depth=4)",
                dt_accuracy, dt_precision, dt_recall, dt_f1, dt_auc])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC"]
)

print("\nModel Performance:\n")
print(results_df)

plt.figure(figsize=(6, 4))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Accuracy Comparison: Logistic Regression vs Decision Tree")
plt.ylabel("Accuracy")
plt.show()
