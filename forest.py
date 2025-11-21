import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
df = pd.read_csv("heart_disease.csv")   

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

plt.figure(figsize=(10, 7))
sns.heatmap(X_train.corr())
plt.title("Correlation Heatmap (Training Data Only)")
plt.show()

corr = df.corr(numeric_only=True)
top5 = corr["target"].abs().sort_values(ascending=False)[1:6]
print("\nTop 5 important features:\n", top5)

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", RandomForestClassifier())
    ]),
    
    "SVM (RBF)": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", SVC(probability=True))
    ])
}

results = []

for name, pipe in models.items():

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

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
plt.title("Accuracy Comparison: Random Forest vs SVM")
plt.ylim(0, 1)
plt.show()
