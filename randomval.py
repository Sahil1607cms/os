import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("breast.csv")
print("Dataset Shape:", df.shape)

feature_names = df.columns[:-1]
target_name = "target"
print("\nFeature Names:")
print(feature_names.tolist())

print("\nTarget Class Distribution:")
print(df["target"].value_counts())

# Correlation
corr_with_target = df.corr()["target"].sort_values(ascending=False)
top5_corr = corr_with_target[1:6]
print("\nTop 5 correlated features with target:")
print(top5_corr)

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr())
plt.title("Correlation Heatmap")
plt.show()

# Introduce missing values
df_missing = df.copy()
np.random.seed(42)

n_missing = int(df_missing.size * 0.025)
rows = np.random.randint(0, df_missing.shape[0], n_missing)
cols = np.random.randint(0, df_missing.shape[1] - 1, n_missing)

for r, c in zip(rows, cols):
    df_missing.iat[r, c] = np.nan

# Split X, y
X = df_missing.drop("target", axis=1)
y = df_missing["target"]

# Impute
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

results = []

# ===================================================
# 🚀 MANUAL SPLIT 1 — 70% TRAIN / 30% TEST
# ===================================================
print("\n========== 70-30 SPLIT ==========")

X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

classes = np.unique(y_train_70)
y_test_30_bin = label_binarize(y_test_30, classes=classes)

# Scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(df["mean radius"], df["mean texture"], c=df["target"])
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Scatter Plot (70-30 split)")
plt.show()

# Box plot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["target"], y=df["mean area"])
plt.xlabel("Target Class")
plt.ylabel("Mean Area")
plt.title("Box Plot (70-30 split)")
plt.show()

# Train models manually
for model_name, model in models.items():
    model.fit(X_train_70, y_train_70)
    preds = model.predict(X_test_30)

    probs = model.predict_proba(X_test_30)[:, 1]

    results.append([
        "70-30",
        model_name,
        accuracy_score(y_test_30, preds),
        precision_score(y_test_30, preds),
        recall_score(y_test_30, preds),
        f1_score(y_test_30, preds),
        roc_auc_score(y_test_30, probs),
    ])

# ===================================================
# 🚀 MANUAL SPLIT 2 — 80% TRAIN / 20% TEST
# ===================================================
print("\n========== 80-20 SPLIT ==========")

X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

classes = np.unique(y_train_80)
y_test_20_bin = label_binarize(y_test_20, classes=classes)

# Scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(df["mean radius"], df["mean texture"], c=df["target"])
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Scatter Plot (80-20 split)")
plt.show()

# Box plot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["target"], y=df["mean area"])
plt.xlabel("Target Class")
plt.ylabel("Mean Area")
plt.title("Box Plot (80-20 split)")
plt.show()

# Train models manually
for model_name, model in models.items():
    model.fit(X_train_80, y_train_80)
    preds = model.predict(X_test_20)

    probs = model.predict_proba(X_test_20)[:, 1]

    results.append([
        "80-20",
        model_name,
        accuracy_score(y_test_20, preds),
        precision_score(y_test_20, preds),
        recall_score(y_test_20, preds),
        f1_score(y_test_20, preds),
        roc_auc_score(y_test_20, probs),
    ])

# Final results table
results_df = pd.DataFrame(
    results,
    columns=["Split", "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
)

print("\nPerformance Results:")
print(results_df)
