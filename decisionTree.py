import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

df.rename(columns={"median_house_value": "MedHouseVal"}, inplace=True)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = ["ocean_proximity"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", DecisionTreeRegressor(max_depth=6, random_state=42))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": preds[:10]
})
print("\nActual vs Predicted (first 10 rows):\n")
print(comparison)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# Scatter plot – Actual vs Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=preds)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Seaborn)")
plt.show()

onehot_features = model.named_steps["preprocess"] \
                        .named_transformers_["cat"] \
                        .named_steps["encoder"] \
                        .get_feature_names_out(categorical_features)

all_features = list(numeric_features) + list(onehot_features)
importances = model.named_steps["regressor"].feature_importances_

feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(5)

# Plot top 5 important features
plt.figure(figsize=(8, 4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, orient="h")
plt.title("Top 5 Feature Importances (Seaborn)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
