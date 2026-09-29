import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# Load Dataset
df = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())


# Data Visualization

sns.boxplot(x=df["Glucose"])
plt.title("Glucose Distribution")
plt.show()

sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()


features = df.drop(columns="Outcome").columns

df[features].hist(
    figsize=(14, 12),
    bins=20,
    kde=True
)

plt.suptitle(
    "Distribution of Diabetes Dataset Features",
    fontsize=16
)

plt.tight_layout()
plt.show()


# Identify Invalid Zero Values

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

zero_counts = (df[zero_columns] == 0).sum()

print("\nZero Values:")
print(zero_counts)


# Data Cleaning

df_clean = df.copy()

df_clean[zero_columns] = df_clean[zero_columns].replace(
    0,
    np.nan
)

print("\nMissing Values after replacing invalid zeros:")
print(df_clean.isnull().sum())


# Replace Missing Values with Median

for col in zero_columns:
    df_clean[col] = df_clean[col].fillna(
        df_clean[col].median()
    )

print("\nMissing Values after median replacement:")
print(df_clean.isnull().sum())


# Separate Features and Target

X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]

print("\nX shape:", X.shape)
print("y Shape:", y.shape)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Check Class Distribution

print("\nTraining distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting distribution:")
print(y_test.value_counts(normalize=True))


# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# KNN Classification

k = 5

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_scaled, y_train)


# Prediction

y_pred = knn.predict(X_test_scaled)

print("\nPrediction:")
print(y_pred)


# Prediction Probabilities

y_prob = knn.predict_proba(X_test_scaled)[:, 1]

print("\nPrediction Probabilities:")
print(y_prob)


# Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")


# Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NoDiabetic", "Diabetic"]
    )
)


# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Confusion Matrix Visualization

plt.figure(figsize=(7, 6))

class_names = [
    "NoDiabetic",
    "Diabetic"
]

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()


# Find Best K using Train-Test Accuracy

k_values = range(1, 31)

train_accuracy = []
test_accuracy = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    train_accuracy.append(
        model.score(
            X_train_scaled,
            y_train
        )
    )

    test_accuracy.append(
        model.score(
            X_test_scaled,
            y_test
        )
    )


# Plot Accuracy vs K

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    k_values,
    test_accuracy,
    marker="s",
    label="Testing Accuracy"
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy VS K")

plt.legend()
plt.grid(True)

plt.show()


# Best K based on Test Accuracy

best_k_accuracy = k_values[
    np.argmax(test_accuracy)
]

best_accuracy = max(test_accuracy)

print("\nBest K:", best_k_accuracy)
print("Best Test Accuracy:", best_accuracy)


# Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = []

for k in range(1, 31):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    cv_scores.append(
        scores.mean()
    )


# Cross Validation Plot

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, 31),
    cv_scores,
    marker="o"
)

plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Finding the Best K Value")

plt.grid(True)

plt.show()


# Best K based on Cross Validation

best_k_cv = np.argmax(cv_scores) + 1

print(
    "\nBest K Based on Cross Validation:",
    best_k_cv
)

print(
    "Best CV Accuracy:",
    max(cv_scores)
)


# Grid Search

param_grid = {
    "n_neighbors": list(range(3, 31, 2)),
    "weights": [
        "uniform",
        "distance"
    ],
    "metric": [
        "euclidean",
        "manhattan",
        "minkowski"
    ],
    "p": [
        1,
        2
    ]
}


grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(
    X_train_scaled,
    y_train
)

print("\nGrid Search completed.")


# Best Parameters

print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    "\nBest Cross-Validation Accuracy:",
    grid_search.best_score_
)
