import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load Dataset
df = pd.read_csv("seeds.csv")


# Display Dataset
df.head()
df.tail()
df.shape
df.info()
df.describe()


# Check Missing Values
df.isnull().sum()


# Class Distribution
print(df["Class (1, 2, 3)"].value_counts().sort_index())

class_counts = df["Class (1, 2, 3)"].value_counts().sort_index()

plt.figure(figsize=(6, 6))

plt.pie(
    class_counts,
    labels=["Class 1", "Class 2", "Class 3"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Class Distribution")
plt.show()


# Distribution of Input Features
df.drop("Class (1, 2, 3)", axis=1).hist(
    figsize=(14, 10),
    bins=15
)

plt.suptitle("Distribution of Input Features")
plt.tight_layout()
plt.show()


# Boxplot of Features
plt.figure(figsize=(14, 7))

sns.boxplot(data=df.drop("Class (1, 2, 3)", axis=1))

plt.title("Boxplot of Seeds Features")
plt.xticks(rotation=45)
plt.show()


# Individual Feature Boxplots
features = df.drop("Class (1, 2, 3)", axis=1).columns

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[feature])

    plt.title("Boxplot of {feature}")
    plt.xlabel(feature)
    plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 7))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# Scatter Plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Area",
    y="Perimeter",
    hue="Class (1, 2, 3)",
    s=70
)

plt.title("Area vs Perimeter")
plt.show()


# Pair Plot
sns.pairplot(
    df,
    hue="Class (1, 2, 3)",
    diag_kind="hist"
)

plt.show()


# Separate Input and Target
X = df.drop("Class (1, 2, 3)", axis=1)
y_original = df["Class (1, 2, 3)"]

print("Input shape:", X.shape)
print("Target shape:", y_original.shape)


# Encode Classes
y = y_original - 1

print("Original classes:")
print(sorted(y_original.unique()))

print("\nANN encoded classes:")
print(sorted(y.unique()))


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# Display Scaled Data
print("First 5 scaled training samples:")
print(X_train_scaled[:5])


# Build ANN Model
model = Sequential([
    Input(shape=(7,)),

    Dense(16, activation="relu"),

    Dense(8, activation="relu"),

    Dense(3, activation="softmax")
])


# Display Model Summary
model.summary()


# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Early Stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# Train Model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# Training and Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Valiation Accuracy"
)

plt.title("ANN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()


# Training and Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Valiation Loss"
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()


# Evaluate Model
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Prediction Probabilities
y_prob = model.predict(
    X_test_scaled,
    verbose=1
)

print(y_prob[:5])


# Predicted Encoded Classes
y_pred = np.argmax(
    y_prob,
    axis=1
)

print("Predicted encoded classes:")
print(y_pred[:10])


# Convert Back to Original Classes
y_pred_original = y_pred + 1
y_test_original = y_test + 1

print("Predicted classes:")
print(y_pred_original[:10])


# Actual vs Predicted Comparison
comparison = pd.DataFrame({
    "Actual Class": y_test_original.values,
    "Predicted Class": y_pred_original
})

comparison.head(15)


# Accuracy
accuracy = accuracy_score(
    y_test_original,
    y_pred_original
)

print("Accuracy:", accuracy)


# Precision
precision = precision_score(
    y_test_original,
    y_pred_original,
    average="weighted"
)

print("Precision:", precision)


# Recall
recall = recall_score(
    y_test_original,
    y_pred_original,
    average="weighted"
)

print("Recall:", recall)


# F1 Score
f1 = f1_score(
    y_test_original,
    y_pred_original,
    average="weighted"
)

print("F1 Score:", f1)
