# Lab 4 - Decision Tree Classification

## Aim

To implement the Decision Tree Classification algorithm on a dataset using Python and Scikit-learn.

## Objective

The objective of this practical is to:

- Understand the working of Decision Tree Classification.
- Load and explore a classification dataset.
- Split the dataset into training and testing sets.
- Train a Decision Tree Classifier.
- Predict the classes of test data.
- Evaluate the model using accuracy, confusion matrix, and classification report.
- Visualize the trained Decision Tree.

---

## Dataset Used

### Iris Dataset

The Iris dataset is a commonly used classification dataset containing measurements of iris flowers.

It contains four input features:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

The target variable contains three classes:

- Setosa
- Versicolor
- Virginica

The dataset contains 150 observations.

The dataset is available directly through Scikit-learn, so no external CSV file is required.

---

## Technologies and Libraries Used

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## Algorithm Used

### Decision Tree Classification

A Decision Tree is a supervised machine learning algorithm used for classification and regression.

For classification, the algorithm creates a tree-like structure consisting of:

- Root Node
- Internal Nodes
- Branches
- Leaf Nodes

The model used in this practical uses the **Gini Impurity** criterion for selecting the best splits.

The maximum depth of the tree is set to 3 to keep the model simple and interpretable.

---

## Procedure

1. Import the required Python libraries.
2. Load the Iris dataset using Scikit-learn.
3. Separate the features and target variable.
4. Display the first five records.
5. Split the dataset into training and testing data.
6. Create a Decision Tree Classifier.
7. Train the model using the training data.
8. Predict the classes for the testing data.
9. Calculate the accuracy of the model.
10. Generate the confusion matrix.
11. Generate the classification report.
12. Visualize the Decision Tree.

---

## Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The training data is used to train the Decision Tree model, while the testing data is used to evaluate its performance on unseen data.

---

## Model Configuration

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
