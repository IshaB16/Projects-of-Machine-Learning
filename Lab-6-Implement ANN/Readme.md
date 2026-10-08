# Lab-6: Perceptron Learning Algorithm

## Aim

To implement the **Perceptron Learning Algorithm** for classification using Python and Machine Learning libraries.

## Objective

* To understand the working of the Perceptron Learning Algorithm.
* To load and preprocess a dataset.
* To train a classification model.
* To evaluate the model using classification metrics.
* To visualize and analyze the model performance.

## Dataset Used

The **Seeds dataset** is used for this experiment.

The dataset contains measurements of different types of seeds. The features used include:

* Area
* Perimeter
* Compactness
* Length of kernel
* Width of kernel
* Asymmetry coefficient
* Length of kernel groove

The target variable is:

* **Class (1, 2, 3)**

The dataset contains **210 records** and three classes.

## Technologies / Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

## Perceptron Learning Algorithm

A **Perceptron** is a simple supervised machine learning algorithm used for classification.

It calculates a weighted sum of the input features:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

The output is then determined using an activation function.

The weights are updated when the predicted output is different from the actual output.

### Weight Update Rule

```text
wᵢ = wᵢ + η(y - ŷ)xᵢ
```

where:

* `wᵢ` = weight
* `η` = learning rate
* `y` = actual output
* `ŷ` = predicted output
* `xᵢ` = input feature

## Procedure

1. Import the required Python libraries.
2. Load the Seeds dataset using Pandas.
3. Display the first and last few records.
4. Separate the input features and target variable.
5. Split the dataset into training and testing sets.
6. Standardize the input features.
7. Train the Perceptron/classification model.
8. Make predictions on the test data.
9. Evaluate the model using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
10. Visualize the model performance.

## Evaluation Metrics

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many of the samples predicted as a class actually belong to that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

It is the harmonic mean of precision and recall.

```text
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions for each class.

## Expected Output

The implementation produces:

* Dataset preview
* Preprocessed data
* Model predictions
* Accuracy score
* Precision score
* Recall score
* F1-score
* Classification report
* Confusion matrix
* Graphical visualizations

## Conclusion

The **Perceptron Learning Algorithm** was implemented for classification using the Seeds dataset. The data was loaded, preprocessed, and divided into training and testing sets. The model was trained and evaluated using different classification metrics. The experiment helped in understanding how a Perceptron performs classification by learning weights from the input data.

## Files

```text
Lab-6/
│
├── B_B2_20_ML_LAB6.ipynb
├── seeds.csv
└── README.md
```

## Author

**Name:** Isha Bhangre
