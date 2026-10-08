Lab-7: Naïve Bayes Classification

Aim

To implement the Naïve Bayes Classification algorithm using a suitable dataset.

Dataset Used

The Raisin Dataset is used for this experiment.

The dataset contains numerical measurements of raisins and a target column named Class. The two raisin classes used in the notebook are:

Kecimen

Besni

The dataset file used by the code is:

Raisin_Dataset.xlsx

Algorithm Used

The notebook implements the Gaussian Naïve Bayes classifier using:

from sklearn.naive_bayes import GaussianNB

Gaussian Naïve Bayes is suitable for numerical features and assumes that the features follow a Gaussian (normal) distribution within each class.

Libraries Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Features Used

The numerical features used for classification are:

Area

MajorAxisLength

MinorAxisLength

Eccentricity

ConvexArea

Extent

Perimeter

The target variable is:

Class

Procedure

Import the required Python libraries.

Load the Raisin dataset from the Excel file.

Inspect the dataset using head(), tail(), shape(), info() and describe().

Check duplicate values, missing values, data types and class distribution.

Visualize the dataset using bar plots, pie charts, histograms, box plots, scatter plots and a correlation heatmap.

Encode the categorical class labels using LabelEncoder.

Separate the input features X and target variable y.

Split the data into training and testing sets using an 80:20 ratio.

Train a GaussianNB model.

Predict the classes for the test data.

Evaluate the model using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Compare training and testing accuracy.

Test different var_smoothing values for the Gaussian Naïve Bayes model.

Enter new raisin measurements manually and predict its class.

Model

The classifier used is:

gnb = GaussianNB()
gnb.fit(X_train, y_train)

The model predicts the class using the probability of each class given the observed feature values.

Evaluation Metrics

Accuracy

Accuracy represents the proportion of correctly classified samples.

Accuracy = Correct Predictions / Total Predictions

Precision

Precision measures how many predicted positive samples are actually positive.

Recall

Recall measures how many actual positive samples are correctly identified.

F1-Score

F1-score combines precision and recall into a single measure.

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each raisin class.

Hyperparameter Tuning

The notebook tests different values of the var_smoothing parameter:

1e-12
1e-10
1e-9
1e-8
1e-7
1e-6
1e-5
1e-4
1e-3
1e-2

The Accuracy, Precision, Recall and F1-Score are calculated for each value.

Manual Prediction

The program also accepts new raisin measurements from the user:

Area

Major Axis Length

Minor Axis Length

Eccentricity

Convex Area

Extent

Perimeter

The trained Gaussian Naïve Bayes model then predicts the raisin class.

Expected Output

The program produces:

Dataset information and statistics

Class distribution visualizations

Feature distributions

Correlation heatmap

Scatter plots

Class encoding

Training and testing sample counts

Predicted labels

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Training and testing accuracy

Hyperparameter comparison results

Prediction for a manually entered raisin sample

Conclusion

The Gaussian Naïve Bayes Classification algorithm was successfully implemented using the Raisin Dataset. The dataset was explored and preprocessed, categorical class labels were encoded, and the data was divided into training and testing sets. The Gaussian Naïve Bayes model was trained and evaluated using different classification metrics. Hyperparameter tuning using var_smoothing was also performed, and the trained model was used to predict the class of a new raisin sample.

Files

Lab-7/
│
├── lab7_naive_bayes.py
├── Raisin_Dataset.xlsx
├── README.md
└── requirements.txt

Author

Name: Isha Bhangre
