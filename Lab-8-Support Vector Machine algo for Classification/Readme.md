Machine Learning Lab 8 — SVM Classification

Aim
To implement Support Vector Machine (SVM) classification on the AI4I 2020 Predictive Maintenance Dataset to predict machine failure.

Dataset
ai4i2020 - ai4i2020.csv contains 10,000 records and 14 columns. The notebook uses six input features:
- Type (categorical: L, M, H)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
The target is Machine failure (0 = no failure, 1 = failure). There are 9,661 non-failure records and 339 failure records, so the target is imbalanced.

Algorithms and Libraries
- SVC with a linear kernel
- SVC with an RBF kernel
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn: train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, evaluation metrics and GridSearchCV
- 
Workflow
1. Load and inspect the dataset.
2. Check shape, data types, missing values and duplicate rows.
3. Explore machine-failure distribution and feature relationships.
4. Split the data into 80% training and 20% testing data using random_state=42 and stratification.
5. Standardize numerical features and one-hot encode Type.
6. Train Linear and RBF SVM pipelines.
7. Evaluate accuracy, precision, recall, F1-score, classification report and confusion matrix.
8. Compare both kernels.
   

With the 80:20 stratified split and random_state=42:
Kernel	Accuracy	Precision	Recall	F1-score
Linear	0.9660	0.0000	0.0000	0.0000
RBF	0.9720	0.8750	0.2059	0.3333


The Linear model predicted only the majority class in this run. The RBF model detected some failure cases but missed others. Since failure cases are rare, accuracy alone is not sufficient; failure-class recall and F1-score matter too.
Hyperparameter Tuning
For a pipeline step named classifier, GridSearchCV parameter names must use double underscores, e.g. classifier__kernel, classifier__C, and classifier__gamma. Update the parameter grid to use these names before running the tuning cell.


Conclusion
This practical compares linear and RBF SVM kernels for machine-failure prediction. The results illustrate the challenge of an imbalanced dataset and the importance of evaluating failure-class recall and F1-score in addition to overall accuracy.
