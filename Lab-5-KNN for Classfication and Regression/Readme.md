# Lab-5: K-Nearest Neighbor (KNN) for Classification and Regression

## Lab Aim

To implement the **K-Nearest Neighbor (KNN)** algorithm for both **Classification** and **Regression** problems using Python and machine learning libraries.

---

## Objectives

1. To understand the working principle of the K-Nearest Neighbor algorithm.
2. To implement KNN for classification.
3. To implement KNN for regression.
4. To understand the importance of the value of **K**.
5. To evaluate the performance of KNN models using suitable evaluation metrics.
6. To visualize and analyze the obtained results.

---

## Theory

### What is K-Nearest Neighbor (KNN)?

K-Nearest Neighbor (KNN) is a **supervised machine learning algorithm** used for both classification and regression problems.

KNN makes predictions based on the nearest data points in the training dataset.

The basic idea is:

> Similar data points are likely to have similar outputs.

The algorithm calculates the distance between the new data point and existing training data points. It then selects the **K nearest neighbors** and uses them to make the prediction.

---

### KNN for Classification

In classification, KNN assigns a class to a new data point based on the majority class among its K nearest neighbors.

For example, if K = 5 and the five nearest neighbors contain:

- 3 points from Class A
- 2 points from Class B

The new data point will be classified as **Class A**.

---

### KNN for Regression

In regression, KNN predicts a numerical value by calculating the average value of the K nearest neighbors.
