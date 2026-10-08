import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)

#Display Settings
pd.set_option('display.max_columns', None)

df = pd.read_excel("Raisin_Dataset.xlsx")

df.head()

df.tail()

df.duplicated().sum()

df["Class"].unique()

df.shape

df.info()

df.describe()

df["Class"].value_counts()

df.columns

df.isnull().sum()

df.dtypes

plt.figure(figsize=(7,5))

sns.countplot(data=df,x="Class")

plt.title("Distribution of Raisin Classes")
plt.xlabel("Raisin Class")
plt.ylabel("Number of Samples")

plt.show()

class_counts = df["Class"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(class_counts,labels=class_counts.index,autopct="%1.1f%%", startangle=90)

plt.title("Raisin Class Distribution")

plt.show()

import matplotlib.pyplot as plt

numerical_features = df.select_dtypes(include='number').columns

df[numerical_features].hist(
    figsize=(15, 12),
    bins=20,
    color='yellow'
)

plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))

for i, column in enumerate(numerical_features, 1):

    plt.subplot(3,3,i)

    sns.boxplot(y=df[column],color='red')

    plt.title(column)
plt.tight_layout()
plt.show()

for column in numerical_features:

    plt.figure(figsize=(8,5))

    sns.histplot(data=df,x=column,hue="Class",kde=True,bins=25)

    plt.title(f"{column} Distribution by Class")

    plt.show()

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(10,8))

sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="magma")

plt.title("Correlation Matrix of Raisin Features")

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(data=df,x="Area",y="Perimeter",palette={'Kecimen': 'Green', 'Besni':'Red'}, 
hue="Class",
s=70
)

plt.title("Area vs Perimeter")
plt.show()

plt.figure(figsize=(8,6))

sns.scatterplot(data=df,x="MajorAxisLength",y="MinorAxisLength",hue="Class",s=70)

plt.title("Major Axis Length vs Minor Axis Length")

plt.show()

# data preprocessing
label_encoder = LabelEncoder()

df["Class_encoded"] = label_encoder.fit_transform(df["Class"])

print(df[["Class", "Class_encoded"]].head(10))

print("Class mapping:")

for index,class_name in enumerate(label_encoder.classes_):
    print(index, "=", class_name)

X = df[numerical_features]

y = df["Class_encoded"]

print("X shape:",X.shape)
print("y shape:",y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

print("Training samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])

GaussianNB()

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

print("Predicted labels:")
print(y_pred)

comparison = pd.DataFrame({"Actual":y_test.values,"Predicted":y_pred})

print(comparison.head(5))

comparison["Actual_Class"] = label_encoder.inverse_transform(comparison["Actual"])

comparison["Predicted_Class"] = label_encoder.inverse_transform(comparison["Predicted"])

comparison.head(5)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Accuracy (%) : ", accuracy * 100)

precision = precision_score(y_test,y_pred)
print("Precision:",precision)

recall = recall_score(y_test,y_pred)
print("Recall:",recall)

f1 = f1_score(y_test,y_pred)
print("F1-Score:",f1)

print("==== MODEL PERFORMANCE ====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall  : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Dark2",xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

from sklearn.metrics import accuracy_score

y_train_pred = gnb.predict(X_train)

y_test_pred = gnb.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy :", train_accuracy)
print("Testing Accuracy :", test_accuracy)

GaussianNB(priors=None)

GaussianNB(var_smoothing=1e-9)

print(gnb.get_params())

smoothing_values = [1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
results = []

for value in smoothing_values:
    model = GaussianNB(
        var_smoothing=value
    )
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    acc =  accuracy_score(y_test,prediction)
    prec = precision_score(y_test,prediction)
    rec = recall_score(y_test,prediction)
    f1_value = f1_score(y_test,prediction)
    results.append([value,acc,prec,rec,f1_value])

results_df = pd.DataFrame(

    results,
    columns=["var_smoothing","Accuracy","Precision","Recall","F1-Score"])

results_df

area = float(input("Enter Area: "))
major_axis = float(input("Enter Major Axis Length: "))
minor_axis = float(input("Enter Minor Axis Length: "))
eccentricity = float(input("Enter Eccentricity: "))
convex_area = float(input("Enter Convex Area: "))
extent = float(input("Enter Extent: "))
perimeter = float(input("Enter Perimeter: "))

new_sample = pd.DataFrame({'Area' : [area], 'MajorAxisLength' : [major_axis],
                           'MinorAxisLength':[minor_axis],
                           'Eccentricity':[eccentricity],
                           'ConvexArea':[convex_area],
                           'Extent':[extent],
                           'Perimeter':[perimeter]
                          })

prediction = gnb.predict(new_sample)
predicted_class = label_encoder.inverse_transform(prediction)
print("\nPredicted Raisin Class:",predicted_class[0])
