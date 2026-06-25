#Titanic Survival Prediction.

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble impoirt RandonForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import consufion_matrix


#Reading the csv file into a variable
titanic = pd.read_csv("./train_and_test2.csv")

#Viewing the 1st five rows
titanic.head()

#Viewing the last 5 rows
titanic.tail()

#Getting more info about the rows
titanic.info()

#viewing the list of columns
titanic.columns

#correcting names and dropping unwanted columns
titanic = titanic.rename(columns = {"2urvived" : "Survived"})

cols_to_drop = [col for col in titanic.columns if "zero" in col]

titanic = titanic.drop(columns = cols_to_drop)

titanic =  titanic.drop(columns = "Passengerid")

#Checking null values
titanic.isnull().sum()

#adding values in place of null values
tianic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])

#Data Visualization
sns.countplot(data = titanic, x = "Survived", Hue = "Sex")
plt.title("Survival count by Sex")
plt.show()

sns.countplot(data = titanic, x = "Survived", hue = "Pclass" )
plt.title("Survival count by Class")
plt.show()

sns.histplot(data = tianic, x = "Age", hue = "Survived", multiple = "stack", palatte = "coolwarm", kde = True)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.plot()

sns.boxplot(data = titanic, x = "Survived", y = "Fare", palette = "Set2")
plt.title("Fare Paid vs Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare Paid")
plt.ylim(0, 300)
plt.show()

#Preparing data for training and testing
X = titanic.drop("Survved", axis = 1)
y = titanic["Survived"]

#Splitting data
X-train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.2, random_state = 42
)

#Creating and training model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

#Making predictions and evaluating performance
y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#Trying a different model and then making predictions and evaluating performance
rf_model = RandomForestClassifier()
rf_model(X_train, y_train)

y_pred_rf = rf_model.predict(y_test, y_pred_rf)

accuracu_rf = accuracy_score(y_test, y_pred_rf)

#Confusion Matrices for both models
confuson_matrix(y_test, y_pred)
confusion_matrix(y_test, y_pred_rf)
