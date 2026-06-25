# Titanic Survival Prediction 🚢

## Project Overview

This project aims to predict whether a passenger survived the Titanic disaster using machine learning models.

---

## Data Preprocessing

* Cleaned dataset and removed irrelevant columns
* Handled missing values (e.g., Embarked column)
* Selected relevant features such as Age, Fare, Sex, and Pclass

---

## Models Used

Two models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier

---

## Model Evaluation

### Logistic Regression

* Accuracy: ~76.7%
* Strong at predicting **non-survivors**
* Weak at predicting **survivors**

### Random Forest

* Accuracy: ~70.2%
* Better at predicting **survivors**
* Higher false positives for non-survivors

---

## Key Insights

* Logistic Regression performed better overall in terms of accuracy
* The dataset is imbalanced (more non-survivors), which influenced model behavior
* Random Forest provided more balanced predictions but lower overall accuracy
* Features like **Sex and Pclass** likely had strong influence on survival

---

## Conclusion

Logistic Regression was selected as the final model due to its higher accuracy. However, Random Forest demonstrated better performance in identifying survivors, highlighting the trade-off between overall accuracy and class-specific performance.

---

## Future Improvements

* Handle class imbalance more effectively
* Perform hyperparameter tuning
* Try additional models (e.g., Gradient Boosting)
* Deploy the model using a web app (Streamlit)

