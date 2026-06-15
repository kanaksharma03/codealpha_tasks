# Disease Prediction using Machine Learning

## About the Project

This project was developed as part of my CodeAlpha Machine Learning Internship. The aim of the project is to predict whether a person is likely to have diabetes based on different medical attributes.

The project uses machine learning classification algorithms and compares their performance on the diabetes dataset.

## Dataset Used

Pima Indians Diabetes Dataset

The dataset contains information such as:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The target column is **Outcome**, where:

* 0 = No Diabetes
* 1 = Diabetes

## Steps Performed

1. Downloaded and loaded the dataset using Pandas.
2. Explored the dataset using shape, info and describe functions.
3. Separated features and target values.
4. Split the dataset into training and testing data.
5. Trained multiple machine learning models.
6. Evaluated model performance using different metrics.
7. Compared the results of different algorithms.
8. Saved the trained model for future use.
9. Predicted diabetes for a new patient using the trained model.

## Algorithms Used

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)

## Evaluation Methods

* Accuracy Score
* Confusion Matrix
* Classification Report
* Feature Importance

## Files Included

* disease_prediction.py → Main project code
* diabetes_model.pkl → Saved Random Forest model
* README.md → Project documentation

## Output

The models were trained successfully and their accuracies were compared. The trained model can also be used to predict whether a new patient is likely to have diabetes.

## Internship

CodeAlpha Machine Learning Internship

## Author

Kanak Sharma
