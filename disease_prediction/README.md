# Disease Prediction Using Machine Learning

## Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship. The aim of the project is to predict whether a person is diabetic or non-diabetic based on medical information using machine learning techniques.

The model uses different health-related parameters such as glucose level, blood pressure, BMI, age, and other medical factors to make predictions.

## Dataset

The Pima Indians Diabetes Dataset was used for this project.

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target Variable

Outcome

* 0 = Non-Diabetic
* 1 = Diabetic

## What I Did

* Loaded and explored the dataset
* Checked dataset information and statistics
* Separated features and target variables
* Split the dataset into training and testing sets
* Trained multiple machine learning models
* Evaluated model performance using different metrics
* Compared model accuracies
* Analyzed feature importance
* Saved the trained model using Joblib
* Predicted results for new patient data

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Models Used

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

## Evaluation Techniques

* Accuracy Score
* Confusion Matrix
* Classification Report
* Feature Importance Analysis

## Files

* disease_prediction.py → Main project code
* diabetes_model.pkl → Saved model
* README.md → Project documentation

## Results

The models were successfully trained on the diabetes dataset and their performances were compared. Among the models used, Random Forest provided the best overall performance and was used for making predictions on new patient data.

## Internship

This project was completed as part of the CodeAlpha Machine Learning Internship.