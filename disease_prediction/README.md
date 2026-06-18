# Disease Prediction using Machine Learning

## Project Overview

This project was developed during my CodeAlpha Machine Learning Internship. The goal of the project is to predict whether a person is likely to have diabetes using medical data and machine learning algorithms.

The dataset contains different health-related attributes such as glucose level, blood pressure, BMI, age, and other medical factors. These attributes are used to train machine learning models and make predictions.

## Dataset

Pima Indians Diabetes Dataset

### Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target Variable

Outcome

- 0 = Non-Diabetic
- 1 = Diabetic

## Work Done

- Loaded and explored the dataset using Pandas
- Checked dataset information and statistics
- Separated features and target values
- Split the dataset into training and testing sets
- Trained multiple machine learning models
- Evaluated model performance using different metrics
- Compared model accuracies
- Saved the trained model using Joblib
- Predicted the result for a new patient

## Algorithms Used

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

## Evaluation Techniques

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance Analysis

## Files

- disease_prediction.py → Main project code
- diabetes_model.pkl → Saved Random Forest model
- README.md → Project documentation

## Results

The models were trained successfully on the diabetes dataset and their performance was compared. Random Forest gave the best overall performance and was also used for predicting a new patient's result.

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Joblib

## Author

Kanak Sharma

CodeAlpha Machine Learning Internship
