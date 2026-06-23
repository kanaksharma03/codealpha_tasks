# Credit Scoring Model

## Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship. The aim of the project is to predict the creditworthiness of a customer using Machine Learning techniques. Based on customer information and financial details, the model classifies whether a customer is a good credit risk or a bad credit risk.

## Dataset

The German Credit Dataset was used for this project. The dataset contains information such as customer status, credit history, loan amount, age, employment details, and credit risk.

## What I Did

* Loaded and explored the dataset
* Separated features and target variables
* Converted categorical data into numerical format using one-hot encoding
* Split the dataset into training and testing sets
* Trained a Random Forest Classifier model
* Trained a Logistic Regression model for comparison
* Evaluated model performance using accuracy score
* Generated a confusion matrix and classification report
* Analyzed feature importance
* Saved the trained model using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Models Used

1. Random Forest Classifier
2. Logistic Regression

## Results

The models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

The Random Forest model performed well in predicting customer credit risk and was saved for future use.

## How to Run

```bash
pip install -r requirements.txt
python creditscoring.py
```

## Internship

This project was completed as part of the CodeAlpha Machine Learning Internship.