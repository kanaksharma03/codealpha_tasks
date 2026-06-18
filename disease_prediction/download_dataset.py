# dataset download
import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

df = pd.read_csv(url)

df.to_csv("diabetes.csv", index=False)

print("Dataset downloaded successfully!")

#load dataset
import pandas as pd

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv("diabetes.csv", names=columns)

print(df.shape)

print(df.info())

print(df.describe())

#features and labels
X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print(X.head())
print(y.head())

#training and testing split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#training of first model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")

#predictions
predictions = model.predict(X_test)

print(predictions[:10])

#accuracy check
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

#confusion matrices
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

#classification
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))

#comparison of models
print("Logistic Regression:", accuracy)
print("Random Forest:", accuracy_score(y_test, rf_predictions))

# feature importance
import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False))

#save the model
import joblib

joblib.dump(rf_model, "diabetes_model.pkl")

print("Model saved!")

#svm model
from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

#prediction of new patient 

new_patient = [[2, 120, 70, 20, 79, 25.0, 0.5, 30]]

prediction = rf_model.predict(new_patient)

if prediction[0] == 1:
    print("Patient is likely Diabetic")
else:
    print("Patient is likely Non-Diabetic")

#probability of new patient

probability = rf_model.predict_proba(new_patient)

print("Probability:", probability)

print("Chance of No Diabetes:", round(probability[0][0] * 100, 2), "%")
print("Chance of Diabetes:", round(probability[0][1] * 100, 2), "%")