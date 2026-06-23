# load dataset
import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

df = pd.read_csv(url)

df.to_csv("credit_data.csv", index=False)

print("Dataset downloaded successfully!")

# load dataset again

df = pd.read_csv("credit_data.csv")

print(df.shape)

print(df.info())

print(df.describe())

print(df.columns)

#separate features and target variable
# Features and Target

X = df.drop("credit_risk", axis=1)

y = df["credit_risk"]

print("features and target variable separated successfully!")
print(X.head())
print(y.head())

#convert categorical variables to numerical 

X = pd.get_dummies(X, drop_first=True)

print(X.head())
print(X.shape)

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#train random forest classifier model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model trained successfully!")

#prediction and probability
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

#feature importance
import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False))

#save model
import joblib
joblib.dump(model, "credit_scoring_model.pkl")
print("Model Saved Successfully!")

#predict a new customer's credit risk
new_customer = [[
    24,      # duration
    5000,    # amount
    2,       # installment_rate
    2,       # present_residence
    35,      # age
    1,       # number_credits
    1,       # people_liable

    # Remaining dummy columns
] ]

#logistic regressison model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

#compare model
print("Random Forest Accuracy:", accuracy)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

#graph
import matplotlib.pyplot as plt

importance.sort_values(by="Importance", ascending=False).head(10).plot(
    x="Feature",
    y="Importance",
    kind="bar"
)

plt.show()