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