import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

data = pd.read_csv("data/housing.csv")

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model,"models/model.pkl")

print("Training completed")