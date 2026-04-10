# Generated from: Train.ipynb
# Converted at: 2026-04-09T10:30:44.522Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_excel("global_inflation_post_covid.csv.xlsx")
df.head()

# Data Understanding

df.info()
df.describe()
df.isnull().sum()

# Data Cleaning

# Remove missing values
df = df.dropna()

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

df.head()

# Feature Selection

# Target variable
y = df['inflation_rate']

# Features
X = df.drop('inflation_rate', axis=1)

print("Features:", X.columns)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (Linear Regression)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Model Evaluation

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Actual vs Predicted Visualization

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Inflation")
plt.ylabel("Predicted Inflation")
plt.title("Actual vs Predicted")
plt.show()

# Save Model

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save Predictions

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("inflation_predictions.csv", index=False)

# Test with New Input

sample = [[1.2, 3.4, 5.6, 2.1, 4.5, 6.7, 2.2, 1.1]]

sample_df = pd.DataFrame(sample, columns=X.columns)

sample_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_scaled)

print("Predicted Inflation:", prediction[0])
print("Predicted Inflation:", round(prediction[0], 2))