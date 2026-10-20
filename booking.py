import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv("booking.csv")


df = df.drop(columns="Booking_ID", axis=1)

df["date of reservation"] = pd.to_datetime(df["date of reservation"], errors="coerce")
df["reservation_year"] = df["date of reservation"].dt.year
df["reservation_month"] = df["date of reservation"].dt.month
df["reservation_week"] = df["date of reservation"].dt.dayofweek
df = df.drop(columns=["date of reservation"])

df = df.dropna()


df["reservation_year"] = df["reservation_year"].fillna(df["reservation_year"].median())
df["reservation_month"] = df["reservation_month"].fillna(df["reservation_month"].median())
df["reservation_week"] = df["reservation_week"].fillna(df["reservation_week"].median())

df.columns = df.columns.str.strip()


df["total_nights"] = df["number of weekend nights"] + df["number of week nights"]
df["total_family_member"] = df["number of adults"] + df["number of children"]


df["booking status"] = df["booking status"].map(
    {"Not_Canceled": 0, "Canceled": 1}
)

# One-hot encoding for categorical variables
df = pd.get_dummies(
    data=df,
    columns=["type of meal", "room type", "market segment type"],
    drop_first=True,
    dtype=int,
)


X = df.drop("booking status", axis=1)
y = df["booking status"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
}

for name, model in models.items():
  model.fit(x_train_scaled, y_train)
  y_pred = model.predict(x_test_scaled)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(f"\n{name}")
  print(confusion_matrix(y_test, y_pred))


x_train_unscaled = X_train.copy()
x_test_unscaled = X_test.copy()

modelsss = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
}

for name, model in modelsss.items():
  model.fit(x_train_unscaled, y_train)
  y_pred = model.predict(x_test_unscaled)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(f"\n{name}")
  print(confusion_matrix(y_test, y_pred))


ada_model = AdaBoostClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=6,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
)

ada_model.fit(x_train_unscaled, y_train)
xgb_model.fit(x_train_unscaled, y_train)


y_pred_xgb = xgb_model.predict(x_test_unscaled)
print("====== FINAL MODEL PERFORMANCE ======")
print(classification_report(y_test, y_pred_xgb))
print("====== CONFUSION MATRIX ======")
print(confusion_matrix(y_test, y_pred_xgb))


joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
print("Model, scaler, and columns successfully saved!")
