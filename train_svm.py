#train_svm.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("parkinsons.data")
X = df.drop(columns=["name", "status"])
y = df["status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm_model = SVC(probability=True, kernel='rbf', C=1.0)
svm_model.fit(X_scaled, y)

joblib.dump(svm_model, "svm_model.pkl")
