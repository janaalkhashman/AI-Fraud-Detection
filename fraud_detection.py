import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
data = pd.read_csv("creditcard.csv")
y= data['Class']
X=data.drop(columns=['Class'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 10})
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
y_pred = model.predict(X_test_scaled)
print(pd.Series(y_pred).value_counts())

joblib.dump(model, "fraud_model.joblib")
