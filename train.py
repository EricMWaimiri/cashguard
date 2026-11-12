import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

orders = pd.read_csv("orders.csv")
X = orders.drop(columns=["is_fraud"])
y = orders["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
print("Fraud model accuracy:", round(accuracy_score(y_test, p > 0.5), 3),
      "| AUC:", round(roc_auc_score(y_test, p), 3))
joblib.dump((model, list(X.columns)), "fraud_model.joblib")
print("Saved fraud_model.joblib")