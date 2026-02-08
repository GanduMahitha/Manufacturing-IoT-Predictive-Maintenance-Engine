import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/manufacturing_predictive_maintenance_dataset.csv")

X = df.drop("failure", axis=1)
y = df["failure"]

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Baseline Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict_proba(X_test)[:, 1]
print("Logistic Regression PR-AUC:",
      average_precision_score(y_test, lr_pred))

# Baseline Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
print("Random Forest PR-AUC:",
      average_precision_score(y_test, rf_pred))

# Final Model: XGBoost
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_test)[:, 1]

print("XGBoost PR-AUC:",
      average_precision_score(y_test, xgb_pred))

print("\nFinal Model Classification Report:")
print(classification_report(y_test, xgb.predict(X_test)))
