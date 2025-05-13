"""
train_model.py
Loads the Kaggle bank customer churn dataset, preprocesses features, trains a Random Forest model, evaluates it, and saves the model as churn_model.pkl.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (update the path as needed)
df = pd.read_csv('Bank Customer Churn Prediction.csv')

# Features and target
features = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]
target = 'Exited'

# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])

# Feature scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Train/test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
plt.figure(figsize=(8, 5))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=np.array(features)[indices], y=importances[indices])
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save model and preprocessors
joblib.dump({
    'model': model,
    'scaler': scaler,
    'le_gender': le_gender,
    'le_geo': le_geo,
    'features': features
}, 'churn_model.pkl')
print('Model and preprocessors saved to churn_model.pkl')
