import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
data = pd.read_csv(r"D:\SACAIM\Digital Health Assistant\DHA datasets\breast-cancer.csv") 

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling with mean for numerical features.")
    data.fillna(data.mean(), inplace=True)

# Convert 'diagnosis' column to numeric (0 for 'B' and 1 for 'M')
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Feature and target separation
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: Accuracy = {acc:.2f}")

# Select the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy = {results[best_model_name]:.2f}")

# Save the pipeline if preprocessing is involved
pipeline = Pipeline([('scaler', scaler), ('model', best_model)])
with open('breastCancerPrediction.sav', 'wb') as f:
    pickle.dump(pipeline, f)

# Visualizations
# Confusion Matrix
y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Benign", "Malignant"], 
            yticklabels=["Benign", "Malignant"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
if hasattr(best_model, "predict_proba"):  # Ensure the model supports predict_proba
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {best_model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
