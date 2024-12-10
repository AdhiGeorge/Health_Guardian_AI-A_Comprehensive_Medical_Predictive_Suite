import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

def check_missing_values(data):
    """
    Check for missing values in the dataset
    Returns True if missing values exist, False otherwise
    """
    missing_counts = data.isnull().sum()
    return missing_counts[missing_counts > 0].any()

# Load the data
data = pd.read_csv(r'D:\SACAIM\Digital Health Assistant\DHA datasets\diabetes.csv')

# Check and handle missing values
if check_missing_values(data):
    print("Missing values detected in the dataset.")
else:
    print("No missing values found in the dataset.")

# Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Store results
model_results = {}

# Create preprocessing pipeline
def create_pipeline(classifier):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), X.columns)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

# Evaluate models
for name, model in models.items():
    # Create pipeline for each model
    pipeline = create_pipeline(model)
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    model_results[name] = {
        'accuracy': accuracy,
        'pipeline': pipeline
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Select best model
best_model_name = max(model_results, key=lambda k: model_results[k]['accuracy'])
best_pipeline = model_results[best_model_name]['pipeline']
best_accuracy = model_results[best_model_name]['accuracy']

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Visualizations
plt.figure(figsize=(15,5))

# 1. Correlation Heatmap
plt.subplot(131)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Feature Correlation Heatmap')

# 2. Model Comparison
plt.subplot(132)
accuracies = [model_results[model]['accuracy'] for model in models]
plt.bar(models.keys(), accuracies)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')

# Save the best pipeline
pipeline_filename = 'diabetesPrediction.sav'
pickle.dump(best_pipeline, open(pipeline_filename, 'wb'))
print(f"\nBest Pipeline ({best_model_name}) saved as {pipeline_filename}")