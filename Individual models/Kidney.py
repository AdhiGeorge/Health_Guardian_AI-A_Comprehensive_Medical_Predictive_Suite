import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import pickle

def clean_numeric_column(series):
    """
    Clean numeric column by replacing non-numeric values
    """
    # Replace '?' or '\t?' with NaN
    series = series.replace(['?', '\t?'], np.nan)
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(series, errors='coerce')

def preprocess_data(data):
    """
    Preprocess the data:
    - Handle categorical variables
    - Encode target variable
    - Handle missing values
    """
    # Clean numeric columns
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_columns:
        data[col] = clean_numeric_column(data[col])
    
    # Encode categorical columns
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = data[col].fillna('unknown')  # Fill missing categorical values
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Drop unnecessary columns
    data = data.drop(['id'], axis=1, errors='ignore')
    
    # Fill numeric missing values with median
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    return data, label_encoders

# Load the data
data = pd.read_csv(r'D:\SACAIM\Digital Health Assistant\DHA datasets\kidney_disease.csv')

# Preprocess the data
data, label_encoders = preprocess_data(data)

# Split features and target
X = data.drop('classification', axis=1)
y = data['classification']

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

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
        'pipeline': pipeline,
        'predictions': y_pred
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

# 1. Model Comparison Bar Plot
plt.subplot(132)
accuracies = [model_results[model]['accuracy'] for model in models]
plt.bar(models.keys(), accuracies)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')

# 2. Confusion Matrix for Best Model
plt.subplot(133)
cm = confusion_matrix(y_test, model_results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix\n{best_model_name}')

# 3. Classification Report for Best Model
plt.subplot(131)
print(classification_report(y_test, model_results[best_model_name]['predictions'])) 


plt.tight_layout()
plt.show()

# Feature Importance (for Random Forest)
if best_model_name == 'Random Forest':
    importances = best_pipeline.named_steps['classifier'].feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ROC Curve
plt.figure(figsize=(8,6))
y_pred_proba = best_pipeline.predict_proba(X_test)[:,1]
# Specify pos_label explicitly
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=2)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Prepare for saving
full_pipeline_data = {
    'pipeline': best_pipeline,
    'label_encoders': label_encoders,
    'target_encoder': le_target
}

# Save the best pipeline and encoders
pipeline_filename = 'kidneyDiseasePrediction.sav'
pickle.dump(full_pipeline_data, open(pipeline_filename, 'wb'))
print(f"\nBest Pipeline ({best_model_name}) saved as {pipeline_filename}")