import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Read the dataset
df = pd.read_csv(r"D:\SACAIM\Digital Health Assistant\DHA datasets\Disease_symptom_and_patient_profile_dataset.csv")

# Drop the output variable
df = df.drop('Outcome Variable', axis=1)

# Initialize label encoders for categorical columns
le_disease = LabelEncoder()
le_fever = LabelEncoder() 
le_cough = LabelEncoder()
le_fatigue = LabelEncoder()
le_breathing = LabelEncoder()
le_gender = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()

# Encode categorical columns
df['Disease'] = le_disease.fit_transform(df['Disease'])
df['Fever'] = le_fever.fit_transform(df['Fever'])
df['Cough'] = le_cough.fit_transform(df['Cough'])
df['Fatigue'] = le_fatigue.fit_transform(df['Fatigue'])
df['Difficulty Breathing'] = le_breathing.fit_transform(df['Difficulty Breathing'])
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Blood Pressure'] = le_bp.fit_transform(df['Blood Pressure'])
df['Cholesterol Level'] = le_cholesterol.fit_transform(df['Cholesterol Level'])

# Split features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian NB': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate models
best_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f'\nBest model: {type(best_model).__name__}')
print(f'Best accuracy: {best_accuracy}')

# Create dictionary with model and encoders
model_dict = {
    'model': best_model,
    'disease_encoder': le_disease,
    'fever_encoder': le_fever,
    'cough_encoder': le_cough,
    'fatigue_encoder': le_fatigue,
    'breathing_encoder': le_breathing,
    'gender_encoder': le_gender,
    'bp_encoder': le_bp,
    'cholesterol_encoder': le_cholesterol
}

# Save model and encoders
pickle.dump(model_dict, open('symptomChecker.sav', 'wb'))