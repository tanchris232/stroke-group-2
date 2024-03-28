import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
# df = pd.read_csv('path_to_your_data.csv')

# Example DataFrame
data = {
    'age': [25, 45, 35, np.nan, 60],
    'hypertension': [0, 1, 0, 0, 1],
    'heart_disease': [0, 0, 0, 1, 1],
    'avg_glucose_level': [80, 200, 150, 140, 180],
    'bmi': [22, np.nan, 28, 27, 30],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'smoking_status': ['never smoked', 'smokes', 'formerly smoked', np.nan, 'never smoked'],
    'stroke': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical columns (scale them)
numerical_features = ['age', 'avg_glucose_level', 'bmi']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Define preprocessing for categorical columns (encode them)
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
