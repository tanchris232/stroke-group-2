import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Load the dataset
file_path = '/path/to/your/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Drop the 'id' column as it's not useful for prediction
df.drop('id', axis=1, inplace=True)

# Handle missing values in 'bmi' column
# Assuming missing values are already represented as NaN, otherwise you might need to replace other representations with np.nan
df['bmi'].replace('N/A', np.nan, inplace=True)  # Example if 'N/A' were used for missing values
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')  # Ensure 'bmi' is numeric and coerce any errors into NaN
df.dropna(subset=['bmi'], inplace=True)  # Drop rows with NaN in 'bmi'

# Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical columns (scale them)
numerical_features = ['age', 'avg_glucose_level', 'bmi']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputing just in case
    ('scaler', StandardScaler())])

# Define preprocessing for categorical columns (encode them)
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputing just in case
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
