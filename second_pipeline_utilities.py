from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def preprocess_stroke_data(stroke_df):
    """
    Preprocesses stroke data by handling missing values, encoding categorical variables,
    and splitting into features and target.
    """
    # Handle missing 'bmi' values represented as 'N/A' or NaN
    stroke_df['bmi'].replace('N/A', np.nan, inplace=True)
    stroke_df['bmi'] = pd.to_numeric(stroke_df['bmi'], errors='coerce')
    stroke_df.dropna(subset=['bmi'], inplace=True)
    
    # Define features and target
    X = stroke_df.drop(columns=['id', 'stroke'])
    y = stroke_df['stroke'].values
    
    # Define numerical and categorical features
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                            'work_type', 'Residence_type', 'smoking_status']
    
    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Combine preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def stroke_model_generator(stroke_df):
    """
    Preprocesses the stroke data, trains classification models, and evaluates them.
    """
    X_train, X_test, y_train, y_test = preprocess_stroke_data(stroke_df)
    
    # Define a pipeline with preprocessing and a classifier
    steps_rf = [('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))]
    steps_lr = [('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))]
    
    pipeline_rf = Pipeline(steps_rf)
    pipeline_lr = Pipeline(steps_lr)
    
    # Train the models
    pipeline_rf.fit(X_train, y_train)
    pipeline_lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred_rf = pipeline_rf.predict(X_test)
    y_pred_lr = pipeline_lr.predict(X_test)
    
    # Evaluate the models
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
    
    print(f"Random Forest - Accuracy: {accuracy_rf}, ROC AUC: {roc_auc_rf}")
    print(f"Logistic Regression - Accuracy: {accuracy_lr}, ROC AUC: {roc_auc_lr}")
    
    return pipeline_rf, pipeline_lr

if __name__ == "__main__":
    stroke_df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    stroke_model_generator(stroke_df)
