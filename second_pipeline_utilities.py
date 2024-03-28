from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
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
    
    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def stroke_model_generator(stroke_df):
    """
    Preprocesses the stroke data, trains classification models, and evaluates them.
    """
    X_train, X_test, y_train, y_test = preprocess_stroke_data(stroke_df)
    
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
            ('cat', OneHotEncoder(), make_column_selector(dtype_include=object))],
        remainder='passthrough')  
    # Include this to avoid dropping columns not specified
    
    # Apply scaling to all features after preprocessing
    all_features_scaler = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('scaler', StandardScaler(with_mean=False))])

    # Define pipelines with the correct preprocessor and model
    pipeline_rf = Pipeline([
        ('all_features_scaler', all_features_scaler),
        ('classifier', RandomForestClassifier(random_state=42))])
    pipeline_lr = Pipeline([
        ('all_features_scaler', all_features_scaler),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
    
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
