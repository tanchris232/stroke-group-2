from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to apply LabelEncoder across multiple columns
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for c in X.columns:
            le = LabelEncoder()
            le.fit(X[c].astype(str))
            self.encoders[c] = le
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for c in X.columns:
            le = self.encoders[c]
            X[c] = le.transform(X[c].astype(str))
        return X

def preprocess_stroke_data(stroke_df):
    """
    Preprocesses stroke data by handling missing values and splitting into features and target.
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
    
    # Define the preprocessor for numerical features
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', MultiColumnLabelEncoder(), categorical_features)],
        remainder='passthrough')
    
    # Define pipelines with the correct preprocessor and model
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))])
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
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

    
    print(f"Random Forest - Accuracy: {accuracy_rf}")
    print(f"Logistic Regression - Accuracy: {accuracy_lr}")
    
    return pipeline_rf, pipeline_lr


if __name__ == "__main__":
    stroke_df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    stroke_model_generator(stroke_df)
