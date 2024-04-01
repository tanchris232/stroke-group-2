import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column].astype(str))
            self.encoders[column] = le
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        for column, encoder in self.encoders.items():
            X_copy[column] = encoder.transform(X_copy[column].astype(str))
        return X_copy

def preprocess_data(stroke_df):
    """Preprocesses the given dataframe for binary classification."""
    # Drop 'BMI' column and any rows with missing values
    stroke_df.drop(columns=['bmi'], inplace=True)
    stroke_df.dropna(inplace=True)
    
    # Define features (X) and target (y)
    X = stroke_df.drop(columns=['id', 'stroke'])  # Dropping 'id' as it's not a feature and 'stroke' as it's the target
    y = stroke_df['stroke']  # 'stroke' column is the target
    
    # Select categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Define transformers for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Optionally handles any additional missing values
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Optionally handles any additional missing values
        ('labelencoder', MultiColumnLabelEncoder()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # ColumnTransformer to apply the transformations to the respective column types
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return X, y, preprocessor

def train_and_evaluate_model(data_file):
    # Load and preprocess the dataset
    stroke_df = pd.read_csv(data_file)
    X, y, preprocessor = preprocess_data(stroke_df)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=42, max_iter=1000))  # Increased max_iter for convergence if necessary
    ])
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    predictions = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model_pipeline.predict_proba(X_test)[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC Score: {roc_auc}")

