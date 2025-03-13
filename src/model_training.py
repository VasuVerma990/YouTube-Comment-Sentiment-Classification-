import yaml
import logging
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb
import shutil  # Required for zipping model before logging

from zenml import step
from zenml.client import Client
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin 


with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_name = config["model"]["model_name"]
fine_tuning = config["model"]["fine_tuning"]
enable_tracking = config["experiment_tracking"]["enable_tracking"]

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=enable_tracking)
def training_models(
    Data: pd.DataFrame, model_name: str  # Fix: Ensure config is passed as a string
) -> Tuple[ClassifierMixin, np.ndarray, np.ndarray]:
    """
    Args:
        Data: pd.DataFrame containing features and sentiment labels.
        config: str representing the model to train.

    Returns:
        Trained model, X_test, y_test, and metadata dictionary.
    """
    try:
        X = np.array(Data["features"].tolist())  
        y = Data["Sentiment"].values

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if mlflow.active_run():
            mlflow.end_run()
            
        mlflow.set_experiment("YouTubeClassifier")
        
        with mlflow.start_run():
            # Select model based on config
            if model_name == "logistic_regression":
                model = LogisticRegression()
                model.fit(X_train,y_train)
                mlflow.sklearn.log_model(model, "model")                
            elif model_name == "random_forest":
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_train,y_train)
                mlflow.sklearn.log_model(model, "model")
            elif model_name == "svc":
                model = SVC(kernel="sigmoid", probability=True)
                model.fit(X_train,y_train)
                mlflow.sklearn.log_model(model, "model")
            elif model_name == "multinomialNB":
                model = MultinomialNB()
                model.fit(X_train,y_train)
                mlflow.sklearn.log_model(model, "model")
            elif model_name == "xgb":
                model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss")
                model.fit(X_train,y_train)
                mlflow.sklearn.log_model(model, "model")
            else:
                raise ValueError("Model name not supported")
            
        return model, X_test, y_test
    
    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise e
