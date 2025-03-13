import os
import yaml
import logging
import mlflow
import joblib
import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from zenml import step
from zenml.client import Client

with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)
enable_tracking = config["experiment_tracking"]["enable_tracking"]

experiment_tracker = Client().active_stack.experiment_tracker

with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract config settings
model_name = config["model"]["model_name"]


class Accuracy:
    def calculate_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class Precision:
    def calculate_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average="weighted")

class Recall:
    def calculate_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average="weighted")

class F1Score:
    def calculate_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="weighted")




@step(experiment_tracker=experiment_tracker.name, enable_cache= enable_tracking)
def evaluate_model(
    model: ClassifierMixin,
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1_score"]
]:
    """
    Evaluates the classification model and logs metrics to MLflow.

    Args:
        model: Classifier to evaluate.
        model_name: Name of the model.
        x_test: Test data.
        y_test: True labels.

    Returns:
        accuracy: float
        precision: float
        recall: float
        f1_score: float
    """
    model_dir = r"C:\Users\Vasu\Desktop\youtube\models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")

    try:
        # Make predictions
        y_pred = model.predict(x_test)

        # Compute evaluation metrics using defined classes
        accuracy = Accuracy().calculate_score(y_test, y_pred)
        precision = Precision().calculate_score(y_test, y_pred)
        recall = Recall().calculate_score(y_test, y_pred)
        f1_score_value = F1Score().calculate_score(y_test, y_pred)

        # Log metrics in MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score_value)

        # Save model to MLflow
        joblib.dump(model, model_path)
        print("saved")
        return accuracy, precision, recall, f1_score_value

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise e
