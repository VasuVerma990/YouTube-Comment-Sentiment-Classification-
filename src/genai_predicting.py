import pandas as pd
import mlflow
import time

from src.Genai_predictor import ClassifierGenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("genai")

generation_config = {"temperature": 0.5, "max_output_tokens": 10}
model_name = 'genai'

@step
def genai_pred(df: pd.DataFrame, generation_config: dict) -> pd.DataFrame:
    classifier = ClassifierGenAI(api_key,generation_config)
    df = classifier.classifying_text(df,"Comment","predicted_sentiment")
    return df

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model_genai(df: pd.DataFrame) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1_score"]
]:
    """Evaluates sentiment classification accuracy using sklearn metrics."""
    true_labels = df["Sentiment"]
    predicted_labels = df["predicted_sentiment"]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("YoutubeCommentAnalysis")

    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    return accuracy, precision, recall, f1