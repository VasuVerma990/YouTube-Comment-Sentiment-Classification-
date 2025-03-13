import yaml
import mlflow
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

from src.ingest_data import ingest_data
from src.cleaning import clean_dataframe
from src.feature_engineering import word_to_numbers
from src.model_training import training_models
from src.evaluation import evaluate_model
from src.genai_predicting import genai_pred, evaluate_model_genai

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
generation_config = {"temperature": 0.5, "max_output_tokens": 10}

# Load configuration
with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)

feature_method = config["feature_engineering"]["method"] 
model_name = config["model"]["model_name"]

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def train_pipeline():
    """
    Training pipeline for classification (argument-free).

    Steps:
        1. Ingest Data
        2. Clean Data
        3. Convert Text Features
        4. Train Model
        5. Evaluate Model

    Returns:
        accuracy: float
        precision: float
        recall: float
        f1_score: float
    """
    df, col_name = ingest_data() 
    df = clean_dataframe(df, col_name)  
    if model_name != 'genai':
        df = word_to_numbers(df, col_name, feature_method) 
        model, X_test, y_test = training_models(df, model_name) 
        accuracy, precision, recall, f1_score = evaluate_model(model, model_name, X_test, y_test)
    else:
        df = genai_pred(df,generation_config)
        accuracy, precision, recall, f1_score = evaluate_model_genai(df)

    return accuracy, precision, recall, f1_score


