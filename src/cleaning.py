import re
import pandas as pd
import yaml
from zenml import step
from langdetect import detect

with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)
enable_tracking = config["experiment_tracking"]["enable_tracking"]


def clean_text(text: str) -> str:
    """
    Cleans a given text by:
    - Removing special characters, punctuation, and numbers
    - Converting text to lowercase
    - Removing extra spaces

    Parameters:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""  # Return empty string if text is None or not a string

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
@step(enable_cache=enable_tracking)
def clean_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Cleans the specified text column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the text column to clean.

    Returns:
        pd.DataFrame: A DataFrame with the cleaned text.
    """
    df = df.dropna().copy()
    df.drop_duplicates(inplace=True)
    df["language"] = df["Comment"].apply(detect_language)
    df = df[df["language"].isin(["en"])]
    df.drop(columns=["language"], inplace=True)
    df[text_column] = df[text_column].apply(clean_text)  

    return df.iloc[0:50,:]

