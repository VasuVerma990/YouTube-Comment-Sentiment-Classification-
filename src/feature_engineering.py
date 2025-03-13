import yaml
import pandas as pd
import numpy as np
import mlflow

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

# Load configuration
with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)

feature_method = config["feature_engineering"]["method"] 
enable_tracking = config["experiment_tracking"]["enable_tracking"]

class FeatureEngineering:
    def __init__(self):
        self.bow_vectorizer = CountVectorizer(max_features=5000)  
        self.word2vec_model = None
        self.vector_size = 100  # Fixed size for Word2Vec embeddings

    def bow_features(self, texts):
        return self.bow_vectorizer.fit_transform(texts).toarray()

    def train_word2vec(self, texts):
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=self.vector_size, 
                                       window=5, min_count=1, workers=4)

    def get_word2vec_features(self, texts):
        tokenized_texts = [text.split() for text in texts]
        features = []
        
        for words in tokenized_texts:
            word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))  
            else:
                features.append(np.zeros(self.vector_size))  
        
        return np.array(features)  

    def extract_features(self, df, text_column="text", method="bow"):
        if method == "bow":
            df["features"] = list(self.bow_features(df[text_column]))
        elif method == "word2vec":
            self.train_word2vec(df[text_column])  
            df["features"] = list(self.get_word2vec_features(df[text_column]))
        return df


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def word_to_numbers(df: pd.DataFrame, text_column: str, method: str = None) -> pd.DataFrame:
    """
    Converts text to numerical features using either BoW or Word2Vec.

    Args:
        df: Input DataFrame containing text data.
        text_column: The column in df that contains text.
        method: "bow" for Bag of Words, "word2vec" for Word2Vec. If None, uses config.

    Returns:
        df: DataFrame with extracted numerical features.
    """
    if method is None:
        method = feature_method  
    
    fe = FeatureEngineering()  
    df_transformed = fe.extract_features(df.copy(), text_column=text_column, method=method)

    mlflow.log_param("feature_extraction_method", method)

    return df_transformed
