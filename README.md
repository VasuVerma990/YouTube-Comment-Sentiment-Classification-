# YouTube Comment Sentiment Classification

## 📌 Project Overview

This project focuses on classifying YouTube comments into three categories: Neutral, Positive, and Negative. The classification is performed using multiple machine learning models and ZenML to orchestrate ML workflows. Additionally, MLflow is used for experiment tracking, model logging, and evaluation metric storage.

## 🚀 Features

* Data Exploration: Exploratory data analysis (EDA) to understand the dataset.
* Training Pipeline:
    * Data Ingestion: Load dataset from Kaggle into a Pandas DataFrame.
    * Data Cleaning: Preprocess and clean YouTube comments.
    * Feature Engineering:
        * Bag of Words (BoW)
        * Word2Vec
        * TF-IDF (Term Frequency-Inverse Document Frequency)
    * Model Training:
        * Logistic Regression
        * Random Forest
        * Support Vector Classifier (SVC)
        * Multinomial Naïve Bayes
        * XGBoost
        * Model training is automatically logged using MLflow Autologging.
    * Model Evaluation:
        * Evaluate model performance on the test set.
        * Metrics are stored in the MLflow artifact store.
    * GenAI for Classification:
        * Option to classify comments using a Generative AI model.
        * Requires an API key stored in the .env file.
    
## 📜 Dataset

* The dataset used in this project is sourced from Kaggle.
* https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset

## 🛠️ Configuration

Modify the config.yaml file to choose the model and feature engineering method.

## 🔑 GenAI Integration

* To classify comments using a Generative AI model, add your API key to .env
* Note: If you use a paid API key, there will be no quota limitations.

## 🔧 Installation & Setup

* 1️⃣ Install dependencies
    pip install -r requirements.txt

* 2️⃣ Set up ZenML
    zenml init

* 3️⃣ Run the training pipeline
    python run_pipeline.py

* 4️⃣ View MLflow logs
    mlflow ui --backend-store-uri {get_tracking_uri()}

## 📌 Future Enhancements

* Deployment (not included in the current scope).
* Hyperparameter tuning for better model performance.
* More deep learning models for improved sentiment classification.