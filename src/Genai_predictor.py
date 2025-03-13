import google.generativeai as genai
import pandas as pd

class ClassifierGenAI:
    def __init__(self, api_key, generation_config, model_name="gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.generation_config = generation_config
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
        )

    def classifier(self, content):
        prompt = f"Classify the sentiment of this YouTube comment as positive, negative, or neutral only in small letters and give only one words answer:\n\n'{content}'"
        response = self.model.generate_content(prompt)
        
        return response.text.strip() if response and response.text else "Unknown"

    def classifying_text(self, df, text_column, sentiment_predicted):
        df[sentiment_predicted] = df[text_column].apply(self.classifier)
        return df