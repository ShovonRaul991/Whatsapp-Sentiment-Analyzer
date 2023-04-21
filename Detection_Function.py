import joblib
from sklearn.pipeline import Pipeline
import os

cwd = os.getcwd()
file_path_language = os.path.join(cwd,"Model_Creation/ML_Models/Language_detection_model")
file_path_sentiment = os.path.join(cwd,"Model_Creation/ML_Models/Sentiment_detection_model")
language_dec = joblib.load(open(file_path_language, "rb"))
sentiment_dec = joblib.load(open(file_path_sentiment, 'rb'))

def Detect_The_lang(text):
    text = [text]
    result = language_dec.predict(text)[0]
    return result


def Detect_The_senti(text):
    text = [text]
    result = sentiment_dec.predict(text)[0]
    return result

