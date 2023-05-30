# imports
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# initializing a huggingface pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# loading the dataset
df = pd.read_csv('language-project/in/headlines.csv')

# i want to be looking at articles from 2021, as they are the newest and as there are most occurrences of these
df_2021 = df[df['time'].str.startswith('2021')]

# Using function from earlier to classify emotions for the new df_2021 dataset
def classify_emotions(data):
    emotions = []
    for headline in data['headline_no_site']:
        result = emotion_classifier(headline)
        emotion = result[0]['label']
        emotions.append(emotion)
    return emotions

classified_headlines = classify_emotions(df_2021)

# Now i want to compare the classified headlines based on country
# Creating an empty dictionary to store the classified headlines for each country
country_headlines = {}

