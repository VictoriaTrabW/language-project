import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Initializing a Hugging Face pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Loading the dataset
df = pd.read_csv('language-project/in/headlines.csv')

# Filtering articles from 2021
df_2021 = df[df['time'].str.startswith('2021')]

# Function to classify emotions
def classify_emotions(data):
    return [emotion_classifier(headline)[0]['label'] for headline in data['headline_no_site']]

# Classifying emotions for 2021 articles
classified_headlines = classify_emotions(df_2021)

# Creating a dictionary to store the classified headlines for each country
country_headlines = {}

# Iterating over the rows of df_2021 dataset
for _, row in df_2021.iterrows():
    country = row['country']
    emotion = classified_headlines.pop(0)  # Get the next emotion from the list
    country_headlines.setdefault(country, []).append(emotion)

# Investigating the countries and the number of articles for each country
for country, headlines in country_headlines.items():
    num_articles = len(headlines)
    print(f"Country: {country} | Number of Articles: {num_articles}")

# Creating a plot to visualize the number of articles for the countries represented in 2021
countries = list(country_headlines.keys())
num_articles = [len(headlines) for headlines in country_headlines.values()]

plt.bar(countries, num_articles)
plt.xlabel('Country')
plt.ylabel('Number of Articles')
plt.title('Number of Articles per Country')

# Saving the figure
plt.savefig('language-project/out/articles_per_country.png')

# Creating a table to store emotion distributions for all countries
emotion_distributions = {}

# Calculating and visualizing the emotion distribution for each country
for country, emotions in country_headlines.items():
    emotion_counts = pd.Series(emotions).value_counts()
    normalized_emotion_counts = pd.Series(emotions).value_counts(normalize=True)
    
    # Creating a bar plot for the emotion distribution
    plt.figure()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title(f'Emotion Distribution for {country} Headlines')
    plt.xticks(rotation=45)
    
    # Saving the plot
    plt.savefig(f'language-project/out/emotion_distribution_{country}.png')
    
    # Storing the emotion distribution in the table
    emotion_distributions[country] = normalized_emotion_counts

# Creating a table to display the normalized emotion distributions for all countries
emotion_table = pd.DataFrame(emotion_distributions)

# Save the emotion table as a CSV file
emotion_table.to_csv('language-project/out/emotion_table.csv', index=True)