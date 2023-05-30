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

# Iterating over the rows of the df_2021 dataset
for _, row in df_2021.iterrows():
    country = row['country']
    headline = row['headline_no_site']
    result = emotion_classifier(headline)
    emotion = result[0]['label']
    
    # Check if the country is already a key in the dictionary
    if country in country_headlines:
        country_headlines[country].append(emotion)
    else:
        country_headlines[country] = [emotion]

# iterating over the dictionary to investigate which countries are present and how many articles are from each country
for country, headlines in country_headlines.items():
    num_articles = len(headlines)
    print(f"Country: {country} | Number of Articles: {num_articles}")

# creating a plot to visualise the amount of articles for the countries represented in 2021
countries = ['USA', 'India', 'UK', 'South Africa']
num_articles = [7747, 12622, 10539, 2553]

plt.bar(countries, num_articles)
plt.xlabel('Country')
plt.ylabel('Number of Articles')
plt.title('Number of Articles per Country')

# saving figure to out folder
plt.savefig('language-project/out/articles_per_country.png')

# the number of articles is important to remember when comparing the countries, because there is
# an obvious difference, which will be reflected in the number of classified emotions because they are 
# represented differently

# Now i want to compare the classified headlines based on country

# Creating an empty dictionary to store the classified headlines for each country
country_emotions = {}

# Iterate over the rows of df_2021 and store the emotions for each country
index = 0
for _, row in df_2021.iterrows():
    country = row['country']
    emotion = classified_headlines[index]
    index += 1
    if country in country_emotions:
        country_emotions[country].append(emotion)
    else:
        country_emotions[country] = [emotion]

# Visualize the emotion distribution for each country
for country, emotions in country_emotions.items():
    emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
    
    # Create bar plot
    plt.figure()
    sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()))
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title(f'Emotion Distribution for {country} Headlines')
    plt.xticks(rotation=45)
    
    # Save the plot
    plt.savefig(f'language-project/out/emotion_distribution_{country}.png')