# Using Huggingface to investigate emotions in headlines about women in different countries
## Purpose
This is my self-assigned project, which can be seen as a continuation of assignment 4, in that the methods are similar, but the dataset is different, which makes room for using Huggingface for classifying emotions in other ways.

The project aims to analyze the emotions conveyed in news headlines from different countries. It utilizes a pre-trained emotion classification model from the Hugging Face library to classify the emotions associated with each headline. The analysis focuses on articles from the year 2021 as they represent the most recent and contains most articles.

Overall, this project seeks to investigate which emotions are more dominant in headlines speaking of women.

## Steps
This script analyzes the emotions conveyed in news headlines from different countries. It loads a dataset of headlines, filters it for articles from 2021, and classifies the emotions using Huggingface as an emotion classification model. It then creates visualizations of the number of articles per country and the emotion distribution for each country. The script also calculates and displays the normalized emotion distribution in a table. 

## Repository structure
-	In folder: contains the dataset headlines.csv.
-	out folder: Contains the outputs generated during the execution of the scripts. This includes csv files with classified emotions and some visualisations.
-	src folder: Contains the main script to investigate the dataset. This includes the pipeline for the classifier and the visualisation of the results.
-	Setup and reproducibility files:
o	Requirements.txt file: Lists the required programs and packages to run the code. 
o	Can be installed with: pip install -r requirements.txt
-	README.md file: Contains the assignment details, dependencies, additional notes, and reflections on the output. 
## Dependencies and data
The project has been run through UCloud in the Coder Python app (1.78.2), and the neccesary programs are listed in requirements.txt.

The data has been obtained from Kaggle.com at https://www.kaggle.com/datasets/thedevastator/women-in-headlines-bias?select=headlines.csv

The dataset was created by Amber Thomas and contains headlines where women are mentioned in articles from different countries from 2012-2021. The dataset has been downloaded and is in the in folder. I will only be using a subset of the data.

Link for Huggingface documentation: https://huggingface.co/docs/transformers/main_classes/pipelines#natural-language-processing

## Reflections and methods
By following these steps, the script enables an analysis of the emotions expressed in news headlines from different countries, providing insights into the emotional content of the articles. The generated plots and tables offer visual and tabular representations of the emotion distributions, aiding in the comparative analysis of emotions across countries. An important part of the data analysis was realising the differing amount of articles per country which could potentially skewer the impression of the distribution of emotions. 

Generally, the most dominant emotion for the headlines from all the four countries, is neutral. The least occurring classified emotion is surprise. 
For the headlines from India, sadness, joy and anger are most dominant after the neutral classification. However, compared to the other countries, India’s distribution is much more even.
In South Africa, the emotions sadness and anger dominates the classification of the headlines followed by disgust. The distribution here is more uneven with joy and surprise being the least present classifications. In the UK, a lot of the headlines are classified with the emotions sadness and fear, while disgust, anger and joy are more evenly distributed. In the US, the most dominant emotions are sadness and anger, with joy actually being more present in the data than disgust.

To get a greater understanding of why many of the headlines are classified as negative it would be relevant to analyze the dataset differently to explore which topics were discussed in the articles. Perhaps some countries were going through some particular events.
