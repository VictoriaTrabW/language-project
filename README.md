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


By following these steps, the script enables an analysis of the emotions expressed in news headlines from different countries, providing insights into the emotional content of the articles. The generated plots and tables offer visual and tabular representations of the emotion distributions, aiding in the comparative analysis of emotions across countries.
