# Fake News Detection using Logistic Regression

## Overview
This project detects whether a news article is Fake or True using Machine Learning.

## Dataset
The project uses the following datasets:

- Fake.csv
- True.csv

These datasets are not included in this repository because of their large size.

Download them and place them in the project folder before running the code.


## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

## How It Works

1. Load Fake.csv and True.csv
2. Add labels to both datasets
3. Combine both datasets
4. Convert text into numerical features using TF-IDF
5. Train Logistic Regression model
6. Test the model accuracy
7. Predict new article examples

## How to Run

Install required packages:

```bash
pip install -r requirements.txt