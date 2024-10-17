# IMDb Movie Review Sentiment Analysis

This project demonstrates how to perform sentiment analysis on movie reviews using the **IMDb dataset**. The goal is to classify reviews as positive or negative using a machine learning model. The model is built using the **Naive Bayes** algorithm, which is commonly used for text classification tasks.

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)

## Dataset
The dataset used for this project is the **IMDb movie reviews dataset**. It contains a large number of movie reviews labeled as either **positive** or **negative**.

You can get the dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Dataset Features:
- **Review Text**: The textual content of the movie review.
- **Sentiment**: The label for each review, either "positive" or "negative."

## Project Workflow
The notebook was built following these steps:

### 1. Getting the Dataset
The IMDb dataset is loaded and inspected for any missing values or data quality issues.

### 2. Visualizing Data
Basic data visualization techniques are used to understand the distribution of sentiments and word usage in the reviews. Common tools used include word clouds and histograms.

### 3. Data Preprocessing
- **Text Cleaning**: Lowercasing, removing special characters, and stop words.
- **Tokenization**: Splitting the text into individual words or tokens.
- **Vectorization**: Converting the text data into numerical features using techniques such as **TF-IDF** (Term Frequency-Inverse Document Frequency) to make it suitable for the Naive Bayes model.

### 4. Train-Test Split
The dataset is split into training and testing sets to evaluate the model's performance. Typically, a 80-20 split is used, but this can be adjusted as needed.

### 5. Build the Model with Naive Bayes
The **Naive Bayes** algorithm, particularly the **Multinomial Naive Bayes** variant, is used to build the sentiment classification model. This model works well for text data, particularly when the features are frequencies of words or TF-IDF scores.

### 6. Model Evaluation
The model is evaluated using several key metrics:
- **Accuracy**: The percentage of correctly classified reviews.
- **Precision, Recall, and F1-Score**: To assess the balance between positive and negative sentiment prediction.
- **Confusion Matrix**: To visualize the number of correct and incorrect predictions.

### 7. Testing Our Model
The trained model is tested on a separate test set to evaluate its real-world performance. Sample predictions are displayed to demonstrate how the model classifies new reviews.

## Installation

To run this project, you will need the following Python libraries:
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **NLTK** (for natural language processing)
- **Scikit-learn** (for machine learning)
  
You can install the required dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```
## Usage
1. Clone the repository:
```bash
Copy code
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```
2. Run the notebook: Open the notebook file imdb_sentiment_analysis.ipynb in Jupyter and run the cells step by step to follow the entire analysis.

3. Customize the model: You can modify the code to try different machine learning models or experiment with text preprocessing methods.

## Model Evaluation
The following metrics are used to evaluate the performance of the Naive Bayes model:
- Accuracy: The overall correctness of the model’s predictions.
- Precision: The percentage of correct positive predictions.
- Recall: The percentage of actual positives that were correctly identified.
- F1-Score: The harmonic mean of precision and recall.
- Confusion Matrix: A table showing the number of correct and incorrect predictions.
