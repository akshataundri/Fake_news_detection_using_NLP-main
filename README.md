# Fake_news_detection_using_NLP

# Fake News Detection Using NLP and Machine Learning

## Overview
This project focuses on detecting fake news using **Natural Language Processing (NLP)** and **Machine Learning**. The model classifies news articles as either **Fake News** or **Factual News** based on textual content.

## Features
- **Data Preprocessing**: Text cleaning, tokenization, lemmatization, and stopword removal.
- **POS Tagging & Named Entity Recognition (NER)**: Extracts linguistic insights.
- **Sentiment Analysis**: Evaluates sentiment polarity of news articles.
- **Topic Modeling**: Uses **Latent Dirichlet Allocation (LDA)** to identify key topics.
- **Feature Extraction**: Uses **TF-IDF Vectorization** for text representation.
- **Model Training**: Logistic Regression classifier.
- **Evaluation Metrics**: Accuracy score, classification report.

## Installation
### Prerequisites
Ensure you have Python installed. You can install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn nltk spacy vaderSentiment gensim scikit-learn
```
Additionally, download NLTK stopwords and Punkt tokenizer:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

## Dataset
The dataset should contain a **CSV file** with at least the following columns:
- `text`: The news article content.
- `fake_or_factual`: Label indicating whether the news is **Fake News** or **Factual News**.

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```
### 2. Run the Code
Execute the script to preprocess data, train the model, and evaluate performance.
```bash
python fake_news_detection.py
```

## Results
- Accuracy Score: **(To be updated after training)**
- Confusion Matrix & Classification Report

## Future Improvements
- Implement **Deep Learning models** (LSTMs, Transformers).
- Fine-tune with **BERT-based models**.
- Improve feature extraction techniques.

## Contributing
Pull requests are welcome! If you have any suggestions, feel free to contribute.

## License
This project is licensed under the **MIT License**.

---
**Author:** akshataudri 


