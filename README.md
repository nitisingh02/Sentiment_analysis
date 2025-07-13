
# 💬 Sentiment Analysis of Amazon Product Reviews Using Naive Bayes

Classify Amazon product reviews as **Positive**, **Neutral**, or **Negative** using **Natural Language Processing** and a **Multinomial Naive Bayes** classifier.

---

## 🔗 Live App

👉 [Click here to try the app](https://sentimentanalysis-nb.streamlit.app/)

---
## 📊 Dataset

- **Name**: Consumer Reviews of Amazon Products (Datafiniti)
- **Format**: CSV (zipped in repo)
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

Columns used:
- `reviews.rating`
- `reviews.text`
---

## 🧽 Preprocessing

- Removed missing entries  
- Converted ratings to sentiment labels (`positive`, `neutral`, `negative`)  
- Balanced the dataset via upsampling  
- Cleaned and stemmed review text  
- Removed stop words  
- Applied **negation handling** (e.g., “not bad”)  
- Extracted features using **TF-IDF** with unigrams, bigrams, trigrams

---

## 🤖 Model

- **Classifier:** Multinomial Naive Bayes  
- **Vectorization:** TF-IDF  
- **Accuracy:** ~69%  
- **Evaluation Metrics:** Accuracy, Precision, Recall, Confusion Matrix  
- Tuned via alpha values and tricky review injections

---

## 🎯 Key Features

- **NLP-based text classification**  
- **Negation-aware sentiment detection**  
- Styled and animated **Streamlit GUI**  
- Custom confidence scores & emoji feedback  
- Handles phrases like *“not bad”, “wasn’t great”* correctly

---

## 🖼️ GUI Preview

Run `app.py` and interact with a styled Streamlit interface featuring:

- 🎨 Animated gradient background  
- ✨ Sentiment predictions with emoji  
- 📈 Confidence scores displayed

---

## 🧩 Project Structure

```
📁 sentiment-analysis-naive-bayes/
├── app.py                # Streamlit frontend
├── nb_model.pkl          # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── 1429_1.csv (or .csv.gz)
├── requirements.txt
├── README.md
└── .gitignore
```
---
 ## 🚀 Getting Started

```bash
git clone https://github.com/your-username/sentiment-analyzer-app.git
cd sentiment-analyzer-app
pip install -r requirements.txt
streamlit run app.py
---

## 📜 License

This project is for academic and learning purposes only.
=======

