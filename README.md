# 🧠 Sentiment Analysis on Product Reviews

A Machine Learning and Deep Learning project that predicts whether a product review is **Positive** or **Negative** using Natural Language Processing (NLP).

---

## 🚀 Features

- Text preprocessing using NLP techniques
- TF-IDF Vectorization
- Multiple Machine Learning Models:
  - Naive Bayes
  - Logistic Regression
  - LSTM (Deep Learning)
- Model Comparison Visualization
- WordCloud Visualization
- Confusion Matrix
- Interactive Web App using Streamlit

---

## 📊 Model Performance

| Model | Accuracy |
|------|----------|
| Naive Bayes | 88.1% |
| Logistic Regression | 91.9% |
| LSTM | 91.6% |

---

## 🖥️ Web App Demo

Users can enter a product review and instantly receive sentiment prediction.

Example:

Input:
"This product is amazing!"

Output:
Positive 😊

---

## 📂 Project Structure
Sentiment_analysis/

│── app.py
│── sentiment_analysis.py
│── sentiment_model.pkl
│── tfidf_vectorizer.pkl
│── confusion_matrix.png
│── model_comparison.png
│── wordcloud.png
│── requirements.txt
│── README.md


---

## 🛠️ Technologies Used

- Python
- NLP (NLTK)
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Matplotlib
- Seaborn

---

## ▶️ How to Run the Project

### Step 1 — Install dependencies
pip install -r requirements.txt


### Step 2 — Run the Streamlit app


streamlit run app.py


---

## 📌 Dataset Used

Amazon Fine Food Reviews Dataset  
Source: Kaggle

---

## 🎯 Future Improvements

- Deploy to Streamlit Cloud
- Add Neutral Sentiment
- Improve UI design
- Use BERT-based models

---

## 👩‍💻 Author

**Khushi Dogra**  
Aspiring Data Scientist
