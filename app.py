import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load trained model
model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

tfidf = pickle.load(
    open("tfidf_vectorizer.pkl", "rb")
)

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ======================
# STREAMLIT UI
# ======================

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠"
)

st.markdown("""
# 🧠 Sentiment Analysis on Product Reviews

### Built using Machine Learning & Deep Learning  
**Models Used:**  
- Naive Bayes  
- Logistic Regression  
- LSTM  

Enter a review below to predict sentiment.
""")

user_input = st.text_area(
    "Enter your review here:"
)

st.caption("Try examples like:")
st.code("This product is amazing and worth every penny")
st.code("Terrible quality, completely useless")

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")

    else:

        cleaned_text = clean_text(user_input)

        vectorized_text = tfidf.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            vectorized_text
        )[0]

        if prediction == 1:

            st.success("Positive Review 😊")

        else:

            st.error("Negative Review 😞")
            
st.info("""
### 📊 Model Performance

- Naive Bayes: **88.1%**
- Logistic Regression: **91.9%**
- LSTM: **91.6%**
""")