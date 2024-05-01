import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def transformText(text):
    # Lower
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)

    alnumText = []
    # Removing Special Characters
    for i in text:
        if i.isalnum():
            alnumText.append(i)

    text = alnumText[:]
    alnumText.clear()
    # Removing Stop Words & Punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            alnumText.append(i)

    text = alnumText[:]
    alnumText.clear()
    # Stemming
    ps = PorterStemmer()
    for i in text:
        alnumText.append(ps.stem(i))

    return " ".join(alnumText)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transformedSMS = transformText(input_sms)
    # 2. Vectorize
    vectorInput = tfidf.transform([transformedSMS])
    # 3. Predict
    result = model.predict(vectorInput)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Ham")