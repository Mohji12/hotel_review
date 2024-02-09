import streamlit as st
import pickle 


# Load the model and vectorizer from disk
    

    

def load_tfidf():
    tfidf = pickle.load(open("tf_idf2.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("Naive_Bayes.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Negative" if prediction == 1 else "Positive"
    return class_name

st.header("Hotel Review Classifier App")

st.subheader("Input your text")

text_input = st.text_input("Enter your text")

if text_input is not None:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        if result == 'Negative':
            st.success("The result is "+ result + ":slightly_frowning_face:.")
        else:
            st.success("The result is "+ result + ":smile:.")
