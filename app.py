import streamlit as st
import joblib

# Load saved model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")
st.write("Detect Spam Emails using Machine Learning")

st.sidebar.title("Project Information")
st.sidebar.success("Developer: Aruna Sanda")
st.sidebar.write("Algorithm: Naive Bayes")
st.sidebar.write("Accuracy: 99.19%")

message = st.text_area("Enter your Email or SMS")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        vector = vectorizer.transform([message])

        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚫 Spgit --versionam Email")
        else:
            st.success("✅ Not Spam Email")