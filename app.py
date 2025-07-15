import streamlit as st
import joblib
import re
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load model and vectorizer
model = joblib.load("spam_detector_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# Load full data to calculate performance
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
df["message"] = df["message"].astype(str)
df["message"] = df["message"].str.lower()
df["message"] = df["message"].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
df["clean_message"] = df["message"]
X = vectorizer.transform(df["clean_message"])
y = df["label_num"]
y_pred = model.predict(X)

# Preprocess function
def clean_message(message):
    message = message.lower()
    message = re.sub(r'[^a-z\s]', '', message)
    return message

# Streamlit UI
st.title("Spam Message Detector")

# Model Performance
st.markdown("### Model Performance on Full Dataset:")
st.write(f"**Accuracy:** {accuracy_score(y, y_pred) * 100} %")

report = classification_report(y, y_pred, digits=2, output_dict=False)
st.text("Classification Report:\n" + report)

# Interface
st.markdown("---")
st.markdown("### Spam Detection Interface")
st.markdown("Type a message and press **Detect**. Type `'exit'` to stop.")

# Initialize session state for loop-style input
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter message:")

if st.button("Detect"):
    if user_input.strip().lower() == "exit":
        st.info("Exiting Spam Detector... Thank you!")
    elif user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_message(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        label = "Ham." if prediction == 0 else "Spam!"
        st.session_state.history.append((user_input, label))

# Show previous messages and predictions
if st.session_state.history:
    st.markdown("### Detection Log")
    for i, (msg, result) in enumerate(st.session_state.history[::-1], 1):
        st.markdown(f"**{i}.** `{msg}` → **{result}**")
