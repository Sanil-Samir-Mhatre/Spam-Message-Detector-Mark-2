import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load model and vectorizer
model = joblib.load("spam_detector_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data again to show accuracy
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Preprocess the dataset
df['message'] = df['message'].astype(str).str.lower()
df['message'] = df['message'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
df['tokens'] = df['message'].apply(lambda x: [word for word in x.split() if word not in stop_words])
df['clean_message'] = df['tokens'].apply(lambda x: ' '.join(x))

# Vectorize and evaluate
X = vectorizer.transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})
y_pred = model.predict(X)

print("\nModel Performance on Full Dataset:")
print("Accuracy:", accuracy_score(y, y_pred) * 100, "%")
print("\nClassification Report:\n", classification_report(y, y_pred))

# Function to clean and predict
def clean_and_predict(message):
    message = str(message).lower()
    message = re.sub(r'[^a-z\s]', '', message)
    tokens = [word for word in message.split() if word not in stop_words]
    cleaned = ' '.join(tokens)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return "Spam" if prediction == 1 else "Ham"

# Interactive loop
print("\nSpam Detector is ready.")
print("Type a message to test, or type 'exit' to stop.\n")

while True:
    msg = input("Enter message: ")
    if msg.lower() == "exit":
        print("Exiting Spam Detector.......Thank You.")
        break
    result = clean_and_predict(msg)
    print("Prediction:", result)
    print("-" * 40)
