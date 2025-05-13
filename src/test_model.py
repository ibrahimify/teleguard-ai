import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK stopwords if not already present
nltk.download('stopwords', quiet=True)

# Load the Voting Classifier model and vectorizer
try:
    model = joblib.load("models/spam_classifier_voting.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("✅ Models and vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models or vectorizer: {e}")
    exit()

# Text preprocessing
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Predict message using VotingClassifier
def predict_message(message):
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return "Spam" if prediction == 1 else "Ham"

# Sample messages
messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still on for coffee tomorrow?",
    "FREE entry into our £250 weekly competition just text WIN to 80086 NOW.",
    "Can you please send me the project files?",
    "Get 50% discount on all products. Visit our website now!"
]

# Run predictions
print("\n📨 Model Predictions:")
for msg in messages:
    result = predict_message(msg)
    print(f"Message: {msg}\nPrediction: {result}\n")
