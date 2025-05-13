import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources
nltk.download('stopwords')

# Load both models
try:
    model_lr = joblib.load("models/spam_classifier_lr.pkl")
    model_nb = joblib.load("models/spam_classifier_nb.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("Models and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models or vectorizer: {e}")
    exit()

# Text cleaning function
ps = PorterStemmer()

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = ''.join([char for char in text if not char.isdigit()])  # Remove numbers
    words = text.split()  # Split into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [ps.stem(word) for word in words]  # Stemming
    return ' '.join(words)

def predict_message(message):
    cleaned_message = clean_text(message)
    features = vectorizer.transform([cleaned_message]).toarray()
    prediction_lr = model_lr.predict(features)
    prediction_nb = model_nb.predict(features)
    return ("Spam" if prediction_lr[0] == 1 else "Ham", 
            "Spam" if prediction_nb[0] == 1 else "Ham")

# Sample test cases
messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still on for coffee tomorrow?",
    "FREE entry into our £250 weekly competition just text WIN to 80086 NOW.",
    "Can you please send me the project files?",
    "Get 50% discount on all products. Visit our website now!"
]

print("\nModel Predictions:")
for msg in messages:
    result_lr, result_nb = predict_message(msg)
    print(f"Message: {msg}\nLogistic Regression: {result_lr}\nNaive Bayes: {result_nb}\n")
