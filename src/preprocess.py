import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
ps = PorterStemmer()

# Load the cleaned data
try:
    data = pd.read_csv("data/cleaned_spam.csv")
    print("Cleaned data loaded successfully!")
except Exception as e:
    print(f"Error loading cleaned data: {e}")
    exit()

# Step 1: Text Cleaning Function
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters except $, %, and !
    text = re.sub(r"[^A-Za-z0-9\s$%!]", '', text)
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into words
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]  # Stemming and remove stopwords
    return ' '.join(words)

# Apply text cleaning
data['message'] = data['message'].apply(clean_text)
print("\nText cleaned!")

# Step 2: Vectorization using TF-IDF with Bigrams and Trigrams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = vectorizer.fit_transform(data['message']).toarray()
y = data['label']

print(f"\nShape of TF-IDF matrix: {X.shape}")

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets!")

# Step 4: Save Preprocessed Data and Vectorizer
joblib.dump((X_train, X_test, y_train, y_test), "data/preprocessed_data.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\nPreprocessed data and vectorizer saved successfully!")
