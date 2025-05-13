import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources
nltk.download('stopwords')

# Load the Voting Classifier model and TF-IDF vectorizer
model = joblib.load("models/spam_classifier_voting.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Text cleaning function
ps = PorterStemmer()

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters except $
    text = re.sub(r"[^A-Za-z0-9\s$]", '', text)
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [ps.stem(word) for word in words]  # Stemming
    return ' '.join(words)

def predict_message(message):
    cleaned_message = clean_text(message)
    features = vectorizer.transform([cleaned_message]).toarray()
    prediction = model.predict(features)
    return "Spam" if prediction[0] == 1 else "Ham"

# Start Command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Hi! I am TeleGuard AI. Send me a message, and I'll tell you if it's spam or not!")

# Handle incoming messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    prediction = predict_message(user_message)
    response = f"🔍 Analyzing...\n\nMessage: {user_message}\nPrediction: {prediction}"
    await update.message.reply_text(response)

# Telegram bot token
TOKEN = "7302014354:AAFTsMB-M5-plhPamydEoJSKHKQ2K4u1enA"

# Set up the bot
app = ApplicationBuilder().token(TOKEN).build()

# Add handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Start the bot
print("🤖 TeleGuard AI Bot is running...")
app.run_polling()
