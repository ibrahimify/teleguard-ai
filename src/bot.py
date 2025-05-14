import logging
import joblib
import string
import re
import nltk
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

#  NLTK setup
nltk.download('stopwords')

#  Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

#  Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)

#  Load model/vectorizer
try:
    model = joblib.load("models/spam_classifier_voting.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model/vectorizer: {e}")
    exit()

#  Preprocessing
ps = PorterStemmer()
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if not c.isdigit()])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join([ps.stem(w) for w in words])

def predict_message(message):
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(features)[0]
    return "🚫 Spam" if prediction == 1 else "✅ Ham"

#  Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to TeleGuard AI 🤖! Send me a message and I’ll tell you if it’s spam.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    result = predict_message(user_message)
    logging.info(f"User: {update.effective_user.username}, Message: '{user_message}', Prediction: {result}")
    await update.message.reply_text(f"Prediction: {result}")

#  Load .env and token
if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN environment variable not set.")
        exit()

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot is polling...")
    app.run_polling()
