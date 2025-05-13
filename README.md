# TeleGuard AI

**TeleGuard AI** is a lightweight and intelligent Telegram bot that classifies incoming messages as **Spam** or **Ham (Not Spam)** using a trained machine learning model.

## Features
- Real-time spam detection using Telegram Bot API
- Preprocessing pipeline: text cleaning, TF-IDF vectorization
- ML models used: Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost
- Trained on a dataset of 5,574 messages with synthetic spam augmentation.  
- Uses TF-IDF vectorization with bigrams and trigrams for better spam detection. 
- VotingClassifier used for improved prediction accuracy
- Easy-to-deploy with virtual environment setup

## Quickstart

```bash
git clone https://github.com/ibrahimify/teleguard-ai.git
cd teleguard-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training the Model

```bash
python src/preprocess.py       # Clean data and vectorize
python src/train_model.py      # Train and save the model
```

## Run the Telegram Bot

```bash
python src/telegram_bot.py
```

Set your Telegram bot token inside `telegram_bot.py` or as an environment variable.

## Project Structure

```
teleguard-ai/
├── data/                    # Input dataset
├── models/                  # Saved models
├── src/                     # Source code
│   ├── preprocess.py
│   ├── train_model.py
│   ├── test_model.py
│   └── telegram_bot.py
├── requirements.txt
└── README.md
```

## Issues  
- Currently, some promotional spam messages are being misclassified as Ham.  
- Planned improvements: Use advanced NLP techniques and better training data augmentation.  

## Contributing  
Feel free to fork this repository and make improvements.