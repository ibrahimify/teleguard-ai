# TeleGuard AI

Real-time Telegram bot that detects spam with a TF-IDF + Logistic Regression model.

## Tech stack
- Python 3.11, scikit-learn, pandas, numpy
- python-telegram-bot
- Docker (coming soon) · GitHub Actions CI (coming soon)

## Quick start
```bash
git clone https://github.com/<your-username>/teleguard-ai
cd teleguard-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py          # build model  (todo)
python bot.py            # run bot     (todo)
