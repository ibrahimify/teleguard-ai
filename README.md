# TeleGuard AI

Real-time Telegram bot that detects spam using a TF-IDF + Logistic Regression model.

## Tech Stack
- Python 3.11, scikit-learn, pandas, numpy
- python-telegram-bot
- Docker (coming soon)
- GitHub Actions CI (coming soon)

## Quick Start

```bash
git clone https://github.com/ibrahimify/teleguard-ai
cd teleguard-ai
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py        # Build model (TODO)
python bot.py          # Run bot (TODO)
