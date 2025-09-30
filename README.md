# TradeBot - Render demo (Simulation)

This repo contains a simulated trading bot pre-configured to run on Render as a Web Service.
It uses Binance public klines (1m) to evaluate EMA/RSI/volume signals and simulates spot trades
with an initial virtual balance (default $10).

Endpoints:
- /status -> JSON status (balance, current trade, stats, recent trades)
- /download_trades -> download trades.csv

Deployment (short):
1. Create a new GitHub repo and upload these files.
2. On Render create a **New Web Service**, connect to the repo, branch `main`.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn main:app --bind 0.0.0.0:$PORT`
5. Add Environment Variables in Render dashboard (see .env.example)
6. Create service and open https://<your-service>.onrender.com/status
