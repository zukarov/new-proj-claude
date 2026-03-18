Show the current state of the trading bot — open positions, daily P&L, and session stats.

Execute: `trading-bot status`

Then read `logs/positions.json` if it exists and summarize:
- Number of open positions and their details (market, action, size, entry price, unrealized P&L)
- Today's realized P&L vs the daily target ($300)
- Progress toward the daily profit target as a percentage
- Any positions approaching the stop-loss threshold (20% down)
- Recommendation: should the user adjust any positions?
