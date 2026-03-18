Run pre-flight connection checks for the trading bot and report results.

Execute: `trading-bot check`

Report the results clearly:
- Which checks passed (Anthropic API, Polymarket CLOB, Polygon RPC, USDC balance, Wallet Auth, News RSS)
- Which checks failed and what the fix is
- Whether the bot is ready to run in live mode
- Current DRY_RUN setting in .env

If any checks fail, provide the exact steps to fix them.
