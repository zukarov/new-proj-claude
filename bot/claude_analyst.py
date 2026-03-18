import asyncio
import json
import logging
from enum import Enum
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from .config import TradingConfig
from .polymarket_client import MarketSummary, OrderBookSnapshot
from .risk_manager import TradingAction
from .sentiment_client import MarketSentiment

logger = logging.getLogger(__name__)


class TradingDecision(BaseModel):
    action: TradingAction
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_probability: float = Field(ge=0.0, le=1.0)
    market_price: float
    edge: float
    reasoning: str
    key_factors: list[str]
    concerns: list[str]
    suggested_size_pct: float = Field(ge=0.0, le=0.20)
    time_horizon_hours: Optional[int] = None


# Tool schema Claude must call to emit its decision
TRADING_DECISION_TOOL = {
    "name": "submit_trading_decision",
    "description": (
        "Submit your final trading decision for a Polymarket prediction market. "
        "You MUST call this tool to provide a structured decision. "
        "Only recommend BUY_YES or BUY_NO when you have a quantifiable edge "
        "(your estimated probability differs from market price by at least 5 percentage points)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["BUY_YES", "BUY_NO", "HOLD"],
                "description": "Trading action to take",
            },
            "confidence": {
                "type": "number",
                "description": "Your confidence in this decision (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "estimated_probability": {
                "type": "number",
                "description": "Your estimate of the true YES probability (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "market_price": {
                "type": "number",
                "description": "Current market price for YES (0.0-1.0)",
            },
            "edge": {
                "type": "number",
                "description": (
                    "Your edge: estimated_probability - market_price "
                    "(positive = YES underpriced)"
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "1-3 sentence summary of your core reasoning",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-5 bullet points of supporting evidence",
            },
            "concerns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 risks or reasons this analysis could be wrong",
            },
            "suggested_size_pct": {
                "type": "number",
                "description": (
                    "Suggested position size as fraction of bankroll (0.0-0.20). "
                    "Use Half-Kelly sizing."
                ),
                "minimum": 0.0,
                "maximum": 0.20,
            },
            "time_horizon_hours": {
                "type": "integer",
                "description": "Expected time until the market resolves or you would exit (hours)",
                "minimum": 1,
            },
        },
        "required": [
            "action",
            "confidence",
            "estimated_probability",
            "market_price",
            "edge",
            "reasoning",
            "key_factors",
            "concerns",
            "suggested_size_pct",
        ],
    },
}

SYSTEM_PROMPT = """You are a disciplined quantitative analyst specializing in prediction market \
arbitrage on Polymarket.

Your mandate is value betting — you trade ONLY when you have a measurable probabilistic edge \
over the market.

Analytical framework:
1. Estimate the true probability of the YES outcome using all available information.
2. Compare your estimate to the current market price (implied probability).
3. Your edge = your_estimate - market_price.
4. Only recommend BUY_YES if edge > +0.05 (market underprices YES).
5. Only recommend BUY_NO if edge < -0.05 (market overprices YES, so NO is underpriced).
6. Otherwise HOLD — no edge, no trade.

Use Half-Kelly criterion for sizing:
  kelly = edge / (1 - market_price)
  half_kelly = kelly * 0.5
  Cap suggested_size_pct at 0.10 (10% of bankroll maximum).

Be calibrated and skeptical. Distinguish between:
- Information you actually know vs. information you are inferring.
- High-certainty forecasts vs. low-certainty speculation.
- Liquid markets (reliable price signals) vs. illiquid ones (noisy).

Always call the submit_trading_decision tool with your final answer."""


class ClaudeAnalyst:
    def __init__(self, config: TradingConfig) -> None:
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def _build_market_context(
        self,
        market: MarketSummary,
        order_book: OrderBookSnapshot,
        portfolio_context: dict,
        sentiment: Optional[MarketSentiment] = None,
    ) -> str:
        sentiment_section = ""
        if sentiment and sentiment.has_news:
            sentiment_section = f"""
## Recent News & Social Sentiment
Search query used: "{sentiment.query}"

{sentiment.as_text()}

Use these headlines to calibrate your probability estimate. Ask yourself:
- Do the headlines confirm or contradict the market consensus?
- Is the news already priced in, or does it represent new information?
- What is the direction of recent sentiment (positive/negative for YES)?
"""

        return f"""## Market to Analyze

**Question:** {market.question}
**Market closes:** {market.end_date_iso}
**24h Volume:** ${market.volume_24h:,.0f} USDC
**Liquidity:** ${market.liquidity:,.0f} USDC
{sentiment_section}
## Order Book
- YES best ask: {order_book.yes_best_ask}
- YES best bid: {order_book.yes_best_bid}
- Midpoint (implied probability of YES): {order_book.midpoint}
- Bid-ask spread: {order_book.spread}

## Portfolio Context
- Available bankroll: ${portfolio_context.get('available_usdc', 0):,.2f} USDC
- Open positions: {portfolio_context.get('open_positions', 0)}/{portfolio_context.get('max_positions', 10)}
- Today's P&L: ${portfolio_context.get('daily_pnl', 0):+.2f} USDC

Analyze this market and call submit_trading_decision with your verdict.
"""

    async def analyze_market(
        self,
        market: MarketSummary,
        order_book: OrderBookSnapshot,
        portfolio_context: dict,
        sentiment: Optional[MarketSentiment] = None,
    ) -> TradingDecision:
        """
        Deep analysis with adaptive thinking + tool use.
        Optionally includes recent news sentiment in the prompt.
        Returns a fully typed TradingDecision.
        """
        user_content = self._build_market_context(market, order_book, portfolio_context, sentiment)

        def _call_claude() -> TradingDecision:
            response = self._client.messages.create(
                model=self._config.CLAUDE_MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=[TRADING_DECISION_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_content}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_trading_decision":
                    return TradingDecision(**block.input)
            raise ValueError(
                "Claude did not call submit_trading_decision — unexpected response"
            )

        decision = await asyncio.to_thread(_call_claude)
        logger.info(
            "Claude decision: action=%s edge=%.3f confidence=%.2f market=%s",
            decision.action.value,
            decision.edge,
            decision.confidence,
            market.question[:60],
        )
        return decision

    async def batch_filter_markets(
        self,
        markets: list[MarketSummary],
    ) -> list[MarketSummary]:
        """
        Lightweight pass: ask Claude which markets are worth deep analysis.
        No tools, no thinking — fast and cheap cost-control gate.
        """
        summaries = "\n".join(
            f"{i + 1}. [{m.liquidity:,.0f} USDC liq] {m.question} (closes {m.end_date_iso})"
            for i, m in enumerate(markets)
        )
        prompt = (
            "You are a prediction market analyst. From the list below, select the indices "
            "of markets most likely to have mispriced probabilities — based on recency, "
            "complexity, and information asymmetry. Return ONLY a JSON array of integers.\n\n"
            f"{summaries}"
        )

        def _filter_sync() -> list[int]:
            resp = self._client.messages.create(
                model=self._config.CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                return list(range(min(len(markets), 10)))
            indices = json.loads(text[start:end])
            return [i - 1 for i in indices if 1 <= i <= len(markets)]

        indices = await asyncio.to_thread(_filter_sync)
        selected = [markets[i] for i in indices if i < len(markets)]
        logger.info("Quick filter: %d -> %d markets for deep analysis", len(markets), len(selected))
        return selected
