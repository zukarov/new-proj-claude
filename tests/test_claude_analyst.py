import pytest
from unittest.mock import MagicMock, patch
from decimal import Decimal

from bot.claude_analyst import ClaudeAnalyst, TradingDecision
from bot.polymarket_client import MarketSummary, OrderBookSnapshot
from bot.risk_manager import TradingAction


def make_market(question: str = "Will X happen?") -> MarketSummary:
    return MarketSummary(
        condition_id="test_condition_001",
        question=question,
        tokens=[
            {"token_id": "yes_token_001", "outcome": "Yes", "price": 0.55},
            {"token_id": "no_token_001", "outcome": "No", "price": 0.45},
        ],
        volume_24h=50000.0,
        liquidity=25000.0,
        end_date_iso="2026-06-30T00:00:00Z",
        active=True,
    )


def make_order_book(token_id: str = "yes_token_001") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        yes_best_ask=Decimal("0.57"),
        yes_best_bid=Decimal("0.53"),
        midpoint=Decimal("0.55"),
        spread=Decimal("0.04"),
    )


def make_mock_claude_response(action: str = "BUY_YES") -> MagicMock:
    """Build a mock Anthropic API response with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_trading_decision"
    tool_block.input = {
        "action": action,
        "confidence": 0.78,
        "estimated_probability": 0.70,
        "market_price": 0.55,
        "edge": 0.15,
        "reasoning": "The market underprices the probability of this event occurring.",
        "key_factors": ["Strong historical precedent", "Recent news supports YES"],
        "concerns": ["Uncertainty in timing"],
        "suggested_size_pct": 0.05,
        "time_horizon_hours": 72,
    }
    response = MagicMock()
    response.content = [tool_block]
    return response


class TestClaudeAnalyst:
    @pytest.mark.asyncio
    async def test_analyze_market_returns_trading_decision(self, config):
        analyst = ClaudeAnalyst(config)

        with patch.object(analyst._client.messages, "create", return_value=make_mock_claude_response()):
            decision = await analyst.analyze_market(
                market=make_market(),
                order_book=make_order_book(),
                portfolio_context={"available_usdc": 1000.0, "open_positions": 2, "max_positions": 10, "daily_pnl": 0.0},
            )

        assert isinstance(decision, TradingDecision)
        assert decision.action == TradingAction.BUY_YES
        assert decision.confidence == 0.78
        assert decision.edge == 0.15

    @pytest.mark.asyncio
    async def test_analyze_market_hold_decision(self, config):
        analyst = ClaudeAnalyst(config)

        with patch.object(analyst._client.messages, "create", return_value=make_mock_claude_response("HOLD")):
            decision = await analyst.analyze_market(
                market=make_market(),
                order_book=make_order_book(),
                portfolio_context={"available_usdc": 1000.0, "open_positions": 0, "max_positions": 10, "daily_pnl": 0.0},
            )

        assert decision.action == TradingAction.HOLD

    @pytest.mark.asyncio
    async def test_analyze_market_raises_if_no_tool_call(self, config):
        analyst = ClaudeAnalyst(config)

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I think you should buy."
        bad_response = MagicMock()
        bad_response.content = [text_block]

        with patch.object(analyst._client.messages, "create", return_value=bad_response):
            with pytest.raises(ValueError, match="submit_trading_decision"):
                await analyst.analyze_market(
                    market=make_market(),
                    order_book=make_order_book(),
                    portfolio_context={},
                )

    @pytest.mark.asyncio
    async def test_batch_filter_returns_subset(self, config):
        analyst = ClaudeAnalyst(config)
        markets = [make_market(f"Market {i}") for i in range(10)]

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[1, 3, 5, 7]"  # Select markets 0, 2, 4, 6 (0-indexed)
        filter_response = MagicMock()
        filter_response.content = [text_block]

        with patch.object(analyst._client.messages, "create", return_value=filter_response):
            selected = await analyst.batch_filter_markets(markets)

        assert len(selected) == 4
        assert selected[0].question == "Market 0"
        assert selected[1].question == "Market 2"

    @pytest.mark.asyncio
    async def test_batch_filter_handles_invalid_json(self, config):
        analyst = ClaudeAnalyst(config)
        markets = [make_market(f"Market {i}") for i in range(5)]

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'd suggest looking at markets 1 and 3"  # No JSON array
        filter_response = MagicMock()
        filter_response.content = [text_block]

        with patch.object(analyst._client.messages, "create", return_value=filter_response):
            selected = await analyst.batch_filter_markets(markets)

        # Falls back to returning first N markets
        assert isinstance(selected, list)
        assert len(selected) <= len(markets)
