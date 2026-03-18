import pytest
from bot.risk_manager import PortfolioState, RejectionReason, RiskManager, TradingAction


def make_state(
    available_usdc: float = 1000.0,
    total_value: float = 1000.0,
    open_positions: int = 0,
    daily_pnl: float = 0.0,
    peak_value: float = 1000.0,
) -> PortfolioState:
    return PortfolioState(
        available_usdc=available_usdc,
        total_portfolio_value=total_value,
        open_position_count=open_positions,
        daily_pnl=daily_pnl,
        peak_portfolio_value=peak_value,
    )


class TestRiskManager:
    def test_hold_is_always_rejected(self, config):
        rm = RiskManager(config)
        state = make_state()
        verdict = rm.approve_trade(
            action=TradingAction.HOLD,
            confidence=0.9,
            estimated_probability=0.7,
            market_price=0.5,
            edge=0.2,
            state=state,
            spread=0.01,
        )
        assert not verdict.approved
        assert verdict.rejection_reason is None
        assert "HOLD" in verdict.notes

    def test_circuit_breaker_triggers_on_large_drawdown(self, config):
        rm = RiskManager(config)
        # Peak $1000, current $800 = 20% drawdown (> 15% threshold)
        state = make_state(available_usdc=800, total_value=800, peak_value=1000)
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.9,
            estimated_probability=0.7,
            market_price=0.5,
            edge=0.2,
            state=state,
            spread=0.01,
        )
        assert not verdict.approved
        assert verdict.rejection_reason == RejectionReason.CIRCUIT_BREAKER

    def test_circuit_breaker_does_not_trigger_on_small_drawdown(self, config):
        rm = RiskManager(config)
        # Peak $1000, current $900 = 10% drawdown (< 15% threshold)
        state = make_state(available_usdc=900, total_value=900, peak_value=1000)
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.9,
            estimated_probability=0.7,
            market_price=0.5,
            edge=0.2,
            state=state,
            spread=0.01,
        )
        # Should not hit circuit breaker (may still hit other rules)
        assert verdict.rejection_reason != RejectionReason.CIRCUIT_BREAKER

    def test_insufficient_edge_rejected(self, config):
        rm = RiskManager(config)
        state = make_state()
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.9,
            estimated_probability=0.55,
            market_price=0.52,
            edge=0.03,  # < 0.05 threshold
            state=state,
            spread=0.01,
        )
        assert not verdict.approved
        assert verdict.rejection_reason == RejectionReason.INSUFFICIENT_EDGE

    def test_low_confidence_rejected(self, config):
        rm = RiskManager(config)
        state = make_state()
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.50,  # < 0.55 threshold
            estimated_probability=0.65,
            market_price=0.50,
            edge=0.15,
            state=state,
            spread=0.01,
        )
        assert not verdict.approved
        assert verdict.rejection_reason == RejectionReason.LOW_CONFIDENCE

    def test_spread_too_wide_rejected(self, config):
        rm = RiskManager(config)
        state = make_state()
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.8,
            estimated_probability=0.65,
            market_price=0.50,
            edge=0.15,
            state=state,
            spread=0.08,  # > 0.05 threshold
        )
        assert not verdict.approved
        assert verdict.rejection_reason == RejectionReason.SPREAD_TOO_WIDE

    def test_position_limit_rejected(self, config):
        rm = RiskManager(config)
        # config has MAX_OPEN_POSITIONS=5
        state = make_state(open_positions=5)
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.8,
            estimated_probability=0.65,
            market_price=0.50,
            edge=0.15,
            state=state,
            spread=0.01,
        )
        assert not verdict.approved
        assert verdict.rejection_reason == RejectionReason.POSITION_LIMIT_REACHED

    def test_valid_trade_approved(self, config):
        rm = RiskManager(config)
        state = make_state(available_usdc=1000.0, total_value=1000.0)
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.80,
            estimated_probability=0.70,
            market_price=0.50,
            edge=0.20,  # Strong edge
            state=state,
            spread=0.01,
        )
        assert verdict.approved
        assert verdict.final_size_usdc > 0
        assert 0.0 < verdict.final_price < 1.0

    def test_kelly_sizing_is_bounded(self, config):
        rm = RiskManager(config)
        state = make_state(available_usdc=10000.0)
        verdict = rm.approve_trade(
            action=TradingAction.BUY_YES,
            confidence=0.95,
            estimated_probability=0.90,
            market_price=0.50,
            edge=0.40,  # Huge edge
            state=state,
            spread=0.01,
        )
        assert verdict.approved
        # Size should be capped at 2% of $10,000 = $200
        assert verdict.final_size_usdc <= 200.0

    def test_stop_loss_triggers(self, config):
        rm = RiskManager(config)
        # Entry $0.60, current $0.40 = 33% loss > 20% threshold
        assert rm.should_stop_loss(entry_price=0.60, current_price=0.40, side="BUY_YES")

    def test_stop_loss_does_not_trigger_small_loss(self, config):
        rm = RiskManager(config)
        # Entry $0.60, current $0.55 = 8% loss < 20% threshold
        assert not rm.should_stop_loss(entry_price=0.60, current_price=0.55, side="BUY_YES")

    def test_stop_loss_buy_no(self, config):
        rm = RiskManager(config)
        # BUY_NO: loss when price goes UP from entry
        # Entry $0.40, current $0.55 = price rose against us
        pct = (0.55 - 0.40) / 0.40  # 37.5% adverse move
        assert rm.should_stop_loss(entry_price=0.40, current_price=0.55, side="BUY_NO")
