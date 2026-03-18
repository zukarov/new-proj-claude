import logging
from dataclasses import dataclass, field
from enum import Enum

from .config import TradingConfig

logger = logging.getLogger(__name__)


class TradingAction(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


class RejectionReason(str, Enum):
    INSUFFICIENT_EDGE = "insufficient_edge"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    MAX_EXPOSURE_EXCEEDED = "max_exposure_exceeded"
    CIRCUIT_BREAKER = "circuit_breaker"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    LOW_CONFIDENCE = "low_confidence"
    SPREAD_TOO_WIDE = "spread_too_wide"


@dataclass
class RiskVerdict:
    approved: bool
    final_size_usdc: float
    final_price: float
    rejection_reason: RejectionReason | None = None
    notes: str = ""


@dataclass
class PortfolioState:
    available_usdc: float
    total_portfolio_value: float
    open_position_count: int
    daily_pnl: float
    peak_portfolio_value: float
    positions: dict = field(default_factory=dict)


class RiskManager:
    def __init__(self, config: TradingConfig) -> None:
        self._config = config

    def _compute_kelly_size(
        self,
        edge: float,
        market_price: float,
        available_usdc: float,
    ) -> float:
        """
        Half-Kelly position sizing capped at config maximums.
        kelly_fraction = edge / (1 - market_price)  [for YES bets]
        """
        if edge <= 0 or market_price >= 1.0:
            return 0.0
        kelly_fraction = edge / (1.0 - market_price)
        half_kelly = kelly_fraction * float(self._config.KELLY_FRACTION)
        capped = min(half_kelly, float(self._config.MAX_SINGLE_POSITION_PCT))
        capped = min(capped, float(self._config.MAX_PORTFOLIO_RISK_PCT))
        return available_usdc * capped

    def check_circuit_breaker(self, state: PortfolioState) -> bool:
        """Return True if circuit breaker should STOP all trading."""
        if state.peak_portfolio_value <= 0:
            return False
        drawdown = (
            (state.peak_portfolio_value - state.total_portfolio_value)
            / state.peak_portfolio_value
        )
        return drawdown >= float(self._config.MAX_DRAWDOWN_PCT)

    def approve_trade(
        self,
        action: TradingAction,
        confidence: float,
        estimated_probability: float,
        market_price: float,
        edge: float,
        state: PortfolioState,
        spread: float | None,
    ) -> RiskVerdict:
        """
        Evaluate a trading decision against all risk rules.
        Returns a RiskVerdict — NEVER raises, always returns a decision.
        """
        # 1. Circuit breaker check
        if self.check_circuit_breaker(state):
            logger.warning("Circuit breaker triggered — halting all trades")
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.CIRCUIT_BREAKER,
                notes=f"Drawdown exceeded {float(self._config.MAX_DRAWDOWN_PCT) * 100:.0f}%",
            )

        # 2. Hold signals pass through immediately
        if action == TradingAction.HOLD:
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                notes="HOLD signal",
            )

        # 3. Minimum edge check
        if abs(edge) < float(self._config.MIN_EDGE_THRESHOLD):
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.INSUFFICIENT_EDGE,
                notes=f"Edge {edge:.3f} < threshold {self._config.MIN_EDGE_THRESHOLD}",
            )

        # 4. Minimum confidence check
        if confidence < 0.55:
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.LOW_CONFIDENCE,
                notes=f"Confidence {confidence:.2f} below 0.55 floor",
            )

        # 5. Spread check (avoid illiquid markets)
        if spread is not None and spread > 0.05:
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.SPREAD_TOO_WIDE,
                notes=f"Spread {spread:.3f} > 0.05 threshold",
            )

        # 6. Open position count
        if state.open_position_count >= self._config.MAX_OPEN_POSITIONS:
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.POSITION_LIMIT_REACHED,
                notes=f"Already at max {self._config.MAX_OPEN_POSITIONS} open positions",
            )

        # 7. Kelly sizing
        edge_for_kelly = abs(edge)
        effective_market_price = (
            market_price if action == TradingAction.BUY_YES else (1.0 - market_price)
        )
        size_usdc = self._compute_kelly_size(
            edge=edge_for_kelly,
            market_price=effective_market_price,
            available_usdc=state.available_usdc,
        )

        if size_usdc < 5.0:  # Polymarket practical minimum ~$5
            return RiskVerdict(
                approved=False,
                final_size_usdc=0.0,
                final_price=market_price,
                rejection_reason=RejectionReason.INSUFFICIENT_FUNDS,
                notes=f"Calculated size ${size_usdc:.2f} below $5 minimum",
            )

        # Determine execution price with small slippage buffer
        slippage = 0.002  # 0.2% buffer
        final_price = (
            min(market_price + slippage, 0.99)
            if action == TradingAction.BUY_YES
            else max(market_price - slippage, 0.01)
        )

        logger.info(
            "Trade approved",
            extra={
                "action": action.value,
                "edge": edge,
                "size_usdc": size_usdc,
                "final_price": final_price,
            },
        )
        return RiskVerdict(
            approved=True,
            final_size_usdc=size_usdc,
            final_price=final_price,
            notes=f"Kelly-sized: edge={edge:.3f}, size=${size_usdc:.2f}",
        )

    def should_stop_loss(self, entry_price: float, current_price: float, side: str) -> bool:
        """Return True if an open position should be exited due to stop-loss."""
        if entry_price <= 0:
            return False
        if side == "BUY_YES":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
        return loss_pct >= float(self._config.STOP_LOSS_PCT)
