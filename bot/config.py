from decimal import Decimal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class TradingConfig(BaseSettings):
    # --- Polymarket / Ethereum ---
    POLYGON_PRIVATE_KEY: str = "0x0000000000000000000000000000000000000000000000000000000000000001"
    POLYMARKET_FUNDER: str = "0x0000000000000000000000000000000000000000"
    CLOB_HOST: str = "https://clob.polymarket.com"
    CHAIN_ID: int = 137
    SIGNATURE_TYPE: int = 1  # 1 = email/Magic wallet

    # --- Anthropic ---
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # --- Risk Parameters ---
    MAX_PORTFOLIO_RISK_PCT: Decimal = Decimal("0.02")    # 2% per trade max
    MAX_SINGLE_POSITION_PCT: Decimal = Decimal("0.10")  # 10% of bankroll cap
    MIN_EDGE_THRESHOLD: Decimal = Decimal("0.05")       # 5 pp minimum edge
    KELLY_FRACTION: Decimal = Decimal("0.5")            # Half-Kelly
    MAX_OPEN_POSITIONS: int = 10
    STOP_LOSS_PCT: Decimal = Decimal("0.20")            # Exit if down 20%
    MAX_DRAWDOWN_PCT: Decimal = Decimal("0.15")         # Circuit breaker

    # --- Bot Behavior ---
    DRY_RUN: bool = True             # Default SAFE: no real orders
    SCAN_INTERVAL_SECONDS: int = 300  # How often to fetch markets
    MAX_MARKETS_PER_SCAN: int = 20   # Claude analyses per cycle (cost control)
    MIN_MARKET_LIQUIDITY_USDC: float = 1000.0

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_bot.jsonl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("MIN_EDGE_THRESHOLD")
    @classmethod
    def edge_must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("MIN_EDGE_THRESHOLD must be positive")
        return v

    @field_validator("KELLY_FRACTION")
    @classmethod
    def kelly_must_be_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") < v <= Decimal("1")):
            raise ValueError("KELLY_FRACTION must be between 0 and 1")
        return v


_config: TradingConfig | None = None


def get_config() -> TradingConfig:
    """Cached singleton — call once at startup, pass around."""
    global _config
    if _config is None:
        _config = TradingConfig()
    return _config
