from decimal import Decimal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class TradingConfig(BaseSettings):
    # --- Polymarket / Ethereum ---
    # Required for live trading. Must be set in .env when DRY_RUN=false.
    POLYGON_PRIVATE_KEY: str = ""
    POLYMARKET_FUNDER: str = ""  # Your wallet address (holds USDC on Polygon)
    CLOB_HOST: str = "https://clob.polymarket.com"
    CHAIN_ID: int = 137
    SIGNATURE_TYPE: int = 1  # 1 = email/Magic wallet; 0 = EOA

    # Polygon RPC endpoint for querying on-chain USDC balance
    POLYGON_RPC_URL: str = "https://polygon-rpc.com"

    # --- Anthropic ---
    # Required always. Get yours at https://console.anthropic.com/
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
    DRY_RUN: bool = True             # Default SAFE: no real orders sent
    SCAN_INTERVAL_SECONDS: int = 300  # How often to scan markets (seconds)
    MAX_MARKETS_PER_SCAN: int = 20   # Claude analyses per cycle (controls AI cost)
    MIN_MARKET_LIQUIDITY_USDC: float = 1000.0

    # Simulated balance used only in DRY_RUN mode (no real money)
    DRY_RUN_BALANCE_USDC: float = 1000.0

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

    @model_validator(mode="after")
    def validate_live_credentials(self) -> "TradingConfig":
        """
        Fail fast if live mode is enabled but credentials are missing or placeholder.
        In DRY_RUN mode, missing credentials are fine — they are never used.
        """
        if not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
            )

        if not self.DRY_RUN:
            # Private key check
            if not self.POLYGON_PRIVATE_KEY or self.POLYGON_PRIVATE_KEY in ("", "0x"):
                raise ValueError(
                    "POLYGON_PRIVATE_KEY is not set. "
                    "Required for live trading. Add it to your .env file."
                )
            # Wallet address check
            if not self.POLYMARKET_FUNDER or len(self.POLYMARKET_FUNDER) != 42:
                raise ValueError(
                    "POLYMARKET_FUNDER is not a valid Ethereum address (42 chars, 0x-prefixed). "
                    "Set it in your .env file to your Polygon wallet address."
                )

        return self


_config: TradingConfig | None = None


def get_config() -> TradingConfig:
    """Cached singleton — call once at startup, pass around."""
    global _config
    if _config is None:
        _config = TradingConfig()
    return _config
