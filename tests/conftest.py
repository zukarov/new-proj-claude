import pytest
from bot.config import TradingConfig


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        POLYGON_PRIVATE_KEY="0x0000000000000000000000000000000000000000000000000000000000000001",
        POLYMARKET_FUNDER="0x0000000000000000000000000000000000000000",
        ANTHROPIC_API_KEY="test-key",
        DRY_RUN=True,
        MAX_OPEN_POSITIONS=5,
    )
