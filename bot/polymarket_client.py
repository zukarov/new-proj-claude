import asyncio
import logging
from decimal import Decimal
from typing import Optional

import aiohttp
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import TradingConfig

# USDC.e (bridged USDC) on Polygon — the collateral token used by Polymarket
_USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_USDC_DECIMALS = 6
# ERC20 balanceOf(address) selector
_BALANCE_OF_SELECTOR = "0x70a08231"

logger = logging.getLogger(__name__)


class MarketSummary(BaseModel):
    condition_id: str
    question: str
    tokens: list[dict]
    volume_24h: float
    liquidity: float
    end_date_iso: str
    active: bool


class OrderBookSnapshot(BaseModel):
    token_id: str
    yes_best_ask: Optional[Decimal] = None
    yes_best_bid: Optional[Decimal] = None
    midpoint: Optional[Decimal] = None
    spread: Optional[Decimal] = None


class TradeResult(BaseModel):
    order_id: str
    status: str
    token_id: str
    side: str
    size: float
    price: float
    dry_run: bool


class PolymarketClient:
    def __init__(self, config: TradingConfig) -> None:
        self._config = config
        self._client = None

    async def initialize(self) -> None:
        """Set up client and derive API credentials. Call once at startup."""
        if self._config.DRY_RUN:
            logger.info("PolymarketClient: DRY RUN mode — skipping live auth")
            return

        def _init_sync():
            from py_clob_client.client import ClobClient

            client = ClobClient(
                self._config.CLOB_HOST,
                key=self._config.POLYGON_PRIVATE_KEY,
                chain_id=self._config.CHAIN_ID,
                signature_type=self._config.SIGNATURE_TYPE,
                funder=self._config.POLYMARKET_FUNDER,
            )
            client.set_api_creds(client.create_or_derive_api_creds())
            return client

        self._client = await asyncio.to_thread(_init_sync)
        logger.info("PolymarketClient initialized (live mode)")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_active_markets(
        self,
        limit: int = 20,
        min_liquidity: float = 1000.0,
    ) -> list[MarketSummary]:
        """Fetch and filter active markets by liquidity."""
        if self._config.DRY_RUN or self._client is None:
            return self._mock_markets(limit)

        raw = await asyncio.to_thread(self._client.get_simplified_markets)
        markets = []
        for m in raw.get("data", []):
            if not m.get("active"):
                continue
            if float(m.get("liquidity", 0)) < min_liquidity:
                continue
            markets.append(
                MarketSummary(
                    condition_id=m["condition_id"],
                    question=m["question"],
                    tokens=m.get("tokens", []),
                    volume_24h=float(m.get("volume24hr", 0)),
                    liquidity=float(m.get("liquidity", 0)),
                    end_date_iso=m.get("end_date_iso", ""),
                    active=True,
                )
            )
        return markets[:limit]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_order_book_snapshot(self, token_id: str) -> OrderBookSnapshot:
        """Get best bid/ask and midpoint for a given token."""
        if self._config.DRY_RUN or self._client is None:
            return self._mock_order_book(token_id)

        book = await asyncio.to_thread(self._client.get_order_book, token_id)
        bids = sorted(book.bids, key=lambda x: float(x.price), reverse=True)
        asks = sorted(book.asks, key=lambda x: float(x.price))

        best_bid = Decimal(str(bids[0].price)) if bids else None
        best_ask = Decimal(str(asks[0].price)) if asks else None
        midpoint = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        return OrderBookSnapshot(
            token_id=token_id,
            yes_best_ask=best_ask,
            yes_best_bid=best_bid,
            midpoint=midpoint,
            spread=spread,
        )

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> TradeResult:
        """Place a GTC limit order. In dry-run, returns a simulated result."""
        if self._config.DRY_RUN or self._client is None:
            logger.info(
                f"[DRY RUN] Would place {side} order: {size:.4f} shares @ {price:.4f} "
                f"on token {token_id[:12]}..."
            )
            return TradeResult(
                order_id=f"DRY-{token_id[:8]}",
                status="dry_run",
                token_id=token_id,
                side=side,
                size=size,
                price=price,
                dry_run=True,
            )

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        clob_side = BUY if side == "BUY" else SELL

        def _place_sync() -> dict:
            order_args = OrderArgs(price=price, size=size, side=clob_side, token_id=token_id)
            signed = self._client.create_order(order_args)
            return self._client.post_order(signed, OrderType.GTC)

        resp = await asyncio.to_thread(_place_sync)
        return TradeResult(
            order_id=resp.get("orderID", "unknown"),
            status=resp.get("status", "unknown"),
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            dry_run=False,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if self._config.DRY_RUN or self._client is None:
            return True
        result = await asyncio.to_thread(self._client.cancel, order_id)
        return result.get("canceled", False)

    async def get_open_orders(self) -> list[dict]:
        if self._config.DRY_RUN or self._client is None:
            return []
        return await asyncio.to_thread(
            lambda: self._client.get_orders().get("data", [])
        )

    async def get_usdc_balance(self) -> float:
        """
        Return the USDC.e balance (in dollars) of the funder wallet.

        In DRY_RUN mode, returns the configured DRY_RUN_BALANCE_USDC value.
        In live mode, queries the Polygon RPC directly via eth_call on the
        USDC.e contract — no extra dependency needed beyond aiohttp.
        """
        if self._config.DRY_RUN:
            return self._config.DRY_RUN_BALANCE_USDC

        # Encode balanceOf(address) call data
        # Pad the wallet address to 32 bytes (strip 0x prefix, left-pad with zeros)
        wallet = self._config.POLYMARKET_FUNDER.lower().removeprefix("0x")
        call_data = _BALANCE_OF_SELECTOR + wallet.zfill(64)

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {"to": _USDC_CONTRACT, "data": call_data},
                "latest",
            ],
            "id": 1,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.POLYGON_RPC_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
            raw_hex = data.get("result", "0x0")
            raw_int = int(raw_hex, 16)
            balance = raw_int / (10 ** _USDC_DECIMALS)
            logger.info("USDC balance fetched", balance_usdc=balance, wallet=self._config.POLYMARKET_FUNDER)
            return balance
        except Exception as exc:
            logger.error("Failed to fetch USDC balance", error=str(exc))
            return 0.0

    # --- Mock data for dry-run mode ---

    def _mock_markets(self, limit: int) -> list[MarketSummary]:
        import random

        questions = [
            "Will the Fed cut rates in Q1 2026?",
            "Will Bitcoin reach $120,000 before April 2026?",
            "Will SpaceX Starship complete an orbital flight in 2026?",
            "Will the US unemployment rate exceed 5% in 2026?",
            "Will OpenAI release GPT-5 before June 2026?",
            "Will the S&P 500 close above 6000 by end of Q1 2026?",
            "Will there be a US recession declared in 2026?",
            "Will Apple release AR glasses in 2026?",
            "Will the Euro drop below 1.0 USD in Q1 2026?",
            "Will a major AI safety incident occur in 2026?",
        ]
        markets = []
        for i, q in enumerate(questions[:limit]):
            price = round(random.uniform(0.15, 0.85), 2)
            markets.append(
                MarketSummary(
                    condition_id=f"mock_condition_{i:04d}",
                    question=q,
                    tokens=[
                        {"token_id": f"mock_yes_{i:04d}", "outcome": "Yes", "price": price},
                        {"token_id": f"mock_no_{i:04d}", "outcome": "No", "price": round(1.0 - price, 2)},
                    ],
                    volume_24h=round(random.uniform(5000, 200000), 0),
                    liquidity=round(random.uniform(2000, 500000), 0),
                    end_date_iso="2026-06-30T00:00:00Z",
                    active=True,
                )
            )
        return markets

    def _mock_order_book(self, token_id: str) -> OrderBookSnapshot:
        import random

        mid = round(random.uniform(0.2, 0.8), 3)
        half_spread = round(random.uniform(0.005, 0.02), 3)
        best_bid = Decimal(str(max(0.01, mid - half_spread)))
        best_ask = Decimal(str(min(0.99, mid + half_spread)))
        return OrderBookSnapshot(
            token_id=token_id,
            yes_best_ask=best_ask,
            yes_best_bid=best_bid,
            midpoint=(best_bid + best_ask) / 2,
            spread=best_ask - best_bid,
        )
