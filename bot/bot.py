import asyncio
import json
import logging
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .claude_analyst import ClaudeAnalyst, TradingDecision
from .config import TradingConfig, get_config
from .polymarket_client import PolymarketClient, TradeResult
from .risk_manager import PortfolioState, RiskManager, TradingAction

logger = structlog.get_logger(__name__)


class PositionTracker:
    """In-memory position tracking with JSON persistence."""

    PERSIST_PATH = Path("logs/positions.json")

    def __init__(self) -> None:
        self.positions: dict[str, dict] = {}
        self.daily_pnl: float = 0.0
        self.peak_value: float = 0.0
        self._load_from_disk()

    def open_position(
        self,
        token_id: str,
        action: str,
        size_usdc: float,
        price: float,
        market_question: str,
        order_id: str,
        dry_run: bool,
    ) -> None:
        self.positions[token_id] = {
            "token_id": token_id,
            "question": market_question[:100],
            "action": action,
            "size_usdc": size_usdc,
            "entry_price": price,
            "current_price": price,
            "shares": size_usdc / price if price > 0 else 0,
            "order_id": order_id,
            "opened_at": time.time(),
            "dry_run": dry_run,
            "unrealized_pnl": 0.0,
        }
        self._save_to_disk()

    def update_price(self, token_id: str, current_price: float) -> None:
        if token_id not in self.positions:
            return
        pos = self.positions[token_id]
        entry = pos["entry_price"]
        shares = pos["shares"]
        if pos["action"] == "BUY_YES":
            pos["unrealized_pnl"] = (current_price - entry) * shares
        else:
            pos["unrealized_pnl"] = (entry - current_price) * shares
        pos["current_price"] = current_price

    def close_position(self, token_id: str, exit_price: float) -> float:
        """Returns realized P&L."""
        if token_id not in self.positions:
            return 0.0
        pos = self.positions.pop(token_id)
        shares = pos["shares"]
        if pos["action"] == "BUY_YES":
            pnl = (exit_price - pos["entry_price"]) * shares
        else:
            pnl = (pos["entry_price"] - exit_price) * shares
        self.daily_pnl += pnl
        self._save_to_disk()
        return pnl

    def get_total_unrealized_pnl(self) -> float:
        return sum(p["unrealized_pnl"] for p in self.positions.values())

    def _save_to_disk(self) -> None:
        self.PERSIST_PATH.parent.mkdir(exist_ok=True)
        with open(self.PERSIST_PATH, "w") as f:
            json.dump(
                {"positions": self.positions, "daily_pnl": self.daily_pnl},
                f,
                indent=2,
            )

    def _load_from_disk(self) -> None:
        if self.PERSIST_PATH.exists():
            try:
                with open(self.PERSIST_PATH) as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.daily_pnl = data.get("daily_pnl", 0.0)
                logger.info(
                    "Loaded positions from disk",
                    count=len(self.positions),
                    daily_pnl=self.daily_pnl,
                )
            except Exception as exc:
                logger.warning("Failed to load positions from disk", error=str(exc))


class TradingBot:
    def __init__(self, config: Optional[TradingConfig] = None) -> None:
        self.config = config or get_config()
        self.polymarket = PolymarketClient(self.config)
        self.analyst = ClaudeAnalyst(self.config)
        self.risk = RiskManager(self.config)
        self.tracker = PositionTracker()
        self.scheduler = AsyncIOScheduler()
        self._running = False
        self._scan_count = 0
        self._stats: dict = defaultdict(int)

    async def initialize(self) -> None:
        await self.polymarket.initialize()
        logger.info("Bot initialized", dry_run=self.config.DRY_RUN)

    def _build_portfolio_state(self, available_usdc: float) -> PortfolioState:
        total_value = available_usdc + sum(
            p["size_usdc"] for p in self.tracker.positions.values()
        )
        if total_value > self.tracker.peak_value:
            self.tracker.peak_value = total_value
        return PortfolioState(
            available_usdc=available_usdc,
            total_portfolio_value=total_value,
            open_position_count=len(self.tracker.positions),
            daily_pnl=self.tracker.daily_pnl,
            peak_portfolio_value=self.tracker.peak_value,
            positions=self.tracker.positions,
        )

    async def _check_stop_losses(self) -> None:
        """Update prices and exit positions that hit stop-loss."""
        for token_id, pos in list(self.tracker.positions.items()):
            try:
                book = await self.polymarket.get_order_book_snapshot(token_id)
                if not book.midpoint:
                    continue
                current_price = float(book.midpoint)
                self.tracker.update_price(token_id, current_price)

                if self.risk.should_stop_loss(pos["entry_price"], current_price, pos["action"]):
                    logger.warning(
                        "Stop-loss triggered",
                        token_id=token_id,
                        entry_price=pos["entry_price"],
                        current_price=current_price,
                    )
                    if not pos["dry_run"]:
                        exit_side = "SELL" if pos["action"] == "BUY_YES" else "BUY"
                        await self.polymarket.place_limit_order(
                            token_id=token_id,
                            side=exit_side,
                            price=current_price,
                            size=pos["shares"],
                        )
                    pnl = self.tracker.close_position(token_id, current_price)
                    logger.info("Position closed via stop-loss", pnl=pnl)
                    self._stats["stop_losses"] += 1
            except Exception as exc:
                logger.error("Stop-loss check failed", token_id=token_id, error=str(exc))

    async def run_scan_cycle(self) -> dict:
        """One full market scan + analysis + execution cycle. Returns cycle stats."""
        self._scan_count += 1
        cycle_stats = {"scanned": 0, "analyzed": 0, "traded": 0, "rejected": 0}

        # Step 1: Check stop-losses on existing positions
        await self._check_stop_losses()

        # Step 2: Fetch active markets
        markets = await self.polymarket.get_active_markets(
            limit=self.config.MAX_MARKETS_PER_SCAN * 3,
            min_liquidity=self.config.MIN_MARKET_LIQUIDITY_USDC,
        )
        cycle_stats["scanned"] = len(markets)

        # Skip markets already in portfolio
        held_token_ids = set(self.tracker.positions.keys())
        markets = [
            m for m in markets
            if not any(
                t.get("token_id") in held_token_ids for t in m.tokens
            )
        ]

        # Step 3: Claude quick filter (only if we have more than MAX_MARKETS_PER_SCAN)
        if len(markets) > self.config.MAX_MARKETS_PER_SCAN:
            markets = await self.analyst.batch_filter_markets(markets)
        markets = markets[: self.config.MAX_MARKETS_PER_SCAN]

        # Step 4: Deep analysis + execution per market
        available_usdc = 1000.0  # Default; real use: fetch on-chain USDC balance
        portfolio_state = self._build_portfolio_state(available_usdc)

        # Circuit breaker check before processing any markets
        if self.risk.check_circuit_breaker(portfolio_state):
            logger.error("Circuit breaker active — skipping this scan cycle")
            self._stats["circuit_breaker_trips"] += 1
            return cycle_stats

        for market in markets:
            try:
                yes_token = market.tokens[0] if market.tokens else None
                if not yes_token:
                    continue

                token_id = yes_token.get("token_id", "")
                if not token_id:
                    continue

                book = await self.polymarket.get_order_book_snapshot(token_id)
                if not book.midpoint:
                    continue

                portfolio_context = {
                    "available_usdc": portfolio_state.available_usdc,
                    "open_positions": portfolio_state.open_position_count,
                    "max_positions": self.config.MAX_OPEN_POSITIONS,
                    "daily_pnl": portfolio_state.daily_pnl,
                }

                # Claude deep analysis
                decision: TradingDecision = await self.analyst.analyze_market(
                    market, book, portfolio_context
                )
                cycle_stats["analyzed"] += 1
                self._stats["decisions"] += 1

                # Risk gating
                spread = float(book.spread) if book.spread else None
                verdict = self.risk.approve_trade(
                    action=decision.action,
                    confidence=decision.confidence,
                    estimated_probability=decision.estimated_probability,
                    market_price=decision.market_price,
                    edge=decision.edge,
                    state=portfolio_state,
                    spread=spread,
                )

                if not verdict.approved:
                    cycle_stats["rejected"] += 1
                    self._stats["rejections"] += 1
                    logger.debug(
                        "Trade rejected",
                        reason=verdict.rejection_reason,
                        notes=verdict.notes,
                        market=market.question[:60],
                    )
                    continue

                # Execute
                side = "BUY" if decision.action == TradingAction.BUY_YES else "BUY"
                # For BUY_NO we buy the NO token (second token)
                if decision.action == TradingAction.BUY_NO:
                    no_token = market.tokens[1] if len(market.tokens) > 1 else None
                    if not no_token:
                        continue
                    trade_token_id = no_token.get("token_id", token_id)
                else:
                    trade_token_id = token_id

                shares = verdict.final_size_usdc / verdict.final_price if verdict.final_price > 0 else 0
                if shares <= 0:
                    continue

                result: TradeResult = await self.polymarket.place_limit_order(
                    token_id=trade_token_id,
                    side=side,
                    price=verdict.final_price,
                    size=shares,
                )

                self.tracker.open_position(
                    token_id=trade_token_id,
                    action=decision.action.value,
                    size_usdc=verdict.final_size_usdc,
                    price=verdict.final_price,
                    market_question=market.question,
                    order_id=result.order_id,
                    dry_run=self.config.DRY_RUN,
                )

                cycle_stats["traded"] += 1
                self._stats["trades"] += 1

                logger.info(
                    "Order placed",
                    question=market.question[:60],
                    action=decision.action.value,
                    size_usdc=verdict.final_size_usdc,
                    price=verdict.final_price,
                    dry_run=self.config.DRY_RUN,
                    reasoning=decision.reasoning[:100],
                )

                # Refresh portfolio state after trade
                portfolio_state = self._build_portfolio_state(
                    portfolio_state.available_usdc - verdict.final_size_usdc
                )

            except Exception as exc:
                logger.error(
                    "Market analysis failed",
                    question=market.question[:60],
                    error=str(exc),
                )
                self._stats["errors"] += 1

        logger.info(
            "Scan cycle complete",
            cycle=self._scan_count,
            **cycle_stats,
        )
        return cycle_stats

    async def start(self) -> None:
        """Start the bot with scheduled scan cycles. Blocks until stopped."""
        self._running = True

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        self.scheduler.add_job(
            self.run_scan_cycle,
            trigger="interval",
            seconds=self.config.SCAN_INTERVAL_SECONDS,
            id="market_scan",
            max_instances=1,  # Prevent overlapping scans
        )
        self.scheduler.start()
        logger.info(
            "Bot started",
            scan_interval_s=self.config.SCAN_INTERVAL_SECONDS,
            dry_run=self.config.DRY_RUN,
        )

        # Run first cycle immediately
        await self.run_scan_cycle()

        # Keep alive
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        self.scheduler.shutdown(wait=False)
        logger.info("Bot stopped gracefully", stats=dict(self._stats))

    def get_dashboard_data(self) -> dict:
        """Snapshot of current state for the CLI dashboard."""
        return {
            "positions": list(self.tracker.positions.values()),
            "daily_pnl": self.tracker.daily_pnl,
            "unrealized_pnl": self.tracker.get_total_unrealized_pnl(),
            "scan_count": self._scan_count,
            "stats": dict(self._stats),
            "dry_run": self.config.DRY_RUN,
        }
