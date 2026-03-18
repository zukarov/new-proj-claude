"""
Connection validator — run before starting the bot to confirm every
external dependency is reachable and credentials are valid.

Usage:
    trading-bot check          # via CLI entry point
    python -m bot.setup_check  # direct execution
"""

import asyncio
import sys
from dataclasses import dataclass

import aiohttp
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

_CHECK = "[bold green]  OK  [/]"
_FAIL  = "[bold red] FAIL [/]"
_SKIP  = "[dim] SKIP [/]"


@dataclass
class CheckResult:
    name: str
    status: str   # "ok" | "fail" | "skip"
    detail: str


async def check_env_file() -> CheckResult:
    """Verify .env exists and required keys are present."""
    from pathlib import Path
    env_path = Path(".env")
    if not env_path.exists():
        return CheckResult(
            "Config (.env)",
            "fail",
            ".env file not found — run: cp .env.example .env  then fill in values",
        )
    content = env_path.read_text()
    missing = []
    for key in ("ANTHROPIC_API_KEY", "POLYGON_PRIVATE_KEY", "POLYMARKET_FUNDER"):
        if key not in content or f"{key}=YOUR" in content or f"{key}=0x0000" in content or f"{key}=sk-ant-YOUR" in content:
            missing.append(key)
    if missing:
        return CheckResult(
            "Config (.env)",
            "fail",
            f"Placeholder values still set for: {', '.join(missing)}",
        )
    return CheckResult("Config (.env)", "ok", "All required keys present")


async def check_anthropic_api() -> CheckResult:
    """Verify Anthropic API key with a minimal API call."""
    try:
        from .config import get_config
        config = get_config()
    except Exception as exc:
        return CheckResult("Anthropic API", "fail", f"Config load failed: {exc}")

    if not config.ANTHROPIC_API_KEY:
        return CheckResult("Anthropic API", "fail", "ANTHROPIC_API_KEY is empty")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        def _ping() -> str:
            resp = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=8,
                messages=[{"role": "user", "content": "reply: ok"}],
            )
            return resp.content[0].text.strip()

        reply = await asyncio.to_thread(_ping)
        return CheckResult("Anthropic API", "ok", f"Model {config.CLAUDE_MODEL} reachable (reply: {reply!r})")
    except Exception as exc:
        return CheckResult("Anthropic API", "fail", str(exc)[:120])


async def check_polymarket_clob() -> CheckResult:
    """Verify Polymarket CLOB is reachable (no auth required)."""
    url = "https://clob.polymarket.com/markets?limit=1"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    count = len(data.get("data", []))
                    return CheckResult(
                        "Polymarket CLOB",
                        "ok",
                        f"Reachable — returned {count} market(s)",
                    )
                return CheckResult(
                    "Polymarket CLOB", "fail", f"HTTP {resp.status}"
                )
    except Exception as exc:
        return CheckResult("Polymarket CLOB", "fail", str(exc)[:120])


async def check_polygon_rpc() -> CheckResult:
    """Verify Polygon RPC responds to eth_blockNumber."""
    try:
        from .config import get_config
        config = get_config()
    except Exception as exc:
        return CheckResult("Polygon RPC", "skip", f"Config unavailable: {exc}")

    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.POLYGON_RPC_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                data = await resp.json()
        block_hex = data.get("result", "0x0")
        block_num = int(block_hex, 16)
        return CheckResult("Polygon RPC", "ok", f"Block #{block_num:,} on chain {config.CHAIN_ID}")
    except Exception as exc:
        return CheckResult("Polygon RPC", "fail", str(exc)[:120])


async def check_usdc_balance() -> CheckResult:
    """Check USDC.e balance on the funder wallet (live mode only)."""
    try:
        from .config import get_config
        config = get_config()
    except Exception as exc:
        return CheckResult("USDC Balance", "skip", f"Config unavailable: {exc}")

    if config.DRY_RUN:
        return CheckResult(
            "USDC Balance",
            "skip",
            f"DRY_RUN=true — simulated balance: ${config.DRY_RUN_BALANCE_USDC:,.2f}",
        )

    try:
        from .polymarket_client import PolymarketClient
        client = PolymarketClient(config)
        balance = await client.get_usdc_balance()
        if balance <= 0:
            return CheckResult(
                "USDC Balance",
                "fail",
                f"Balance is ${balance:.2f} — fund your wallet with USDC.e on Polygon",
            )
        return CheckResult("USDC Balance", "ok", f"${balance:,.2f} USDC.e available")
    except Exception as exc:
        return CheckResult("USDC Balance", "fail", str(exc)[:120])


async def check_polymarket_auth() -> CheckResult:
    """Verify wallet credentials by deriving API credentials (live mode only)."""
    try:
        from .config import get_config
        config = get_config()
    except Exception as exc:
        return CheckResult("Wallet Auth", "skip", f"Config unavailable: {exc}")

    if config.DRY_RUN:
        return CheckResult("Wallet Auth", "skip", "DRY_RUN=true — skipping live auth test")

    try:
        from .polymarket_client import PolymarketClient
        client = PolymarketClient(config)
        await client.initialize()
        return CheckResult("Wallet Auth", "ok", f"Credentials valid for {config.POLYMARKET_FUNDER[:12]}...")
    except Exception as exc:
        return CheckResult("Wallet Auth", "fail", str(exc)[:120])


async def check_news_feed() -> CheckResult:
    """Verify Google News RSS is reachable."""
    url = "https://news.google.com/rss/search?q=prediction+market&hl=en-US&gl=US&ceid=US:en"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    return CheckResult("News RSS", "ok", "Google News RSS reachable")
                return CheckResult("News RSS", "fail", f"HTTP {resp.status}")
    except Exception as exc:
        return CheckResult("News RSS", "fail", str(exc)[:120])


async def run_all_checks() -> list[CheckResult]:
    results = await asyncio.gather(
        check_env_file(),
        check_anthropic_api(),
        check_polymarket_clob(),
        check_polygon_rpc(),
        check_usdc_balance(),
        check_polymarket_auth(),
        check_news_feed(),
        return_exceptions=False,
    )
    return list(results)


def print_results(results: list[CheckResult]) -> bool:
    table = Table(box=box.ROUNDED, title="Connection Check", title_style="bold blue")
    table.add_column("Check", style="bold", min_width=20)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Detail")

    all_ok = True
    for r in results:
        if r.status == "ok":
            badge = _CHECK
        elif r.status == "fail":
            badge = _FAIL
            all_ok = False
        else:
            badge = _SKIP
        table.add_row(r.name, badge, r.detail)

    console.print()
    console.print(table)
    console.print()

    if all_ok:
        console.print("[bold green]All checks passed.[/] Ready to start the bot.\n")
    else:
        console.print("[bold red]Some checks failed.[/] Fix the issues above before running live.\n")

    return all_ok


async def async_main() -> int:
    console.print("\n[bold blue]Running pre-flight checks...[/]\n")
    results = await run_all_checks()
    ok = print_results(results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
