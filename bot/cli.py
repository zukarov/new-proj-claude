import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .bot import TradingBot
from .config import get_config

console = Console()


def build_header(dry_run: bool, scan_count: int, sentiment_enabled: bool) -> Panel:
    mode = "[bold yellow]DRY RUN[/]" if dry_run else "[bold red]LIVE TRADING[/]"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    news = "[green]NEWS ON[/]" if sentiment_enabled else "[dim]NEWS OFF[/]"
    return Panel(
        Text.from_markup(
            f"  Claude + Polymarket Bot  |  {mode}  |  {ts}  "
            f"|  Scans: [cyan]{scan_count}[/]  |  {news}"
        ),
        style="bold blue",
        box=box.HEAVY,
    )


def build_positions_table(positions: list[dict]) -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True)
    table.add_column("Market", style="cyan", ratio=4)
    table.add_column("Action", justify="center", ratio=1)
    table.add_column("Size $", justify="right", ratio=1)
    table.add_column("Entry", justify="right", ratio=1)
    table.add_column("Current", justify="right", ratio=1)
    table.add_column("P&L $", justify="right", ratio=1)
    table.add_column("DRY", justify="center", ratio=1)

    if not positions:
        table.add_row("[dim]No open positions[/]", "", "", "", "", "", "")
    else:
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0.0)
            pnl_style = "green" if pnl >= 0 else "red"
            action = pos.get("action", "")
            action_style = "green" if "YES" in action else "magenta"
            question = pos.get("question", "")
            q_short = (question[:52] + "...") if len(question) > 55 else question
            is_dry = "Y" if pos.get("dry_run") else "N"
            table.add_row(
                q_short,
                Text(action, style=action_style),
                f"${pos.get('size_usdc', 0):.2f}",
                f"{pos.get('entry_price', 0):.3f}",
                f"{pos.get('current_price', 0):.3f}",
                Text(f"${pnl:+.2f}", style=pnl_style),
                is_dry,
            )
    return Panel(table, title="Open Positions", border_style="blue")


def build_stats_panel(data: dict) -> Panel:
    stats = data.get("stats", {})
    daily_pnl = data.get("daily_pnl", 0.0)
    unrealized = data.get("unrealized_pnl", 0.0)
    total_pnl = daily_pnl + unrealized
    daily_target = data.get("daily_target", 0.0)
    pnl_style = "green" if total_pnl >= 0 else "red"

    # Progress bar toward daily target
    target_line = ""
    if daily_target > 0:
        pct = min(daily_pnl / daily_target, 1.0)
        bar_filled = int(pct * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        target_style = "green" if pct >= 1.0 else "yellow"
        target_line = (
            f"\nDaily target:      [{target_style}]{bar}[/] "
            f"[{target_style}]${daily_pnl:.0f}/${daily_target:.0f}[/]"
        )

    lines = [
        f"Decisions made:    [cyan]{stats.get('decisions', 0)}[/]",
        f"Trades executed:   [green]{stats.get('trades', 0)}[/]",
        f"Rejected by risk:  [yellow]{stats.get('rejections', 0)}[/]",
        f"Stop-losses hit:   [red]{stats.get('stop_losses', 0)}[/]",
        f"Errors:            [red]{stats.get('errors', 0)}[/]",
        f"Circuit trips:     [red]{stats.get('circuit_breaker_trips', 0)}[/]",
        "",
        f"Realized P&L:      [{pnl_style}]${daily_pnl:+.2f}[/]",
        f"Unrealized P&L:    [{pnl_style}]${unrealized:+.2f}[/]",
        f"Total P&L:         [{pnl_style}]${total_pnl:+.2f}[/]",
    ]
    if target_line:
        lines.append(target_line)

    return Panel("\n".join(lines), title="Session Stats", border_style="green")


def render_layout(bot: TradingBot) -> Layout:
    data = bot.get_dashboard_data()
    layout = Layout()
    layout.split_column(
        Layout(build_header(data["dry_run"], data["scan_count"], data.get("sentiment_enabled", False)), size=3),
        Layout(build_positions_table(data["positions"]), ratio=3),
        Layout(build_stats_panel(data), ratio=2),
    )
    return layout


@click.group()
def cli() -> None:
    """Claude + Polymarket Automated Trading Bot"""


@cli.command()
@click.option("--dry-run/--no-dry-run", default=None, help="Override DRY_RUN from .env")
@click.option("--once", is_flag=True, help="Run a single scan cycle and exit")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def run(dry_run: bool | None, once: bool, log_level: str | None) -> None:
    """Start the trading bot (24h continuous with live dashboard)."""
    logging.basicConfig(
        level=getattr(logging, log_level or "INFO"),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = get_config()
    if dry_run is not None:
        config.DRY_RUN = dry_run

    mode_label = "DRY RUN" if config.DRY_RUN else "LIVE TRADING"
    console.print(f"\n[bold blue]Claude + Polymarket Bot[/] — [bold yellow]{mode_label}[/]\n")

    if config.DAILY_PROFIT_TARGET_USDC > 0:
        console.print(
            f"Daily profit target: [green]${config.DAILY_PROFIT_TARGET_USDC:,.2f}[/] "
            f"(resets midnight UTC)\n"
        )

    bot = TradingBot(config)

    async def _main() -> None:
        await bot.initialize()

        if once:
            stats = await bot.run_scan_cycle()
            console.print(f"\nScan complete: {stats}")
            return

        bot_task = asyncio.create_task(bot.start())

        with Live(render_layout(bot), refresh_per_second=0.5, screen=True) as live:
            try:
                while not bot_task.done():
                    live.update(render_layout(bot))
                    await asyncio.sleep(2)
            except (KeyboardInterrupt, asyncio.CancelledError):
                await bot.stop()

        if bot_task.exception():
            raise bot_task.exception()

    asyncio.run(_main())


@cli.command()
def status() -> None:
    """Show current positions and P&L from persisted state."""
    p = Path("logs/positions.json")
    if not p.exists():
        console.print("[yellow]No positions file found. Has the bot run yet?[/]")
        return
    data = json.loads(p.read_text())
    positions = data.get("positions", {})
    daily_pnl = data.get("daily_pnl", 0.0)

    console.print(f"\n[bold]Open positions:[/] {len(positions)}")
    console.print(f"[bold]Daily P&L:[/] [{'green' if daily_pnl >= 0 else 'red'}]${daily_pnl:+.2f}[/]\n")

    if positions:
        table = Table(box=box.SIMPLE)
        table.add_column("Market")
        table.add_column("Action")
        table.add_column("Size $", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Unrealized P&L", justify="right")
        for pos in positions.values():
            pnl = pos.get("unrealized_pnl", 0.0)
            table.add_row(
                pos.get("question", "")[:60],
                pos.get("action", ""),
                f"${pos.get('size_usdc', 0):.2f}",
                f"{pos.get('entry_price', 0):.3f}",
                Text(f"${pnl:+.2f}", style="green" if pnl >= 0 else "red"),
            )
        console.print(table)


@cli.command()
def check() -> None:
    """Run pre-flight connection checks (Anthropic, Polymarket, Polygon, News)."""
    from .setup_check import async_main
    import sys
    sys.exit(asyncio.run(async_main()))


def main() -> None:
    cli()
