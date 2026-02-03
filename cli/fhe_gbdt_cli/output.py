"""
Output formatting utilities for FHE-GBDT CLI.

Provides consistent output formatting across all commands including:
- Table formatting using rich
- JSON output
- Progress bars
- Status indicators
"""

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


# Global console instance
console = Console()
error_console = Console(stderr=True)


class OutputFormatter:
    """Handles output formatting for CLI commands."""

    def __init__(self, format: str = "table", quiet: bool = False):
        """
        Initialize the output formatter.

        Args:
            format: Output format ('table' or 'json')
            quiet: Suppress non-essential output
        """
        self.format = format
        self.quiet = quiet

    def print_success(self, message: str) -> None:
        """Print a success message."""
        if self.quiet:
            return
        if self.format == "json":
            self._print_json({"status": "success", "message": message})
        else:
            console.print(f"[green]\u2713[/green] {message}")

    def print_error(self, message: str, details: Optional[str] = None) -> None:
        """Print an error message."""
        if self.format == "json":
            data = {"status": "error", "message": message}
            if details:
                data["details"] = details
            self._print_json(data, file=sys.stderr)
        else:
            error_console.print(f"[red]\u2717 Error:[/red] {message}")
            if details:
                error_console.print(f"  [dim]{details}[/dim]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if self.quiet:
            return
        if self.format == "json":
            self._print_json({"status": "warning", "message": message})
        else:
            console.print(f"[yellow]\u26a0[/yellow] {message}")

    def print_info(self, message: str) -> None:
        """Print an informational message."""
        if self.quiet:
            return
        if self.format == "json":
            self._print_json({"status": "info", "message": message})
        else:
            console.print(f"[blue]\u2139[/blue] {message}")

    def print_table(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[Dict[str, str]]] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Print data as a formatted table or JSON.

        Args:
            data: List of dictionaries to display
            columns: Column definitions with 'key', 'header', and optional 'style'
            title: Optional table title
        """
        if not data:
            if self.format == "json":
                self._print_json([])
            else:
                console.print("[dim]No data to display[/dim]")
            return

        if self.format == "json":
            self._print_json(data)
            return

        # Auto-detect columns if not provided
        if columns is None:
            columns = [{"key": k, "header": k.replace("_", " ").title()} for k in data[0].keys()]

        table = Table(title=title, show_header=True, header_style="bold cyan")

        for col in columns:
            table.add_column(col["header"], style=col.get("style", ""))

        for row in data:
            values = []
            for col in columns:
                value = row.get(col["key"], "")
                values.append(self._format_value(value))
            table.add_row(*values)

        console.print(table)

    def print_dict(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
    ) -> None:
        """
        Print a dictionary as a formatted table or JSON.

        Args:
            data: Dictionary to display
            title: Optional title
        """
        if self.format == "json":
            self._print_json(data)
            return

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in data.items():
            table.add_row(
                key.replace("_", " ").title(),
                self._format_value(value),
            )

        console.print(table)

    def print_panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "blue",
    ) -> None:
        """Print content in a panel."""
        if self.quiet:
            return
        if self.format == "json":
            self._print_json({"title": title, "content": content})
        else:
            console.print(Panel(content, title=title, border_style=style))

    def print_json_raw(self, data: Any) -> None:
        """Print raw JSON data regardless of output format setting."""
        self._print_json(data)

    def _print_json(self, data: Any, file: Any = None) -> None:
        """Print data as formatted JSON."""
        output = json.dumps(data, indent=2, default=str)
        if file:
            print(output, file=file)
        else:
            console.print(output, highlight=False)

    def _format_value(self, value: Any) -> str:
        """Format a value for table display."""
        if value is None:
            return "[dim]-[/dim]"
        elif isinstance(value, bool):
            return "[green]Yes[/green]" if value else "[red]No[/red]"
        elif isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        else:
            return str(value)


@contextmanager
def progress_spinner(message: str) -> Generator[Progress, None, None]:
    """
    Context manager for a simple spinner progress indicator.

    Args:
        message: Message to display while spinning

    Yields:
        Progress instance for additional control
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description=message, total=None)
        yield progress


@contextmanager
def progress_bar(
    total: int,
    description: str = "Processing",
) -> Generator[Progress, None, None]:
    """
    Context manager for a progress bar.

    Args:
        total: Total number of items
        description: Description text

    Yields:
        Progress instance
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        progress.add_task(description=description, total=total)
        yield progress


def format_bytes(num_bytes: Union[int, float]) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: Union[int, float]) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(timestamp: Union[str, datetime, None]) -> str:
    """Format a timestamp for display."""
    if timestamp is None:
        return "-"
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return timestamp
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_status(status: str) -> Text:
    """Format a status string with appropriate coloring."""
    status_colors = {
        "active": "green",
        "ready": "green",
        "completed": "green",
        "success": "green",
        "running": "blue",
        "pending": "yellow",
        "queued": "yellow",
        "compiling": "yellow",
        "processing": "yellow",
        "failed": "red",
        "error": "red",
        "cancelled": "red",
        "expired": "dim",
        "inactive": "dim",
    }

    color = status_colors.get(status.lower(), "white")
    return Text(status.upper(), style=color)


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Prompt user for confirmation.

    Args:
        message: Confirmation message
        default: Default value if user presses enter

    Returns:
        True if confirmed, False otherwise
    """
    suffix = "[Y/n]" if default else "[y/N]"
    response = console.input(f"{message} {suffix}: ").strip().lower()

    if not response:
        return default
    return response in ("y", "yes")


def print_separator() -> None:
    """Print a horizontal separator line."""
    console.print("-" * 60, style="dim")
