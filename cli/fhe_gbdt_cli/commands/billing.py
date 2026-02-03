"""Billing and subscription commands."""

import click
import json
import sys
from typing import Optional

from ..output import console, print_error, print_success, print_warning, format_table


@click.group()
def billing():
    """Manage billing and subscriptions.

    \b
    Commands for managing your subscription, viewing usage,
    and handling billing operations.
    """
    pass


@billing.command()
@click.pass_context
def plans(ctx: click.Context):
    """List available subscription plans.

    \b
    Example:
      fhe-gbdt billing plans
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        plans = client.list_plans()

        if output_format == "json":
            console.print(json.dumps(plans, indent=2, default=str))
        else:
            console.print("\n[bold]Available Plans[/bold]\n")

            for plan in plans:
                price = plan.get("price_cents", 0) / 100
                limit = plan.get("prediction_limit", 0)
                limit_str = f"{limit:,}/month" if limit > 0 else "Unlimited"

                console.print(f"  [bold]{plan.get('name', 'Unknown').title()}[/bold]")
                console.print(f"    Price: ${price:.2f}/month")
                console.print(f"    Predictions: {limit_str}")
                console.print(f"    {plan.get('description', '')}")
                console.print()

    except Exception as e:
        print_error(f"Failed to list plans: {e}")
        sys.exit(1)


@billing.command()
@click.pass_context
def subscription(ctx: click.Context):
    """View current subscription details.

    \b
    Example:
      fhe-gbdt billing subscription
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        sub = client.get_subscription()

        if not sub:
            console.print("No active subscription.")
            console.print("\nView available plans with: fhe-gbdt billing plans")
            return

        if output_format == "json":
            console.print(json.dumps(sub, indent=2, default=str))
        else:
            console.print("\n[bold]Subscription Details[/bold]\n")
            console.print(f"  Plan: {sub.get('plan_name', 'Unknown')}")
            console.print(f"  Status: {sub.get('status', 'Unknown')}")
            console.print(f"  Current Period: {sub.get('period_start', '')} - {sub.get('period_end', '')}")

            if sub.get("cancel_at_period_end"):
                print_warning("\n  Subscription will cancel at end of period")

    except Exception as e:
        print_error(f"Failed to get subscription: {e}")
        sys.exit(1)


@billing.command()
@click.pass_context
def usage(ctx: click.Context):
    """View current usage statistics.

    \b
    Example:
      fhe-gbdt billing usage
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        usage = client.get_usage()

        if output_format == "json":
            console.print(json.dumps(usage, indent=2, default=str))
        else:
            predictions_used = usage.get("predictions_count", 0)
            predictions_limit = usage.get("predictions_limit", 0)

            if predictions_limit > 0:
                pct = (predictions_used / predictions_limit) * 100
                pct_str = f"{pct:.1f}%"
            else:
                pct_str = "N/A (unlimited)"

            console.print("\n[bold]Usage Statistics[/bold]\n")
            console.print(f"  Current Period: {usage.get('period_start', '')} - {usage.get('period_end', '')}")
            console.print(f"\n  Predictions: {predictions_used:,} / {predictions_limit:,} ({pct_str})")

            # Progress bar
            if predictions_limit > 0:
                bar_width = 40
                filled = int((predictions_used / predictions_limit) * bar_width)
                bar = "█" * min(filled, bar_width) + "░" * (bar_width - filled)

                if pct > 80:
                    console.print(f"  [{bar}] [yellow]{pct_str}[/yellow]")
                else:
                    console.print(f"  [{bar}] {pct_str}")

            overage = usage.get("overage_count", 0)
            if overage > 0:
                overage_cost = usage.get("overage_cost_cents", 0) / 100
                print_warning(f"\n  Overage: {overage:,} predictions (${overage_cost:.2f})")

    except Exception as e:
        print_error(f"Failed to get usage: {e}")
        sys.exit(1)


@billing.command()
@click.option("--limit", "-l", default=10, help="Number of invoices to show")
@click.pass_context
def invoices(ctx: click.Context, limit: int):
    """List billing invoices.

    \b
    Example:
      fhe-gbdt billing invoices
      fhe-gbdt billing invoices --limit 20
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        invoices = client.list_invoices(limit=limit)

        if output_format == "json":
            console.print(json.dumps(invoices, indent=2, default=str))
        elif not invoices:
            console.print("No invoices found.")
        else:
            headers = ["Invoice", "Date", "Amount", "Status"]
            rows = [
                [
                    inv.get("id", "")[:12],
                    inv.get("created_at", "")[:10] if inv.get("created_at") else "",
                    f"${inv.get('total_cents', 0) / 100:.2f}",
                    inv.get("status", ""),
                ]
                for inv in invoices
            ]
            console.print(format_table(headers, rows))

    except Exception as e:
        print_error(f"Failed to list invoices: {e}")
        sys.exit(1)


@billing.command()
@click.argument("plan_id")
@click.pass_context
def upgrade(ctx: click.Context, plan_id: str):
    """Upgrade to a different plan.

    \b
    Example:
      fhe-gbdt billing upgrade pro
    """
    config = ctx.obj.get("config")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)

        # Get plan details
        plans = client.list_plans()
        plan = next((p for p in plans if p.get("name") == plan_id or p.get("id") == plan_id), None)

        if not plan:
            print_error(f"Plan '{plan_id}' not found")
            console.print("View available plans with: fhe-gbdt billing plans")
            sys.exit(1)

        price = plan.get("price_cents", 0) / 100
        console.print(f"\nUpgrading to {plan.get('name', 'Unknown')} (${price:.2f}/month)")

        if not click.confirm("Proceed?"):
            console.print("Cancelled.")
            return

        # Create checkout session
        result = client.create_checkout_session(plan_id)

        console.print(f"\nComplete checkout at:")
        console.print(f"  {result.get('checkout_url', 'N/A')}")

    except Exception as e:
        print_error(f"Upgrade failed: {e}")
        sys.exit(1)


@billing.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def cancel(ctx: click.Context, force: bool):
    """Cancel subscription at end of billing period.

    \b
    Example:
      fhe-gbdt billing cancel
    """
    config = ctx.obj.get("config")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    if not force:
        print_warning("This will cancel your subscription at the end of the current billing period.")
        console.print("You will retain access until then.")

        if not click.confirm("Cancel subscription?"):
            console.print("Cancelled.")
            return

    try:
        from ..client import create_client

        client = create_client(config)
        client.cancel_subscription()
        print_success("Subscription will be canceled at end of billing period.")

    except Exception as e:
        print_error(f"Cancellation failed: {e}")
        sys.exit(1)
