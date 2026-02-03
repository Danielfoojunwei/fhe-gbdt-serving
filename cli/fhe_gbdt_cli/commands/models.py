"""Model management commands."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from ..output import console, print_error, print_success, print_warning, format_table


@click.group()
def models():
    """Manage GBDT models.

    \b
    Commands for registering, compiling, and managing machine learning
    models for encrypted inference.

    \b
    Workflow:
      1. Register a model: fhe-gbdt models register model.json
      2. Compile for FHE: fhe-gbdt models compile <model-id>
      3. Check status: fhe-gbdt models status <compiled-model-id>
    """
    pass


@models.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Model name")
@click.option(
    "--library",
    "-l",
    type=click.Choice(["xgboost", "lightgbm", "catboost"]),
    help="GBDT library type (auto-detected if not specified)",
)
@click.option("--tag", "-t", multiple=True, help="Tags for the model")
@click.option("--description", "-d", help="Model description")
@click.pass_context
def register(
    ctx: click.Context,
    model_file: str,
    name: str,
    library: Optional[str],
    tag: tuple,
    description: Optional[str],
):
    """Register a new GBDT model.

    \b
    Supported formats:
      - XGBoost: JSON or binary (.json, .bin, .model)
      - LightGBM: JSON or text (.json, .txt)
      - CatBoost: JSON or binary (.json, .cbm)

    \b
    Examples:
      fhe-gbdt models register model.json --name my-classifier
      fhe-gbdt models register model.bin -n fraud-detector -l xgboost
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    model_path = Path(model_file)

    # Auto-detect library type if not specified
    if not library:
        library = detect_library_type(model_path)
        if not library:
            print_error("Could not detect library type. Please specify with --library")
            sys.exit(1)
        if verbose:
            console.print(f"Detected library type: {library}")

    try:
        from ..client import create_client

        client = create_client(config)

        with console.status(f"Registering model '{name}'..."):
            result = client.register_model(
                name=name,
                model_path=model_path,
                library_type=library,
                tags=list(tag),
                description=description,
            )

        if output_format == "json":
            console.print(json.dumps(result, indent=2, default=str))
        else:
            print_success(f"Model registered successfully!")
            console.print(f"\n  Model ID: [bold]{result['model_id']}[/bold]")
            console.print(f"  Name: {name}")
            console.print(f"  Library: {library}")
            console.print(f"\nNext step: Compile the model for FHE inference:")
            console.print(f"  fhe-gbdt models compile {result['model_id']}")

    except Exception as e:
        print_error(f"Registration failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@models.command()
@click.argument("model_id")
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["fast", "balanced", "accurate"]),
    default="balanced",
    help="Optimization profile",
)
@click.option("--wait/--no-wait", default=True, help="Wait for compilation to complete")
@click.option("--timeout", default=300, help="Compilation timeout in seconds")
@click.pass_context
def compile(
    ctx: click.Context,
    model_id: str,
    profile: str,
    wait: bool,
    timeout: int,
):
    """Compile a model for FHE inference.

    \b
    Profiles:
      fast     - Lower precision, faster inference (~50ms)
      balanced - Good balance of speed and accuracy (~60ms)
      accurate - Higher precision, slower inference (~80ms)

    \b
    Examples:
      fhe-gbdt models compile abc123
      fhe-gbdt models compile abc123 --profile fast
      fhe-gbdt models compile abc123 --no-wait
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)

        # Start compilation
        with console.status(f"Starting compilation (profile: {profile})..."):
            result = client.compile_model(model_id=model_id, profile=profile)

        compiled_id = result["compiled_model_id"]
        console.print(f"Compilation started: {compiled_id}")

        if not wait:
            console.print(f"\nCheck status with:")
            console.print(f"  fhe-gbdt models status {compiled_id}")
            return

        # Wait for completion
        import time

        start_time = time.time()
        with console.status("Compiling...") as status:
            while time.time() - start_time < timeout:
                status_result = client.get_compile_status(compiled_id)
                status_text = status_result.get("status", "unknown")

                if status_text == "successful":
                    break
                elif status_text == "failed":
                    print_error(f"Compilation failed: {status_result.get('error', 'Unknown error')}")
                    sys.exit(1)

                progress = status_result.get("progress", 0)
                status.update(f"Compiling... {progress}%")
                time.sleep(2)
            else:
                print_error("Compilation timed out")
                sys.exit(1)

        if output_format == "json":
            console.print(json.dumps(status_result, indent=2, default=str))
        else:
            elapsed = time.time() - start_time
            print_success(f"Compilation complete! ({elapsed:.1f}s)")
            console.print(f"\n  Compiled Model ID: [bold]{compiled_id}[/bold]")
            console.print(f"  Plan ID: {status_result.get('plan_id', 'N/A')}")
            console.print(f"\nNext steps:")
            console.print(f"  1. Generate keys: fhe-gbdt keys generate")
            console.print(f"  2. Make predictions: fhe-gbdt predict {compiled_id} --input data.json")

    except Exception as e:
        print_error(f"Compilation failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@models.command()
@click.argument("compiled_model_id")
@click.pass_context
def status(ctx: click.Context, compiled_model_id: str):
    """Check compilation status of a model.

    \b
    Example:
      fhe-gbdt models status abc123
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        result = client.get_compile_status(compiled_model_id)

        if output_format == "json":
            console.print(json.dumps(result, indent=2, default=str))
        else:
            status_text = result.get("status", "unknown")
            console.print(f"\n[bold]Compilation Status[/bold]")
            console.print(f"  ID: {compiled_model_id}")
            console.print(f"  Status: {status_text}")

            if status_text == "successful":
                console.print(f"  Plan ID: {result.get('plan_id', 'N/A')}")
            elif status_text == "failed":
                console.print(f"  Error: {result.get('error', 'Unknown')}")
            else:
                console.print(f"  Progress: {result.get('progress', 0)}%")

    except Exception as e:
        print_error(f"Failed to get status: {e}")
        sys.exit(1)


@models.command(name="list")
@click.option("--limit", "-l", default=20, help="Maximum models to list")
@click.option("--tag", "-t", help="Filter by tag")
@click.pass_context
def list_models(ctx: click.Context, limit: int, tag: Optional[str]):
    """List registered models.

    \b
    Example:
      fhe-gbdt models list
      fhe-gbdt models list --tag production
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        models = client.list_models(limit=limit, tag=tag)

        if output_format == "json":
            console.print(json.dumps(models, indent=2, default=str))
        elif not models:
            console.print("No models found.")
        else:
            headers = ["ID", "Name", "Library", "Status", "Created"]
            rows = [
                [
                    m.get("model_id", "")[:12],
                    m.get("name", ""),
                    m.get("library_type", ""),
                    m.get("status", ""),
                    m.get("created_at", "")[:10] if m.get("created_at") else "",
                ]
                for m in models
            ]
            console.print(format_table(headers, rows))

    except Exception as e:
        print_error(f"Failed to list models: {e}")
        sys.exit(1)


@models.command()
@click.argument("model_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx: click.Context, model_id: str, force: bool):
    """Delete a registered model.

    \b
    Example:
      fhe-gbdt models delete abc123
    """
    config = ctx.obj.get("config")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    if not force:
        if not click.confirm(f"Delete model {model_id}?"):
            console.print("Cancelled.")
            return

    try:
        from ..client import create_client

        client = create_client(config)
        client.delete_model(model_id)
        print_success(f"Model {model_id} deleted.")

    except Exception as e:
        print_error(f"Failed to delete model: {e}")
        sys.exit(1)


def detect_library_type(model_path: Path) -> Optional[str]:
    """Detect GBDT library type from file content."""
    try:
        content = model_path.read_bytes()

        # Try to parse as JSON
        if model_path.suffix in [".json", ".txt"]:
            try:
                data = json.loads(content)

                # XGBoost JSON format
                if "learner" in data or "version" in data:
                    return "xgboost"

                # LightGBM JSON format
                if "name" in data and data.get("name") == "tree":
                    return "lightgbm"
                if "tree_info" in data:
                    return "lightgbm"

                # CatBoost JSON format
                if "model_info" in data or "oblivious_trees" in data:
                    return "catboost"

            except json.JSONDecodeError:
                pass

        # Binary format detection
        if model_path.suffix == ".cbm":
            return "catboost"
        if model_path.suffix in [".bin", ".model"]:
            # Check for XGBoost binary magic
            if content[:4] == b"binf":
                return "xgboost"

        return None

    except Exception:
        return None
