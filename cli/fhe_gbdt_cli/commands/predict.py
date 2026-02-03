"""Prediction commands."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from ..output import console, print_error, print_success, print_warning


@click.command()
@click.argument("compiled_model_id")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), help="Input features JSON file")
@click.option("--features", "-f", help="Inline features as JSON array")
@click.option("--output", "-o", type=click.Path(), help="Output file for predictions")
@click.option("--decrypt/--no-decrypt", default=True, help="Decrypt results locally")
@click.option("--key-dir", "-k", type=click.Path(exists=True), help="Directory containing keys")
@click.pass_context
def predict(
    ctx: click.Context,
    compiled_model_id: str,
    input_file: Optional[str],
    features: Optional[str],
    output: Optional[str],
    decrypt: bool,
    key_dir: Optional[str],
):
    """Make encrypted predictions.

    \b
    Run inference on encrypted data using a compiled FHE model.
    Your data never leaves your machine unencrypted.

    \b
    Input Formats:
      - Single sample: [1.0, 2.0, 3.0, 4.0]
      - Multiple samples: [[1.0, 2.0], [3.0, 4.0]]
      - Named features: {"feature1": 1.0, "feature2": 2.0}

    \b
    Examples:
      fhe-gbdt predict abc123 --input data.json
      fhe-gbdt predict abc123 --features "[1.0, 2.0, 3.0, 4.0]"
      fhe-gbdt predict abc123 -i data.json -o predictions.json
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    # Parse input features
    if input_file:
        try:
            feature_data = json.loads(Path(input_file).read_text())
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in input file: {e}")
            sys.exit(1)
    elif features:
        try:
            feature_data = json.loads(features)
        except json.JSONDecodeError as e:
            print_error(f"Invalid features JSON: {e}")
            sys.exit(1)
    else:
        print_error("Must provide --input or --features")
        sys.exit(1)

    # Normalize features to list of lists
    if isinstance(feature_data, dict):
        # Named features - convert to list
        feature_list = [list(feature_data.values())]
    elif isinstance(feature_data, list):
        if len(feature_data) > 0 and not isinstance(feature_data[0], list):
            # Single sample
            feature_list = [feature_data]
        else:
            feature_list = feature_data
    else:
        print_error("Features must be a list or dict")
        sys.exit(1)

    # Find keys directory
    if key_dir:
        keys_path = Path(key_dir)
    elif config:
        keys_path = Path(config.key_directory)
    else:
        keys_path = Path.home() / ".fhe-gbdt" / "keys"

    secret_key_path = keys_path / "secret.key"

    if decrypt and not secret_key_path.exists():
        print_error(f"Secret key not found at {secret_key_path}")
        console.print("Generate keys first with: fhe-gbdt keys generate")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)

        num_samples = len(feature_list)
        num_features = len(feature_list[0]) if feature_list else 0

        console.print(f"Predicting {num_samples} sample(s) with {num_features} features...")

        # Encrypt features
        with console.status("Encrypting features..."):
            secret_key = secret_key_path.read_bytes() if decrypt else None
            encrypted_payload = client.encrypt_features(
                features=feature_list,
                secret_key=secret_key,
            )

        # Make prediction
        with console.status("Running encrypted inference..."):
            result = client.predict(
                compiled_model_id=compiled_model_id,
                encrypted_payload=encrypted_payload,
            )

        latency = result.get("latency_ms", 0)

        # Decrypt if requested
        if decrypt:
            with console.status("Decrypting results..."):
                predictions = client.decrypt_results(
                    encrypted_result=result.get("encrypted_result"),
                    secret_key=secret_key,
                )
        else:
            predictions = {"encrypted": True, "payload": result.get("encrypted_result")}

        # Output results
        if output:
            output_path = Path(output)
            output_data = {
                "model_id": compiled_model_id,
                "predictions": predictions,
                "latency_ms": latency,
                "samples": num_samples,
            }
            output_path.write_text(json.dumps(output_data, indent=2, default=str))
            print_success(f"Predictions saved to {output}")
        elif output_format == "json":
            console.print(json.dumps({
                "predictions": predictions,
                "latency_ms": latency,
            }, indent=2, default=str))
        else:
            print_success(f"Prediction complete! (latency: {latency:.1f}ms)")
            console.print("\n[bold]Predictions:[/bold]")

            if isinstance(predictions, list):
                for i, pred in enumerate(predictions):
                    if isinstance(pred, list):
                        console.print(f"  Sample {i+1}: {pred}")
                    else:
                        console.print(f"  Sample {i+1}: {pred:.6f}")
            else:
                console.print(f"  {predictions}")

    except Exception as e:
        print_error(f"Prediction failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@click.command(name="batch")
@click.argument("compiled_model_id")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), required=True, help="Input CSV/JSON file")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file for predictions")
@click.option("--batch-size", "-b", default=100, help="Batch size for processing")
@click.option("--format", "output_fmt", type=click.Choice(["csv", "json"]), default="csv", help="Output format")
@click.pass_context
def batch_predict(
    ctx: click.Context,
    compiled_model_id: str,
    input_file: str,
    output: str,
    batch_size: int,
    output_fmt: str,
):
    """Run batch predictions on a file.

    \b
    Process large datasets in batches with encrypted inference.

    \b
    Examples:
      fhe-gbdt batch abc123 -i data.csv -o predictions.csv
      fhe-gbdt batch abc123 -i data.json -o predictions.json --batch-size 50
    """
    config = ctx.obj.get("config")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output)

    try:
        # Load input data
        if input_path.suffix == ".csv":
            import csv
            with open(input_path) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                data = [[float(x) for x in row] for row in reader]
        else:
            data = json.loads(input_path.read_text())
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

        total_samples = len(data)
        console.print(f"Processing {total_samples} samples in batches of {batch_size}...")

        from ..client import create_client
        client = create_client(config)

        # Load keys
        keys_path = Path(config.key_directory) if config else Path.home() / ".fhe-gbdt" / "keys"
        secret_key = (keys_path / "secret.key").read_bytes()

        all_predictions = []
        total_latency = 0

        with console.status("Processing batches...") as status:
            for i in range(0, total_samples, batch_size):
                batch = data[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_samples + batch_size - 1) // batch_size

                status.update(f"Processing batch {batch_num}/{total_batches}...")

                encrypted = client.encrypt_features(batch, secret_key)
                result = client.predict(compiled_model_id, encrypted)
                predictions = client.decrypt_results(result["encrypted_result"], secret_key)

                all_predictions.extend(predictions)
                total_latency += result.get("latency_ms", 0)

        # Save output
        if output_fmt == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["prediction"])
                for pred in all_predictions:
                    writer.writerow([pred])
        else:
            output_path.write_text(json.dumps({
                "model_id": compiled_model_id,
                "predictions": all_predictions,
                "total_samples": total_samples,
                "total_latency_ms": total_latency,
            }, indent=2))

        print_success(f"Batch prediction complete!")
        console.print(f"  Samples processed: {total_samples}")
        console.print(f"  Total latency: {total_latency:.1f}ms")
        console.print(f"  Output saved to: {output_path}")

    except Exception as e:
        print_error(f"Batch prediction failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
