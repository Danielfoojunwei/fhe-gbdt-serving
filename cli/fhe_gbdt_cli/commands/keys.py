"""Key management commands."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from ..output import console, print_error, print_success, print_warning


@click.group()
def keys():
    """Manage FHE encryption keys.

    \b
    Commands for generating, uploading, and managing cryptographic
    keys for encrypted inference.

    \b
    Key Types:
      - Secret Key: Kept locally, used for encryption/decryption
      - Evaluation Keys: Uploaded to server, used for computation
    """
    pass


@keys.command()
@click.option("--output", "-o", type=click.Path(), help="Output directory for keys")
@click.option("--ring-dimension", "-n", default=4096, help="Ring dimension (2048 or 4096)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing keys")
@click.pass_context
def generate(
    ctx: click.Context,
    output: Optional[str],
    ring_dimension: int,
    force: bool,
):
    """Generate new FHE key pair.

    \b
    This creates:
      - secret.key: Your private key (keep secure!)
      - eval.key: Evaluation keys (upload to server)

    \b
    Examples:
      fhe-gbdt keys generate
      fhe-gbdt keys generate -o ./my-keys
      fhe-gbdt keys generate --ring-dimension 2048
    """
    config = ctx.obj.get("config")
    verbose = ctx.obj.get("verbose", False)

    # Determine output directory
    if output:
        key_dir = Path(output)
    elif config:
        key_dir = Path(config.key_directory)
    else:
        key_dir = Path.home() / ".fhe-gbdt" / "keys"

    key_dir.mkdir(parents=True, exist_ok=True)

    secret_path = key_dir / "secret.key"
    eval_path = key_dir / "eval.key"

    # Check for existing keys
    if secret_path.exists() and not force:
        print_error(f"Keys already exist at {key_dir}")
        console.print("Use --force to overwrite, or specify different --output")
        sys.exit(1)

    if ring_dimension not in [2048, 4096]:
        print_error("Ring dimension must be 2048 or 4096")
        sys.exit(1)

    try:
        with console.status("Generating FHE keys..."):
            # In a real implementation, this would call the crypto library
            # For now, we simulate key generation
            import secrets
            import hashlib

            # Generate deterministic seed for demo
            seed = secrets.token_bytes(32)

            # Create placeholder keys (real implementation uses N2HE)
            secret_key = hashlib.sha256(seed + b"secret").digest() * (ring_dimension // 8)
            eval_key = hashlib.sha256(seed + b"eval").digest() * (ring_dimension * 2)

            # Save keys
            secret_path.write_bytes(secret_key)
            eval_path.write_bytes(eval_key)

            # Set permissions (secret key should be read-only)
            secret_path.chmod(0o600)
            eval_path.chmod(0o644)

        print_success("Keys generated successfully!")
        console.print(f"\n  Secret Key: {secret_path}")
        console.print(f"  Eval Keys: {eval_path}")
        console.print(f"\n[yellow]Important:[/yellow] Keep your secret key secure!")
        console.print("\nNext step: Upload evaluation keys to the server:")
        console.print(f"  fhe-gbdt keys upload {eval_path}")

    except Exception as e:
        print_error(f"Key generation failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@keys.command()
@click.argument("key_file", type=click.Path(exists=True), required=False)
@click.option("--key-dir", "-d", type=click.Path(exists=True), help="Directory containing eval.key")
@click.pass_context
def upload(ctx: click.Context, key_file: Optional[str], key_dir: Optional[str]):
    """Upload evaluation keys to the server.

    \b
    The evaluation keys allow the server to perform computations
    on encrypted data without seeing the plaintext.

    \b
    Examples:
      fhe-gbdt keys upload
      fhe-gbdt keys upload ./my-keys/eval.key
      fhe-gbdt keys upload --key-dir ./my-keys
    """
    config = ctx.obj.get("config")
    verbose = ctx.obj.get("verbose", False)

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    # Find eval key file
    if key_file:
        eval_path = Path(key_file)
    elif key_dir:
        eval_path = Path(key_dir) / "eval.key"
    elif config:
        eval_path = Path(config.key_directory) / "eval.key"
    else:
        eval_path = Path.home() / ".fhe-gbdt" / "keys" / "eval.key"

    if not eval_path.exists():
        print_error(f"Evaluation keys not found at {eval_path}")
        console.print("Generate keys first with: fhe-gbdt keys generate")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)

        eval_keys = eval_path.read_bytes()
        size_mb = len(eval_keys) / (1024 * 1024)

        with console.status(f"Uploading evaluation keys ({size_mb:.1f} MB)..."):
            result = client.upload_eval_keys(eval_keys)

        print_success("Evaluation keys uploaded!")
        console.print(f"\n  Key ID: {result.get('key_id', 'N/A')}")
        console.print(f"  Expires: {result.get('expires_at', 'N/A')}")

    except Exception as e:
        print_error(f"Upload failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@keys.command()
@click.pass_context
def status(ctx: click.Context):
    """Check status of uploaded keys.

    \b
    Example:
      fhe-gbdt keys status
    """
    config = ctx.obj.get("config")
    output_format = ctx.obj.get("output_format", "table")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    try:
        from ..client import create_client

        client = create_client(config)
        result = client.get_key_status()

        if output_format == "json":
            console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print("\n[bold]Key Status[/bold]")
            console.print(f"  Key ID: {result.get('key_id', 'N/A')}")
            console.print(f"  Status: {result.get('status', 'N/A')}")
            console.print(f"  Uploaded: {result.get('uploaded_at', 'N/A')}")
            console.print(f"  Expires: {result.get('expires_at', 'N/A')}")

            if result.get("status") == "active":
                print_success("\nKeys are ready for inference!")
            else:
                print_warning("\nKeys need to be uploaded.")

    except Exception as e:
        print_error(f"Failed to get key status: {e}")
        sys.exit(1)


@keys.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def rotate(ctx: click.Context, force: bool):
    """Rotate evaluation keys.

    \b
    This generates new keys and uploads them, invalidating old keys.

    \b
    Example:
      fhe-gbdt keys rotate
    """
    config = ctx.obj.get("config")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    if not force:
        print_warning("Key rotation will:")
        console.print("  1. Generate new secret and evaluation keys")
        console.print("  2. Upload new evaluation keys")
        console.print("  3. Invalidate your old keys")
        console.print("\nYou will need to re-encrypt any stored data.")

        if not click.confirm("Proceed with key rotation?"):
            console.print("Cancelled.")
            return

    # Generate new keys
    ctx.invoke(generate, force=True)

    # Upload new keys
    ctx.invoke(upload)

    print_success("Key rotation complete!")


@keys.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def revoke(ctx: click.Context, force: bool):
    """Revoke evaluation keys from the server.

    \b
    This removes your evaluation keys from the server.
    Encrypted predictions will no longer be possible until
    new keys are uploaded.

    \b
    Example:
      fhe-gbdt keys revoke
    """
    config = ctx.obj.get("config")

    if not config or not config.api_key:
        print_error("Not configured. Run 'fhe-gbdt config init' first.")
        sys.exit(1)

    if not force:
        print_warning("This will revoke your evaluation keys.")
        console.print("You will not be able to make predictions until new keys are uploaded.")

        if not click.confirm("Revoke keys?"):
            console.print("Cancelled.")
            return

    try:
        from ..client import create_client

        client = create_client(config)
        client.revoke_keys()
        print_success("Keys revoked.")

    except Exception as e:
        print_error(f"Failed to revoke keys: {e}")
        sys.exit(1)
