"""
Compiler Service - Hosted SaaS Control Plane

Provides authenticated, multi-tenant model compilation endpoints.
This service runs on the vendor's cloud and never touches customer data --
it only processes model structure (tree topology, thresholds, leaf values).

Endpoints:
    POST /v1/compile       - Compile a GBDT model to FHE-optimized plan
    POST /v1/license/issue - Issue a license token for on-prem runtime
    GET  /health           - Health check
    GET  /v1/license/validate - Validate a license token (for testing)
"""

import hashlib
import json
import logging
import os
import time
import functools
from flask import Flask, jsonify, request, abort

from .compiler import Compiler
from .ir import ObliviousPlanIR

# License and plan encryption modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from licensing.license_token import LicenseIssuer, LicenseValidator, LicenseTokenError
from licensing.plan_encryption import PlanEncryptor

logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---

LICENSE_SIGNING_KEY = os.getenv(
    "LICENSE_SIGNING_KEY",
    "dev-signing-key-change-in-production-minimum-32-chars",
)

# Tenant API keys: tenant_id -> api_key (in production, backed by Vault/DB)
TENANT_API_KEYS = {}

# Pre-populate from environment: TENANT_KEY_<id>=<secret>
for key, value in os.environ.items():
    if key.startswith("TENANT_KEY_"):
        tenant_id = key[len("TENANT_KEY_"):].lower().replace("_", "-")
        TENANT_API_KEYS[tenant_id] = value

# Always allow test tenants in development
if os.getenv("DEPLOYMENT_ENV", "development") == "development":
    TENANT_API_KEYS.setdefault("test-tenant", "test-secret")
    TENANT_API_KEYS.setdefault("demo-tenant", "demo-secret")
    TENANT_API_KEYS.setdefault("integration-test", "test-secret")

# --- Singletons ---

compiler_instance = Compiler()
license_issuer = LicenseIssuer(LICENSE_SIGNING_KEY)
license_validator = LicenseValidator(LICENSE_SIGNING_KEY)


# --- Auth Middleware ---

def require_tenant_auth(f):
    """Decorator that validates API key and injects tenant_id."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key", "")
        parts = api_key.split(".", 1)
        if len(parts) != 2:
            abort(401, description="Missing or malformed API key (expected tenant_id.secret)")

        tenant_id, secret = parts
        expected = TENANT_API_KEYS.get(tenant_id)
        if expected is None or expected != secret:
            logger.warning("AUTH_FAILURE: tenant=%s", tenant_id)
            abort(403, description="Invalid API key")

        logger.info("AUTH_SUCCESS: tenant=%s endpoint=%s", tenant_id, request.path)
        return f(tenant_id=tenant_id, *args, **kwargs)
    return decorated


# --- Endpoints ---

@app.route("/health")
def health():
    return jsonify({"status": "OK", "service": "compiler"})


@app.route("/v1/compile", methods=["POST"])
@require_tenant_auth
def compile_model(tenant_id: str):
    """
    Compile a GBDT model to an FHE-optimized execution plan.

    Request JSON:
        model_content: base64-encoded model file content
        library_type: "xgboost" | "lightgbm" | "catboost"
        profile: "latency" | "throughput"
        deployment_secret: (optional) encrypt the plan for a specific deployment

    Returns:
        compiled_plan: JSON execution plan (or encrypted bytes if deployment_secret provided)
        compiled_model_id: unique ID for this compiled plan
        plan_hash: SHA-256 hash for integrity verification
    """
    data = request.get_json(force=True)

    model_content = data.get("model_content")
    library_type = data.get("library_type", "xgboost")
    profile = data.get("profile", "latency")
    deployment_secret = data.get("deployment_secret")

    if not model_content:
        abort(400, description="model_content is required")

    # Decode content
    import base64
    try:
        content_bytes = base64.b64decode(model_content)
    except Exception:
        # Try as raw JSON string
        content_bytes = model_content.encode("utf-8") if isinstance(model_content, str) else model_content

    # Compile
    try:
        plan = compiler_instance.compile(content_bytes, library_type, profile)
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        logger.error("COMPILE_ERROR: tenant=%s error=%s", tenant_id, e)
        abort(500, description="Compilation failed")

    plan_json = plan.to_json()
    plan_hash = hashlib.sha256(plan_json.encode()).hexdigest()

    response = {
        "compiled_model_id": plan.compiled_model_id,
        "plan_hash": plan_hash,
        "tenant_id": tenant_id,
        "profile": profile,
        "library_type": library_type,
        "num_trees": plan.num_trees,
    }

    # Optionally encrypt the plan for a specific deployment
    if deployment_secret:
        encryptor = PlanEncryptor(tenant_id, deployment_secret)
        encrypted = encryptor.encrypt(plan_json)
        response["encrypted_plan"] = encrypted.serialize().hex()
        response["deployment_id"] = encrypted.deployment_id
        response["encrypted"] = True
    else:
        response["compiled_plan"] = json.loads(plan_json)
        response["encrypted"] = False

    logger.info(
        "COMPILE_SUCCESS: tenant=%s model_id=%s trees=%d encrypted=%s",
        tenant_id, plan.compiled_model_id, plan.num_trees, response["encrypted"],
    )
    return jsonify(response)


@app.route("/v1/license/issue", methods=["POST"])
@require_tenant_auth
def issue_license(tenant_id: str):
    """
    Issue a license token for the on-prem runtime.

    Request JSON:
        model_ids: list of compiled model IDs to authorize
        max_predictions: (optional) prediction cap (default: 1,000,000)
        validity_hours: (optional) token validity in hours (default: 72)

    Returns:
        token: signed JWT license token
        license_id: unique license identifier
        expires_at: expiration timestamp
    """
    data = request.get_json(force=True)

    model_ids = data.get("model_ids", ["*"])
    max_predictions = data.get("max_predictions", 1_000_000)
    validity_hours = data.get("validity_hours", 72)

    if not isinstance(model_ids, list) or not model_ids:
        abort(400, description="model_ids must be a non-empty list")

    token = license_issuer.issue(
        tenant_id=tenant_id,
        model_ids=model_ids,
        max_predictions=max_predictions,
        validity_hours=validity_hours,
    )

    # Decode to get license_id and expires_at for the response
    claims = license_validator.validate(token)

    logger.info(
        "LICENSE_ISSUED: tenant=%s license_id=%s models=%s max_pred=%d hours=%d",
        tenant_id, claims.license_id, model_ids, max_predictions, validity_hours,
    )

    return jsonify({
        "token": token,
        "license_id": claims.license_id,
        "tenant_id": tenant_id,
        "model_ids": model_ids,
        "max_predictions": max_predictions,
        "expires_at": claims.expires_at,
    })


@app.route("/v1/license/validate", methods=["POST"])
@require_tenant_auth
def validate_license(tenant_id: str):
    """
    Validate a license token (for testing/debugging).

    Request JSON:
        token: the license token to validate
        model_id: (optional) check if a specific model is authorized

    Returns:
        valid: bool
        claims: decoded token claims (if valid)
        error: error message (if invalid)
    """
    data = request.get_json(force=True)
    token = data.get("token", "")
    model_id = data.get("model_id")

    try:
        claims = license_validator.validate(token, model_id=model_id)
        if claims.tenant_id != tenant_id:
            return jsonify({
                "valid": False,
                "error": "Token tenant_id does not match authenticated tenant",
            }), 403

        return jsonify({
            "valid": True,
            "claims": claims.to_dict(),
        })
    except LicenseTokenError as e:
        return jsonify({
            "valid": False,
            "error": str(e),
        }), 400


# --- App Runner ---

if __name__ == "__main__":
    port = int(os.getenv("COMPILER_PORT", "5000"))
    debug = os.getenv("DEPLOYMENT_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
