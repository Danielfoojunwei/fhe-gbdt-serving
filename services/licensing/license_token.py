"""
License Token Module for FHE-GBDT Serving

Issues and validates time-bound, metered JWT license tokens that on-prem
runtime instances require to execute predictions.

Token claims:
- tenant_id: The licensed tenant
- model_ids: List of compiled model IDs authorized for this license
- max_predictions: Maximum prediction count before renewal required
- exp: Expiration timestamp (UTC)
- iat: Issued-at timestamp (UTC)
- license_id: Unique license identifier for audit trail
"""

import hashlib
import hmac
import json
import time
import uuid
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class LicenseClaims:
    """Claims embedded in a license token."""
    tenant_id: str
    license_id: str
    model_ids: List[str]
    max_predictions: int
    issued_at: float
    expires_at: float
    features: List[str] = field(default_factory=lambda: ["predict"])

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def allows_model(self, model_id: str) -> bool:
        if "*" in self.model_ids:
            return True
        return model_id in self.model_ids

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LicenseClaims":
        return cls(
            tenant_id=data["tenant_id"],
            license_id=data["license_id"],
            model_ids=data["model_ids"],
            max_predictions=data["max_predictions"],
            issued_at=data["issued_at"],
            expires_at=data["expires_at"],
            features=data.get("features", ["predict"]),
        )


class LicenseTokenError(Exception):
    """Base exception for license token operations."""
    pass


class LicenseExpiredError(LicenseTokenError):
    pass


class LicenseInvalidError(LicenseTokenError):
    pass


class LicenseModelNotAuthorizedError(LicenseTokenError):
    pass


class LicensePredictionCapExceededError(LicenseTokenError):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


class LicenseIssuer:
    """
    Issues signed license tokens. Runs on the vendor's cloud (SaaS control plane).

    Tokens are HMAC-SHA256 signed JWTs. In production, this would use
    RSA/EC asymmetric keys so the on-prem runtime can verify without
    holding the signing key. HMAC is used here for simplicity.
    """

    def __init__(self, signing_key: str):
        if not signing_key or len(signing_key) < 32:
            raise ValueError("Signing key must be at least 32 characters")
        self._key = signing_key.encode("utf-8")

    def issue(
        self,
        tenant_id: str,
        model_ids: List[str],
        max_predictions: int = 1_000_000,
        validity_hours: int = 72,
        features: Optional[List[str]] = None,
    ) -> str:
        """Issue a signed license token."""
        now = time.time()
        claims = LicenseClaims(
            tenant_id=tenant_id,
            license_id=str(uuid.uuid4()),
            model_ids=model_ids,
            max_predictions=max_predictions,
            issued_at=now,
            expires_at=now + (validity_hours * 3600),
            features=features or ["predict"],
        )
        return self._sign(claims)

    def _sign(self, claims: LicenseClaims) -> str:
        header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = _b64url_encode(json.dumps(claims.to_dict()).encode())
        signing_input = f"{header}.{payload}"
        signature = hmac.new(
            self._key, signing_input.encode("utf-8"), hashlib.sha256
        ).digest()
        return f"{signing_input}.{_b64url_encode(signature)}"


class LicenseValidator:
    """
    Validates license tokens. Runs on the customer's on-prem gateway/runtime.

    Tracks prediction counts per license to enforce metering caps.
    Supports offline grace periods when the license server is unreachable.
    """

    def __init__(self, signing_key: str, offline_grace_hours: int = 72):
        if not signing_key or len(signing_key) < 32:
            raise ValueError("Signing key must be at least 32 characters")
        self._key = signing_key.encode("utf-8")
        self._offline_grace_hours = offline_grace_hours
        self._prediction_counts: Dict[str, int] = {}
        self._cached_claims: Optional[LicenseClaims] = None
        self._cache_time: float = 0.0
        self._last_successful_validation: float = 0.0

    def validate(self, token: str, model_id: Optional[str] = None) -> LicenseClaims:
        """
        Validate a license token and return its claims.

        Raises LicenseTokenError subclasses on failure.
        """
        claims = self._verify_signature(token)

        # Check expiration with offline grace period
        now = time.time()
        if claims.is_expired():
            # Allow offline grace period if we had a recent successful validation
            grace_deadline = self._last_successful_validation + (
                self._offline_grace_hours * 3600
            )
            if now > grace_deadline or self._last_successful_validation == 0.0:
                raise LicenseExpiredError(
                    f"License {claims.license_id} expired at "
                    f"{claims.expires_at}, grace period exhausted"
                )

        # Check model authorization
        if model_id and not claims.allows_model(model_id):
            raise LicenseModelNotAuthorizedError(
                f"Model {model_id} not authorized under license {claims.license_id}"
            )

        # Check prediction cap
        current_count = self._prediction_counts.get(claims.license_id, 0)
        if current_count >= claims.max_predictions:
            raise LicensePredictionCapExceededError(
                f"License {claims.license_id} prediction cap "
                f"({claims.max_predictions}) exceeded"
            )

        # Track successful validation
        self._last_successful_validation = now
        self._cached_claims = claims
        self._cache_time = now

        return claims

    def record_prediction(self, license_id: str) -> int:
        """Record a prediction and return the new count."""
        count = self._prediction_counts.get(license_id, 0) + 1
        self._prediction_counts[license_id] = count
        return count

    def get_prediction_count(self, license_id: str) -> int:
        """Get the current prediction count for a license."""
        return self._prediction_counts.get(license_id, 0)

    def get_cached_claims(self) -> Optional[LicenseClaims]:
        """Return cached claims for offline operation."""
        if self._cached_claims is None:
            return None
        # Only return cached claims within grace period
        elapsed = time.time() - self._cache_time
        if elapsed > self._offline_grace_hours * 3600:
            return None
        return self._cached_claims

    def _verify_signature(self, token: str) -> LicenseClaims:
        """Verify HMAC signature and decode claims."""
        parts = token.split(".")
        if len(parts) != 3:
            raise LicenseInvalidError("Malformed token: expected 3 parts")

        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            self._key, signing_input.encode("utf-8"), hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise LicenseInvalidError("Invalid token signature")

        # Decode claims
        try:
            payload = json.loads(_b64url_decode(payload_b64))
            return LicenseClaims.from_dict(payload)
        except (json.JSONDecodeError, KeyError) as e:
            raise LicenseInvalidError(f"Invalid token payload: {e}")
