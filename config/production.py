"""
Production Configuration Module

This module provides production-ready configuration settings for the FHE-GBDT
serving system, following best practices from:
- Cloud Security Alliance FHE Guidelines
- ISO/NIST cryptographic standards
- OWASP security recommendations

Usage:
    from config.production import ProductionConfig
    config = ProductionConfig.from_env()
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityLevel(Enum):
    """Security level configurations."""
    STANDARD = "standard"      # 128-bit security
    HIGH = "high"              # 192-bit security
    MAXIMUM = "maximum"        # 256-bit security


@dataclass
class CryptoConfig:
    """Cryptographic configuration for FHE operations."""

    # RLWE parameters
    ring_dimension: int = 2048       # N: polynomial ring dimension
    ciphertext_modulus: int = 2**32  # q: ciphertext modulus
    plaintext_modulus: int = 2**16   # t: plaintext modulus
    gaussian_sigma: float = 3.2      # σ: Gaussian error standard deviation

    # Security parameters
    security_level: SecurityLevel = SecurityLevel.STANDARD

    # Performance tuning
    enable_batching: bool = True
    batch_size: int = 1024
    enable_key_switching: bool = True
    bootstrap_threshold: int = 5     # Noise budget before bootstrapping

    # Key management
    key_rotation_days: int = 90
    eval_key_cache_size: int = 100

    @classmethod
    def for_security_level(cls, level: SecurityLevel) -> "CryptoConfig":
        """Get crypto config for a security level."""
        configs = {
            SecurityLevel.STANDARD: cls(
                ring_dimension=2048,
                ciphertext_modulus=2**32,
            ),
            SecurityLevel.HIGH: cls(
                ring_dimension=4096,
                ciphertext_modulus=2**64,
            ),
            SecurityLevel.MAXIMUM: cls(
                ring_dimension=8192,
                ciphertext_modulus=2**128,
            ),
        }
        return configs.get(level, configs[SecurityLevel.STANDARD])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: int = 100
    burst_size: int = 200
    per_tenant: bool = True
    enable_adaptive: bool = False
    backoff_factor: float = 1.5


@dataclass
class TLSConfig:
    """TLS/mTLS configuration."""
    enabled: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    require_client_cert: bool = True
    min_version: str = "TLS1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256",
    ])


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration."""
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_audit_logs: bool = True

    # Tracing
    otel_endpoint: Optional[str] = None
    trace_sample_rate: float = 0.1

    # Metrics
    prometheus_port: int = 9090
    metrics_prefix: str = "fhe_gbdt"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    redact_sensitive_data: bool = True


@dataclass
class ModelValidationConfig:
    """Model validation rules for FHE compatibility."""
    max_tree_depth: int = 10
    max_num_trees: int = 100
    max_num_features: int = 1000
    max_model_size_mb: int = 100

    allowed_model_types: List[str] = field(default_factory=lambda: [
        "xgboost",
        "lightgbm",
        "catboost",
    ])

    # Performance constraints
    max_inference_latency_ms: int = 1000
    max_batch_size: int = 1000


@dataclass
class ProductionConfig:
    """Complete production configuration."""

    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION

    # Service configuration
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080
    runtime_host: str = "localhost"
    runtime_port: int = 9000
    registry_host: str = "localhost"
    registry_port: int = 8081
    keystore_host: str = "localhost"
    keystore_port: int = 8082

    # Sub-configurations
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    model_validation: ModelValidationConfig = field(default_factory=ModelValidationConfig)

    # Request limits
    max_request_size_mb: int = 64
    request_timeout_seconds: int = 30

    # Health check
    health_check_interval_seconds: int = 10
    health_check_timeout_seconds: int = 5

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Load configuration from environment variables."""
        env = os.getenv("DEPLOYMENT_ENV", "production").lower()
        environment = DeploymentEnvironment(env)

        config = cls(environment=environment)

        # Override from environment
        config.gateway_host = os.getenv("GATEWAY_HOST", config.gateway_host)
        config.gateway_port = int(os.getenv("GATEWAY_PORT", config.gateway_port))
        config.runtime_host = os.getenv("RUNTIME_HOST", config.runtime_host)
        config.runtime_port = int(os.getenv("RUNTIME_PORT", config.runtime_port))

        # TLS configuration
        config.tls.cert_file = os.getenv("MTLS_CERT_FILE")
        config.tls.key_file = os.getenv("MTLS_KEY_FILE")
        config.tls.ca_file = os.getenv("MTLS_CA_FILE")
        config.tls.enabled = config.tls.cert_file is not None

        # Observability
        config.observability.otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        config.observability.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Apply environment-specific defaults
        if environment == DeploymentEnvironment.DEVELOPMENT:
            config = cls._apply_dev_defaults(config)
        elif environment == DeploymentEnvironment.STAGING:
            config = cls._apply_staging_defaults(config)
        else:
            config = cls._apply_prod_defaults(config)

        return config

    @staticmethod
    def _apply_dev_defaults(config: "ProductionConfig") -> "ProductionConfig":
        """Apply development environment defaults."""
        config.tls.enabled = False
        config.tls.require_client_cert = False
        config.rate_limit.requests_per_second = 1000
        config.observability.trace_sample_rate = 1.0
        config.observability.log_level = "DEBUG"
        return config

    @staticmethod
    def _apply_staging_defaults(config: "ProductionConfig") -> "ProductionConfig":
        """Apply staging environment defaults."""
        config.observability.trace_sample_rate = 0.5
        config.observability.log_level = "DEBUG"
        return config

    @staticmethod
    def _apply_prod_defaults(config: "ProductionConfig") -> "ProductionConfig":
        """Apply production environment defaults."""
        config.tls.enabled = True
        config.tls.require_client_cert = True
        config.observability.redact_sensitive_data = True
        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        # TLS validation
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if not self.tls.enabled:
                issues.append("ERROR: TLS must be enabled in production")
            if not self.tls.require_client_cert:
                issues.append("WARNING: mTLS should be required in production")

        # Crypto validation
        if self.crypto.ring_dimension < 2048:
            issues.append("ERROR: Ring dimension must be >= 2048 for security")

        # Rate limit validation
        if self.rate_limit.requests_per_second > 10000:
            issues.append("WARNING: Very high rate limit may impact system stability")

        # Model validation
        if self.model_validation.max_tree_depth > 15:
            issues.append("WARNING: Tree depth > 15 may cause slow FHE inference")

        return issues

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary (for logging/debugging)."""
        return {
            "environment": self.environment.value,
            "gateway": f"{self.gateway_host}:{self.gateway_port}",
            "tls_enabled": self.tls.enabled,
            "crypto_security_level": self.crypto.security_level.value,
            "rate_limit": self.rate_limit.requests_per_second,
            "max_request_size_mb": self.max_request_size_mb,
        }


def validate_and_log_config() -> ProductionConfig:
    """Load, validate, and log configuration."""
    config = ProductionConfig.from_env()

    # Validate
    issues = config.validate()
    for issue in issues:
        if issue.startswith("ERROR"):
            logger.error(issue)
        else:
            logger.warning(issue)

    # Log summary (without sensitive data)
    logger.info(f"Configuration loaded: {config.to_dict()}")

    return config
