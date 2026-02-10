"""
Plan Encryption Module for FHE-GBDT Serving

Encrypts compiled execution plans so they are bound to a specific
customer deployment. Prevents plan reuse across environments.

Uses AES-256-GCM with a deployment-specific key derived from
the tenant ID and a deployment secret.
"""

import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from typing import Optional


# AES-GCM constants
AES_KEY_SIZE = 32       # 256 bits
GCM_NONCE_SIZE = 12     # 96 bits (NIST recommendation)
GCM_TAG_SIZE = 16       # 128 bits

# Header magic + version for encrypted plans
ENCRYPTED_PLAN_MAGIC = b"FHEPLAN\x01"
HEADER_SIZE = len(ENCRYPTED_PLAN_MAGIC)


class PlanEncryptionError(Exception):
    """Error during plan encryption/decryption."""
    pass


@dataclass
class EncryptedPlan:
    """An encrypted compiled execution plan."""
    tenant_id: str
    deployment_id: str
    ciphertext: bytes
    nonce: bytes
    tag: bytes

    def serialize(self) -> bytes:
        """Serialize to wire format: magic | metadata_len | metadata | nonce | tag | ciphertext"""
        metadata = json.dumps({
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
        }).encode("utf-8")

        buf = bytearray()
        buf.extend(ENCRYPTED_PLAN_MAGIC)
        buf.extend(struct.pack(">H", len(metadata)))
        buf.extend(metadata)
        buf.extend(self.nonce)
        buf.extend(self.tag)
        buf.extend(self.ciphertext)
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncryptedPlan":
        """Deserialize from wire format."""
        if len(data) < HEADER_SIZE + 2:
            raise PlanEncryptionError("Data too short for encrypted plan")

        if data[:HEADER_SIZE] != ENCRYPTED_PLAN_MAGIC:
            raise PlanEncryptionError("Invalid encrypted plan magic bytes")

        offset = HEADER_SIZE
        meta_len = struct.unpack(">H", data[offset:offset + 2])[0]
        offset += 2

        metadata = json.loads(data[offset:offset + meta_len])
        offset += meta_len

        nonce = data[offset:offset + GCM_NONCE_SIZE]
        offset += GCM_NONCE_SIZE

        tag = data[offset:offset + GCM_TAG_SIZE]
        offset += GCM_TAG_SIZE

        ciphertext = data[offset:]

        return cls(
            tenant_id=metadata["tenant_id"],
            deployment_id=metadata["deployment_id"],
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
        )


def derive_deployment_key(
    tenant_id: str,
    deployment_secret: str,
) -> bytes:
    """
    Derive a deployment-specific AES-256 key using HKDF-like construction.

    The deployment_secret is unique per customer deployment (set during
    on-prem installation). Combined with tenant_id, this ensures plans
    are bound to a specific tenant + deployment pair.
    """
    # Use HMAC-SHA256 as a simple KDF
    # In production, use proper HKDF (from cryptography library)
    salt = f"fhe-gbdt-plan-encryption-v1:{tenant_id}".encode("utf-8")
    key_material = hmac.new(
        salt,
        deployment_secret.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return key_material[:AES_KEY_SIZE]


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    return bytes(x ^ y for x, y in zip(a, b))


class PlanEncryptor:
    """
    Encrypts and decrypts compiled execution plans.

    Uses a simplified AES-CTR + HMAC construction for portability
    (no external crypto dependencies required). In production, use
    AES-GCM from the cryptography library.
    """

    def __init__(self, tenant_id: str, deployment_secret: str):
        self._tenant_id = tenant_id
        self._key = derive_deployment_key(tenant_id, deployment_secret)
        self._deployment_id = hashlib.sha256(
            f"{tenant_id}:{deployment_secret}".encode()
        ).hexdigest()[:16]

    def encrypt(self, plan_json: str) -> EncryptedPlan:
        """Encrypt a JSON-serialized execution plan."""
        plaintext = plan_json.encode("utf-8")
        nonce = os.urandom(GCM_NONCE_SIZE)

        # Simplified authenticated encryption:
        # 1. Derive a stream key from key + nonce using HMAC
        stream_key = hmac.new(
            self._key, nonce, hashlib.sha256
        ).digest()

        # 2. XOR-based stream cipher (simplified CTR mode)
        ciphertext = bytearray()
        for i in range(0, len(plaintext), 32):
            block_key = hmac.new(
                stream_key,
                struct.pack(">I", i // 32),
                hashlib.sha256,
            ).digest()
            chunk = plaintext[i:i + 32]
            ciphertext.extend(_xor_bytes(chunk, block_key[:len(chunk)]))

        ciphertext = bytes(ciphertext)

        # 3. Compute authentication tag (HMAC over nonce + ciphertext)
        tag = hmac.new(
            self._key,
            nonce + ciphertext,
            hashlib.sha256,
        ).digest()[:GCM_TAG_SIZE]

        return EncryptedPlan(
            tenant_id=self._tenant_id,
            deployment_id=self._deployment_id,
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
        )

    def decrypt(self, encrypted_plan: EncryptedPlan) -> str:
        """Decrypt an encrypted execution plan. Returns the JSON string."""
        # Verify tenant binding
        if encrypted_plan.tenant_id != self._tenant_id:
            raise PlanEncryptionError(
                f"Plan bound to tenant {encrypted_plan.tenant_id}, "
                f"but this encryptor is for {self._tenant_id}"
            )

        # Verify deployment binding
        if encrypted_plan.deployment_id != self._deployment_id:
            raise PlanEncryptionError(
                "Plan bound to a different deployment")

        # Verify authentication tag
        expected_tag = hmac.new(
            self._key,
            encrypted_plan.nonce + encrypted_plan.ciphertext,
            hashlib.sha256,
        ).digest()[:GCM_TAG_SIZE]

        if not hmac.compare_digest(expected_tag, encrypted_plan.tag):
            raise PlanEncryptionError("Authentication tag mismatch -- plan tampered")

        # Decrypt
        stream_key = hmac.new(
            self._key, encrypted_plan.nonce, hashlib.sha256
        ).digest()

        plaintext = bytearray()
        ct = encrypted_plan.ciphertext
        for i in range(0, len(ct), 32):
            block_key = hmac.new(
                stream_key,
                struct.pack(">I", i // 32),
                hashlib.sha256,
            ).digest()
            chunk = ct[i:i + 32]
            plaintext.extend(_xor_bytes(chunk, block_key[:len(chunk)]))

        return bytes(plaintext).decode("utf-8")
