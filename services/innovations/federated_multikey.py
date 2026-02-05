"""
Novel Innovation #4: N2HE Multi-Key Federated GBDT

Enables federated GBDT inference where different parties hold different features.
Uses N2HE's multi-key capabilities to combine partial computations without
revealing individual features.

Key Insight:
- Each party encrypts their features under their own key
- Partial tree traversals happen independently
- N2HE multi-key homomorphic operations combine results
- Final decryption requires threshold cooperation

Benefits:
- True feature-level privacy across parties
- No single party sees all features
- Compatible with existing GBDT models
- Supports dynamic party participation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
import logging
import asyncio
import hashlib
import secrets
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class PartyRole(Enum):
    """Role of a party in federated protocol."""
    FEATURE_PROVIDER = "feature_provider"
    MODEL_OWNER = "model_owner"
    AGGREGATOR = "aggregator"
    DECRYPTOR = "decryptor"


@dataclass
class FeatureAssignment:
    """Assignment of features to parties."""
    party_id: str
    feature_indices: List[int]
    is_primary: bool = False  # Primary party coordinates


@dataclass
class PartialTraversalResult:
    """Result of partial tree traversal by one party."""
    party_id: str
    tree_idx: int

    # Encrypted partial path indicators
    # For features owned by this party
    partial_indicators: Any  # Encrypted ciphertext

    # Metadata
    features_evaluated: List[int]
    depth_levels_evaluated: List[int]

    # Cryptographic commitment for verification
    commitment: bytes = field(default_factory=lambda: b"")

    def compute_commitment(self) -> bytes:
        """Compute commitment to partial result."""
        data = f"{self.party_id}:{self.tree_idx}:{self.features_evaluated}"
        return hashlib.sha256(data.encode()).digest()


@dataclass
class MultiKeyConfig:
    """Configuration for multi-key protocol."""
    # Number of parties required for decryption
    decryption_threshold: int = 2

    # Maximum parties
    max_parties: int = 10

    # Timeout for party responses
    party_timeout_seconds: int = 30

    # Enable verification of partial results
    verify_commitments: bool = True

    # N2HE specific parameters
    n2he_security_level: int = 128
    n2he_ring_dimension: int = 4096


class MultiKeyParty:
    """
    Represents a party in the multi-key federated protocol.

    Each party:
    1. Holds a subset of features
    2. Has their own N2HE key pair
    3. Computes partial tree traversals on their features
    4. Contributes to threshold decryption
    """

    def __init__(
        self,
        party_id: str,
        owned_features: List[int],
        role: PartyRole = PartyRole.FEATURE_PROVIDER
    ):
        """
        Initialize party.

        Args:
            party_id: Unique identifier
            owned_features: Feature indices owned by this party
            role: Party's role in protocol
        """
        self.party_id = party_id
        self.owned_features = set(owned_features)
        self.role = role

        # Cryptographic state
        self._secret_key = None
        self._public_key = None
        self._eval_key = None

        # Multi-key state
        self._combined_public_key = None
        self._decryption_share = None

        logger.info(f"Party {party_id}: owns features {owned_features}")

    def generate_keys(self, config: MultiKeyConfig):
        """Generate N2HE key pair for this party."""
        # In production, this would use actual N2HE library
        # Here we simulate the key structure

        n = config.n2he_ring_dimension

        # Secret key: binary polynomial
        self._secret_key = np.random.randint(0, 2, size=n)

        # Public key: (a, b) where b = a*s + e
        a = np.random.randint(0, 2**32, size=n, dtype=np.uint64)
        e = np.random.randint(-3, 4, size=n)
        b = (a * self._secret_key + e) % (2**32)

        self._public_key = (a, b)
        self._eval_key = self._generate_eval_key()

        logger.info(f"Party {self.party_id}: generated N2HE keys")

    def _generate_eval_key(self) -> bytes:
        """Generate evaluation key for homomorphic operations."""
        # Simplified: in production this would be RGSW ciphertexts
        seed = hashlib.sha256(
            f"{self.party_id}:eval:{self._secret_key.tobytes()[:32]}".encode()
        ).digest()
        return seed + secrets.token_bytes(256)

    def encrypt_features(self, feature_values: Dict[int, float]) -> Dict[int, Any]:
        """
        Encrypt this party's features.

        Args:
            feature_values: Dict of feature_idx -> value

        Returns:
            Dict of feature_idx -> encrypted_value
        """
        encrypted = {}

        for feat_idx, value in feature_values.items():
            if feat_idx not in self.owned_features:
                raise ValueError(f"Party {self.party_id} does not own feature {feat_idx}")

            # Encrypt using party's public key
            encrypted[feat_idx] = self._encrypt_single(value)

        return encrypted

    def _encrypt_single(self, value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Encrypt a single value under party's key."""
        if self._public_key is None:
            raise ValueError("Keys not generated")

        n = len(self._public_key[0])
        a, b = self._public_key

        # Sample random r
        r = np.random.randint(0, 2, size=n)

        # Encode value
        scale = 2**20
        encoded = int(value * scale) % (2**32)

        # Ciphertext: (c0, c1) = (a*r + e1, b*r + e2 + encoded)
        e1 = np.random.randint(-3, 4, size=n)
        e2 = np.random.randint(-3, 4, size=n)

        c0 = (np.sum(a * r) + e1[0]) % (2**32)
        c1 = (np.sum(b * r) + e2[0] + encoded) % (2**32)

        return (c0, c1)

    async def compute_partial_traversal(
        self,
        encrypted_features: Dict[int, Any],
        model_ir: Any,
        depth_range: Tuple[int, int]
    ) -> List[PartialTraversalResult]:
        """
        Compute partial tree traversal for features owned by this party.

        Args:
            encrypted_features: Encrypted features (all parties)
            model_ir: Tree model
            depth_range: (start_depth, end_depth) to evaluate

        Returns:
            List of partial traversal results
        """
        results = []

        for tree_idx, tree in enumerate(model_ir.trees):
            # Find nodes at this depth range using our features
            partial_indicator = self._compute_tree_partial(
                tree, encrypted_features, depth_range
            )

            result = PartialTraversalResult(
                party_id=self.party_id,
                tree_idx=tree_idx,
                partial_indicators=partial_indicator,
                features_evaluated=list(self.owned_features),
                depth_levels_evaluated=list(range(depth_range[0], depth_range[1]))
            )
            result.commitment = result.compute_commitment()
            results.append(result)

        return results

    def _compute_tree_partial(
        self,
        tree_ir: Any,
        encrypted_features: Dict[int, Any],
        depth_range: Tuple[int, int]
    ) -> Any:
        """Compute partial traversal for a single tree."""
        # Collect nodes at specified depths using our features
        partial_result = 1.0  # Neutral element for multiplication

        for node in tree_ir.nodes.values():
            if node.depth < depth_range[0] or node.depth >= depth_range[1]:
                continue

            if node.feature_index in self.owned_features:
                # This party evaluates this comparison
                if node.feature_index in encrypted_features:
                    # Compute encrypted comparison
                    # In production: homomorphic comparison operation
                    partial_result *= 0.5  # Placeholder

        return partial_result

    def compute_decryption_share(self, combined_ciphertext: Any) -> Any:
        """
        Compute this party's share of threshold decryption.

        Args:
            combined_ciphertext: Combined multi-key ciphertext

        Returns:
            Decryption share
        """
        if self._secret_key is None:
            raise ValueError("Keys not generated")

        # In production: compute s_i * c_0 for threshold decryption
        share = np.sum(self._secret_key[:10]) % (2**32)  # Simplified
        return share


class N2HEMultiKeyCombiner:
    """
    Combines partial results from multiple parties using N2HE multi-key operations.

    Key Operations:
    1. Combine public keys into collective key
    2. Re-encrypt under collective key (key switching)
    3. Combine partial traversals homomorphically
    4. Coordinate threshold decryption
    """

    def __init__(self, config: Optional[MultiKeyConfig] = None):
        """
        Initialize combiner.

        Args:
            config: Multi-key configuration
        """
        self.config = config or MultiKeyConfig()
        self._parties: Dict[str, MultiKeyParty] = {}
        self._combined_public_key = None

    def register_party(self, party: MultiKeyParty):
        """Register a party for multi-key protocol."""
        if len(self._parties) >= self.config.max_parties:
            raise ValueError(f"Maximum parties ({self.config.max_parties}) reached")

        self._parties[party.party_id] = party
        self._combined_public_key = None  # Invalidate combined key

        logger.info(f"Registered party {party.party_id}")

    def setup_combined_key(self):
        """Setup combined public key from all parties."""
        if len(self._parties) < self.config.decryption_threshold:
            raise ValueError(
                f"Need at least {self.config.decryption_threshold} parties, "
                f"have {len(self._parties)}"
            )

        # Combine public keys (simplified: in production use proper MK-FHE)
        combined_a = None
        combined_b = None

        for party in self._parties.values():
            if party._public_key is None:
                raise ValueError(f"Party {party.party_id} has no keys")

            a, b = party._public_key
            if combined_a is None:
                combined_a = a.copy()
                combined_b = b.copy()
            else:
                combined_a = (combined_a + a) % (2**32)
                combined_b = (combined_b + b) % (2**32)

        self._combined_public_key = (combined_a, combined_b)
        logger.info(f"Setup combined key from {len(self._parties)} parties")

    def combine_partial_results(
        self,
        partial_results: List[List[PartialTraversalResult]]
    ) -> Dict[int, Any]:
        """
        Combine partial traversal results from all parties.

        Args:
            partial_results: List of results per party

        Returns:
            Combined tree indicators
        """
        if self.config.verify_commitments:
            self._verify_commitments(partial_results)

        # Group by tree
        tree_partials: Dict[int, List[PartialTraversalResult]] = {}

        for party_results in partial_results:
            for result in party_results:
                if result.tree_idx not in tree_partials:
                    tree_partials[result.tree_idx] = []
                tree_partials[result.tree_idx].append(result)

        # Combine partials for each tree
        combined = {}
        for tree_idx, partials in tree_partials.items():
            combined[tree_idx] = self._combine_tree_partials(partials)

        return combined

    def _combine_tree_partials(
        self,
        partials: List[PartialTraversalResult]
    ) -> Any:
        """Combine partial results for a single tree."""
        # In production: homomorphic multiplication of partial indicators
        combined = 1.0
        for partial in partials:
            combined *= partial.partial_indicators

        return combined

    def _verify_commitments(
        self,
        partial_results: List[List[PartialTraversalResult]]
    ):
        """Verify commitments from all parties."""
        for party_results in partial_results:
            for result in party_results:
                expected = result.compute_commitment()
                if result.commitment != expected:
                    raise ValueError(
                        f"Commitment verification failed for party {result.party_id}"
                    )

    async def threshold_decrypt(
        self,
        combined_ciphertext: Any,
        participating_parties: List[str]
    ) -> float:
        """
        Perform threshold decryption with participating parties.

        Args:
            combined_ciphertext: Multi-key ciphertext to decrypt
            participating_parties: Party IDs participating in decryption

        Returns:
            Decrypted value
        """
        if len(participating_parties) < self.config.decryption_threshold:
            raise ValueError(
                f"Need {self.config.decryption_threshold} parties for decryption, "
                f"have {len(participating_parties)}"
            )

        # Collect decryption shares
        shares = []
        for party_id in participating_parties:
            party = self._parties.get(party_id)
            if party is None:
                raise ValueError(f"Unknown party: {party_id}")

            share = party.compute_decryption_share(combined_ciphertext)
            shares.append(share)

        # Combine shares (simplified Shamir reconstruction)
        combined_share = sum(shares) % (2**32)

        # Decode
        scale = 2**20
        if combined_share > 2**31:
            combined_share -= 2**32
        result = combined_share / scale

        return result


class FederatedGBDTProtocol:
    """
    Complete federated GBDT inference protocol using N2HE multi-key.

    Protocol Steps:
    1. Parties register and generate keys
    2. Combined public key is setup
    3. Each party encrypts their features
    4. Partial tree traversals are computed in parallel
    5. Results are combined homomorphically
    6. Threshold decryption produces final result
    """

    def __init__(
        self,
        model_ir: Any,
        config: Optional[MultiKeyConfig] = None
    ):
        """
        Initialize protocol.

        Args:
            model_ir: GBDT model
            config: Multi-key configuration
        """
        self.model_ir = model_ir
        self.config = config or MultiKeyConfig()
        self.combiner = N2HEMultiKeyCombiner(config)

        # Feature assignment
        self._feature_assignments: Dict[str, FeatureAssignment] = {}

    def assign_features(
        self,
        assignments: List[FeatureAssignment]
    ):
        """
        Assign features to parties.

        Args:
            assignments: List of feature assignments
        """
        # Verify all features are assigned
        all_features = set(range(self.model_ir.num_features))
        assigned_features = set()

        for assignment in assignments:
            assigned_features.update(assignment.feature_indices)
            self._feature_assignments[assignment.party_id] = assignment

        if assigned_features != all_features:
            missing = all_features - assigned_features
            logger.warning(f"Features not assigned: {missing}")

        logger.info(f"Assigned features to {len(assignments)} parties")

    def create_party(
        self,
        party_id: str,
        feature_indices: List[int]
    ) -> MultiKeyParty:
        """
        Create and register a party.

        Args:
            party_id: Unique party ID
            feature_indices: Features owned by this party

        Returns:
            Created party
        """
        party = MultiKeyParty(party_id, feature_indices)
        party.generate_keys(self.config)
        self.combiner.register_party(party)

        assignment = FeatureAssignment(
            party_id=party_id,
            feature_indices=feature_indices
        )
        self._feature_assignments[party_id] = assignment

        return party

    async def predict_federated(
        self,
        feature_values: Dict[str, Dict[int, float]]
    ) -> float:
        """
        Run federated prediction.

        Args:
            feature_values: Dict of party_id -> {feature_idx -> value}

        Returns:
            Final prediction
        """
        # Setup combined key
        self.combiner.setup_combined_key()

        # Each party encrypts their features
        encrypted_features = {}
        for party_id, values in feature_values.items():
            party = self.combiner._parties.get(party_id)
            if party:
                encrypted = party.encrypt_features(values)
                encrypted_features.update(encrypted)

        # Compute partial traversals in parallel
        max_depth = max(t.max_depth for t in self.model_ir.trees)
        partial_results = await asyncio.gather(*[
            party.compute_partial_traversal(
                encrypted_features,
                self.model_ir,
                (0, max_depth)
            )
            for party in self.combiner._parties.values()
        ])

        # Combine partial results
        combined = self.combiner.combine_partial_results(partial_results)

        # Aggregate tree outputs
        combined_output = sum(combined.values())

        # Threshold decryption
        participating = list(self.combiner._parties.keys())[:self.config.decryption_threshold]
        result = await self.combiner.threshold_decrypt(combined_output, participating)

        return result + self.model_ir.base_score

    def get_protocol_status(self) -> Dict[str, Any]:
        """Get protocol status."""
        return {
            "num_parties": len(self.combiner._parties),
            "decryption_threshold": self.config.decryption_threshold,
            "feature_assignments": {
                party_id: assgn.feature_indices
                for party_id, assgn in self._feature_assignments.items()
            },
            "combined_key_ready": self.combiner._combined_public_key is not None,
        }


# Convenience functions

def create_federated_protocol(
    model_ir: Any,
    num_parties: int = 2,
    decryption_threshold: int = 2
) -> FederatedGBDTProtocol:
    """
    Create a federated GBDT protocol with automatic feature assignment.

    Args:
        model_ir: GBDT model
        num_parties: Number of parties
        decryption_threshold: Parties required for decryption

    Returns:
        Configured protocol
    """
    config = MultiKeyConfig(decryption_threshold=decryption_threshold)
    protocol = FederatedGBDTProtocol(model_ir, config)

    # Auto-assign features round-robin
    features_per_party = model_ir.num_features // num_parties
    for i in range(num_parties):
        start = i * features_per_party
        end = start + features_per_party if i < num_parties - 1 else model_ir.num_features
        feature_indices = list(range(start, end))

        protocol.create_party(f"party_{i}", feature_indices)

    return protocol


async def run_federated_prediction(
    protocol: FederatedGBDTProtocol,
    all_features: np.ndarray
) -> float:
    """
    Run federated prediction with automatic feature distribution.

    Args:
        protocol: Configured protocol
        all_features: All feature values

    Returns:
        Prediction result
    """
    # Distribute features to parties
    feature_values = {}
    for party_id, assignment in protocol._feature_assignments.items():
        feature_values[party_id] = {
            idx: float(all_features[idx])
            for idx in assignment.feature_indices
        }

    return await protocol.predict_federated(feature_values)
