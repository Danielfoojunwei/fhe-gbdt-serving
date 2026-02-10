"""
Novel Innovation #5: Bootstrapping-Aligned Tree Architecture

Design GBDT trees to align with FHE bootstrapping boundaries, minimizing
noise accumulation and enabling deeper computations.

Key Insight:
- Each tree level consumes noise (especially step functions)
- Bootstrapping refreshes noise but is expensive
- Trees can be designed so natural depth boundaries align with bootstrap points
- Multiple shallow forests can replace one deep ensemble

Benefits:
- Predictable noise consumption per forest
- Optimal placement of bootstrapping operations
- Support for arbitrarily deep ensembles
- Better noise budget utilization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseConsumptionModel:
    """Models noise consumption for FHE operations."""
    initial_noise_bits: float = 3.2  # Initial encryption noise
    step_function_bits: float = 8.0  # Per step function
    addition_bits: float = 0.1  # Per addition
    plain_mult_bits: float = 10.0  # Per plaintext multiplication
    rotation_bits: float = 0.5  # Per rotation
    total_budget_bits: float = 31.0  # log2(q) - 1 safety margin

    @property
    def available_budget(self) -> float:
        """Available noise budget after initial encryption."""
        return self.total_budget_bits - self.initial_noise_bits

    def levels_before_bootstrap(self) -> int:
        """Calculate tree levels possible before bootstrapping needed."""
        available = self.available_budget
        per_level = self.step_function_bits + self.addition_bits
        return int(available / per_level)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap-aligned trees."""
    # Noise model
    noise_model: NoiseConsumptionModel = field(default_factory=NoiseConsumptionModel)

    # Maximum depth per forest chunk
    max_chunk_depth: Optional[int] = None  # Auto-computed from noise model

    # Minimum trees per chunk
    min_trees_per_chunk: int = 10

    # Bootstrap margin (bits to reserve for aggregation)
    bootstrap_margin_bits: float = 5.0

    # Enable interleaved forest execution
    interleaved_execution: bool = True

    def __post_init__(self):
        if self.max_chunk_depth is None:
            # Auto-compute based on noise model
            available = self.noise_model.available_budget - self.bootstrap_margin_bits
            per_level = self.noise_model.step_function_bits
            self.max_chunk_depth = max(1, int(available / per_level))


@dataclass
class ForestChunk:
    """A chunk of trees that fits within one noise budget cycle."""
    chunk_id: int
    trees: List[Any]  # TreeIR objects
    max_depth: int
    estimated_noise_bits: float

    # Bootstrap metadata
    requires_bootstrap_before: bool = False
    requires_bootstrap_after: bool = False


@dataclass
class NoiseAlignedForest:
    """A forest structured for optimal noise budget usage."""
    chunks: List[ForestChunk]
    total_trees: int
    total_depth_equivalent: int

    # Execution plan
    bootstrap_points: List[int]  # Chunk indices where bootstrapping occurs

    # Metadata
    config: BootstrapConfig = field(default_factory=BootstrapConfig)


class BootstrapAwareTreeBuilder:
    """
    Builds trees with bootstrapping boundaries in mind.

    Strategies:
    1. Depth chunking: Split deep trees into shallow chunks
    2. Forest chunking: Group trees to fit noise budget
    3. Interleaved boosting: Alternate chunks for gradient updates
    """

    def __init__(self, config: Optional[BootstrapConfig] = None):
        """
        Initialize builder.

        Args:
            config: Bootstrap configuration
        """
        self.config = config or BootstrapConfig()

    def analyze_noise_consumption(
        self,
        model_ir: Any
    ) -> Dict[str, Any]:
        """
        Analyze noise consumption for a model.

        Args:
            model_ir: Parsed ModelIR

        Returns:
            Analysis results
        """
        noise_model = self.config.noise_model

        # Per-tree analysis
        tree_depths = [t.max_depth for t in model_ir.trees]
        max_depth = max(tree_depths)

        # Noise per level
        per_level_noise = noise_model.step_function_bits + noise_model.addition_bits

        # Total noise for deepest path
        deepest_tree_noise = max_depth * per_level_noise

        # Aggregation noise (log2 additions)
        num_trees = len(model_ir.trees)
        aggregation_noise = math.ceil(math.log2(max(num_trees, 1))) * noise_model.addition_bits

        total_noise = noise_model.initial_noise_bits + deepest_tree_noise + aggregation_noise

        # Bootstrap requirements
        needs_bootstrap = total_noise > noise_model.total_budget_bits
        num_bootstraps = math.ceil(
            total_noise / noise_model.available_budget
        ) - 1 if needs_bootstrap else 0

        return {
            "max_depth": max_depth,
            "num_trees": num_trees,
            "per_level_noise": per_level_noise,
            "deepest_tree_noise": deepest_tree_noise,
            "aggregation_noise": aggregation_noise,
            "total_estimated_noise": total_noise,
            "noise_budget": noise_model.total_budget_bits,
            "needs_bootstrap": needs_bootstrap,
            "estimated_bootstraps": num_bootstraps,
            "budget_utilization": total_noise / noise_model.total_budget_bits,
        }

    def partition_into_chunks(
        self,
        model_ir: Any
    ) -> NoiseAlignedForest:
        """
        Partition model into bootstrap-aligned chunks.

        Args:
            model_ir: Parsed ModelIR

        Returns:
            NoiseAlignedForest with chunked trees
        """
        max_chunk_depth = self.config.max_chunk_depth
        noise_model = self.config.noise_model

        chunks = []
        bootstrap_points = []

        current_chunk_trees = []
        current_chunk_noise = noise_model.initial_noise_bits
        chunk_id = 0

        for tree in model_ir.trees:
            tree_noise = tree.max_depth * (
                noise_model.step_function_bits + noise_model.addition_bits
            )

            # Check if adding this tree would exceed budget
            potential_noise = current_chunk_noise + tree_noise + noise_model.addition_bits

            if potential_noise > noise_model.available_budget - self.config.bootstrap_margin_bits:
                # Finalize current chunk
                if current_chunk_trees:
                    chunk = self._create_chunk(
                        chunk_id, current_chunk_trees, current_chunk_noise
                    )
                    chunks.append(chunk)
                    bootstrap_points.append(chunk_id)
                    chunk_id += 1

                # Start new chunk
                current_chunk_trees = [tree]
                current_chunk_noise = noise_model.initial_noise_bits + tree_noise
            else:
                current_chunk_trees.append(tree)
                current_chunk_noise = potential_noise

        # Finalize last chunk
        if current_chunk_trees:
            chunk = self._create_chunk(chunk_id, current_chunk_trees, current_chunk_noise)
            chunks.append(chunk)

        # Mark bootstrap requirements
        for i, chunk in enumerate(chunks):
            chunk.requires_bootstrap_before = i > 0 and i in bootstrap_points
            chunk.requires_bootstrap_after = i < len(chunks) - 1 and i in bootstrap_points

        total_depth_eq = sum(chunk.max_depth * len(chunk.trees) for chunk in chunks)

        forest = NoiseAlignedForest(
            chunks=chunks,
            total_trees=len(model_ir.trees),
            total_depth_equivalent=total_depth_eq,
            bootstrap_points=bootstrap_points[:-1] if bootstrap_points else [],  # No bootstrap after last
            config=self.config
        )

        logger.info(
            f"Partitioned into {len(chunks)} chunks, "
            f"{len(forest.bootstrap_points)} bootstrap points"
        )

        return forest

    def _create_chunk(
        self,
        chunk_id: int,
        trees: List[Any],
        estimated_noise: float
    ) -> ForestChunk:
        """Create a forest chunk."""
        max_depth = max(t.max_depth for t in trees) if trees else 0

        return ForestChunk(
            chunk_id=chunk_id,
            trees=trees,
            max_depth=max_depth,
            estimated_noise_bits=estimated_noise
        )

    def build_depth_chunked_trees(
        self,
        original_trees: List[Any],
        chunk_depth: int
    ) -> List[List[Any]]:
        """
        Split deep trees into multiple shallow trees.

        Each original tree of depth D becomes ceil(D/chunk_depth) shallow trees
        that compute partial traversals.

        Args:
            original_trees: Original deep trees
            chunk_depth: Maximum depth per chunk

        Returns:
            List of tree chunks (each chunk is a list of partial trees)
        """
        all_chunks = []
        max_original_depth = max(t.max_depth for t in original_trees)
        num_depth_chunks = math.ceil(max_original_depth / chunk_depth)

        for chunk_idx in range(num_depth_chunks):
            depth_start = chunk_idx * chunk_depth
            depth_end = min((chunk_idx + 1) * chunk_depth, max_original_depth)

            chunk_trees = []
            for tree in original_trees:
                partial_tree = self._extract_depth_range(tree, depth_start, depth_end)
                if partial_tree is not None:
                    chunk_trees.append(partial_tree)

            all_chunks.append(chunk_trees)

        return all_chunks

    def _extract_depth_range(
        self,
        tree_ir: Any,
        depth_start: int,
        depth_end: int
    ) -> Optional[Any]:
        """Extract nodes in a depth range from a tree."""
        # In a full implementation, this would create a partial TreeIR
        # containing only nodes in the specified depth range

        relevant_nodes = {
            node_id: node
            for node_id, node in tree_ir.nodes.items()
            if depth_start <= node.depth < depth_end
        }

        if not relevant_nodes:
            return None

        # Create partial tree (simplified)
        return type(tree_ir)(
            tree_id=tree_ir.tree_id,
            nodes=relevant_nodes,
            root_id=tree_ir.root_id,
            max_depth=depth_end - depth_start
        )


class BootstrapInterleavedEnsemble:
    """
    Ensemble that interleaves computation with bootstrapping.

    Execution model:
    1. Evaluate chunk 0 (depth 0 to D)
    2. Bootstrap accumulated state
    3. Evaluate chunk 1 (continuing from chunk 0 outputs)
    4. Bootstrap
    5. Continue...

    This enables arbitrarily deep ensembles within fixed noise budget.
    """

    def __init__(
        self,
        aligned_forest: NoiseAlignedForest
    ):
        """
        Initialize interleaved ensemble.

        Args:
            aligned_forest: Bootstrap-aligned forest
        """
        self.forest = aligned_forest
        self._chunk_outputs: Dict[int, np.ndarray] = {}

    def evaluate_plaintext(
        self,
        features: np.ndarray,
        base_score: float = 0.0
    ) -> np.ndarray:
        """
        Evaluate ensemble in plaintext (for validation).

        Args:
            features: Input features
            base_score: Base prediction

        Returns:
            Predictions
        """
        batch_size = features.shape[0]
        accumulated = np.full(batch_size, base_score)

        for chunk in self.forest.chunks:
            # Evaluate this chunk
            chunk_output = self._evaluate_chunk_plaintext(chunk, features)
            accumulated += chunk_output

            # Store for potential residual computation
            self._chunk_outputs[chunk.chunk_id] = chunk_output

            # Simulate bootstrap (no-op in plaintext)
            if chunk.requires_bootstrap_after:
                logger.debug(f"Bootstrap point after chunk {chunk.chunk_id}")

        return accumulated

    def _evaluate_chunk_plaintext(
        self,
        chunk: ForestChunk,
        features: np.ndarray
    ) -> np.ndarray:
        """Evaluate a single chunk in plaintext."""
        batch_size = features.shape[0]
        chunk_output = np.zeros(batch_size)

        for tree in chunk.trees:
            tree_output = self._evaluate_tree_plaintext(tree, features)
            chunk_output += tree_output

        return chunk_output

    def _evaluate_tree_plaintext(
        self,
        tree_ir: Any,
        features: np.ndarray
    ) -> np.ndarray:
        """Evaluate a single tree in plaintext."""
        batch_size = features.shape[0]
        outputs = np.zeros(batch_size)

        for i in range(batch_size):
            outputs[i] = self._traverse_tree(tree_ir, features[i])

        return outputs

    def _traverse_tree(
        self,
        tree_ir: Any,
        sample: np.ndarray
    ) -> float:
        """Traverse tree for a single sample."""
        node = tree_ir.nodes.get(tree_ir.root_id)

        while node is not None:
            if node.leaf_value is not None:
                return node.leaf_value

            feat_idx = node.feature_index
            threshold = node.threshold

            if sample[feat_idx] < threshold:
                next_id = node.left_child_id
            else:
                next_id = node.right_child_id

            node = tree_ir.nodes.get(next_id)

        return 0.0

    def get_execution_plan(self) -> Dict[str, Any]:
        """Get execution plan with bootstrap points."""
        return {
            "num_chunks": len(self.forest.chunks),
            "bootstrap_points": self.forest.bootstrap_points,
            "chunk_depths": [c.max_depth for c in self.forest.chunks],
            "chunk_trees": [len(c.trees) for c in self.forest.chunks],
            "estimated_noise_per_chunk": [c.estimated_noise_bits for c in self.forest.chunks],
            "total_bootstraps": len(self.forest.bootstrap_points),
        }


# Convenience functions

def create_bootstrap_aligned_forest(
    model_ir: Any,
    noise_budget_bits: float = 31.0,
    step_function_bits: float = 8.0
) -> NoiseAlignedForest:
    """
    Create a bootstrap-aligned forest from a model.

    Args:
        model_ir: Parsed ModelIR
        noise_budget_bits: Total noise budget
        step_function_bits: Noise per step function

    Returns:
        NoiseAlignedForest
    """
    noise_model = NoiseConsumptionModel(
        total_budget_bits=noise_budget_bits,
        step_function_bits=step_function_bits
    )
    config = BootstrapConfig(noise_model=noise_model)
    builder = BootstrapAwareTreeBuilder(config)

    return builder.partition_into_chunks(model_ir)


def analyze_bootstrap_requirements(model_ir: Any) -> Dict[str, Any]:
    """
    Analyze bootstrapping requirements for a model.

    Args:
        model_ir: Parsed ModelIR

    Returns:
        Analysis results
    """
    builder = BootstrapAwareTreeBuilder()
    return builder.analyze_noise_consumption(model_ir)


def create_interleaved_ensemble(
    model_ir: Any
) -> BootstrapInterleavedEnsemble:
    """
    Create an interleaved ensemble with automatic bootstrap alignment.

    Args:
        model_ir: Parsed ModelIR

    Returns:
        BootstrapInterleavedEnsemble
    """
    forest = create_bootstrap_aligned_forest(model_ir)
    return BootstrapInterleavedEnsemble(forest)
