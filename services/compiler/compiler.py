from .parser import get_parser
from .optimizer import MOAIOptimizer
from .ir import ModelIR, ModelFamily, ObliviousPlanIR, LinearPlanIR, ExecutionPlanIR
from .linear_optimizer import LinearModelOptimizer
from .random_forest_optimizer import RandomForestOptimizer
from .decision_tree_optimizer import DecisionTreeOptimizer


class Compiler:
    def __init__(self):
        # GBDT optimizers (original)
        self.optimizer_latency = MOAIOptimizer(profile="latency")
        self.optimizer_throughput = MOAIOptimizer(profile="throughput")

        # Linear model optimizers (Phase 2)
        self.linear_latency = LinearModelOptimizer(profile="latency")
        self.linear_throughput = LinearModelOptimizer(profile="throughput")

        # Random forest optimizers (Phase 3)
        self.rf_latency = RandomForestOptimizer(profile="latency")
        self.rf_throughput = RandomForestOptimizer(profile="throughput")

        # Decision tree optimizers (Phase 3)
        self.dt_latency = DecisionTreeOptimizer(profile="latency")
        self.dt_throughput = DecisionTreeOptimizer(profile="throughput")

    def compile(self, content: bytes, library_type: str, profile: str) -> ExecutionPlanIR:
        """
        Compile a model into an FHE execution plan.

        Routes to the appropriate optimizer based on model family:
        - TREE_ENSEMBLE → MOAIOptimizer (GBDT)
        - LINEAR → LinearModelOptimizer (LR, GLM)
        - RANDOM_FOREST → RandomForestOptimizer
        - SINGLE_TREE → DecisionTreeOptimizer
        """
        parser = get_parser(library_type)
        if not parser:
            raise ValueError(f"Unsupported library type: {library_type}")

        model_ir = parser.parse(content)

        return self._optimize(model_ir, profile)

    def _optimize(self, model_ir: ModelIR, profile: str) -> ExecutionPlanIR:
        """Route to correct optimizer based on model family."""
        use_throughput = profile == "throughput"

        if model_ir.model_family == ModelFamily.LINEAR:
            opt = self.linear_throughput if use_throughput else self.linear_latency
            return opt.optimize(model_ir)

        elif model_ir.model_family == ModelFamily.RANDOM_FOREST:
            opt = self.rf_throughput if use_throughput else self.rf_latency
            return opt.optimize(model_ir)

        elif model_ir.model_family == ModelFamily.SINGLE_TREE:
            opt = self.dt_throughput if use_throughput else self.dt_latency
            return opt.optimize(model_ir)

        else:
            # Default: TREE_ENSEMBLE (GBDT)
            opt = self.optimizer_throughput if use_throughput else self.optimizer_latency
            return opt.optimize(model_ir)


compiler = Compiler()
