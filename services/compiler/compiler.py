from .parser import get_parser
from .optimizer import MOAIOptimizer
from .ir import ModelIR, ObliviousPlanIR

class Compiler:
    def __init__(self):
        self.optimizer_latency = MOAIOptimizer(profile="latency")
        self.optimizer_throughput = MOAIOptimizer(profile="throughput")

    def compile(self, content: bytes, library_type: str, profile: str) -> ObliviousPlanIR:
        parser = get_parser(library_type)
        if not parser:
            raise ValueError(f"Unsupported library type: {library_type}")
        
        model_ir = parser.parse(content)
        
        if profile == "throughput":
            return self.optimizer_throughput.optimize(model_ir)
        else:
            return self.optimizer_latency.optimize(model_ir)

compiler = Compiler()
