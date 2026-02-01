import grpc
import gen.inference_pb2 as inference_pb2
import gen.inference_pb2_grpc as inference_pb2_grpc
from .features import FeatureSpec
from .crypto import KeyManager

class FHEGBDTClient:
    def __init__(self, gateway_addr: str, tenant_id: str):
        self.gateway_addr = gateway_addr
        self.tenant_id = tenant_id
        self.key_manager = KeyManager(tenant_id)
        self.channel = grpc.insecure_channel(gateway_addr)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def predict_encrypted(self, compiled_model_id: str, features: List[Dict[str, float]]) -> List[float]:
        """Runs end-to-end encrypted inference."""
        # 1. Preprocess
        # 2. KeyGen (if needed)
        # 3. Encrypt
        payload = self.key_manager.encrypt([1.0]) # Placeholder
        
        # 4. Request
        request = inference_pb2.PredictRequest(
            tenant_id=self.tenant_id,
            compiled_model_id=compiled_model_id,
            profile="latency",
            batch=inference_pb2.CiphertextBatch(
                scheme_id="n2he_default",
                batch_size=len(features),
                payload=payload
            )
        )
        
        response = self.stub.Predict(request)
        
        # 5. Decrypt
        return self.key_manager.decrypt(response.outputs.payload)

client = FHEGBDTClient("localhost:8080", "tenant-1")
