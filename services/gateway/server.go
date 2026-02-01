package main

import (
	"context"
	"log"
	"net"

	"google.golang.org/grpc"
	inf_pb "github.com/fhe-gbdt-serving/proto/inference"
)

type gatewayServer struct {
	inf_pb.UnimplementedInferenceServiceServer
}

func (s *gatewayServer) Predict(ctx context.Context, req *inf_pb.PredictRequest) (*inf_pb.PredictResponse, error) {
	// Structured Audit Log (NO PAYLOADS)
	log.Printf("AUDIT: PredictRequest received. Tenant: %s, Model: %s, Profile: %s", 
		req.TenantId, req.CompiledModelId, req.Profile)
	
	// OpenTelemetry Tracing
	tracer := otel.Tracer("gateway")
	_, span := tracer.Start(ctx, "PredictProxy")
	defer span.End()

	// Request Size Limit (Simplified check)
	if len(req.Batch.Payload) > 64*1024*1024 {
		return nil, grpc.Errorf(codes.InvalidArgument, "Payload too large")
	}

	// 1. Fetch Plan from Registry
	// 2. Fetch Eval Keys from Keystore
	// 3. Forward to Runtime pool
	
	return &inf_pb.PredictResponse{
		Outputs: &inf_pb.CiphertextBatch{
			Payload: []byte("mock-encrypted-result"),
		},
		Stats: &inf_pb.RuntimeStats{
			RuntimeMs: 12.5,
		},
	}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	inf_pb.RegisterInferenceServiceServer(s, &gatewayServer{})
	log.Printf("Gateway service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
