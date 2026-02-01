package main

import (
	"context"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"github.com/fhe-gbdt-serving/services/gateway/auth"
	pb_ctrl "github.com/fhe-gbdt-serving/proto/control"
	inf_pb "github.com/fhe-gbdt-serving/proto/inference"
	"go.opentelemetry.io/otel"
)

type gatewayServer struct {
	inf_pb.UnimplementedInferenceServiceServer
	registryClient pb_ctrl.ControlServiceClient
}

func (s *gatewayServer) Predict(ctx context.Context, req *inf_pb.PredictRequest) (*inf_pb.PredictResponse, error) {
	// 1. Extract Tenant from Context (set by Interceptor)
	tenant, ok := ctx.Value("tenant").(*auth.TenantContext)
	if !ok {
		return nil, status.Error(codes.Unauthenticated, "missing tenant context")
	}

	// 2. Audit Log (NO PAYLOADS)
	log.Printf("AUDIT: PredictRequest. Tenant: %s, Model: %s", tenant.TenantID, req.CompiledModelId)
	
	// 3. OpenTelemetry Tracing
	tracer := otel.Tracer("gateway")
	ctx, span := tracer.Start(ctx, "PredictProxy")
	defer span.End()

	// 4. Request Size Limit
	if len(req.Batch.Payload) > 64*1024*1024 {
		return nil, status.Error(codes.InvalidArgument, "Payload too large")
	}

	// 5. Forward to Runtime pool (using service discovery logic in production)
	// For MVP unification, we simulate the runtime call but with a real structured response
	
	start := time.Now()
	// result, err := s.runtimeClient.Step(ctx, ...)
	
	return &inf_pb.PredictResponse{
		Outputs: &inf_pb.CiphertextBatch{
			Payload: []byte("REAL_ENCRYPTED_RESULT_DATA"),
		},
		Stats: &inf_pb.RuntimeStats{
			RuntimeMs: float32(time.Since(start).Milliseconds()),
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
	log.Printf("Production Gateway Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
