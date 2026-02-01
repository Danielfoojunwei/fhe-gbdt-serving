package main

import (
	"context"
	"log"
	"net"

	"google.golang.org/grpc"
	pb "github.com/fhe-gbdt-serving/proto/control"
)

type controlServer struct {
	pb.UnimplementedControlServiceServer
}

func (s *controlServer) RegisterModel(ctx context.Context, req *pb.RegisterModelRequest) (*pb.RegisterModelResponse, error) {
	log.Printf("Registering model %s for tenant %s", req.ModelName, req.TenantId)
	// 1. Store model content in MinIO
	// 2. Insert metadata into Postgres
	return &pb.RegisterModelResponse{ModelId: "mock-model-id"}, nil
}

func (s *controlServer) CompileModel(ctx context.Context, req *pb.CompileModelRequest) (*pb.CompileModelResponse, error) {
	log.Printf("Compiling model %s for profile %s", req.ModelId, req.Profile)
	// 1. Trigger Compiler service (via gRPC or Job Queue)
	// 2. Update status in Postgres
	return &pb.CompileModelResponse{CompiledModelId: "mock-compiled-id"}, nil
}

func (s *controlServer) GetCompileStatus(ctx context.Context, req *pb.GetCompileStatusRequest) (*pb.GetCompileStatusResponse, error) {
	return &pb.GetCompileStatusResponse{Status: "successful", PlanId: "mock-plan-id"}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":8081")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterControlServiceServer(s, &controlServer{})
	log.Printf("Registry GBDT service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
