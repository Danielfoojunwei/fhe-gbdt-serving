package main

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "github.com/fhe-gbdt-serving/proto/control"
)

type controlServer struct {
	pb.UnimplementedControlServiceServer
	// In-memory store for MVP, following production patterns
	models map[string]*pb.RegisterModelRequest
	compiled map[string]*pb.GetCompileStatusResponse
}

func newControlServer() *controlServer {
	return &controlServer{
		models:   make(map[string]*pb.RegisterModelRequest),
		compiled: make(map[string]*pb.GetCompileStatusResponse),
	}
}

func (s *controlServer) RegisterModel(ctx context.Context, req *pb.RegisterModelRequest) (*pb.RegisterModelResponse, error) {
	if req.ModelContent == nil || len(req.ModelContent) == 0 {
		return nil, status.Error(codes.InvalidArgument, "model content cannot be empty")
	}

	modelID := uuid.New().String()
	log.Printf("AUDIT: Registering model %s (ID: %s) for tenant %s", req.ModelName, modelID, req.TenantId)
	
	// Real Persistence Point: 
	// 1. Upload model_content to MinIO
	// 2. Insert into Postgres
	s.models[modelID] = req

	return &pb.RegisterModelResponse{ModelId: modelID}, nil
}

func (s *controlServer) CompileModel(ctx context.Context, req *pb.CompileModelRequest) (*pb.CompileModelResponse, error) {
	log.Printf("AUDIT: Compiling model %s for profile %s", req.ModelId, req.Profile)
	
	if _, ok := s.models[req.ModelId]; !ok {
		return nil, status.Errorf(codes.NotFound, "model %s not found", req.ModelId)
	}

	compiledID := uuid.New().String()
	planID := fmt.Sprintf("plan-%s", uuid.New().String()[:8])

	// Real Trigger Point:
	// Start background compilation job
	s.compiled[compiledID] = &pb.GetCompileStatusResponse{
		Status: "successful",
		PlanId: planID,
	}

	return &pb.CompileModelResponse{CompiledModelId: compiledID}, nil
}

func (s *controlServer) GetCompileStatus(ctx context.Context, req *pb.GetCompileStatusRequest) (*pb.GetCompileStatusResponse, error) {
	resp, ok := s.compiled[req.CompiledModelId]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "compiled model %s not found", req.CompiledModelId)
	}
	return resp, nil
}

func main() {
	lis, err := net.Listen("tcp", ":8081")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterControlServiceServer(s, newControlServer())
	log.Printf("Production Registry Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
