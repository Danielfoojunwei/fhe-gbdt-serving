package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "github.com/fhe-gbdt-serving/proto/control"
	"github.com/fhe-gbdt-serving/services/registry/db"
)

type controlServer struct {
	pb.UnimplementedControlServiceServer
	// In-memory store as fallback when DB is not available
	models   map[string]*pb.RegisterModelRequest
	compiled map[string]*pb.GetCompileStatusResponse
	
	// Database store (nil if not available)
	store *db.Store
}

func newControlServer() *controlServer {
	server := &controlServer{
		models:   make(map[string]*pb.RegisterModelRequest),
		compiled: make(map[string]*pb.GetCompileStatusResponse),
	}
	
	// Try to connect to database
	store, err := db.NewStore()
	if err != nil {
		log.Printf("WARN: Database not available, using in-memory storage: %v", err)
	} else {
		log.Printf("Connected to PostgreSQL database")
		server.store = store
	}
	
	return server
}

func (s *controlServer) RegisterModel(ctx context.Context, req *pb.RegisterModelRequest) (*pb.RegisterModelResponse, error) {
	if req.ModelContent == nil || len(req.ModelContent) == 0 {
		return nil, status.Error(codes.InvalidArgument, "model content cannot be empty")
	}

	modelID := uuid.New().String()
	log.Printf("AUDIT: Registering model %s (ID: %s) for tenant %s", req.ModelName, modelID, req.TenantId)
	
	if s.store != nil {
		// Ensure tenant exists
		if err := s.store.EnsureTenant(ctx, req.TenantId); err != nil {
			log.Printf("WARN: Failed to ensure tenant: %v", err)
		}
		
		// Store in database
		// TODO: Upload model content to MinIO and store path
		contentPath := fmt.Sprintf("models/%s/%s.bin", req.TenantId, modelID)
		dbModelID, err := s.store.CreateModel(ctx, req.TenantId, req.ModelName, req.LibraryType, contentPath)
		if err != nil {
			log.Printf("ERROR: Failed to persist model: %v", err)
			// Fall back to in-memory
			s.models[modelID] = req
		} else {
			modelID = dbModelID
		}
	} else {
		// In-memory fallback
		s.models[modelID] = req
	}

	return &pb.RegisterModelResponse{ModelId: modelID}, nil
}

func (s *controlServer) CompileModel(ctx context.Context, req *pb.CompileModelRequest) (*pb.CompileModelResponse, error) {
	log.Printf("AUDIT: Compiling model %s for profile %s", req.ModelId, req.Profile)
	
	// Check model exists
	if s.store != nil {
		model, err := s.store.GetModel(ctx, req.ModelId)
		if err != nil {
			log.Printf("ERROR: Failed to get model: %v", err)
		}
		if model == nil {
			// Check in-memory fallback
			if _, ok := s.models[req.ModelId]; !ok {
				return nil, status.Errorf(codes.NotFound, "model %s not found", req.ModelId)
			}
		}
	} else {
		if _, ok := s.models[req.ModelId]; !ok {
			return nil, status.Errorf(codes.NotFound, "model %s not found", req.ModelId)
		}
	}

	compiledID := uuid.New().String()
	planID := fmt.Sprintf("plan-%s", uuid.New().String()[:8])

	if s.store != nil {
		// Store in database
		planPath := fmt.Sprintf("plans/%s.bin", compiledID)
		dbCompiledID, err := s.store.CreateCompiledModel(ctx, req.ModelId, req.Profile, planID, planPath)
		if err != nil {
			log.Printf("ERROR: Failed to persist compiled model: %v", err)
			// Fall back to in-memory
			s.compiled[compiledID] = &pb.GetCompileStatusResponse{
				Status: "successful",
				PlanId: planID,
			}
		} else {
			compiledID = dbCompiledID
			// Mark as successful (in real system, this would be async)
			s.store.UpdateCompiledModelStatus(ctx, compiledID, "successful", "")
		}
	} else {
		// In-memory fallback
		s.compiled[compiledID] = &pb.GetCompileStatusResponse{
			Status: "successful",
			PlanId: planID,
		}
	}

	return &pb.CompileModelResponse{CompiledModelId: compiledID}, nil
}

func (s *controlServer) GetCompileStatus(ctx context.Context, req *pb.GetCompileStatusRequest) (*pb.GetCompileStatusResponse, error) {
	if s.store != nil {
		cm, err := s.store.GetCompiledModel(ctx, req.CompiledModelId)
		if err != nil {
			log.Printf("ERROR: Failed to get compiled model: %v", err)
		}
		if cm != nil {
			return &pb.GetCompileStatusResponse{
				Status: cm.Status,
				PlanId: cm.PlanID,
			}, nil
		}
	}
	
	// Check in-memory fallback
	resp, ok := s.compiled[req.CompiledModelId]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "compiled model %s not found", req.CompiledModelId)
	}
	return resp, nil
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}
	
	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	
	server := newControlServer()
	s := grpc.NewServer()
	pb.RegisterControlServiceServer(s, server)
	
	log.Printf("Production Registry Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

