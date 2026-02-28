package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"

	pb "github.com/fhe-gbdt-serving/proto/control"
	"github.com/fhe-gbdt-serving/services/registry/db"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type controlServer struct {
	pb.UnimplementedControlServiceServer
	models   map[string]*pb.RegisterModelRequest
	compiled map[string]*pb.GetCompileStatusResponse
	store    *db.Store
	dataDir  string
}

func newControlServer() *controlServer {
	server := &controlServer{
		models:   make(map[string]*pb.RegisterModelRequest),
		compiled: make(map[string]*pb.GetCompileStatusResponse),
		dataDir:  os.Getenv("REGISTRY_STORAGE_DIR"),
	}
	if server.dataDir == "" {
		server.dataDir = "./data/registry"
	}
	if err := os.MkdirAll(server.dataDir, 0o700); err != nil {
		log.Fatalf("failed to create registry storage dir %s: %v", server.dataDir, err)
	}

	store, err := db.NewStore()
	if err != nil {
		log.Printf("WARN: Database not available, using in-memory storage: %v", err)
	} else {
		log.Printf("Connected to PostgreSQL database")
		server.store = store
	}

	return server
}

func sanitizePathPart(v string) string {
	v = strings.ReplaceAll(v, "/", "_")
	v = strings.ReplaceAll(v, "..", "_")
	return v
}

func (s *controlServer) persistModelContent(tenantID, modelID string, content []byte) (string, error) {
	path := filepath.Join(s.dataDir, "models", sanitizePathPart(tenantID))
	if err := os.MkdirAll(path, 0o700); err != nil {
		return "", err
	}
	file := filepath.Join(path, modelID+".bin")
	if err := os.WriteFile(file, content, 0o600); err != nil {
		return "", err
	}
	return file, nil
}

func (s *controlServer) RegisterModel(ctx context.Context, req *pb.RegisterModelRequest) (*pb.RegisterModelResponse, error) {
	if req.ModelContent == nil || len(req.ModelContent) == 0 {
		return nil, status.Error(codes.InvalidArgument, "model content cannot be empty")
	}
	if req.TenantId == "" || req.ModelName == "" || req.LibraryType == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id, model_name and library_type are required")
	}

	modelID := uuid.New().String()
	log.Printf("AUDIT: Registering model %s (ID: %s) for tenant %s", req.ModelName, modelID, req.TenantId)

	contentPath, err := s.persistModelContent(req.TenantId, modelID, req.ModelContent)
	if err != nil {
		log.Printf("ERROR: Failed to persist model content: %v", err)
		return nil, status.Error(codes.Internal, "failed to persist model content")
	}

	if s.store != nil {
		if err := s.store.EnsureTenant(ctx, req.TenantId); err != nil {
			log.Printf("WARN: Failed to ensure tenant: %v", err)
		}
		dbModelID, dberr := s.store.CreateModel(ctx, req.TenantId, req.ModelName, req.LibraryType, contentPath)
		if dberr != nil {
			log.Printf("ERROR: Failed to persist model metadata: %v", dberr)
			return nil, status.Error(codes.Internal, "failed to persist model metadata")
		}
		modelID = dbModelID
	}

	s.models[modelID] = req
	return &pb.RegisterModelResponse{ModelId: modelID}, nil
}

func (s *controlServer) CompileModel(ctx context.Context, req *pb.CompileModelRequest) (*pb.CompileModelResponse, error) {
	log.Printf("AUDIT: Compiling model %s for profile %s", req.ModelId, req.Profile)

	if s.store != nil {
		model, err := s.store.GetModel(ctx, req.ModelId)
		if err != nil {
			log.Printf("ERROR: Failed to get model: %v", err)
		}
		if model == nil {
			if _, ok := s.models[req.ModelId]; !ok {
				return nil, status.Errorf(codes.NotFound, "model %s not found", req.ModelId)
			}
		}
	} else if _, ok := s.models[req.ModelId]; !ok {
		return nil, status.Errorf(codes.NotFound, "model %s not found", req.ModelId)
	}

	compiledID := uuid.New().String()
	planID := fmt.Sprintf("plan-%s", uuid.New().String()[:8])

	if s.store != nil {
		planPath := filepath.Join(s.dataDir, "plans", compiledID+".bin")
		if err := os.MkdirAll(filepath.Dir(planPath), 0o700); err != nil {
			return nil, status.Error(codes.Internal, "failed to initialize plan storage")
		}
		dbCompiledID, err := s.store.CreateCompiledModel(ctx, req.ModelId, req.Profile, planID, planPath)
		if err != nil {
			log.Printf("ERROR: Failed to persist compiled model: %v", err)
			return nil, status.Error(codes.Internal, "failed to persist compiled model")
		}
		compiledID = dbCompiledID
		_ = s.store.UpdateCompiledModelStatus(ctx, compiledID, "successful", "")
	}

	s.compiled[compiledID] = &pb.GetCompileStatusResponse{Status: "successful", PlanId: planID}
	return &pb.CompileModelResponse{CompiledModelId: compiledID}, nil
}

func (s *controlServer) GetCompileStatus(ctx context.Context, req *pb.GetCompileStatusRequest) (*pb.GetCompileStatusResponse, error) {
	if s.store != nil {
		cm, err := s.store.GetCompiledModel(ctx, req.CompiledModelId)
		if err != nil {
			log.Printf("ERROR: Failed to get compiled model: %v", err)
		}
		if cm != nil {
			return &pb.GetCompileStatusResponse{Status: cm.Status, PlanId: cm.PlanID}, nil
		}
	}

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
