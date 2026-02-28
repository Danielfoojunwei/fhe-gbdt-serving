package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"

	pb "github.com/fhe-gbdt-serving/proto/crypto"
	"github.com/fhe-gbdt-serving/services/keystore/vault"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type cryptoServer struct {
	pb.UnimplementedCryptoKeyServiceServer
	vaultClient *vault.KeyVaultClient
	storageDir  string
}

func newCryptoServer() *cryptoServer {
	server := &cryptoServer{
		storageDir: os.Getenv("KEYSTORE_STORAGE_DIR"),
	}
	if server.storageDir == "" {
		server.storageDir = "./data/keystore"
	}

	if err := os.MkdirAll(server.storageDir, 0o700); err != nil {
		log.Fatalf("failed to create keystore storage dir %s: %v", server.storageDir, err)
	}

	vaultCfg := vault.Config{
		Address:   os.Getenv("VAULT_ADDR"),
		Token:     os.Getenv("VAULT_TOKEN"),
		MountPath: "transit",
	}

	if vaultCfg.Address != "" {
		client, err := vault.NewKeyVaultClient(vaultCfg)
		if err != nil {
			log.Printf("WARN: Vault not available, using local KEK: %v", err)
		} else {
			server.vaultClient = client
			log.Printf("Connected to Vault at %s", vaultCfg.Address)
		}
	} else {
		log.Printf("WARN: VAULT_ADDR not set, using local KEK (NOT FOR PRODUCTION)")
	}

	return server
}

func recordPath(storageDir, tenantID, compiledModelID string) string {
	safe := func(v string) string {
		v = strings.ReplaceAll(v, "/", "_")
		v = strings.ReplaceAll(v, "..", "_")
		return v
	}
	return filepath.Join(storageDir, fmt.Sprintf("%s__%s.key", safe(tenantID), safe(compiledModelID)))
}

func (s *cryptoServer) UploadEvalKeys(ctx context.Context, req *pb.UploadEvalKeysRequest) (*pb.UploadEvalKeysResponse, error) {
	log.Printf("AUDIT: Uploading eval keys for tenant %s, model %s", req.TenantId, req.CompiledModelId)
	if req.TenantId == "" || req.CompiledModelId == "" || len(req.EvalKeys) == 0 {
		return nil, status.Error(codes.InvalidArgument, "tenant_id, compiled_model_id, and eval_keys are required")
	}

	var record string
	if s.vaultClient != nil {
		ciphertext, err := s.vaultClient.EncryptWithKEK(ctx, req.TenantId, req.EvalKeys)
		if err != nil {
			log.Printf("ERROR: Failed to encrypt eval keys with Vault: %v", err)
			return nil, status.Error(codes.Internal, "encryption failed")
		}
		record = "vault:" + ciphertext
		log.Printf("AUDIT: Encrypted eval keys with Vault for tenant %s (size: %d -> %d)", req.TenantId, len(req.EvalKeys), len(ciphertext))
	} else {
		localKEK := []byte("dev-only-kek-32-bytes-long!!")[:32]
		encrypted, err := EnvelopeEncrypt(req.EvalKeys, localKEK)
		if err != nil {
			return nil, status.Error(codes.Internal, "local encryption failed")
		}
		record = "local:" + base64.StdEncoding.EncodeToString(encrypted)
		log.Printf("WARN: Using local KEK for tenant %s (NOT FOR PRODUCTION)", req.TenantId)
	}

	if err := os.WriteFile(recordPath(s.storageDir, req.TenantId, req.CompiledModelId), []byte(record), 0o600); err != nil {
		log.Printf("ERROR: failed to persist eval keys: %v", err)
		return nil, status.Error(codes.Internal, "failed to persist encrypted eval keys")
	}
	return &pb.UploadEvalKeysResponse{Success: true}, nil
}

func (s *cryptoServer) GetEvalKeys(ctx context.Context, req *pb.GetEvalKeysRequest) (*pb.GetEvalKeysResponse, error) {
	log.Printf("AUDIT: Retrieving eval keys for tenant %s, model %s", req.TenantId, req.CompiledModelId)
	if req.TenantId == "" || req.CompiledModelId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id and compiled_model_id are required")
	}

	recordBytes, err := os.ReadFile(recordPath(s.storageDir, req.TenantId, req.CompiledModelId))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, status.Error(codes.NotFound, "eval keys not found")
		}
		return nil, status.Error(codes.Internal, "failed to read encrypted eval keys")
	}
	record := string(recordBytes)

	if strings.HasPrefix(record, "vault:") {
		if s.vaultClient == nil {
			return nil, status.Error(codes.FailedPrecondition, "vault required to decrypt stored eval keys")
		}
		plain, derr := s.vaultClient.DecryptWithKEK(ctx, req.TenantId, strings.TrimPrefix(record, "vault:"))
		if derr != nil {
			return nil, status.Error(codes.Internal, "failed to decrypt eval keys")
		}
		return &pb.GetEvalKeysResponse{EvalKeys: plain, Found: true}, nil
	}

	if strings.HasPrefix(record, "local:") {
		enc, derr := base64.StdEncoding.DecodeString(strings.TrimPrefix(record, "local:"))
		if derr != nil {
			return nil, status.Error(codes.Internal, "stored key format invalid")
		}
		localKEK := []byte("dev-only-kek-32-bytes-long!!")[:32]
		plain, derr := EnvelopeDecrypt(enc, localKEK)
		if derr != nil {
			return nil, status.Error(codes.Internal, "failed to decrypt eval keys")
		}
		return &pb.GetEvalKeysResponse{EvalKeys: plain, Found: true}, nil
	}

	return nil, status.Error(codes.Internal, "unknown encrypted key record format")
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

	server := newCryptoServer()
	s := grpc.NewServer()
	pb.RegisterCryptoKeyServiceServer(s, server)

	log.Printf("Production Keystore Service listening at %v (Vault: %v)",
		lis.Addr(), server.vaultClient != nil)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
