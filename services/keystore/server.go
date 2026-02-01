package main

import (
	"context"
	"log"
	"net"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "github.com/fhe-gbdt-serving/proto/crypto"
	"github.com/fhe-gbdt-serving/services/keystore/vault"
)

type cryptoServer struct {
	pb.UnimplementedCryptoKeyServiceServer
	vaultClient *vault.KeyVaultClient
}

func newCryptoServer() *cryptoServer {
	server := &cryptoServer{}
	
	// Try to connect to Vault
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

func (s *cryptoServer) UploadEvalKeys(ctx context.Context, req *pb.UploadEvalKeysRequest) (*pb.UploadEvalKeysResponse, error) {
	log.Printf("AUDIT: Uploading eval keys for tenant %s, model %s", req.TenantId, req.CompiledModelId)
	
	// Encrypt eval keys using Vault if available
	if s.vaultClient != nil {
		// Use Vault Transit to encrypt the eval keys
		ciphertext, err := s.vaultClient.EncryptWithKEK(ctx, req.TenantId, req.EvalKeys)
		if err != nil {
			log.Printf("ERROR: Failed to encrypt eval keys with Vault: %v", err)
			return nil, status.Error(codes.Internal, "encryption failed")
		}
		
		// Store encrypted keys (ciphertext includes vault reference)
		log.Printf("AUDIT: Encrypted eval keys with Vault for tenant %s (size: %d -> %d)", 
			req.TenantId, len(req.EvalKeys), len(ciphertext))
		
		// TODO: Store ciphertext in MinIO/Postgres
		_ = ciphertext
	} else {
		// Fallback: Use local KEK (development only)
		localKEK := []byte("dev-only-kek-32-bytes-long!!")[:32]
		encrypted, err := EnvelopeEncrypt(req.EvalKeys, localKEK)
		if err != nil {
			return nil, status.Error(codes.Internal, "local encryption failed")
		}
		
		log.Printf("WARN: Using local KEK for tenant %s (NOT FOR PRODUCTION)", req.TenantId)
		_ = encrypted
	}
	
	return &pb.UploadEvalKeysResponse{Success: true}, nil
}

func (s *cryptoServer) GetEvalKeys(ctx context.Context, req *pb.GetEvalKeysRequest) (*pb.GetEvalKeysResponse, error) {
	log.Printf("AUDIT: Retrieving eval keys for tenant %s, model %s", req.TenantId, req.CompiledModelId)
	
	// TODO: Retrieve encrypted keys from storage and decrypt
	// For now, return empty (would be implemented with storage integration)
	
	return nil, status.Error(codes.NotFound, "eval keys not found")
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

