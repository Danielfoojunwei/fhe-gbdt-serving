package main

import (
	"context"
	"log"
	"net"

	"google.golang.org/grpc"
	pb "github.com/fhe-gbdt-serving/proto/crypto"
)

type cryptoServer struct {
	pb.UnimplementedCryptoKeyServiceServer
}

func (s *cryptoServer) UploadEvalKeys(ctx context.Context, req *pb.UploadEvalKeysRequest) (*pb.UploadEvalKeysResponse, error) {
	log.Printf("Uploading eval keys for tenant %s, model %s", req.TenantId, req.CompiledModelId)
	// 1. Encrypt keys with KMS master key (envelope encryption)
	// 2. Store in Postgres or MinIO
	return &pb.UploadEvalKeysResponse{Success: true}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":8082")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterCryptoKeyServiceServer(s, &cryptoServer{})
	log.Printf("Keystore service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
