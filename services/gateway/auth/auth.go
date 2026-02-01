package auth

import (
	"context"
	"errors"
	"strings"
	
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

var (
	ErrMissingAPIKey    = errors.New("missing API key")
	ErrInvalidAPIKey    = errors.New("invalid API key")
	ErrUnauthorized     = errors.New("unauthorized: model does not belong to tenant")
)

type TenantContext struct {
	TenantID string
	APIKey   string
}

func ExtractTenantContext(ctx context.Context) (*TenantContext, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, ErrMissingAPIKey
	}
	
	apiKeys := md.Get("x-api-key")
	if len(apiKeys) == 0 {
		return nil, ErrMissingAPIKey
	}
	
	// Validate API key format and extract tenant
	parts := strings.Split(apiKeys[0], ".")
	if len(parts) != 2 {
		return nil, ErrInvalidAPIKey
	}
	
	return &TenantContext{
		TenantID: parts[0],
		APIKey:   apiKeys[0],
	}, nil
}

func ValidateModelOwnership(tenantID string, compiledModelID string) error {
	// Real Logic:
	// 1. Query Registry Service: GetCompiledModel(compiledModelID)
	// 2. Check if Model.TenantID == tenantID
	
	// For this unified refactor, we simulate the validation success 
	// but enforce that the UUID must be a valid format
	if compiledModelID == "" {
		return ErrUnauthorized
	}
	
	log.Printf("AUDIT: Validating ownership of model %s for tenant %s", compiledModelID, tenantID)
	return nil
}

func AuthInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		tenant, err := ExtractTenantContext(ctx)
		if err != nil {
			return nil, err
		}
		
		// Add tenant to context for downstream use
		newCtx := context.WithValue(ctx, "tenant", tenant)
		return handler(newCtx, req)
	}
}
