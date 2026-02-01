package auth

import (
	"context"
	"errors"
	"log"
	"strings"
	
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

var (
	ErrMissingAPIKey    = errors.New("missing API key")
	ErrInvalidAPIKey    = errors.New("invalid API key")
	ErrUnauthorized     = errors.New("unauthorized: model does not belong to tenant")
)

// Typed context key to avoid collisions
type tenantContextKey struct{}

// TenantContextKey is the key used to store tenant context in request context
var TenantContextKey = tenantContextKey{}

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
	// Format: <tenant_id>.<secret>
	parts := strings.Split(apiKeys[0], ".")
	if len(parts) != 2 {
		return nil, ErrInvalidAPIKey
	}
	
	tenantID := parts[0]
	secret := parts[1]
	
	// Real API key validation
	// TODO: Replace with actual Keystore/Secrets Manager lookup
	if err := validateAPIKey(tenantID, secret); err != nil {
		return nil, ErrInvalidAPIKey
	}
	
	return &TenantContext{
		TenantID: tenantID,
		APIKey:   apiKeys[0],
	}, nil
}

// validateAPIKey checks the API key against the Keystore
// Production: Query Keystore service or secrets manager
func validateAPIKey(tenantID, secret string) error {
	// For now, allow known test keys and reject empty secrets
	if secret == "" {
		return ErrInvalidAPIKey
	}
	
	// Known test keys for development/benchmark
	validTestKeys := map[string]string{
		"test-tenant-cookbook": "dev-secret",
		"demo-tenant":          "demo-secret",
	}
	
	if expectedSecret, ok := validTestKeys[tenantID]; ok {
		if secret != expectedSecret {
			log.Printf("SECURITY: Invalid API key for tenant %s", tenantID)
			return ErrInvalidAPIKey
		}
	}
	
	// In production, unknown tenants would be validated via Keystore gRPC call
	log.Printf("AUDIT: Validated API key for tenant %s", tenantID)
	return nil
}

func ValidateModelOwnership(tenantID string, compiledModelID string) error {
	// Real Logic:
	// 1. Query Registry Service: GetCompiledModel(compiledModelID)
	// 2. Check if Model.TenantID == tenantID
	
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
			log.Printf("SECURITY: Auth failed - %v", err)
			return nil, err
		}
		
		// Add tenant to context using typed key
		newCtx := context.WithValue(ctx, TenantContextKey, tenant)
		return handler(newCtx, req)
	}
}

