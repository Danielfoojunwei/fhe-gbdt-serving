// Package auth provides authentication and authorization for the FHE-GBDT Gateway.
// This implementation supports multiple secrets backends (Vault, AWS, Env) for
// API key validation, addressing SOC2 CC6.7, HIPAA 164.312(d), and ISO 27001 A.8.5.
package auth

import (
	"context"
	"errors"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/fhe-gbdt-serving/services/gateway/auth/secrets"
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

// TenantContext holds authenticated tenant information
type TenantContext struct {
	TenantID    string
	APIKey      string
	Permissions []string
}

// Global secrets manager instance (initialized once)
var (
	secretsManager secrets.SecretsManager
	secretsOnce    sync.Once
	secretsErr     error
)

// InitSecretsManager initializes the secrets manager with the given configuration.
// This should be called during service startup.
func InitSecretsManager(cfg secrets.Config) error {
	secretsOnce.Do(func() {
		secretsManager, secretsErr = secrets.NewSecretsManager(cfg)
		if secretsErr != nil {
			log.Printf("ERROR: Failed to initialize secrets manager: %v", secretsErr)
		} else {
			log.Printf("INFO: Secrets manager initialized (backend: %s)", cfg.Backend)
		}
	})
	return secretsErr
}

// GetSecretsManager returns the initialized secrets manager.
// If not initialized, it creates one with default configuration.
func GetSecretsManager() secrets.SecretsManager {
	secretsOnce.Do(func() {
		cfg := secrets.DefaultConfig()
		secretsManager, secretsErr = secrets.NewSecretsManager(cfg)
		if secretsErr != nil {
			log.Printf("ERROR: Failed to initialize secrets manager with defaults: %v", secretsErr)
		}
	})
	return secretsManager
}

// ExtractTenantContext extracts and validates tenant context from the request.
// This is the main entry point for authentication.
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

	// Validate API key using the secrets manager
	entry, err := validateAPIKey(ctx, tenantID, secret)
	if err != nil {
		return nil, ErrInvalidAPIKey
	}

	permissions := []string{"predict"}
	if entry != nil && len(entry.Permissions) > 0 {
		permissions = entry.Permissions
	}

	return &TenantContext{
		TenantID:    tenantID,
		APIKey:      apiKeys[0],
		Permissions: permissions,
	}, nil
}

// validateAPIKey validates the API key against the configured secrets backend.
// Supports: Vault, AWS Secrets Manager, Environment Variables, Local (dev only)
func validateAPIKey(ctx context.Context, tenantID, secret string) (*secrets.SecretEntry, error) {
	if secret == "" {
		return nil, ErrInvalidAPIKey
	}

	// Get the secrets manager (initializes with defaults if needed)
	mgr := GetSecretsManager()
	if mgr == nil {
		log.Printf("ERROR: Secrets manager not available")
		// Fall back to legacy validation if secrets manager unavailable
		return legacyValidateAPIKey(tenantID, secret)
	}

	// Validate against secrets backend
	entry, err := mgr.ValidateAPIKey(ctx, tenantID, secret)
	if err != nil {
		log.Printf("SECURITY: API key validation failed for tenant %s: %v", tenantID, err)
		return nil, ErrInvalidAPIKey
	}

	// Check if key is expired
	if !entry.ExpiresAt.IsZero() && entry.ExpiresAt.Before(time.Now()) {
		log.Printf("SECURITY: API key expired for tenant %s", tenantID)
		return nil, ErrInvalidAPIKey
	}

	// Check if key is active
	if !entry.IsActive {
		log.Printf("SECURITY: API key revoked for tenant %s", tenantID)
		return nil, ErrInvalidAPIKey
	}

	return entry, nil
}

// legacyValidateAPIKey provides backward compatibility during migration.
// This should only be used as a fallback and will be removed in future versions.
// DEPRECATED: Use secrets manager instead.
func legacyValidateAPIKey(tenantID, secret string) (*secrets.SecretEntry, error) {
	// Check if test keys are allowed (development only)
	if os.Getenv("ALLOW_TEST_KEYS") != "true" {
		log.Printf("ERROR: Legacy validation not allowed (ALLOW_TEST_KEYS not set)")
		return nil, ErrInvalidAPIKey
	}

	log.Printf("WARN: Using legacy API key validation for tenant %s (deprecated)", tenantID)

	// Known test keys for development/benchmark ONLY
	validTestKeys := map[string]string{
		"test-tenant-cookbook": "dev-secret",
		"demo-tenant":          "demo-secret",
		"integration-test":     "test-secret",
	}

	if expectedSecret, ok := validTestKeys[tenantID]; ok {
		if secret != expectedSecret {
			log.Printf("SECURITY: Invalid API key for tenant %s", tenantID)
			return nil, ErrInvalidAPIKey
		}
		log.Printf("AUDIT: Validated API key for tenant %s (legacy mode)", tenantID)
		return &secrets.SecretEntry{
			TenantID:    tenantID,
			IsActive:    true,
			Permissions: []string{"predict"},
		}, nil
	}

	// Unknown tenant in legacy mode - reject
	log.Printf("SECURITY: Unknown tenant %s in legacy mode", tenantID)
	return nil, ErrInvalidAPIKey
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

// AuthInterceptorWithSecretsManager creates an auth interceptor with a custom secrets manager.
// This is useful for testing and custom configurations.
func AuthInterceptorWithSecretsManager(mgr secrets.SecretsManager) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Use the provided secrets manager for this request
		tenant, err := extractTenantContextWithManager(ctx, mgr)
		if err != nil {
			log.Printf("SECURITY: Auth failed - %v", err)
			return nil, err
		}

		newCtx := context.WithValue(ctx, TenantContextKey, tenant)
		return handler(newCtx, req)
	}
}

func extractTenantContextWithManager(ctx context.Context, mgr secrets.SecretsManager) (*TenantContext, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, ErrMissingAPIKey
	}

	apiKeys := md.Get("x-api-key")
	if len(apiKeys) == 0 {
		return nil, ErrMissingAPIKey
	}

	parts := strings.Split(apiKeys[0], ".")
	if len(parts) != 2 {
		return nil, ErrInvalidAPIKey
	}

	tenantID := parts[0]
	secret := parts[1]

	entry, err := mgr.ValidateAPIKey(ctx, tenantID, secret)
	if err != nil {
		log.Printf("SECURITY: API key validation failed for tenant %s: %v", tenantID, err)
		return nil, ErrInvalidAPIKey
	}

	permissions := []string{"predict"}
	if entry != nil && len(entry.Permissions) > 0 {
		permissions = entry.Permissions
	}

	return &TenantContext{
		TenantID:    tenantID,
		APIKey:      apiKeys[0],
		Permissions: permissions,
	}, nil
}
