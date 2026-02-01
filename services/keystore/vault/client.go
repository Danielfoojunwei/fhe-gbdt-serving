package vault

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	vault "github.com/hashicorp/vault/api"
)

var (
	ErrKeyNotFound    = errors.New("key not found")
	ErrVaultNotReady  = errors.New("vault client not initialized")
	ErrUnwrapFailed   = errors.New("failed to unwrap key")
)

// KeyVaultClient provides secure key management using HashiCorp Vault
type KeyVaultClient struct {
	client     *vault.Client
	mountPath  string
	mu         sync.RWMutex
	keyCache   map[string][]byte
	cacheTTL   time.Duration
	cacheTime  map[string]time.Time
}

// Config holds Vault client configuration
type Config struct {
	Address   string        // Vault server address
	Token     string        // Vault token (or use VAULT_TOKEN env)
	MountPath string        // Transit secrets engine mount path
	CacheTTL  time.Duration // Key cache TTL
}

// NewKeyVaultClient creates a new Vault-backed key manager
func NewKeyVaultClient(cfg Config) (*KeyVaultClient, error) {
	// Use environment variables if not provided
	if cfg.Address == "" {
		cfg.Address = os.Getenv("VAULT_ADDR")
	}
	if cfg.Token == "" {
		cfg.Token = os.Getenv("VAULT_TOKEN")
	}
	if cfg.MountPath == "" {
		cfg.MountPath = "transit"
	}
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = 5 * time.Minute
	}

	if cfg.Address == "" {
		return nil, fmt.Errorf("vault address not configured")
	}

	vaultConfig := vault.DefaultConfig()
	vaultConfig.Address = cfg.Address

	client, err := vault.NewClient(vaultConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}

	if cfg.Token != "" {
		client.SetToken(cfg.Token)
	}

	// Verify connection
	_, err = client.Sys().Health()
	if err != nil {
		log.Printf("WARN: Vault health check failed: %v", err)
	}

	return &KeyVaultClient{
		client:    client,
		mountPath: cfg.MountPath,
		keyCache:  make(map[string][]byte),
		cacheTTL:  cfg.CacheTTL,
		cacheTime: make(map[string]time.Time),
	}, nil
}

// GetKEK retrieves or generates a Key Encryption Key for a tenant
func (kv *KeyVaultClient) GetKEK(ctx context.Context, tenantID string) ([]byte, error) {
	keyName := fmt.Sprintf("tenant-%s-kek", tenantID)

	// Check cache first
	kv.mu.RLock()
	if cached, ok := kv.keyCache[keyName]; ok {
		if time.Since(kv.cacheTime[keyName]) < kv.cacheTTL {
			kv.mu.RUnlock()
			return cached, nil
		}
	}
	kv.mu.RUnlock()

	// Ensure key exists in Vault
	if err := kv.ensureKey(ctx, keyName); err != nil {
		return nil, err
	}

	// Generate a data key (wrapped)
	path := fmt.Sprintf("%s/datakey/plaintext/%s", kv.mountPath, keyName)
	secret, err := kv.client.Logical().WriteWithContext(ctx, path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate data key: %w", err)
	}

	// Extract plaintext key
	plaintextB64, ok := secret.Data["plaintext"].(string)
	if !ok {
		return nil, ErrUnwrapFailed
	}

	keyBytes, err := base64.StdEncoding.DecodeString(plaintextB64)
	if err != nil {
		return nil, fmt.Errorf("failed to decode key: %w", err)
	}

	// Cache the key
	kv.mu.Lock()
	kv.keyCache[keyName] = keyBytes
	kv.cacheTime[keyName] = time.Now()
	kv.mu.Unlock()

	log.Printf("AUDIT: Retrieved KEK for tenant %s from Vault", tenantID)
	return keyBytes, nil
}

// ensureKey creates the transit key if it doesn't exist
func (kv *KeyVaultClient) ensureKey(ctx context.Context, keyName string) error {
	path := fmt.Sprintf("%s/keys/%s", kv.mountPath, keyName)
	
	// Check if key exists
	secret, err := kv.client.Logical().ReadWithContext(ctx, path)
	if err == nil && secret != nil {
		return nil // Key exists
	}

	// Create key
	_, err = kv.client.Logical().WriteWithContext(ctx, path, map[string]interface{}{
		"type":                 "aes256-gcm96",
		"deletion_allowed":     false,
		"exportable":           false,
		"allow_plaintext_backup": false,
	})
	if err != nil {
		return fmt.Errorf("failed to create transit key: %w", err)
	}

	log.Printf("AUDIT: Created new transit key %s in Vault", keyName)
	return nil
}

// EncryptWithKEK encrypts data using the tenant's KEK via Vault Transit
func (kv *KeyVaultClient) EncryptWithKEK(ctx context.Context, tenantID string, plaintext []byte) (string, error) {
	keyName := fmt.Sprintf("tenant-%s-kek", tenantID)
	
	if err := kv.ensureKey(ctx, keyName); err != nil {
		return "", err
	}

	path := fmt.Sprintf("%s/encrypt/%s", kv.mountPath, keyName)
	secret, err := kv.client.Logical().WriteWithContext(ctx, path, map[string]interface{}{
		"plaintext": base64.StdEncoding.EncodeToString(plaintext),
	})
	if err != nil {
		return "", fmt.Errorf("failed to encrypt: %w", err)
	}

	ciphertext, ok := secret.Data["ciphertext"].(string)
	if !ok {
		return "", errors.New("no ciphertext in response")
	}

	return ciphertext, nil
}

// DecryptWithKEK decrypts data using the tenant's KEK via Vault Transit
func (kv *KeyVaultClient) DecryptWithKEK(ctx context.Context, tenantID, ciphertext string) ([]byte, error) {
	keyName := fmt.Sprintf("tenant-%s-kek", tenantID)

	path := fmt.Sprintf("%s/decrypt/%s", kv.mountPath, keyName)
	secret, err := kv.client.Logical().WriteWithContext(ctx, path, map[string]interface{}{
		"ciphertext": ciphertext,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	plaintextB64, ok := secret.Data["plaintext"].(string)
	if !ok {
		return nil, errors.New("no plaintext in response")
	}

	return base64.StdEncoding.DecodeString(plaintextB64)
}

// RotateKEK rotates the tenant's KEK
func (kv *KeyVaultClient) RotateKEK(ctx context.Context, tenantID string) error {
	keyName := fmt.Sprintf("tenant-%s-kek", tenantID)
	path := fmt.Sprintf("%s/keys/%s/rotate", kv.mountPath, keyName)

	_, err := kv.client.Logical().WriteWithContext(ctx, path, nil)
	if err != nil {
		return fmt.Errorf("failed to rotate key: %w", err)
	}

	// Invalidate cache
	kv.mu.Lock()
	delete(kv.keyCache, keyName)
	delete(kv.cacheTime, keyName)
	kv.mu.Unlock()

	log.Printf("AUDIT: Rotated KEK for tenant %s", tenantID)
	return nil
}

// Close releases resources
func (kv *KeyVaultClient) Close() error {
	kv.mu.Lock()
	defer kv.mu.Unlock()
	
	// Clear cache
	kv.keyCache = nil
	kv.cacheTime = nil
	
	return nil
}
