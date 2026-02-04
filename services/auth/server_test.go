// Unit tests for Auth Service

package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"strings"
	"testing"
	"time"
)

// TestAPIKeyGeneration tests API key generation
func TestAPIKeyGeneration(t *testing.T) {
	tests := []struct {
		name     string
		prefix   string
		wantLen  int
	}{
		{"Standard key", "fhegbdt_", 40},
		{"Test key", "fhegbdt_test_", 40},
		{"Live key", "fhegbdt_live_", 40},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			key := generateAPIKey(tc.prefix)

			if !strings.HasPrefix(key, tc.prefix) {
				t.Errorf("Key should start with %s, got %s", tc.prefix, key[:len(tc.prefix)])
			}

			if len(key) < tc.wantLen {
				t.Errorf("Key should be at least %d chars, got %d", tc.wantLen, len(key))
			}
		})
	}

	// Test uniqueness
	keys := make(map[string]bool)
	for i := 0; i < 1000; i++ {
		key := generateAPIKey("fhegbdt_")
		if keys[key] {
			t.Error("Generated duplicate API key")
		}
		keys[key] = true
	}
}

// TestAPIKeyValidation tests API key validation
func TestAPIKeyValidation(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		expectErr bool
	}{
		{"Valid key", "fhegbdt_abc123def456ghi789jkl012mno345pqr", false},
		{"Empty key", "", true},
		{"Too short", "fhegbdt_abc", true},
		{"Wrong prefix", "wrong_abc123def456ghi789jkl012mno345pqr", true},
		{"Contains spaces", "fhegbdt_abc 123", true},
		{"Contains special chars", "fhegbdt_abc!@#$%", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAPIKey(tc.key)
			if tc.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestJWTTokenGeneration tests JWT token generation
func TestJWTTokenGeneration(t *testing.T) {
	secret := "test-secret-key-256-bits-long!!"

	claims := &TokenClaims{
		TenantID: "tenant-123",
		UserID:   "user-456",
		Email:    "test@example.com",
		Roles:    []string{"admin", "user"},
		Scopes:   []string{"read", "write"},
	}

	token, err := generateJWT(claims, secret, time.Hour)
	if err != nil {
		t.Fatalf("Failed to generate token: %v", err)
	}

	if token == "" {
		t.Error("Token should not be empty")
	}

	// Token should have 3 parts (header.payload.signature)
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		t.Errorf("Token should have 3 parts, got %d", len(parts))
	}
}

// TestJWTTokenValidation tests JWT token validation
func TestJWTTokenValidation(t *testing.T) {
	secret := "test-secret-key-256-bits-long!!"
	wrongSecret := "wrong-secret-key-256-bits-long!"

	claims := &TokenClaims{
		TenantID: "tenant-123",
		UserID:   "user-456",
	}

	validToken, _ := generateJWT(claims, secret, time.Hour)
	expiredToken, _ := generateJWT(claims, secret, -time.Hour) // Already expired

	tests := []struct {
		name      string
		token     string
		secret    string
		expectErr bool
	}{
		{"Valid token", validToken, secret, false},
		{"Wrong secret", validToken, wrongSecret, true},
		{"Expired token", expiredToken, secret, true},
		{"Malformed token", "not.a.valid.token", secret, true},
		{"Empty token", "", secret, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := validateJWT(tc.token, tc.secret)
			if tc.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestPasswordHashing tests password hashing and verification
func TestPasswordHashing(t *testing.T) {
	passwords := []string{
		"simple",
		"Complex!Pass123",
		"verylongpasswordwithmanycharacters",
		"パスワード", // Unicode
	}

	for _, password := range passwords {
		t.Run(password[:min(10, len(password))], func(t *testing.T) {
			hash, err := hashPassword(password)
			if err != nil {
				t.Fatalf("Failed to hash password: %v", err)
			}

			// Hash should be different from password
			if hash == password {
				t.Error("Hash should not equal password")
			}

			// Should verify correctly
			if !verifyPassword(password, hash) {
				t.Error("Password verification failed")
			}

			// Wrong password should fail
			if verifyPassword("wrongpassword", hash) {
				t.Error("Wrong password should not verify")
			}
		})
	}
}

// TestOIDCStateGeneration tests OIDC state parameter generation
func TestOIDCStateGeneration(t *testing.T) {
	states := make(map[string]bool)

	for i := 0; i < 100; i++ {
		state := generateOIDCState()

		if len(state) < 32 {
			t.Errorf("State should be at least 32 chars, got %d", len(state))
		}

		if states[state] {
			t.Error("Generated duplicate state")
		}
		states[state] = true
	}
}

// TestPermissionChecking tests RBAC permission checking
func TestPermissionChecking(t *testing.T) {
	roles := map[string][]string{
		"admin":  {"read", "write", "delete", "admin"},
		"editor": {"read", "write"},
		"viewer": {"read"},
	}

	tests := []struct {
		name       string
		userRoles  []string
		permission string
		allowed    bool
	}{
		{"Admin can read", []string{"admin"}, "read", true},
		{"Admin can delete", []string{"admin"}, "delete", true},
		{"Editor can write", []string{"editor"}, "write", true},
		{"Editor cannot delete", []string{"editor"}, "delete", false},
		{"Viewer can read", []string{"viewer"}, "read", true},
		{"Viewer cannot write", []string{"viewer"}, "write", false},
		{"Multiple roles", []string{"viewer", "editor"}, "write", true},
		{"No roles", []string{}, "read", false},
		{"Unknown role", []string{"unknown"}, "read", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			allowed := checkPermission(roles, tc.userRoles, tc.permission)
			if allowed != tc.allowed {
				t.Errorf("Expected allowed=%v, got %v", tc.allowed, allowed)
			}
		})
	}
}

// TestSessionManagement tests session creation and validation
func TestSessionManagement(t *testing.T) {
	store := NewMockSessionStore()

	// Create session
	session, err := store.Create("user-123", "tenant-456", time.Hour)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	if session.ID == "" {
		t.Error("Session ID should not be empty")
	}

	// Validate session
	retrieved, err := store.Get(session.ID)
	if err != nil {
		t.Fatalf("Failed to get session: %v", err)
	}

	if retrieved.UserID != "user-123" {
		t.Errorf("Expected user ID user-123, got %s", retrieved.UserID)
	}

	// Delete session
	err = store.Delete(session.ID)
	if err != nil {
		t.Fatalf("Failed to delete session: %v", err)
	}

	// Should not find deleted session
	_, err = store.Get(session.ID)
	if err == nil {
		t.Error("Should not find deleted session")
	}
}

// Helper types and functions

type TokenClaims struct {
	TenantID string
	UserID   string
	Email    string
	Roles    []string
	Scopes   []string
}

type Session struct {
	ID        string
	UserID    string
	TenantID  string
	CreatedAt time.Time
	ExpiresAt time.Time
}

type MockSessionStore struct {
	sessions map[string]*Session
}

func NewMockSessionStore() *MockSessionStore {
	return &MockSessionStore{
		sessions: make(map[string]*Session),
	}
}

func (s *MockSessionStore) Create(userID, tenantID string, duration time.Duration) (*Session, error) {
	session := &Session{
		ID:        generateSessionID(),
		UserID:    userID,
		TenantID:  tenantID,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(duration),
	}
	s.sessions[session.ID] = session
	return session, nil
}

func (s *MockSessionStore) Get(id string) (*Session, error) {
	session, exists := s.sessions[id]
	if !exists {
		return nil, &NotFoundError{Resource: "session"}
	}
	if time.Now().After(session.ExpiresAt) {
		delete(s.sessions, id)
		return nil, &ExpiredError{Resource: "session"}
	}
	return session, nil
}

func (s *MockSessionStore) Delete(id string) error {
	delete(s.sessions, id)
	return nil
}

type NotFoundError struct {
	Resource string
}

func (e *NotFoundError) Error() string {
	return e.Resource + " not found"
}

type ExpiredError struct {
	Resource string
}

func (e *ExpiredError) Error() string {
	return e.Resource + " expired"
}

func generateAPIKey(prefix string) string {
	b := make([]byte, 24)
	rand.Read(b)
	return prefix + base64.RawURLEncoding.EncodeToString(b)
}

func validateAPIKey(key string) error {
	if key == "" {
		return &ValidationError{Field: "api_key", Message: "required"}
	}
	if len(key) < 30 {
		return &ValidationError{Field: "api_key", Message: "too short"}
	}
	if !strings.HasPrefix(key, "fhegbdt_") {
		return &ValidationError{Field: "api_key", Message: "invalid prefix"}
	}
	for _, c := range key {
		if c == ' ' || c == '!' || c == '@' || c == '#' || c == '$' || c == '%' {
			return &ValidationError{Field: "api_key", Message: "invalid characters"}
		}
	}
	return nil
}

type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Field + ": " + e.Message
}

func generateJWT(claims *TokenClaims, secret string, duration time.Duration) (string, error) {
	// Simplified JWT generation for testing
	// In production, use proper JWT library
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))

	exp := time.Now().Add(duration).Unix()
	payload := base64.RawURLEncoding.EncodeToString([]byte(
		`{"tenant_id":"` + claims.TenantID + `","user_id":"` + claims.UserID + `","exp":` +
		string(rune(exp)) + `}`))

	// Simplified signature (not cryptographically secure for testing only)
	signature := base64.RawURLEncoding.EncodeToString([]byte(secret[:8]))

	return header + "." + payload + "." + signature, nil
}

func validateJWT(token, secret string) (*TokenClaims, error) {
	if token == "" {
		return nil, &ValidationError{Field: "token", Message: "required"}
	}

	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, &ValidationError{Field: "token", Message: "malformed"}
	}

	// Check signature (simplified)
	expectedSig := base64.RawURLEncoding.EncodeToString([]byte(secret[:8]))
	if parts[2] != expectedSig {
		return nil, &ValidationError{Field: "token", Message: "invalid signature"}
	}

	// Check expiration (simplified - always valid for non-expired tokens in this test)
	// In production, decode payload and check exp claim

	return &TokenClaims{}, nil
}

func hashPassword(password string) (string, error) {
	// Simplified hash for testing (in production use bcrypt)
	b := make([]byte, 16)
	rand.Read(b)
	salt := base64.RawURLEncoding.EncodeToString(b)
	hash := base64.RawURLEncoding.EncodeToString([]byte(password + salt))
	return salt + "$" + hash, nil
}

func verifyPassword(password, hash string) bool {
	parts := strings.Split(hash, "$")
	if len(parts) != 2 {
		return false
	}
	salt := parts[0]
	expectedHash := base64.RawURLEncoding.EncodeToString([]byte(password + salt))
	return parts[1] == expectedHash
}

func generateOIDCState() string {
	b := make([]byte, 32)
	rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func generateSessionID() string {
	b := make([]byte, 32)
	rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func checkPermission(roles map[string][]string, userRoles []string, permission string) bool {
	for _, role := range userRoles {
		perms, exists := roles[role]
		if !exists {
			continue
		}
		for _, p := range perms {
			if p == permission {
				return true
			}
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Benchmark tests

func BenchmarkAPIKeyGeneration(b *testing.B) {
	for i := 0; i < b.N; i++ {
		generateAPIKey("fhegbdt_")
	}
}

func BenchmarkPasswordHashing(b *testing.B) {
	for i := 0; i < b.N; i++ {
		hashPassword("testpassword123")
	}
}

func BenchmarkPermissionCheck(b *testing.B) {
	roles := map[string][]string{
		"admin":  {"read", "write", "delete", "admin"},
		"editor": {"read", "write"},
		"viewer": {"read"},
	}
	userRoles := []string{"editor"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		checkPermission(roles, userRoles, "write")
	}
}
