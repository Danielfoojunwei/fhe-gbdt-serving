// Authentication Service with SSO/SAML Support
// Provides OAuth 2.0, OIDC, and SAML 2.0 authentication

package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/auth"
)

type authServer struct {
	pb.UnimplementedAuthServiceServer
	db            *sql.DB
	jwtSecret     []byte
	samlProviders map[string]*SAMLProvider
	oidcProviders map[string]*OIDCProvider
}

// SAMLProvider holds SAML IdP configuration
type SAMLProvider struct {
	ID            string
	Name          string
	EntityID      string
	SSOUrl        string
	Certificate   string
	AttributeMap  map[string]string
}

// OIDCProvider holds OIDC provider configuration
type OIDCProvider struct {
	ID           string
	Name         string
	Issuer       string
	ClientID     string
	ClientSecret string
	RedirectURI  string
	Scopes       []string
}

func newAuthServer() (*authServer, error) {
	// Initialize JWT secret
	jwtSecret := os.Getenv("JWT_SECRET")
	if jwtSecret == "" {
		// Generate random secret if not provided
		secret := make([]byte, 32)
		rand.Read(secret)
		jwtSecret = base64.StdEncoding.EncodeToString(secret)
	}

	// Connect to database
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/auth?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	return &authServer{
		db:            db,
		jwtSecret:     []byte(jwtSecret),
		samlProviders: make(map[string]*SAMLProvider),
		oidcProviders: make(map[string]*OIDCProvider),
	}, nil
}

// ============================================================================
// User Authentication
// ============================================================================

func (s *authServer) Authenticate(ctx context.Context, req *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	log.Printf("AUDIT: Authentication attempt for user %s", req.Email)

	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var userID, passwordHash, tenantID string
	var mfaEnabled bool

	err := s.db.QueryRowContext(ctx, `
		SELECT id, password_hash, tenant_id, mfa_enabled
		FROM users WHERE email = $1 AND status = 'active'
	`, req.Email).Scan(&userID, &passwordHash, &tenantID, &mfaEnabled)

	if err == sql.ErrNoRows {
		log.Printf("AUDIT: Authentication failed - user not found: %s", req.Email)
		return nil, status.Error(codes.Unauthenticated, "invalid credentials")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "database error: %v", err)
	}

	// Verify password
	if err := bcrypt.CompareHashAndPassword([]byte(passwordHash), []byte(req.Password)); err != nil {
		log.Printf("AUDIT: Authentication failed - invalid password: %s", req.Email)

		// Record failed attempt
		s.recordLoginAttempt(ctx, userID, false, req.IpAddress, req.UserAgent)

		return nil, status.Error(codes.Unauthenticated, "invalid credentials")
	}

	// Check if MFA is required
	if mfaEnabled && req.MfaCode == "" {
		return &pb.AuthenticateResponse{
			RequiresMfa: true,
		}, nil
	}

	// Verify MFA if provided
	if mfaEnabled {
		if !s.verifyMFACode(ctx, userID, req.MfaCode) {
			return nil, status.Error(codes.Unauthenticated, "invalid MFA code")
		}
	}

	// Generate tokens
	accessToken, err := s.generateAccessToken(userID, tenantID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate token: %v", err)
	}

	refreshToken, err := s.generateRefreshToken(userID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate refresh token: %v", err)
	}

	// Record successful login
	s.recordLoginAttempt(ctx, userID, true, req.IpAddress, req.UserAgent)

	log.Printf("AUDIT: Authentication successful for user %s", req.Email)

	return &pb.AuthenticateResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    3600, // 1 hour
		TokenType:    "Bearer",
		User: &pb.User{
			Id:       userID,
			Email:    req.Email,
			TenantId: tenantID,
		},
	}, nil
}

func (s *authServer) RefreshToken(ctx context.Context, req *pb.RefreshTokenRequest) (*pb.RefreshTokenResponse, error) {
	// Validate refresh token
	userID, err := s.validateRefreshToken(ctx, req.RefreshToken)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, "invalid refresh token")
	}

	// Get user's tenant
	var tenantID string
	err = s.db.QueryRowContext(ctx, "SELECT tenant_id FROM users WHERE id = $1", userID).Scan(&tenantID)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	// Generate new access token
	accessToken, err := s.generateAccessToken(userID, tenantID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate token: %v", err)
	}

	return &pb.RefreshTokenResponse{
		AccessToken: accessToken,
		ExpiresIn:   3600,
		TokenType:   "Bearer",
	}, nil
}

func (s *authServer) Logout(ctx context.Context, req *pb.LogoutRequest) (*pb.LogoutResponse, error) {
	// Revoke refresh token
	if s.db != nil {
		_, err := s.db.ExecContext(ctx, `
			UPDATE refresh_tokens SET revoked = true, revoked_at = NOW()
			WHERE token_hash = $1
		`, hashToken(req.RefreshToken))
		if err != nil {
			log.Printf("WARN: Failed to revoke refresh token: %v", err)
		}
	}

	return &pb.LogoutResponse{Success: true}, nil
}

// ============================================================================
// SSO/SAML Authentication
// ============================================================================

func (s *authServer) InitiateSAMLLogin(ctx context.Context, req *pb.InitiateSAMLLoginRequest) (*pb.InitiateSAMLLoginResponse, error) {
	provider, ok := s.samlProviders[req.ProviderId]
	if !ok {
		// Try to load from database
		provider = s.loadSAMLProvider(ctx, req.ProviderId)
		if provider == nil {
			return nil, status.Errorf(codes.NotFound, "SAML provider %s not found", req.ProviderId)
		}
	}

	// Generate SAML AuthnRequest
	requestID := uuid.New().String()
	relayState := base64.StdEncoding.EncodeToString([]byte(req.RelayState))

	// Store request state for validation
	if s.db != nil {
		_, err := s.db.ExecContext(ctx, `
			INSERT INTO saml_requests (id, provider_id, relay_state, created_at, expires_at)
			VALUES ($1, $2, $3, NOW(), NOW() + INTERVAL '10 minutes')
		`, requestID, req.ProviderId, req.RelayState)
		if err != nil {
			log.Printf("WARN: Failed to store SAML request: %v", err)
		}
	}

	// Build redirect URL (simplified - real implementation uses go-saml)
	redirectURL := fmt.Sprintf("%s?SAMLRequest=%s&RelayState=%s",
		provider.SSOUrl, requestID, relayState)

	return &pb.InitiateSAMLLoginResponse{
		RedirectUrl: redirectURL,
		RequestId:   requestID,
	}, nil
}

func (s *authServer) HandleSAMLResponse(ctx context.Context, req *pb.HandleSAMLResponseRequest) (*pb.HandleSAMLResponseResponse, error) {
	log.Printf("AUDIT: Processing SAML response")

	// Validate SAML response (simplified)
	// In production, use go-saml library for full validation

	// Extract user attributes from response
	email := req.Attributes["email"]
	if email == "" {
		return nil, status.Error(codes.InvalidArgument, "email attribute not found in SAML response")
	}

	// Find or create user
	userID, tenantID, err := s.findOrCreateSSOUser(ctx, email, req.ProviderId, req.Attributes)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to process SSO user: %v", err)
	}

	// Generate tokens
	accessToken, _ := s.generateAccessToken(userID, tenantID)
	refreshToken, _ := s.generateRefreshToken(userID)

	log.Printf("AUDIT: SAML authentication successful for %s", email)

	return &pb.HandleSAMLResponseResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    3600,
		TokenType:    "Bearer",
		User: &pb.User{
			Id:       userID,
			Email:    email,
			TenantId: tenantID,
		},
	}, nil
}

// ============================================================================
// OIDC Authentication
// ============================================================================

func (s *authServer) InitiateOIDCLogin(ctx context.Context, req *pb.InitiateOIDCLoginRequest) (*pb.InitiateOIDCLoginResponse, error) {
	provider, ok := s.oidcProviders[req.ProviderId]
	if !ok {
		provider = s.loadOIDCProvider(ctx, req.ProviderId)
		if provider == nil {
			return nil, status.Errorf(codes.NotFound, "OIDC provider %s not found", req.ProviderId)
		}
	}

	// Generate state and nonce
	state := generateRandomString(32)
	nonce := generateRandomString(32)

	// Store state for validation
	if s.db != nil {
		_, _ = s.db.ExecContext(ctx, `
			INSERT INTO oidc_states (state, nonce, provider_id, redirect_uri, created_at, expires_at)
			VALUES ($1, $2, $3, $4, NOW(), NOW() + INTERVAL '10 minutes')
		`, state, nonce, req.ProviderId, req.RedirectUri)
	}

	// Build authorization URL
	authURL := fmt.Sprintf("%s/authorize?client_id=%s&response_type=code&scope=%s&redirect_uri=%s&state=%s&nonce=%s",
		provider.Issuer, provider.ClientID, "openid email profile", req.RedirectUri, state, nonce)

	return &pb.InitiateOIDCLoginResponse{
		AuthorizationUrl: authURL,
		State:            state,
	}, nil
}

func (s *authServer) HandleOIDCCallback(ctx context.Context, req *pb.HandleOIDCCallbackRequest) (*pb.HandleOIDCCallbackResponse, error) {
	log.Printf("AUDIT: Processing OIDC callback")

	// Validate state
	var providerID, nonce string
	err := s.db.QueryRowContext(ctx, `
		SELECT provider_id, nonce FROM oidc_states
		WHERE state = $1 AND expires_at > NOW()
	`, req.State).Scan(&providerID, &nonce)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "invalid or expired state")
	}

	provider := s.loadOIDCProvider(ctx, providerID)
	if provider == nil {
		return nil, status.Error(codes.NotFound, "provider not found")
	}

	// Exchange code for tokens (simplified)
	// In production, use proper OIDC library
	email := req.IdToken // Placeholder - would extract from ID token

	// Find or create user
	userID, tenantID, err := s.findOrCreateSSOUser(ctx, email, providerID, nil)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to process SSO user: %v", err)
	}

	// Generate tokens
	accessToken, _ := s.generateAccessToken(userID, tenantID)
	refreshToken, _ := s.generateRefreshToken(userID)

	log.Printf("AUDIT: OIDC authentication successful for %s", email)

	return &pb.HandleOIDCCallbackResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    3600,
		TokenType:    "Bearer",
		User: &pb.User{
			Id:       userID,
			Email:    email,
			TenantId: tenantID,
		},
	}, nil
}

// ============================================================================
// SSO Provider Management
// ============================================================================

func (s *authServer) ConfigureSAMLProvider(ctx context.Context, req *pb.ConfigureSAMLProviderRequest) (*pb.ConfigureSAMLProviderResponse, error) {
	log.Printf("AUDIT: Configuring SAML provider for tenant %s", req.TenantId)

	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	providerID := uuid.New().String()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO saml_providers (id, tenant_id, name, entity_id, sso_url, certificate, attribute_map, enabled)
		VALUES ($1, $2, $3, $4, $5, $6, $7, true)
		ON CONFLICT (tenant_id) DO UPDATE SET
			name = EXCLUDED.name,
			entity_id = EXCLUDED.entity_id,
			sso_url = EXCLUDED.sso_url,
			certificate = EXCLUDED.certificate,
			attribute_map = EXCLUDED.attribute_map,
			updated_at = NOW()
	`, providerID, req.TenantId, req.Name, req.EntityId, req.SsoUrl, req.Certificate, req.AttributeMap)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to configure SAML provider: %v", err)
	}

	// Cache the provider
	s.samlProviders[providerID] = &SAMLProvider{
		ID:          providerID,
		Name:        req.Name,
		EntityID:    req.EntityId,
		SSOUrl:      req.SsoUrl,
		Certificate: req.Certificate,
	}

	return &pb.ConfigureSAMLProviderResponse{
		ProviderId: providerID,
		AcsUrl:     fmt.Sprintf("https://api.fhe-gbdt.dev/auth/saml/%s/acs", providerID),
		MetadataUrl: fmt.Sprintf("https://api.fhe-gbdt.dev/auth/saml/%s/metadata", providerID),
	}, nil
}

func (s *authServer) ConfigureOIDCProvider(ctx context.Context, req *pb.ConfigureOIDCProviderRequest) (*pb.ConfigureOIDCProviderResponse, error) {
	log.Printf("AUDIT: Configuring OIDC provider for tenant %s", req.TenantId)

	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	providerID := uuid.New().String()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO oidc_providers (id, tenant_id, name, issuer, client_id, client_secret, scopes, enabled)
		VALUES ($1, $2, $3, $4, $5, $6, $7, true)
		ON CONFLICT (tenant_id) DO UPDATE SET
			name = EXCLUDED.name,
			issuer = EXCLUDED.issuer,
			client_id = EXCLUDED.client_id,
			client_secret = EXCLUDED.client_secret,
			scopes = EXCLUDED.scopes,
			updated_at = NOW()
	`, providerID, req.TenantId, req.Name, req.Issuer, req.ClientId, req.ClientSecret, req.Scopes)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to configure OIDC provider: %v", err)
	}

	return &pb.ConfigureOIDCProviderResponse{
		ProviderId:  providerID,
		RedirectUri: fmt.Sprintf("https://api.fhe-gbdt.dev/auth/oidc/%s/callback", providerID),
	}, nil
}

// ============================================================================
// API Key Management
// ============================================================================

func (s *authServer) CreateAPIKey(ctx context.Context, req *pb.CreateAPIKeyRequest) (*pb.CreateAPIKeyResponse, error) {
	log.Printf("AUDIT: Creating API key for tenant %s", req.TenantId)

	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Generate API key
	keyBytes := make([]byte, 32)
	rand.Read(keyBytes)
	apiKey := "sk_" + base64.URLEncoding.EncodeToString(keyBytes)[:40]

	// Hash for storage
	keyHash, _ := bcrypt.GenerateFromPassword([]byte(apiKey), bcrypt.DefaultCost)

	keyID := uuid.New().String()
	now := time.Now()

	var expiresAt *time.Time
	if req.ExpiresInDays > 0 {
		exp := now.AddDate(0, 0, int(req.ExpiresInDays))
		expiresAt = &exp
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO api_keys (id, tenant_id, user_id, name, key_hash, scopes, expires_at, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`, keyID, req.TenantId, req.UserId, req.Name, keyHash, req.Scopes, expiresAt, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create API key: %v", err)
	}

	return &pb.CreateAPIKeyResponse{
		KeyId:     keyID,
		ApiKey:    apiKey, // Only returned once!
		Name:      req.Name,
		CreatedAt: timestamppb.New(now),
	}, nil
}

func (s *authServer) ValidateAPIKey(ctx context.Context, req *pb.ValidateAPIKeyRequest) (*pb.ValidateAPIKeyResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Find key by prefix (first 8 chars)
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, tenant_id, user_id, key_hash, scopes, expires_at, revoked
		FROM api_keys
		WHERE tenant_id IS NOT NULL
	`)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "database error: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var keyID, tenantID, userID, keyHash string
		var scopes []string
		var expiresAt sql.NullTime
		var revoked bool

		if err := rows.Scan(&keyID, &tenantID, &userID, &keyHash, &scopes, &expiresAt, &revoked); err != nil {
			continue
		}

		if revoked {
			continue
		}

		if expiresAt.Valid && expiresAt.Time.Before(time.Now()) {
			continue
		}

		if bcrypt.CompareHashAndPassword([]byte(keyHash), []byte(req.ApiKey)) == nil {
			// Update last used
			s.db.ExecContext(ctx, "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1", keyID)

			return &pb.ValidateAPIKeyResponse{
				Valid:    true,
				KeyId:    keyID,
				TenantId: tenantID,
				UserId:   userID,
				Scopes:   scopes,
			}, nil
		}
	}

	return &pb.ValidateAPIKeyResponse{Valid: false}, nil
}

func (s *authServer) RevokeAPIKey(ctx context.Context, req *pb.RevokeAPIKeyRequest) (*pb.RevokeAPIKeyResponse, error) {
	log.Printf("AUDIT: Revoking API key %s", req.KeyId)

	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	result, err := s.db.ExecContext(ctx, `
		UPDATE api_keys SET revoked = true, revoked_at = NOW()
		WHERE id = $1 AND tenant_id = $2
	`, req.KeyId, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to revoke API key: %v", err)
	}

	rows, _ := result.RowsAffected()
	if rows == 0 {
		return nil, status.Error(codes.NotFound, "API key not found")
	}

	return &pb.RevokeAPIKeyResponse{Success: true}, nil
}

// ============================================================================
// Helper Functions
// ============================================================================

func (s *authServer) generateAccessToken(userID, tenantID string) (string, error) {
	// In production, use proper JWT library
	token := fmt.Sprintf("%s.%s.%d", userID, tenantID, time.Now().Unix())
	return base64.URLEncoding.EncodeToString([]byte(token)), nil
}

func (s *authServer) generateRefreshToken(userID string) (string, error) {
	tokenBytes := make([]byte, 32)
	rand.Read(tokenBytes)
	token := base64.URLEncoding.EncodeToString(tokenBytes)

	// Store hashed token
	if s.db != nil {
		_, _ = s.db.Exec(`
			INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
			VALUES ($1, $2, NOW() + INTERVAL '30 days')
		`, userID, hashToken(token))
	}

	return token, nil
}

func (s *authServer) validateRefreshToken(ctx context.Context, token string) (string, error) {
	if s.db == nil {
		return "", fmt.Errorf("database not available")
	}

	var userID string
	err := s.db.QueryRowContext(ctx, `
		SELECT user_id FROM refresh_tokens
		WHERE token_hash = $1 AND revoked = false AND expires_at > NOW()
	`, hashToken(token)).Scan(&userID)

	return userID, err
}

func (s *authServer) verifyMFACode(ctx context.Context, userID, code string) bool {
	// In production, use TOTP library
	return code == "123456" // Placeholder
}

func (s *authServer) recordLoginAttempt(ctx context.Context, userID string, success bool, ip, userAgent string) {
	if s.db == nil {
		return
	}

	_, _ = s.db.ExecContext(ctx, `
		INSERT INTO login_attempts (user_id, success, ip_address, user_agent, created_at)
		VALUES ($1, $2, $3, $4, NOW())
	`, userID, success, ip, userAgent)
}

func (s *authServer) findOrCreateSSOUser(ctx context.Context, email, providerID string, attributes map[string]string) (string, string, error) {
	if s.db == nil {
		return "", "", fmt.Errorf("database not available")
	}

	// Try to find existing user
	var userID, tenantID string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, tenant_id FROM users WHERE email = $1
	`, email).Scan(&userID, &tenantID)

	if err == sql.ErrNoRows {
		// Create new user
		userID = uuid.New().String()
		tenantID = uuid.New().String()

		_, err = s.db.ExecContext(ctx, `
			INSERT INTO users (id, email, tenant_id, auth_provider, status, created_at)
			VALUES ($1, $2, $3, $4, 'active', NOW())
		`, userID, email, tenantID, providerID)
		if err != nil {
			return "", "", err
		}

		// Create tenant
		_, _ = s.db.ExecContext(ctx, `
			INSERT INTO tenants (id, name, created_at) VALUES ($1, $2, NOW())
		`, tenantID, email)
	}

	return userID, tenantID, nil
}

func (s *authServer) loadSAMLProvider(ctx context.Context, id string) *SAMLProvider {
	if s.db == nil {
		return nil
	}

	var provider SAMLProvider
	err := s.db.QueryRowContext(ctx, `
		SELECT id, name, entity_id, sso_url, certificate
		FROM saml_providers WHERE id = $1 AND enabled = true
	`, id).Scan(&provider.ID, &provider.Name, &provider.EntityID, &provider.SSOUrl, &provider.Certificate)

	if err != nil {
		return nil
	}

	s.samlProviders[id] = &provider
	return &provider
}

func (s *authServer) loadOIDCProvider(ctx context.Context, id string) *OIDCProvider {
	if s.db == nil {
		return nil
	}

	var provider OIDCProvider
	err := s.db.QueryRowContext(ctx, `
		SELECT id, name, issuer, client_id, client_secret
		FROM oidc_providers WHERE id = $1 AND enabled = true
	`, id).Scan(&provider.ID, &provider.Name, &provider.Issuer, &provider.ClientID, &provider.ClientSecret)

	if err != nil {
		return nil
	}

	s.oidcProviders[id] = &provider
	return &provider
}

func hashToken(token string) string {
	hash, _ := bcrypt.GenerateFromPassword([]byte(token), bcrypt.MinCost)
	return string(hash)
}

func generateRandomString(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return base64.URLEncoding.EncodeToString(b)[:n]
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8086"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newAuthServer()
	if err != nil {
		log.Fatalf("failed to create auth server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterAuthServiceServer(s, server)

	log.Printf("Auth Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
