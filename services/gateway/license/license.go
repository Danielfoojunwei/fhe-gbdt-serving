// Package license provides license token validation for the on-prem gateway.
//
// The gateway validates license tokens before forwarding prediction requests
// to the runtime. Tokens are issued by the vendor's cloud control plane and
// contain: tenant_id, authorized model_ids, prediction cap, and expiration.
//
// Supports offline grace period: if the license server is unreachable,
// the gateway continues operating using cached token validation for a
// configurable grace period (default: 72 hours).
package license

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

var (
	ErrNoLicense          = errors.New("no license token configured")
	ErrInvalidSignature   = errors.New("invalid license token signature")
	ErrMalformedToken     = errors.New("malformed license token")
	ErrLicenseExpired     = errors.New("license expired and grace period exhausted")
	ErrModelNotAuthorized = errors.New("model not authorized under this license")
	ErrPredictionCap      = errors.New("prediction cap exceeded")
)

// Claims represents the decoded license token claims.
type Claims struct {
	TenantID       string   `json:"tenant_id"`
	LicenseID      string   `json:"license_id"`
	ModelIDs       []string `json:"model_ids"`
	MaxPredictions int64    `json:"max_predictions"`
	IssuedAt       float64  `json:"issued_at"`
	ExpiresAt      float64  `json:"expires_at"`
	Features       []string `json:"features"`
}

// IsExpired returns true if the token has passed its expiration time.
func (c *Claims) IsExpired() bool {
	return float64(time.Now().Unix()) > c.ExpiresAt
}

// AllowsModel returns true if the given model ID is authorized.
func (c *Claims) AllowsModel(modelID string) bool {
	for _, id := range c.ModelIDs {
		if id == "*" || id == modelID {
			return true
		}
	}
	return false
}

// Validator validates license tokens and tracks prediction counts.
// It is safe for concurrent use.
type Validator struct {
	mu sync.Mutex

	signingKey       []byte
	offlineGraceHrs  int
	predictionCounts map[string]int64

	// Cached state for offline operation
	cachedClaims               *Claims
	cacheTime                  time.Time
	lastSuccessfulValidation   time.Time
}

// NewValidator creates a new license validator.
func NewValidator(signingKey string, offlineGraceHours int) (*Validator, error) {
	if len(signingKey) < 32 {
		return nil, fmt.Errorf("signing key must be at least 32 characters")
	}
	return &Validator{
		signingKey:       []byte(signingKey),
		offlineGraceHrs:  offlineGraceHours,
		predictionCounts: make(map[string]int64),
	}, nil
}

// Validate verifies the token signature, expiration, model authorization, and prediction cap.
func (v *Validator) Validate(token string, modelID string) (*Claims, error) {
	if token == "" {
		return nil, ErrNoLicense
	}

	claims, err := v.verifySignature(token)
	if err != nil {
		return nil, err
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	// Check expiration with offline grace period
	if claims.IsExpired() {
		graceDeadline := v.lastSuccessfulValidation.Add(
			time.Duration(v.offlineGraceHrs) * time.Hour,
		)
		if time.Now().After(graceDeadline) || v.lastSuccessfulValidation.IsZero() {
			return nil, ErrLicenseExpired
		}
		log.Printf("LICENSE: Token expired but within grace period (license=%s)", claims.LicenseID)
	}

	// Check model authorization
	if modelID != "" && !claims.AllowsModel(modelID) {
		return nil, ErrModelNotAuthorized
	}

	// Check prediction cap
	count := v.predictionCounts[claims.LicenseID]
	if count >= claims.MaxPredictions {
		return nil, ErrPredictionCap
	}

	// Update cache
	v.lastSuccessfulValidation = time.Now()
	v.cachedClaims = claims
	v.cacheTime = time.Now()

	return claims, nil
}

// RecordPrediction increments the prediction counter for a license. Returns new count.
func (v *Validator) RecordPrediction(licenseID string) int64 {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.predictionCounts[licenseID]++
	return v.predictionCounts[licenseID]
}

// GetPredictionCount returns the current prediction count for a license.
func (v *Validator) GetPredictionCount(licenseID string) int64 {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.predictionCounts[licenseID]
}

// GetCachedClaims returns cached claims for offline operation.
func (v *Validator) GetCachedClaims() *Claims {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.cachedClaims == nil {
		return nil
	}
	graceDeadline := v.cacheTime.Add(time.Duration(v.offlineGraceHrs) * time.Hour)
	if time.Now().After(graceDeadline) {
		return nil
	}
	return v.cachedClaims
}

func (v *Validator) verifySignature(token string) (*Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, ErrMalformedToken
	}

	headerB64, payloadB64, sigB64 := parts[0], parts[1], parts[2]

	// Verify HMAC-SHA256 signature
	signingInput := headerB64 + "." + payloadB64
	mac := hmac.New(sha256.New, v.signingKey)
	mac.Write([]byte(signingInput))
	expectedSig := mac.Sum(nil)

	actualSig, err := base64URLDecode(sigB64)
	if err != nil {
		return nil, ErrMalformedToken
	}

	if !hmac.Equal(expectedSig, actualSig) {
		return nil, ErrInvalidSignature
	}

	// Decode claims
	payloadBytes, err := base64URLDecode(payloadB64)
	if err != nil {
		return nil, ErrMalformedToken
	}

	var claims Claims
	if err := json.Unmarshal(payloadBytes, &claims); err != nil {
		return nil, fmt.Errorf("invalid token payload: %w", err)
	}

	return &claims, nil
}

func base64URLDecode(s string) ([]byte, error) {
	// Add padding if needed
	switch len(s) % 4 {
	case 2:
		s += "=="
	case 3:
		s += "="
	}
	return base64.URLEncoding.DecodeString(s)
}
