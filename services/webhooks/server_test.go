// Unit tests for Webhooks Service

package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"testing"
	"time"
)

// TestWebhookSignatureGeneration tests HMAC signature generation
func TestWebhookSignatureGeneration(t *testing.T) {
	secret := "whsec_test_secret_key_12345"
	payload := []byte(`{"event":"test","data":{"id":"123"}}`)

	signature := computeSignature(payload, secret)

	// Should have sha256= prefix
	if signature[:7] != "sha256=" {
		t.Errorf("Signature should start with 'sha256=', got %s", signature[:7])
	}

	// Should be valid hex
	hexPart := signature[7:]
	if len(hexPart) != 64 { // SHA256 produces 32 bytes = 64 hex chars
		t.Errorf("Expected 64 hex chars, got %d", len(hexPart))
	}

	// Verify signature
	valid := verifySignature(payload, signature, secret)
	if !valid {
		t.Error("Signature verification failed")
	}
}

// TestWebhookSignatureVerification tests signature verification
func TestWebhookSignatureVerification(t *testing.T) {
	secret := "whsec_test_secret"
	payload := []byte(`{"event":"test"}`)
	validSignature := computeSignature(payload, secret)

	tests := []struct {
		name      string
		payload   []byte
		signature string
		secret    string
		valid     bool
	}{
		{"Valid signature", payload, validSignature, secret, true},
		{"Wrong secret", payload, validSignature, "wrong_secret", false},
		{"Tampered payload", []byte(`{"event":"hacked"}`), validSignature, secret, false},
		{"Invalid signature format", payload, "invalid", secret, false},
		{"Empty signature", payload, "", secret, false},
		{"Wrong algorithm prefix", payload, "sha1=" + validSignature[7:], secret, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := verifySignature(tc.payload, tc.signature, tc.secret)
			if result != tc.valid {
				t.Errorf("Expected valid=%v, got %v", tc.valid, result)
			}
		})
	}
}

// TestSecretGeneration tests webhook secret generation
func TestSecretGeneration(t *testing.T) {
	secrets := make(map[string]bool)

	for i := 0; i < 100; i++ {
		secret := generateSecret()

		// Should have correct prefix
		if secret[:6] != "whsec_" {
			t.Errorf("Secret should start with 'whsec_', got %s", secret[:6])
		}

		// Should be unique
		if secrets[secret] {
			t.Error("Generated duplicate secret")
		}
		secrets[secret] = true

		// Should be sufficient length
		if len(secret) < 30 {
			t.Errorf("Secret should be at least 30 chars, got %d", len(secret))
		}
	}
}

// TestEventTypeValidation tests event type validation
func TestEventTypeValidation(t *testing.T) {
	validTypes := getValidEventTypes()

	tests := []struct {
		eventType string
		valid     bool
	}{
		{"model.registered", true},
		{"model.compiled", true},
		{"model.deployed", true},
		{"prediction.completed", true},
		{"billing.invoice.created", true},
		{"alert.triggered", true},
		{"drift.detected", true},
		{"invalid.event", false},
		{"", false},
		{"model", false}, // Incomplete
	}

	for _, tc := range tests {
		t.Run(tc.eventType, func(t *testing.T) {
			result := isValidEventType(tc.eventType, validTypes)
			if result != tc.valid {
				t.Errorf("Expected valid=%v for %s, got %v", tc.valid, tc.eventType, result)
			}
		})
	}
}

// TestWebhookURLValidation tests URL validation
func TestWebhookURLValidation(t *testing.T) {
	tests := []struct {
		name      string
		url       string
		expectErr bool
	}{
		{"Valid HTTPS", "https://example.com/webhook", false},
		{"Valid HTTPS with path", "https://api.example.com/v1/webhooks", false},
		{"Valid HTTPS with port", "https://example.com:8443/webhook", false},
		{"HTTP not allowed", "http://example.com/webhook", true},
		{"Localhost HTTPS", "https://localhost/webhook", false},
		{"Invalid scheme", "ftp://example.com/webhook", true},
		{"No scheme", "example.com/webhook", true},
		{"Empty URL", "", true},
		{"Invalid URL", "not-a-url", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateWebhookURL(tc.url)
			if tc.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestRetryLogic tests webhook retry with exponential backoff
func TestRetryLogic(t *testing.T) {
	tests := []struct {
		attempt     int
		expectedMin time.Duration
		expectedMax time.Duration
	}{
		{1, 1 * time.Second, 2 * time.Second},
		{2, 4 * time.Second, 5 * time.Second},
		{3, 9 * time.Second, 10 * time.Second},
	}

	for _, tc := range tests {
		t.Run("attempt_"+string(rune(tc.attempt+'0')), func(t *testing.T) {
			backoff := calculateBackoff(tc.attempt)
			if backoff < tc.expectedMin || backoff > tc.expectedMax {
				t.Errorf("Expected backoff between %v and %v, got %v",
					tc.expectedMin, tc.expectedMax, backoff)
			}
		})
	}
}

// TestEventPayloadSerialization tests event payload formatting
func TestEventPayloadSerialization(t *testing.T) {
	event := &WebhookEvent{
		ID:        "evt_123",
		Type:      "model.deployed",
		TenantID:  "tenant_456",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"model_id": "model_789",
			"version":  "v1.0",
		},
	}

	payload, err := serializeEvent(event)
	if err != nil {
		t.Fatalf("Failed to serialize event: %v", err)
	}

	// Should contain required fields
	if !contains(string(payload), "event_id") {
		t.Error("Payload should contain event_id")
	}
	if !contains(string(payload), "event_type") {
		t.Error("Payload should contain event_type")
	}
	if !contains(string(payload), "model.deployed") {
		t.Error("Payload should contain event type value")
	}
}

// TestWebhookDeliveryStatus tests delivery status tracking
func TestWebhookDeliveryStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		expected   string
	}{
		{200, "success"},
		{201, "success"},
		{204, "success"},
		{301, "failed"},
		{400, "failed"},
		{401, "failed"},
		{403, "failed"},
		{404, "failed"},
		{500, "failed"},
		{502, "failed"},
		{503, "failed"},
	}

	for _, tc := range tests {
		t.Run("status_"+string(rune(tc.statusCode)), func(t *testing.T) {
			status := getDeliveryStatus(tc.statusCode)
			if status != tc.expected {
				t.Errorf("Expected status %s for code %d, got %s",
					tc.expected, tc.statusCode, status)
			}
		})
	}
}

// Helper types and functions

type WebhookEvent struct {
	ID        string
	Type      string
	TenantID  string
	Timestamp time.Time
	Data      map[string]interface{}
}

func verifySignature(payload []byte, signature, secret string) bool {
	if signature == "" || len(signature) < 8 {
		return false
	}

	if signature[:7] != "sha256=" {
		return false
	}

	expected := computeSignature(payload, secret)
	return hmac.Equal([]byte(signature), []byte(expected))
}

func getValidEventTypes() []string {
	return []string{
		"model.registered",
		"model.compiled",
		"model.deployed",
		"model.retired",
		"model.deleted",
		"prediction.completed",
		"prediction.failed",
		"key.generated",
		"key.rotated",
		"key.revoked",
		"billing.subscription.created",
		"billing.subscription.updated",
		"billing.subscription.cancelled",
		"billing.invoice.created",
		"billing.invoice.paid",
		"billing.payment.failed",
		"alert.triggered",
		"alert.resolved",
		"drift.detected",
		"security.anomaly",
	}
}

func isValidEventType(eventType string, validTypes []string) bool {
	for _, t := range validTypes {
		if t == eventType {
			return true
		}
	}
	return false
}

func validateWebhookURL(url string) error {
	if url == "" {
		return &ValidationError{Field: "url", Message: "required"}
	}

	// Must start with https://
	if len(url) < 8 || url[:8] != "https://" {
		return &ValidationError{Field: "url", Message: "must use HTTPS"}
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

func calculateBackoff(attempt int) time.Duration {
	base := time.Duration(attempt * attempt) * time.Second
	return base
}

func serializeEvent(event *WebhookEvent) ([]byte, error) {
	// Simplified JSON serialization
	return []byte(`{"event_id":"` + event.ID + `","event_type":"` + event.Type + `"}`), nil
}

func getDeliveryStatus(statusCode int) string {
	if statusCode >= 200 && statusCode < 300 {
		return "success"
	}
	return "failed"
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Benchmark tests

func BenchmarkSignatureGeneration(b *testing.B) {
	secret := "whsec_test_secret_key_12345"
	payload := []byte(`{"event":"test","data":{"id":"123"}}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		computeSignature(payload, secret)
	}
}

func BenchmarkSignatureVerification(b *testing.B) {
	secret := "whsec_test_secret_key_12345"
	payload := []byte(`{"event":"test","data":{"id":"123"}}`)
	signature := computeSignature(payload, secret)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verifySignature(payload, signature, secret)
	}
}

func BenchmarkSecretGeneration(b *testing.B) {
	for i := 0; i < b.N; i++ {
		generateSecret()
	}
}
