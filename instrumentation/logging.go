package main

import (
	"log"
	"strings"
)

var forbiddenKeys = []string{
	"payload", "ciphertext", "feature", "plaintext", "secret_key", "eval_key",
}

var allowedKeys = map[string]bool{
	"request_id":        true,
	"tenant_id":         true,
	"compiled_model_id": true,
	"profile":           true,
	"batch_size":        true,
	"latency_ms":        true,
	"status":            true,
	"error_code":        true,
	"message":           true,
}

func AuditLog(fields map[string]interface{}) {
	for key := range fields {
		lowerKey := strings.ToLower(key)
		for _, forbidden := range forbiddenKeys {
			if strings.Contains(lowerKey, forbidden) {
				log.Fatalf("SECURITY VIOLATION: Forbidden key '%s' in log", key)
			}
		}
		if !allowedKeys[key] {
			log.Printf("WARNING: Unrecognized log key '%s', consider adding to allowlist", key)
		}
	}
	// Emit structured log (JSON)
	log.Printf("AUDIT: %+v", fields)
}
