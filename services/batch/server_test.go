// Unit tests for Batch Inference Service
// Tests batch job management, processing, and validation

package main

import (
	"encoding/json"
	"testing"
)

// ============================================================================
// Input Format Tests
// ============================================================================

func TestInputFormatValidation(t *testing.T) {
	validFormats := []string{"json", "csv", "parquet"}

	tests := []struct {
		name   string
		format string
		valid  bool
	}{
		{"JSON format", "json", true},
		{"CSV format", "csv", true},
		{"Parquet format", "parquet", true},
		{"invalid format", "xml", false},
		{"empty format", "", false},
		{"uppercase format", "JSON", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, f := range validFormats {
				if f == tt.format {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("format %q validity = %v, want %v",
					tt.format, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Output Format Tests
// ============================================================================

func TestOutputFormatValidation(t *testing.T) {
	validFormats := []string{"json", "csv", "parquet", "jsonl"}

	tests := []struct {
		name   string
		format string
		valid  bool
	}{
		{"JSON format", "json", true},
		{"CSV format", "csv", true},
		{"Parquet format", "parquet", true},
		{"JSON Lines format", "jsonl", true},
		{"invalid format", "binary", false},
		{"empty format", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, f := range validFormats {
				if f == tt.format {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("format %q validity = %v, want %v",
					tt.format, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Job Status Tests
// ============================================================================

func TestJobStatusTransitions(t *testing.T) {
	tests := []struct {
		name        string
		from        string
		to          string
		validChange bool
	}{
		{"pending to running", "pending", "running", true},
		{"running to completed", "running", "completed", true},
		{"running to failed", "running", "failed", true},
		{"running to cancelled", "running", "cancelled", true},
		{"pending to cancelled", "pending", "cancelled", true},
		{"completed to running", "completed", "running", false},
		{"failed to running", "failed", "running", false},
		{"cancelled to running", "cancelled", "running", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := isValidJobStatusTransition(tt.from, tt.to)
			if valid != tt.validChange {
				t.Errorf("isValidJobStatusTransition(%q, %q) = %v, want %v",
					tt.from, tt.to, valid, tt.validChange)
			}
		})
	}
}

func isValidJobStatusTransition(from, to string) bool {
	validTransitions := map[string][]string{
		"pending":             {"running", "cancelled"},
		"running":             {"completed", "completed_with_errors", "failed", "cancelled"},
		"completed":           {},
		"completed_with_errors": {},
		"failed":              {},
		"cancelled":           {},
	}

	allowed, ok := validTransitions[from]
	if !ok {
		return false
	}

	for _, s := range allowed {
		if s == to {
			return true
		}
	}
	return false
}

func TestJobStatuses(t *testing.T) {
	validStatuses := []string{"pending", "running", "completed", "completed_with_errors", "failed", "cancelled"}

	tests := []struct {
		name   string
		status string
		valid  bool
	}{
		{"pending status", "pending", true},
		{"running status", "running", true},
		{"completed status", "completed", true},
		{"completed_with_errors status", "completed_with_errors", true},
		{"failed status", "failed", true},
		{"cancelled status", "cancelled", true},
		{"invalid status", "processing", false},
		{"empty status", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, s := range validStatuses {
				if s == tt.status {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("status %q validity = %v, want %v",
					tt.status, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Progress Calculation Tests
// ============================================================================

func TestProgressCalculation(t *testing.T) {
	tests := []struct {
		name      string
		processed int64
		failed    int64
		total     int64
		expected  float32
	}{
		{"no progress", 0, 0, 100, 0},
		{"half done", 50, 0, 100, 50},
		{"all processed", 100, 0, 100, 100},
		{"with failures", 80, 20, 100, 100},
		{"partial with failures", 40, 10, 100, 50},
		{"zero total", 0, 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			progress := calculateProgress(tt.processed, tt.failed, tt.total)
			if progress != tt.expected {
				t.Errorf("calculateProgress(%d, %d, %d) = %f, want %f",
					tt.processed, tt.failed, tt.total, progress, tt.expected)
			}
		})
	}
}

func calculateProgress(processed, failed, total int64) float32 {
	if total == 0 {
		return 0
	}
	return float32(processed+failed) / float32(total) * 100
}

// ============================================================================
// Batch Size Validation Tests
// ============================================================================

func TestBatchSizeValidation(t *testing.T) {
	tests := []struct {
		name         string
		batchSize    int32
		defaultSize  int32
		maxSize      int32
		effectiveSize int32
	}{
		{"zero uses default", 0, 100, 1000, 100},
		{"negative uses default", -1, 100, 1000, 100},
		{"within limits", 500, 100, 1000, 500},
		{"exceeds max", 2000, 100, 1000, 1000},
		{"at max", 1000, 100, 1000, 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := getEffectiveBatchSize(tt.batchSize, tt.defaultSize, tt.maxSize)
			if size != tt.effectiveSize {
				t.Errorf("getEffectiveBatchSize(%d, %d, %d) = %d, want %d",
					tt.batchSize, tt.defaultSize, tt.maxSize, size, tt.effectiveSize)
			}
		})
	}
}

func getEffectiveBatchSize(batchSize, defaultSize, maxSize int32) int32 {
	if batchSize <= 0 {
		return defaultSize
	}
	if batchSize > maxSize {
		return maxSize
	}
	return batchSize
}

// ============================================================================
// URL Validation Tests
// ============================================================================

func TestURLValidation(t *testing.T) {
	tests := []struct {
		name  string
		url   string
		valid bool
	}{
		{"S3 URL", "s3://bucket/path/file.json", true},
		{"GCS URL", "gs://bucket/path/file.json", true},
		{"HTTPS URL", "https://example.com/data.json", true},
		{"HTTP URL", "http://example.com/data.json", false}, // Insecure
		{"empty URL", "", false},
		{"relative path", "/path/to/file", false},
		{"no scheme", "bucket/path/file", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateInputURL(tt.url)
			if valid != tt.valid {
				t.Errorf("validateInputURL(%q) = %v, want %v",
					tt.url, valid, tt.valid)
			}
		})
	}
}

func validateInputURL(url string) bool {
	if url == "" {
		return false
	}
	validPrefixes := []string{"s3://", "gs://", "https://"}
	for _, prefix := range validPrefixes {
		if len(url) >= len(prefix) && url[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

// ============================================================================
// JSON Input Validation Tests
// ============================================================================

func TestJSONInputValidation(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		valid  bool
		count  int
	}{
		{
			name:   "valid JSON array",
			input:  `[{"f1": 1.0, "f2": 2.0}, {"f1": 3.0, "f2": 4.0}]`,
			valid:  true,
			count:  2,
		},
		{
			name:   "empty array",
			input:  `[]`,
			valid:  true,
			count:  0,
		},
		{
			name:   "single record",
			input:  `[{"features": [1.0, 2.0, 3.0]}]`,
			valid:  true,
			count:  1,
		},
		{
			name:   "not an array",
			input:  `{"f1": 1.0}`,
			valid:  false,
			count:  0,
		},
		{
			name:   "invalid JSON",
			input:  `[{"f1": 1.0`,
			valid:  false,
			count:  0,
		},
		{
			name:   "empty string",
			input:  ``,
			valid:  false,
			count:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, count := validateJSONInput([]byte(tt.input))
			if valid != tt.valid {
				t.Errorf("validateJSONInput(%q) validity = %v, want %v",
					tt.input, valid, tt.valid)
			}
			if count != tt.count {
				t.Errorf("validateJSONInput(%q) count = %d, want %d",
					tt.input, count, tt.count)
			}
		})
	}
}

func validateJSONInput(data []byte) (bool, int) {
	if len(data) == 0 {
		return false, 0
	}

	var records []json.RawMessage
	if err := json.Unmarshal(data, &records); err != nil {
		return false, 0
	}

	return true, len(records)
}

// ============================================================================
// Retry Logic Tests
// ============================================================================

func TestRetryConfiguration(t *testing.T) {
	tests := []struct {
		name        string
		maxRetries  int32
		retryDelay  int32
		valid       bool
	}{
		{"default config", 3, 1000, true},
		{"no retries", 0, 0, true},
		{"many retries", 10, 5000, true},
		{"negative retries", -1, 1000, false},
		{"negative delay", 3, -1, false},
		{"too many retries", 100, 1000, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateRetryConfig(tt.maxRetries, tt.retryDelay)
			if valid != tt.valid {
				t.Errorf("validateRetryConfig(%d, %d) = %v, want %v",
					tt.maxRetries, tt.retryDelay, valid, tt.valid)
			}
		})
	}
}

func validateRetryConfig(maxRetries, retryDelay int32) bool {
	if maxRetries < 0 || maxRetries > 20 {
		return false
	}
	if maxRetries > 0 && retryDelay < 0 {
		return false
	}
	return true
}

// ============================================================================
// Cancellation Tests
// ============================================================================

func TestCanCancelJob(t *testing.T) {
	tests := []struct {
		name      string
		status    string
		canCancel bool
	}{
		{"pending job", "pending", true},
		{"running job", "running", true},
		{"completed job", "completed", false},
		{"already cancelled", "cancelled", false},
		{"failed job", "failed", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			canCancel := canCancelJob(tt.status)
			if canCancel != tt.canCancel {
				t.Errorf("canCancelJob(%q) = %v, want %v",
					tt.status, canCancel, tt.canCancel)
			}
		})
	}
}

func canCancelJob(status string) bool {
	return status == "pending" || status == "running"
}

// ============================================================================
// Record Limit Tests
// ============================================================================

func TestRecordLimitValidation(t *testing.T) {
	tests := []struct {
		name      string
		limit     int32
		maxLimit  int32
		effective int32
	}{
		{"zero uses default", 0, 1000, 100},
		{"within limit", 500, 1000, 500},
		{"exceeds limit", 2000, 1000, 1000},
		{"negative", -1, 1000, 100},
	}

	defaultLimit := int32(100)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			effective := getEffectiveLimit(tt.limit, defaultLimit, tt.maxLimit)
			if effective != tt.effective {
				t.Errorf("getEffectiveLimit(%d, %d, %d) = %d, want %d",
					tt.limit, defaultLimit, tt.maxLimit, effective, tt.effective)
			}
		})
	}
}

func getEffectiveLimit(limit, defaultLimit, maxLimit int32) int32 {
	if limit <= 0 {
		return defaultLimit
	}
	if limit > maxLimit {
		return maxLimit
	}
	return limit
}

// ============================================================================
// Final Status Determination Tests
// ============================================================================

func TestFinalStatusDetermination(t *testing.T) {
	tests := []struct {
		name       string
		processed  int64
		failed     int64
		total      int64
		finalStatus string
	}{
		{"all success", 100, 0, 100, "completed"},
		{"some failures", 90, 10, 100, "completed_with_errors"},
		{"all failures", 0, 100, 100, "failed"},
		{"majority failures", 10, 90, 100, "completed_with_errors"},
		{"no records processed", 0, 0, 0, "completed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := determineFinalStatus(tt.processed, tt.failed)
			if status != tt.finalStatus {
				t.Errorf("determineFinalStatus(%d, %d) = %q, want %q",
					tt.processed, tt.failed, status, tt.finalStatus)
			}
		})
	}
}

func determineFinalStatus(processed, failed int64) string {
	if failed > 0 && processed == 0 {
		return "failed"
	}
	if failed > 0 {
		return "completed_with_errors"
	}
	return "completed"
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkCalculateProgress(b *testing.B) {
	for i := 0; i < b.N; i++ {
		calculateProgress(50000, 1000, 100000)
	}
}

func BenchmarkValidateJSONInput(b *testing.B) {
	input := []byte(`[{"f1": 1.0}, {"f1": 2.0}, {"f1": 3.0}]`)
	for i := 0; i < b.N; i++ {
		validateJSONInput(input)
	}
}

func BenchmarkValidateInputURL(b *testing.B) {
	for i := 0; i < b.N; i++ {
		validateInputURL("s3://bucket/path/to/data.json")
	}
}
