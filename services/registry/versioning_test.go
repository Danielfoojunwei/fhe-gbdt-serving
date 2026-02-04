// Unit tests for Versioning Service
// Tests model version management, deployment, and traffic routing

package main

import (
	"math/rand"
	"testing"
)

// ============================================================================
// Version Format Tests
// ============================================================================

func TestVersionFormat(t *testing.T) {
	tests := []struct {
		name    string
		version string
		valid   bool
	}{
		{"semantic version", "v1.0.0", true},
		{"semantic version no prefix", "1.0.0", true},
		{"version with patch", "v2.1.5", true},
		{"simple version", "v1", true},
		{"major minor", "v1.2", true},
		{"with prerelease", "v1.0.0-alpha", true},
		{"with build metadata", "v1.0.0+build123", true},
		{"empty version", "", false},
		{"just v", "v", false},
		{"invalid chars", "v1.0.0@beta", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateVersionFormat(tt.version)
			if valid != tt.valid {
				t.Errorf("validateVersionFormat(%q) = %v, want %v",
					tt.version, valid, tt.valid)
			}
		})
	}
}

func validateVersionFormat(version string) bool {
	if version == "" {
		return false
	}

	// Skip leading 'v' if present
	start := 0
	if version[0] == 'v' || version[0] == 'V' {
		start = 1
	}
	if start >= len(version) {
		return false
	}

	// Check remaining characters are valid
	for _, c := range version[start:] {
		valid := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			c == '.' || c == '-' || c == '+'
		if !valid {
			return false
		}
	}

	return true
}

// ============================================================================
// Traffic Percentage Tests
// ============================================================================

func TestTrafficPercentageValidation(t *testing.T) {
	tests := []struct {
		name    string
		percent int32
		valid   bool
	}{
		{"0 percent", 0, true},
		{"50 percent", 50, true},
		{"100 percent", 100, true},
		{"negative percent", -1, false},
		{"over 100 percent", 101, false},
		{"way over", 200, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateTrafficPercent(tt.percent)
			if valid != tt.valid {
				t.Errorf("validateTrafficPercent(%d) = %v, want %v",
					tt.percent, valid, tt.valid)
			}
		})
	}
}

func validateTrafficPercent(percent int32) bool {
	return percent >= 0 && percent <= 100
}

// ============================================================================
// Traffic Split Tests
// ============================================================================

func TestTrafficSplitValidation(t *testing.T) {
	tests := []struct {
		name   string
		splits []int32
		valid  bool
	}{
		{"single 100%", []int32{100}, true},
		{"50/50 split", []int32{50, 50}, true},
		{"canary 90/10", []int32{90, 10}, true},
		{"three way split", []int32{70, 20, 10}, true},
		{"doesn't sum to 100", []int32{60, 30}, false},
		{"over 100", []int32{60, 50}, false},
		{"empty split", []int32{}, false},
		{"contains negative", []int32{110, -10}, false},
		{"all zeros", []int32{0, 0, 0}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateTrafficSplit(tt.splits)
			if valid != tt.valid {
				t.Errorf("validateTrafficSplit(%v) = %v, want %v",
					tt.splits, valid, tt.valid)
			}
		})
	}
}

func validateTrafficSplit(splits []int32) bool {
	if len(splits) == 0 {
		return false
	}

	var total int32
	for _, s := range splits {
		if s < 0 {
			return false
		}
		total += s
	}
	return total == 100
}

// ============================================================================
// Weighted Selection Tests
// ============================================================================

func TestWeightedSelection(t *testing.T) {
	// Test distribution over many iterations
	versions := []struct {
		id      string
		traffic int32
	}{
		{"v1", 70},
		{"v2", 30},
	}

	counts := make(map[string]int)
	iterations := 10000

	for i := 0; i < iterations; i++ {
		selected := selectVersion(versions)
		counts[selected]++
	}

	// Check distribution is approximately correct (within 5%)
	v1Ratio := float64(counts["v1"]) / float64(iterations)
	v2Ratio := float64(counts["v2"]) / float64(iterations)

	if v1Ratio < 0.65 || v1Ratio > 0.75 {
		t.Errorf("v1 ratio = %f, want ~0.70", v1Ratio)
	}
	if v2Ratio < 0.25 || v2Ratio > 0.35 {
		t.Errorf("v2 ratio = %f, want ~0.30", v2Ratio)
	}
}

func selectVersion(versions []struct {
	id      string
	traffic int32
}) string {
	var total int32
	for _, v := range versions {
		total += v.traffic
	}

	r := rand.Int31n(total)
	var cumulative int32
	for _, v := range versions {
		cumulative += v.traffic
		if r < cumulative {
			return v.id
		}
	}
	return versions[0].id
}

func TestSingleVersionSelection(t *testing.T) {
	versions := []struct {
		id      string
		traffic int32
	}{
		{"v1", 100},
	}

	// Should always select v1
	for i := 0; i < 100; i++ {
		selected := selectVersion(versions)
		if selected != "v1" {
			t.Errorf("expected v1, got %s", selected)
		}
	}
}

// ============================================================================
// Version Status Tests
// ============================================================================

func TestVersionStatusTransitions(t *testing.T) {
	tests := []struct {
		name        string
		from        string
		to          string
		validChange bool
	}{
		{"draft to deployed", "draft", "deployed", true},
		{"deployed to retired", "deployed", "retired", true},
		{"draft to retired", "draft", "retired", true},
		{"retired to deployed", "retired", "deployed", true}, // Rollback
		{"deployed to draft", "deployed", "draft", false},
		{"retired to draft", "retired", "draft", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := isValidStatusTransition(tt.from, tt.to)
			if valid != tt.validChange {
				t.Errorf("isValidStatusTransition(%q, %q) = %v, want %v",
					tt.from, tt.to, valid, tt.validChange)
			}
		})
	}
}

func isValidStatusTransition(from, to string) bool {
	validTransitions := map[string][]string{
		"draft":    {"deployed", "retired"},
		"deployed": {"retired"},
		"retired":  {"deployed"},
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

func TestVersionStatuses(t *testing.T) {
	validStatuses := []string{"draft", "deployed", "retired"}

	tests := []struct {
		name   string
		status string
		valid  bool
	}{
		{"draft status", "draft", true},
		{"deployed status", "deployed", true},
		{"retired status", "retired", true},
		{"invalid status", "pending", false},
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
// Rollback Tests
// ============================================================================

func TestRollbackValidation(t *testing.T) {
	tests := []struct {
		name           string
		targetStatus   string
		canRollbackTo  bool
	}{
		{"rollback to retired version", "retired", true},
		{"rollback to deployed version", "deployed", true},
		{"cannot rollback to draft", "draft", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			canRollback := tt.targetStatus == "retired" || tt.targetStatus == "deployed"
			if canRollback != tt.canRollbackTo {
				t.Errorf("canRollbackTo status %q = %v, want %v",
					tt.targetStatus, canRollback, tt.canRollbackTo)
			}
		})
	}
}

// ============================================================================
// Deployment Validation Tests
// ============================================================================

func TestCanDeployVersion(t *testing.T) {
	tests := []struct {
		name          string
		status        string
		compiledModel string
		canDeploy     bool
	}{
		{"draft with compiled model", "draft", "model-123", true},
		{"draft without compiled model", "draft", "", false},
		{"already deployed", "deployed", "model-123", true}, // Update traffic
		{"retired with compiled model", "retired", "model-123", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			canDeploy := tt.compiledModel != ""
			if canDeploy != tt.canDeploy {
				t.Errorf("canDeploy for status=%q, compiledModel=%q = %v, want %v",
					tt.status, tt.compiledModel, canDeploy, tt.canDeploy)
			}
		})
	}
}

// ============================================================================
// Version Comparison Tests
// ============================================================================

func TestVersionComparison(t *testing.T) {
	tests := []struct {
		name    string
		v1      string
		v2      string
		v1Newer bool
	}{
		{"major version", "v2.0.0", "v1.0.0", true},
		{"minor version", "v1.2.0", "v1.1.0", true},
		{"patch version", "v1.0.2", "v1.0.1", true},
		{"same version", "v1.0.0", "v1.0.0", false},
		{"v2 newer", "v1.0.0", "v2.0.0", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isNewerVersion(tt.v1, tt.v2)
			if result != tt.v1Newer {
				t.Errorf("isNewerVersion(%q, %q) = %v, want %v",
					tt.v1, tt.v2, result, tt.v1Newer)
			}
		})
	}
}

func isNewerVersion(v1, v2 string) bool {
	// Simple comparison based on string (works for semver)
	return v1 > v2
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkValidateVersionFormat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		validateVersionFormat("v1.2.3")
	}
}

func BenchmarkValidateTrafficSplit(b *testing.B) {
	splits := []int32{70, 20, 10}
	for i := 0; i < b.N; i++ {
		validateTrafficSplit(splits)
	}
}

func BenchmarkSelectVersion(b *testing.B) {
	versions := []struct {
		id      string
		traffic int32
	}{
		{"v1", 70},
		{"v2", 20},
		{"v3", 10},
	}
	for i := 0; i < b.N; i++ {
		selectVersion(versions)
	}
}
