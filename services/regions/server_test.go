// Unit tests for Regions Service
// Tests multi-region management, data residency, and geo-routing

package main

import (
	"math"
	"testing"
)

// ============================================================================
// Haversine Distance Tests
// ============================================================================

func TestDistanceCalculation(t *testing.T) {
	tests := []struct {
		name     string
		lat1     float64
		lon1     float64
		lat2     float64
		lon2     float64
		expected float64
		delta    float64
	}{
		{
			name:     "New York to London",
			lat1:     40.7128,
			lon1:     -74.0060,
			lat2:     51.5074,
			lon2:     -0.1278,
			expected: 5570,
			delta:    50, // Allow 50km error
		},
		{
			name:     "Same location",
			lat1:     37.478,
			lon1:     -76.453,
			lat2:     37.478,
			lon2:     -76.453,
			expected: 0,
			delta:    0.1,
		},
		{
			name:     "Singapore to Tokyo",
			lat1:     1.352,
			lon1:     103.820,
			lat2:     35.682,
			lon2:     139.759,
			expected: 5310,
			delta:    50,
		},
		{
			name:     "Frankfurt to Dublin",
			lat1:     50.110,
			lon1:     8.682,
			lat2:     53.349,
			lon2:     -6.260,
			expected: 1090,
			delta:    50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := distance(tt.lat1, tt.lon1, tt.lat2, tt.lon2)
			if math.Abs(result-tt.expected) > tt.delta {
				t.Errorf("distance(%f, %f, %f, %f) = %f, want %f ± %f",
					tt.lat1, tt.lon1, tt.lat2, tt.lon2, result, tt.expected, tt.delta)
			}
		})
	}
}

// ============================================================================
// Region Code Validation Tests
// ============================================================================

func TestRegionCodeValidation(t *testing.T) {
	tests := []struct {
		name  string
		code  string
		valid bool
	}{
		{"valid AWS region", "us-east-1", true},
		{"valid AWS region 2", "eu-west-1", true},
		{"valid AWS region 3", "ap-southeast-1", true},
		{"valid GCP region", "us-central1", true},
		{"valid Azure region", "eastus", true},
		{"empty code", "", false},
		{"code with spaces", "us east 1", false},
		{"code with uppercase", "US-EAST-1", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateRegionCode(tt.code)
			if valid != tt.valid {
				t.Errorf("validateRegionCode(%q) = %v, want %v",
					tt.code, valid, tt.valid)
			}
		})
	}
}

func validateRegionCode(code string) bool {
	if code == "" {
		return false
	}
	for _, c := range code {
		if c == ' ' {
			return false
		}
		if c >= 'A' && c <= 'Z' {
			return false
		}
	}
	return true
}

// ============================================================================
// Data Residency Tests
// ============================================================================

func TestDataResidencyCheck(t *testing.T) {
	tests := []struct {
		name            string
		allowedCountries []string
		blockedCountries []string
		targetCountry   string
		allowed         bool
	}{
		{
			name:            "no restrictions",
			allowedCountries: []string{},
			blockedCountries: []string{},
			targetCountry:   "US",
			allowed:         true,
		},
		{
			name:            "country in allowed list",
			allowedCountries: []string{"US", "CA", "UK"},
			blockedCountries: []string{},
			targetCountry:   "US",
			allowed:         true,
		},
		{
			name:            "country not in allowed list",
			allowedCountries: []string{"US", "CA", "UK"},
			blockedCountries: []string{},
			targetCountry:   "DE",
			allowed:         false,
		},
		{
			name:            "country in blocked list",
			allowedCountries: []string{},
			blockedCountries: []string{"RU", "CN"},
			targetCountry:   "CN",
			allowed:         false,
		},
		{
			name:            "country not in blocked list",
			allowedCountries: []string{},
			blockedCountries: []string{"RU", "CN"},
			targetCountry:   "DE",
			allowed:         true,
		},
		{
			name:            "blocked takes precedence",
			allowedCountries: []string{"US", "CN"},
			blockedCountries: []string{"CN"},
			targetCountry:   "CN",
			allowed:         false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allowed := checkDataResidencyAllowed(tt.allowedCountries, tt.blockedCountries, tt.targetCountry)
			if allowed != tt.allowed {
				t.Errorf("checkDataResidencyAllowed(%v, %v, %q) = %v, want %v",
					tt.allowedCountries, tt.blockedCountries, tt.targetCountry, allowed, tt.allowed)
			}
		})
	}
}

func checkDataResidencyAllowed(allowed, blocked []string, country string) bool {
	// Check blocked first (takes precedence)
	for _, c := range blocked {
		if c == country {
			return false
		}
	}

	// If allowed list is empty, allow all (not blocked)
	if len(allowed) == 0 {
		return true
	}

	// Check if in allowed list
	for _, c := range allowed {
		if c == country {
			return true
		}
	}

	return false
}

// ============================================================================
// Compliance Requirement Tests
// ============================================================================

func TestComplianceRequirements(t *testing.T) {
	tests := []struct {
		name                string
		regionCompliance    []string
		requiredCompliance  []string
		meetsRequirements   bool
	}{
		{
			name:               "no requirements",
			regionCompliance:   []string{"SOC2", "GDPR"},
			requiredCompliance: []string{},
			meetsRequirements:  true,
		},
		{
			name:               "meets all requirements",
			regionCompliance:   []string{"SOC2", "GDPR", "HIPAA"},
			requiredCompliance: []string{"GDPR", "SOC2"},
			meetsRequirements:  true,
		},
		{
			name:               "missing requirement",
			regionCompliance:   []string{"SOC2", "GDPR"},
			requiredCompliance: []string{"HIPAA"},
			meetsRequirements:  false,
		},
		{
			name:               "partial compliance",
			regionCompliance:   []string{"SOC2", "GDPR"},
			requiredCompliance: []string{"SOC2", "HIPAA"},
			meetsRequirements:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meets := meetsComplianceRequirements(tt.regionCompliance, tt.requiredCompliance)
			if meets != tt.meetsRequirements {
				t.Errorf("meetsComplianceRequirements(%v, %v) = %v, want %v",
					tt.regionCompliance, tt.requiredCompliance, meets, tt.meetsRequirements)
			}
		})
	}
}

func meetsComplianceRequirements(regionCompliance, required []string) bool {
	if len(required) == 0 {
		return true
	}

	complianceSet := make(map[string]bool)
	for _, c := range regionCompliance {
		complianceSet[c] = true
	}

	for _, req := range required {
		if !complianceSet[req] {
			return false
		}
	}

	return true
}

// ============================================================================
// Region Selection Tests
// ============================================================================

func TestFindClosestRegion(t *testing.T) {
	regions := []struct {
		code string
		lat  float64
		lon  float64
	}{
		{"us-east-1", 37.478, -76.453},
		{"eu-west-1", 53.349, -6.260},
		{"ap-southeast-1", 1.352, 103.820},
	}

	tests := []struct {
		name         string
		clientLat    float64
		clientLon    float64
		expectedCode string
	}{
		{"client in New York", 40.7128, -74.0060, "us-east-1"},
		{"client in London", 51.5074, -0.1278, "eu-west-1"},
		{"client in Singapore", 1.290, 103.850, "ap-southeast-1"},
		{"client in Tokyo", 35.682, 139.759, "ap-southeast-1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			closest := findClosestRegion(regions, tt.clientLat, tt.clientLon)
			if closest != tt.expectedCode {
				t.Errorf("findClosestRegion for client at (%f, %f) = %q, want %q",
					tt.clientLat, tt.clientLon, closest, tt.expectedCode)
			}
		})
	}
}

func findClosestRegion(regions []struct {
	code string
	lat  float64
	lon  float64
}, clientLat, clientLon float64) string {
	if len(regions) == 0 {
		return ""
	}

	closest := regions[0].code
	minDist := distance(clientLat, clientLon, regions[0].lat, regions[0].lon)

	for _, r := range regions[1:] {
		d := distance(clientLat, clientLon, r.lat, r.lon)
		if d < minDist {
			minDist = d
			closest = r.code
		}
	}

	return closest
}

// ============================================================================
// Region Status Tests
// ============================================================================

func TestRegionStatus(t *testing.T) {
	validStatuses := []string{"active", "maintenance", "disabled"}

	tests := []struct {
		name   string
		status string
		valid  bool
	}{
		{"active status", "active", true},
		{"maintenance status", "maintenance", true},
		{"disabled status", "disabled", true},
		{"invalid status", "offline", false},
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
// Region Tier Tests
// ============================================================================

func TestRegionTier(t *testing.T) {
	validTiers := []string{"primary", "secondary", "edge"}

	tests := []struct {
		name  string
		tier  string
		valid bool
	}{
		{"primary tier", "primary", true},
		{"secondary tier", "secondary", true},
		{"edge tier", "edge", true},
		{"invalid tier", "backup", false},
		{"empty tier", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, tier := range validTiers {
				if tier == tt.tier {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("tier %q validity = %v, want %v",
					tt.tier, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Provider Tests
// ============================================================================

func TestCloudProvider(t *testing.T) {
	validProviders := []string{"aws", "gcp", "azure"}

	tests := []struct {
		name     string
		provider string
		valid    bool
	}{
		{"AWS provider", "aws", true},
		{"GCP provider", "gcp", true},
		{"Azure provider", "azure", true},
		{"invalid provider", "digitalocean", false},
		{"empty provider", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, p := range validProviders {
				if p == tt.provider {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("provider %q validity = %v, want %v",
					tt.provider, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Replication Mode Tests
// ============================================================================

func TestReplicationMode(t *testing.T) {
	validModes := []string{"sync", "async", "eventual"}

	tests := []struct {
		name  string
		mode  string
		valid bool
	}{
		{"sync mode", "sync", true},
		{"async mode", "async", true},
		{"eventual mode", "eventual", true},
		{"invalid mode", "realtime", false},
		{"empty mode", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, m := range validModes {
				if m == tt.mode {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("replication mode %q validity = %v, want %v",
					tt.mode, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Capacity Tests
// ============================================================================

func TestRegionCapacity(t *testing.T) {
	tests := []struct {
		name        string
		maxCapacity int32
		currentLoad int32
		canAccept   bool
		threshold   float64
	}{
		{"under capacity", 10000, 5000, true, 0.8},
		{"at threshold", 10000, 8000, false, 0.8},
		{"over threshold", 10000, 9000, false, 0.8},
		{"empty region", 10000, 0, true, 0.8},
		{"full region", 10000, 10000, false, 0.8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			canAccept := canAcceptLoad(tt.maxCapacity, tt.currentLoad, tt.threshold)
			if canAccept != tt.canAccept {
				t.Errorf("canAcceptLoad(%d, %d, %f) = %v, want %v",
					tt.maxCapacity, tt.currentLoad, tt.threshold, canAccept, tt.canAccept)
			}
		})
	}
}

func canAcceptLoad(maxCapacity, currentLoad int32, threshold float64) bool {
	if maxCapacity == 0 {
		return false
	}
	utilization := float64(currentLoad) / float64(maxCapacity)
	return utilization < threshold
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkDistance(b *testing.B) {
	for i := 0; i < b.N; i++ {
		distance(40.7128, -74.0060, 51.5074, -0.1278)
	}
}

func BenchmarkCheckDataResidency(b *testing.B) {
	allowed := []string{"US", "CA", "UK", "DE", "FR"}
	blocked := []string{"RU", "CN"}
	for i := 0; i < b.N; i++ {
		checkDataResidencyAllowed(allowed, blocked, "US")
	}
}

func BenchmarkMeetsComplianceRequirements(b *testing.B) {
	regionCompliance := []string{"SOC2", "GDPR", "HIPAA", "ISO27001"}
	required := []string{"GDPR", "SOC2"}
	for i := 0; i < b.N; i++ {
		meetsComplianceRequirements(regionCompliance, required)
	}
}
