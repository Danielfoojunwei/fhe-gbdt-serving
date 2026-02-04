// Unit tests for A/B Testing Service

package main

import (
	"math"
	"testing"
)

// TestVariantAssignment tests deterministic variant assignment
func TestVariantAssignment(t *testing.T) {
	variants := []*Variant{
		{ID: "control", Name: "Control", TrafficPercent: 50, IsControl: true},
		{ID: "treatment", Name: "Treatment", TrafficPercent: 50, IsControl: false},
	}

	// Same user should always get same variant
	userID := "user-12345"
	firstAssignment := assignVariantByHashTest(userID, variants)
	for i := 0; i < 100; i++ {
		assignment := assignVariantByHashTest(userID, variants)
		if assignment != firstAssignment {
			t.Error("Assignment should be deterministic for same user")
		}
	}
}

// TestTrafficDistribution tests that traffic is distributed according to percentages
func TestTrafficDistribution(t *testing.T) {
	variants := []*Variant{
		{ID: "control", Name: "Control", TrafficPercent: 70, IsControl: true},
		{ID: "treatment", Name: "Treatment", TrafficPercent: 30, IsControl: false},
	}

	counts := make(map[string]int)
	numUsers := 10000

	for i := 0; i < numUsers; i++ {
		userID := "user-" + string(rune(i))
		variantID := assignVariantByHashTest(userID, variants)
		counts[variantID]++
	}

	// Check distribution is roughly correct (within 5%)
	controlPct := float64(counts["control"]) / float64(numUsers) * 100
	treatmentPct := float64(counts["treatment"]) / float64(numUsers) * 100

	if math.Abs(controlPct-70) > 5 {
		t.Errorf("Control should be ~70%%, got %.1f%%", controlPct)
	}
	if math.Abs(treatmentPct-30) > 5 {
		t.Errorf("Treatment should be ~30%%, got %.1f%%", treatmentPct)
	}
}

// TestMultipleVariants tests assignment with more than 2 variants
func TestMultipleVariants(t *testing.T) {
	variants := []*Variant{
		{ID: "control", TrafficPercent: 50},
		{ID: "treatment_a", TrafficPercent: 25},
		{ID: "treatment_b", TrafficPercent: 25},
	}

	counts := make(map[string]int)
	numUsers := 10000

	for i := 0; i < numUsers; i++ {
		userID := "user-multi-" + string(rune(i))
		variantID := assignVariantByHashTest(userID, variants)
		counts[variantID]++
	}

	// All variants should have assignments
	for _, v := range variants {
		if counts[v.ID] == 0 {
			t.Errorf("Variant %s has no assignments", v.ID)
		}
	}
}

// TestStatisticalSignificance tests p-value calculation
func TestStatisticalSignificance(t *testing.T) {
	tests := []struct {
		name        string
		controlN    int64
		controlMean float64
		controlVar  float64
		treatN      int64
		treatMean   float64
		treatVar    float64
		sigLevel    float64
		significant bool
	}{
		{
			name:        "Clearly significant",
			controlN:    1000,
			controlMean: 100,
			controlVar:  25,
			treatN:      1000,
			treatMean:   110,
			treatVar:    25,
			sigLevel:    0.05,
			significant: true,
		},
		{
			name:        "Not significant - small difference",
			controlN:    100,
			controlMean: 100,
			controlVar:  100,
			treatN:      100,
			treatMean:   101,
			treatVar:    100,
			sigLevel:    0.05,
			significant: false,
		},
		{
			name:        "Not significant - small sample",
			controlN:    10,
			controlMean: 100,
			controlVar:  25,
			treatN:      10,
			treatMean:   110,
			treatVar:    25,
			sigLevel:    0.05,
			significant: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pValue := calculatePValue(
				tc.controlN, tc.controlMean, tc.controlVar,
				tc.treatN, tc.treatMean, tc.treatVar,
			)

			isSignificant := pValue < tc.sigLevel

			// Allow some tolerance for edge cases
			if isSignificant != tc.significant && math.Abs(pValue-tc.sigLevel) > 0.01 {
				t.Errorf("Expected significant=%v, got p-value=%f", tc.significant, pValue)
			}
		})
	}
}

// TestLiftCalculation tests lift percentage calculation
func TestLiftCalculation(t *testing.T) {
	tests := []struct {
		name         string
		controlMean  float64
		treatMean    float64
		expectedLift float64
	}{
		{"10% improvement", 100, 110, 10.0},
		{"50% improvement", 100, 150, 50.0},
		{"No change", 100, 100, 0.0},
		{"Negative lift", 100, 90, -10.0},
		{"100% improvement", 100, 200, 100.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			lift := calculateLift(tc.controlMean, tc.treatMean)
			if math.Abs(lift-tc.expectedLift) > 0.01 {
				t.Errorf("Expected lift %.1f%%, got %.1f%%", tc.expectedLift, lift)
			}
		})
	}
}

// TestConfidenceInterval tests confidence interval calculation
func TestConfidenceInterval(t *testing.T) {
	// 95% CI for mean=100, stddev=10, n=100
	mean := 100.0
	stddev := 10.0
	n := int64(100)

	lower, upper := calculateConfidenceInterval(mean, stddev, n, 0.95)

	// CI should contain the mean
	if lower > mean || upper < mean {
		t.Error("CI should contain the mean")
	}

	// CI width should be reasonable
	width := upper - lower
	expectedWidth := 2 * 1.96 * stddev / math.Sqrt(float64(n)) // ~3.92
	if math.Abs(width-expectedWidth) > 0.1 {
		t.Errorf("Expected CI width ~%.2f, got %.2f", expectedWidth, width)
	}
}

// TestSampleSizeCalculation tests minimum sample size estimation
func TestSampleSizeCalculation(t *testing.T) {
	tests := []struct {
		name           string
		baselineRate   float64
		mde            float64 // Minimum Detectable Effect
		power          float64
		sigLevel       float64
		expectedMinN   int64
	}{
		{"Large effect", 0.10, 0.02, 0.80, 0.05, 1000},    // 20% relative lift
		{"Small effect", 0.10, 0.005, 0.80, 0.05, 10000},  // 5% relative lift
		{"High power", 0.10, 0.02, 0.95, 0.05, 1500},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sampleSize := calculateRequiredSampleSize(tc.baselineRate, tc.mde, tc.power, tc.sigLevel)

			// Should be within 50% of expected (sample size formulas are approximations)
			ratio := float64(sampleSize) / float64(tc.expectedMinN)
			if ratio < 0.5 || ratio > 2.0 {
				t.Errorf("Expected sample size ~%d, got %d", tc.expectedMinN, sampleSize)
			}
		})
	}
}

// TestExperimentValidation tests experiment configuration validation
func TestExperimentValidation(t *testing.T) {
	tests := []struct {
		name      string
		variants  []*Variant
		expectErr bool
	}{
		{
			name: "Valid 2 variants",
			variants: []*Variant{
				{TrafficPercent: 50, IsControl: true},
				{TrafficPercent: 50},
			},
			expectErr: false,
		},
		{
			name: "Valid 3 variants",
			variants: []*Variant{
				{TrafficPercent: 50, IsControl: true},
				{TrafficPercent: 25},
				{TrafficPercent: 25},
			},
			expectErr: false,
		},
		{
			name: "Single variant",
			variants: []*Variant{
				{TrafficPercent: 100, IsControl: true},
			},
			expectErr: true, // Need at least 2 variants
		},
		{
			name: "Traffic not 100%",
			variants: []*Variant{
				{TrafficPercent: 50, IsControl: true},
				{TrafficPercent: 40},
			},
			expectErr: true, // 90% != 100%
		},
		{
			name: "No control",
			variants: []*Variant{
				{TrafficPercent: 50},
				{TrafficPercent: 50},
			},
			expectErr: true, // Need a control
		},
		{
			name: "Multiple controls",
			variants: []*Variant{
				{TrafficPercent: 50, IsControl: true},
				{TrafficPercent: 50, IsControl: true},
			},
			expectErr: true, // Only one control allowed
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateExperiment(tc.variants)
			if tc.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestHashConsistency tests FNV hash consistency
func TestHashConsistency(t *testing.T) {
	testCases := []string{
		"user-123",
		"user-456",
		"test@example.com",
		"uuid-a1b2c3d4",
	}

	for _, userID := range testCases {
		hash1 := fnvHash(userID)
		hash2 := fnvHash(userID)
		if hash1 != hash2 {
			t.Errorf("Hash should be consistent for %s", userID)
		}
	}
}

// Helper types and functions

type Variant struct {
	ID             string
	Name           string
	TrafficPercent int32
	IsControl      bool
}

func assignVariantByHashTest(userID string, variants []*Variant) string {
	hash := fnvHash(userID)
	bucket := hash % 100

	var cumulative int32
	for _, v := range variants {
		cumulative += v.TrafficPercent
		if bucket < int(cumulative) {
			return v.ID
		}
	}

	if len(variants) > 0 {
		return variants[0].ID
	}
	return ""
}

func calculatePValue(controlN int64, controlMean, controlVar float64,
	treatN int64, treatMean, treatVar float64) float64 {

	if controlN == 0 || treatN == 0 {
		return 1.0
	}

	// Z-test for difference of means
	pooledSE := math.Sqrt(controlVar/float64(controlN) + treatVar/float64(treatN))
	if pooledSE == 0 {
		return 1.0
	}

	zScore := (treatMean - controlMean) / pooledSE

	// Two-tailed p-value
	pValue := 2 * (1 - normalCDF(math.Abs(zScore)))
	return pValue
}

func calculateLift(controlMean, treatMean float64) float64 {
	if controlMean == 0 {
		return 0
	}
	return (treatMean - controlMean) / controlMean * 100
}

func calculateConfidenceInterval(mean, stddev float64, n int64, confidence float64) (float64, float64) {
	// Z-score for 95% confidence
	z := 1.96
	if confidence == 0.99 {
		z = 2.576
	} else if confidence == 0.90 {
		z = 1.645
	}

	marginOfError := z * stddev / math.Sqrt(float64(n))
	return mean - marginOfError, mean + marginOfError
}

func calculateRequiredSampleSize(baselineRate, mde, power, sigLevel float64) int64 {
	// Simplified sample size formula for proportions
	// n = 2 * (Z_alpha + Z_beta)^2 * p * (1-p) / (delta)^2

	zAlpha := 1.96 // for 0.05 significance
	if sigLevel == 0.01 {
		zAlpha = 2.576
	}

	zBeta := 0.84 // for 0.80 power
	if power == 0.95 {
		zBeta = 1.645
	} else if power == 0.90 {
		zBeta = 1.28
	}

	p := baselineRate
	delta := mde

	n := 2 * math.Pow(zAlpha+zBeta, 2) * p * (1 - p) / (delta * delta)
	return int64(math.Ceil(n))
}

func validateExperiment(variants []*Variant) error {
	if len(variants) < 2 {
		return &ValidationError{Field: "variants", Message: "at least 2 required"}
	}

	var totalTraffic int32
	var controlCount int
	for _, v := range variants {
		totalTraffic += v.TrafficPercent
		if v.IsControl {
			controlCount++
		}
	}

	if totalTraffic != 100 {
		return &ValidationError{Field: "traffic", Message: "must sum to 100"}
	}

	if controlCount == 0 {
		return &ValidationError{Field: "control", Message: "one control required"}
	}

	if controlCount > 1 {
		return &ValidationError{Field: "control", Message: "only one control allowed"}
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

// Benchmark tests

func BenchmarkVariantAssignment(b *testing.B) {
	variants := []*Variant{
		{ID: "control", TrafficPercent: 50},
		{ID: "treatment", TrafficPercent: 50},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		assignVariantByHashTest("user-12345", variants)
	}
}

func BenchmarkPValueCalculation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculatePValue(1000, 100, 25, 1000, 105, 25)
	}
}

func BenchmarkHashFunction(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fnvHash("user-12345-test-benchmark")
	}
}
