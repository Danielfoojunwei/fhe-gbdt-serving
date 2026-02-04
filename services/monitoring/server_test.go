// Unit tests for Monitoring Service

package main

import (
	"math"
	"testing"
	"time"
)

// TestPSICalculation tests Population Stability Index calculation
func TestPSICalculation(t *testing.T) {
	tests := []struct {
		name     string
		baseline featureStats
		current  featureStats
		maxPSI   float64 // Maximum acceptable PSI
	}{
		{
			name:     "No drift",
			baseline: featureStats{Mean: 100, Stddev: 10},
			current:  featureStats{Mean: 100, Stddev: 10},
			maxPSI:   0.01,
		},
		{
			name:     "Small drift",
			baseline: featureStats{Mean: 100, Stddev: 10},
			current:  featureStats{Mean: 102, Stddev: 11},
			maxPSI:   0.1,
		},
		{
			name:     "Significant drift",
			baseline: featureStats{Mean: 100, Stddev: 10},
			current:  featureStats{Mean: 120, Stddev: 15},
			maxPSI:   0.5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			psi := calculatePSI(tc.baseline, tc.current)
			if psi > tc.maxPSI {
				t.Errorf("PSI %f exceeds expected max %f", psi, tc.maxPSI)
			}
			if psi < 0 {
				t.Errorf("PSI should not be negative, got %f", psi)
			}
		})
	}
}

// TestKSStatisticCalculation tests Kolmogorov-Smirnov statistic calculation
func TestKSStatisticCalculation(t *testing.T) {
	tests := []struct {
		name     string
		baseline featureStats
		current  featureStats
		maxKS    float64
	}{
		{
			name:     "Identical distributions",
			baseline: featureStats{Mean: 50, Stddev: 5},
			current:  featureStats{Mean: 50, Stddev: 5},
			maxKS:    0.01,
		},
		{
			name:     "Shifted mean",
			baseline: featureStats{Mean: 50, Stddev: 5},
			current:  featureStats{Mean: 55, Stddev: 5},
			maxKS:    0.5,
		},
		{
			name:     "Changed variance",
			baseline: featureStats{Mean: 50, Stddev: 5},
			current:  featureStats{Mean: 50, Stddev: 10},
			maxKS:    0.3,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ks := calculateKSStatistic(tc.baseline, tc.current)
			if ks > tc.maxKS {
				t.Errorf("KS statistic %f exceeds expected max %f", ks, tc.maxKS)
			}
			if ks < 0 || ks > 1 {
				t.Errorf("KS statistic should be between 0 and 1, got %f", ks)
			}
		})
	}
}

// TestDriftDetection tests overall drift detection logic
func TestDriftDetection(t *testing.T) {
	tests := []struct {
		name          string
		baselineStats map[string]featureStats
		currentStats  map[string]featureStats
		expectDrift   bool
	}{
		{
			name: "No drift",
			baselineStats: map[string]featureStats{
				"feature1": {Mean: 100, Stddev: 10},
				"feature2": {Mean: 50, Stddev: 5},
			},
			currentStats: map[string]featureStats{
				"feature1": {Mean: 101, Stddev: 10},
				"feature2": {Mean: 50, Stddev: 5},
			},
			expectDrift: false,
		},
		{
			name: "Single feature drift",
			baselineStats: map[string]featureStats{
				"feature1": {Mean: 100, Stddev: 10},
				"feature2": {Mean: 50, Stddev: 5},
			},
			currentStats: map[string]featureStats{
				"feature1": {Mean: 150, Stddev: 20},
				"feature2": {Mean: 50, Stddev: 5},
			},
			expectDrift: true,
		},
		{
			name: "Multiple feature drift",
			baselineStats: map[string]featureStats{
				"feature1": {Mean: 100, Stddev: 10},
				"feature2": {Mean: 50, Stddev: 5},
			},
			currentStats: map[string]featureStats{
				"feature1": {Mean: 150, Stddev: 20},
				"feature2": {Mean: 80, Stddev: 15},
			},
			expectDrift: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			driftDetected := detectDrift(tc.baselineStats, tc.currentStats, 0.2, 0.1)
			if driftDetected != tc.expectDrift {
				t.Errorf("Expected drift=%v, got %v", tc.expectDrift, driftDetected)
			}
		})
	}
}

// TestAlertConditionEvaluation tests alert condition evaluation
func TestAlertConditionEvaluation(t *testing.T) {
	tests := []struct {
		name      string
		value     float64
		condition string
		threshold float64
		triggered bool
	}{
		{"gt triggered", 100, "gt", 50, true},
		{"gt not triggered", 100, "gt", 150, false},
		{"gte triggered equal", 100, "gte", 100, true},
		{"gte triggered greater", 100, "gte", 50, true},
		{"gte not triggered", 100, "gte", 150, false},
		{"lt triggered", 50, "lt", 100, true},
		{"lt not triggered", 150, "lt", 100, false},
		{"lte triggered equal", 100, "lte", 100, true},
		{"lte triggered less", 50, "lte", 100, true},
		{"eq triggered", 100, "eq", 100, true},
		{"eq not triggered", 100, "eq", 50, false},
		{"> alias", 100, ">", 50, true},
		{">= alias", 100, ">=", 100, true},
		{"< alias", 50, "<", 100, true},
		{"<= alias", 100, "<=", 100, true},
		{"== alias", 100, "==", 100, true},
		{"unknown condition", 100, "unknown", 50, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := evaluateCondition(tc.value, tc.condition, tc.threshold)
			if result != tc.triggered {
				t.Errorf("Expected triggered=%v, got %v", tc.triggered, result)
			}
		})
	}
}

// TestLatencyPercentileCalculation tests percentile calculation
func TestLatencyPercentileCalculation(t *testing.T) {
	// Latencies in sorted order
	latencies := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	tests := []struct {
		percentile float64
		expected   float64
		tolerance  float64
	}{
		{0.5, 55, 10},  // p50 (median)
		{0.95, 95, 10}, // p95
		{0.99, 99, 5},  // p99
		{0.0, 10, 1},   // min
		{1.0, 100, 1},  // max
	}

	for _, tc := range tests {
		t.Run("p"+string(rune(int(tc.percentile*100))), func(t *testing.T) {
			result := calculatePercentile(latencies, tc.percentile)
			if math.Abs(result-tc.expected) > tc.tolerance {
				t.Errorf("Expected ~%f, got %f", tc.expected, result)
			}
		})
	}
}

// TestMetricsAggregation tests time-series aggregation
func TestMetricsAggregation(t *testing.T) {
	now := time.Now()
	dataPoints := []MetricsDataPoint{
		{Timestamp: now.Add(-5 * time.Minute), LatencyMs: 50, Count: 100, ErrorCount: 2},
		{Timestamp: now.Add(-4 * time.Minute), LatencyMs: 55, Count: 120, ErrorCount: 1},
		{Timestamp: now.Add(-3 * time.Minute), LatencyMs: 45, Count: 90, ErrorCount: 0},
		{Timestamp: now.Add(-2 * time.Minute), LatencyMs: 60, Count: 110, ErrorCount: 3},
		{Timestamp: now.Add(-1 * time.Minute), LatencyMs: 52, Count: 105, ErrorCount: 1},
	}

	summary := aggregateMetrics(dataPoints)

	expectedTotalCount := int64(525)
	if summary.TotalCount != expectedTotalCount {
		t.Errorf("Expected total count %d, got %d", expectedTotalCount, summary.TotalCount)
	}

	expectedTotalErrors := int64(7)
	if summary.TotalErrors != expectedTotalErrors {
		t.Errorf("Expected total errors %d, got %d", expectedTotalErrors, summary.TotalErrors)
	}

	expectedAvgLatency := (50.0 + 55.0 + 45.0 + 60.0 + 52.0) / 5.0
	if math.Abs(summary.AvgLatency-expectedAvgLatency) > 0.01 {
		t.Errorf("Expected avg latency %f, got %f", expectedAvgLatency, summary.AvgLatency)
	}
}

// TestErrorRateCalculation tests error rate calculation
func TestErrorRateCalculation(t *testing.T) {
	tests := []struct {
		name        string
		total       int64
		errors      int64
		expectedPct float64
	}{
		{"No errors", 1000, 0, 0},
		{"1% errors", 1000, 10, 1.0},
		{"10% errors", 100, 10, 10.0},
		{"All errors", 100, 100, 100.0},
		{"Zero total", 0, 0, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rate := calculateErrorRate(tc.total, tc.errors)
			if math.Abs(rate-tc.expectedPct) > 0.01 {
				t.Errorf("Expected error rate %f%%, got %f%%", tc.expectedPct, rate)
			}
		})
	}
}

// TestHealthStatusDetermination tests health status logic
func TestHealthStatusDetermination(t *testing.T) {
	tests := []struct {
		name       string
		errorRate  float64
		p95Latency float64
		expected   string
	}{
		{"Healthy", 0.1, 50, "healthy"},
		{"Degraded - high latency", 0.1, 200, "degraded"},
		{"Degraded - elevated errors", 2.0, 50, "degraded"},
		{"Unhealthy - high errors", 5.0, 50, "unhealthy"},
		{"Unhealthy - very high latency", 0.1, 500, "unhealthy"},
		{"Unhealthy - both bad", 5.0, 500, "unhealthy"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			status := determineHealthStatus(tc.errorRate, tc.p95Latency)
			if status != tc.expected {
				t.Errorf("Expected status %s, got %s", tc.expected, status)
			}
		})
	}
}

// Helper types and functions

type MetricsDataPoint struct {
	Timestamp  time.Time
	LatencyMs  float64
	Count      int64
	ErrorCount int64
}

type MetricsSummary struct {
	TotalCount  int64
	TotalErrors int64
	AvgLatency  float64
	MinLatency  float64
	MaxLatency  float64
}

func detectDrift(baseline, current map[string]featureStats, psiThreshold, ksThreshold float64) bool {
	for feature, baselineStats := range baseline {
		currentStats, exists := current[feature]
		if !exists {
			continue
		}

		psi := calculatePSI(baselineStats, currentStats)
		ks := calculateKSStatistic(baselineStats, currentStats)

		if psi > psiThreshold || ks > ksThreshold {
			return true
		}
	}
	return false
}

func calculatePercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}

	index := p * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1
	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}

	fraction := index - float64(lower)
	return sorted[lower] + fraction*(sorted[upper]-sorted[lower])
}

func aggregateMetrics(points []MetricsDataPoint) MetricsSummary {
	if len(points) == 0 {
		return MetricsSummary{}
	}

	var summary MetricsSummary
	var totalLatency float64
	summary.MinLatency = points[0].LatencyMs
	summary.MaxLatency = points[0].LatencyMs

	for _, p := range points {
		summary.TotalCount += p.Count
		summary.TotalErrors += p.ErrorCount
		totalLatency += p.LatencyMs

		if p.LatencyMs < summary.MinLatency {
			summary.MinLatency = p.LatencyMs
		}
		if p.LatencyMs > summary.MaxLatency {
			summary.MaxLatency = p.LatencyMs
		}
	}

	summary.AvgLatency = totalLatency / float64(len(points))
	return summary
}

func calculateErrorRate(total, errors int64) float64 {
	if total == 0 {
		return 0
	}
	return float64(errors) / float64(total) * 100
}

func determineHealthStatus(errorRate, p95Latency float64) string {
	// Thresholds
	const (
		errorUnhealthy  = 5.0   // 5%
		errorDegraded   = 1.0   // 1%
		latencyUnhealthy = 300.0 // 300ms
		latencyDegraded  = 150.0 // 150ms
	)

	if errorRate >= errorUnhealthy || p95Latency >= latencyUnhealthy {
		return "unhealthy"
	}
	if errorRate >= errorDegraded || p95Latency >= latencyDegraded {
		return "degraded"
	}
	return "healthy"
}

// Benchmark tests

func BenchmarkPSICalculation(b *testing.B) {
	baseline := featureStats{Mean: 100, Stddev: 10}
	current := featureStats{Mean: 105, Stddev: 12}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculatePSI(baseline, current)
	}
}

func BenchmarkDriftDetection(b *testing.B) {
	baseline := map[string]featureStats{
		"f1": {Mean: 100, Stddev: 10},
		"f2": {Mean: 50, Stddev: 5},
		"f3": {Mean: 200, Stddev: 20},
	}
	current := map[string]featureStats{
		"f1": {Mean: 105, Stddev: 11},
		"f2": {Mean: 52, Stddev: 5},
		"f3": {Mean: 210, Stddev: 22},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detectDrift(baseline, current, 0.2, 0.1)
	}
}

func BenchmarkPercentileCalculation(b *testing.B) {
	latencies := make([]float64, 1000)
	for i := range latencies {
		latencies[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculatePercentile(latencies, 0.95)
	}
}
