// Unit tests for Metering Service

package main

import (
	"context"
	"sync"
	"testing"
	"time"
)

// MockMeteringStore implements an in-memory metering store for testing
type MockMeteringStore struct {
	mu       sync.RWMutex
	usage    map[string]*UsageRecord
	quotas   map[string]*Quota
}

type UsageRecord struct {
	TenantID        string
	Predictions     int64
	ComputeSeconds  float64
	StorageBytes    int64
	PeriodStart     time.Time
	PeriodEnd       time.Time
}

type Quota struct {
	TenantID            string
	MaxPredictions      int64
	MaxComputeSeconds   float64
	MaxStorageBytes     int64
	CurrentPredictions  int64
	CurrentCompute      float64
	CurrentStorage      int64
}

func NewMockMeteringStore() *MockMeteringStore {
	return &MockMeteringStore{
		usage:  make(map[string]*UsageRecord),
		quotas: make(map[string]*Quota),
	}
}

// TestRecordUsage tests usage recording
func TestRecordUsage(t *testing.T) {
	store := NewMockMeteringStore()

	tests := []struct {
		name        string
		tenantID    string
		usageType   string
		quantity    float64
		expectErr   bool
	}{
		{"Valid prediction", "tenant-1", "prediction", 1, false},
		{"Valid batch", "tenant-1", "prediction", 100, false},
		{"Valid compute", "tenant-1", "compute_seconds", 30.5, false},
		{"Valid storage", "tenant-1", "storage_bytes", 1024, false},
		{"Empty tenant", "", "prediction", 1, true},
		{"Negative quantity", "tenant-1", "prediction", -1, true},
		{"Unknown type", "tenant-1", "unknown", 1, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := recordUsage(store, tc.tenantID, tc.usageType, tc.quantity)
			if tc.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestQuotaCheck tests quota enforcement
func TestQuotaCheck(t *testing.T) {
	store := NewMockMeteringStore()

	// Set up quota for tenant
	store.quotas["tenant-1"] = &Quota{
		TenantID:           "tenant-1",
		MaxPredictions:     1000,
		CurrentPredictions: 900,
	}
	store.quotas["tenant-2"] = &Quota{
		TenantID:           "tenant-2",
		MaxPredictions:     1000,
		CurrentPredictions: 1000,
	}
	store.quotas["tenant-unlimited"] = &Quota{
		TenantID:           "tenant-unlimited",
		MaxPredictions:     -1, // unlimited
		CurrentPredictions: 1000000,
	}

	tests := []struct {
		name        string
		tenantID    string
		requested   int64
		allowed     bool
	}{
		{"Under quota", "tenant-1", 50, true},
		{"At limit", "tenant-1", 100, true},
		{"Over quota", "tenant-1", 101, false},
		{"Already at max", "tenant-2", 1, false},
		{"Unlimited plan", "tenant-unlimited", 10000, true},
		{"New tenant (no quota)", "tenant-new", 1, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			allowed := checkQuota(store, tc.tenantID, tc.requested)
			if allowed != tc.allowed {
				t.Errorf("Expected allowed=%v, got %v", tc.allowed, allowed)
			}
		})
	}
}

// TestUsageAggregation tests usage aggregation over time periods
func TestUsageAggregation(t *testing.T) {
	records := []*UsageRecord{
		{TenantID: "tenant-1", Predictions: 100, PeriodStart: time.Now().Add(-24 * time.Hour)},
		{TenantID: "tenant-1", Predictions: 200, PeriodStart: time.Now().Add(-12 * time.Hour)},
		{TenantID: "tenant-1", Predictions: 150, PeriodStart: time.Now().Add(-1 * time.Hour)},
		{TenantID: "tenant-2", Predictions: 500, PeriodStart: time.Now().Add(-6 * time.Hour)},
	}

	// Aggregate for tenant-1
	total := aggregateUsage(records, "tenant-1", time.Now().Add(-48*time.Hour), time.Now())
	if total != 450 {
		t.Errorf("Expected total 450 for tenant-1, got %d", total)
	}

	// Aggregate for tenant-2
	total = aggregateUsage(records, "tenant-2", time.Now().Add(-48*time.Hour), time.Now())
	if total != 500 {
		t.Errorf("Expected total 500 for tenant-2, got %d", total)
	}

	// Aggregate with time filter
	total = aggregateUsage(records, "tenant-1", time.Now().Add(-6*time.Hour), time.Now())
	if total != 150 {
		t.Errorf("Expected total 150 for last 6 hours, got %d", total)
	}
}

// TestUsageRateLimiting tests rate limiting logic
func TestUsageRateLimiting(t *testing.T) {
	limiter := NewRateLimiter(100, time.Minute) // 100 requests per minute

	// Should allow initial requests
	for i := 0; i < 100; i++ {
		if !limiter.Allow("tenant-1") {
			t.Errorf("Request %d should be allowed", i)
		}
	}

	// Should block after limit
	if limiter.Allow("tenant-1") {
		t.Error("Request 101 should be blocked")
	}

	// Different tenant should be allowed
	if !limiter.Allow("tenant-2") {
		t.Error("Different tenant should be allowed")
	}
}

// TestConcurrentUsageRecording tests thread safety
func TestConcurrentUsageRecording(t *testing.T) {
	store := NewMockMeteringStore()
	store.quotas["tenant-1"] = &Quota{
		TenantID:           "tenant-1",
		MaxPredictions:     -1, // unlimited
		CurrentPredictions: 0,
	}

	var wg sync.WaitGroup
	numGoroutines := 100
	requestsPerGoroutine := 100

	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				recordUsageConcurrent(store, "tenant-1", 1)
			}
		}()
	}
	wg.Wait()

	expected := int64(numGoroutines * requestsPerGoroutine)
	actual := store.quotas["tenant-1"].CurrentPredictions
	if actual != expected {
		t.Errorf("Expected %d predictions, got %d", expected, actual)
	}
}

// TestUsageSummaryCalculation tests usage summary generation
func TestUsageSummaryCalculation(t *testing.T) {
	records := []*DetailedUsage{
		{Type: "prediction", Quantity: 1000, Timestamp: time.Now().Add(-1 * time.Hour)},
		{Type: "prediction", Quantity: 500, Timestamp: time.Now().Add(-30 * time.Minute)},
		{Type: "compute_seconds", Quantity: 3600, Timestamp: time.Now().Add(-1 * time.Hour)},
		{Type: "storage_bytes", Quantity: 1024 * 1024 * 100, Timestamp: time.Now()},
	}

	summary := calculateUsageSummary(records)

	if summary.TotalPredictions != 1500 {
		t.Errorf("Expected 1500 predictions, got %d", summary.TotalPredictions)
	}
	if summary.TotalComputeSeconds != 3600 {
		t.Errorf("Expected 3600 compute seconds, got %f", summary.TotalComputeSeconds)
	}
	if summary.TotalStorageBytes != 1024*1024*100 {
		t.Errorf("Expected %d storage bytes, got %d", 1024*1024*100, summary.TotalStorageBytes)
	}
}

// Helper types and functions for testing

type DetailedUsage struct {
	Type      string
	Quantity  float64
	Timestamp time.Time
}

type UsageSummary struct {
	TotalPredictions    int64
	TotalComputeSeconds float64
	TotalStorageBytes   int64
}

type RateLimiter struct {
	limit    int
	window   time.Duration
	counters map[string]*counter
	mu       sync.Mutex
}

type counter struct {
	count    int
	resetAt  time.Time
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		limit:    limit,
		window:   window,
		counters: make(map[string]*counter),
	}
}

func (r *RateLimiter) Allow(key string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	c, exists := r.counters[key]
	if !exists || now.After(c.resetAt) {
		r.counters[key] = &counter{count: 1, resetAt: now.Add(r.window)}
		return true
	}

	if c.count >= r.limit {
		return false
	}

	c.count++
	return true
}

func recordUsage(store *MockMeteringStore, tenantID, usageType string, quantity float64) error {
	if tenantID == "" {
		return &ValidationError{Field: "tenant_id", Message: "required"}
	}
	if quantity < 0 {
		return &ValidationError{Field: "quantity", Message: "must be non-negative"}
	}
	validTypes := map[string]bool{"prediction": true, "compute_seconds": true, "storage_bytes": true}
	if !validTypes[usageType] {
		return &ValidationError{Field: "usage_type", Message: "unknown type"}
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

func checkQuota(store *MockMeteringStore, tenantID string, requested int64) bool {
	store.mu.RLock()
	defer store.mu.RUnlock()

	quota, exists := store.quotas[tenantID]
	if !exists {
		return true // No quota = allowed
	}
	if quota.MaxPredictions == -1 {
		return true // Unlimited
	}
	return quota.CurrentPredictions+requested <= quota.MaxPredictions
}

func aggregateUsage(records []*UsageRecord, tenantID string, start, end time.Time) int64 {
	var total int64
	for _, r := range records {
		if r.TenantID == tenantID && r.PeriodStart.After(start) && r.PeriodStart.Before(end) {
			total += r.Predictions
		}
	}
	return total
}

func recordUsageConcurrent(store *MockMeteringStore, tenantID string, quantity int64) {
	store.mu.Lock()
	defer store.mu.Unlock()
	if q, exists := store.quotas[tenantID]; exists {
		q.CurrentPredictions += quantity
	}
}

func calculateUsageSummary(records []*DetailedUsage) *UsageSummary {
	summary := &UsageSummary{}
	for _, r := range records {
		switch r.Type {
		case "prediction":
			summary.TotalPredictions += int64(r.Quantity)
		case "compute_seconds":
			summary.TotalComputeSeconds += r.Quantity
		case "storage_bytes":
			summary.TotalStorageBytes += int64(r.Quantity)
		}
	}
	return summary
}

// Benchmark tests

func BenchmarkQuotaCheck(b *testing.B) {
	store := NewMockMeteringStore()
	store.quotas["tenant-1"] = &Quota{
		TenantID:           "tenant-1",
		MaxPredictions:     1000000,
		CurrentPredictions: 500000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		checkQuota(store, "tenant-1", 1)
	}
}

func BenchmarkRateLimiter(b *testing.B) {
	limiter := NewRateLimiter(1000000, time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		limiter.Allow("tenant-1")
	}
}
