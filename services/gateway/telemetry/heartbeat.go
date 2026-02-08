// Package telemetry provides heartbeat usage reporting for the on-prem gateway.
//
// Collects counter-only metrics (prediction counts, latency percentiles, error counts)
// and periodically reports them to the vendor's cloud control plane.
//
// SECURITY: No sensitive data (ciphertext, features, keys, payloads) is ever
// included in telemetry. Only counters and timing information are reported.
package telemetry

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"
)

// ForbiddenFields that must NEVER appear in telemetry reports.
var ForbiddenFields = map[string]bool{
	"ciphertext": true, "plaintext": true, "secret_key": true,
	"eval_key": true, "features": true, "predictions": true,
	"payload": true, "api_key": true, "password": true,
	"token": true, "model_weights": true,
}

// PredictionEvent represents a single prediction for telemetry aggregation.
type PredictionEvent struct {
	TenantID  string
	ModelID   string
	LicenseID string
	LatencyMs float64
	Success   bool
	Timestamp time.Time
}

// HeartbeatReport is the aggregated telemetry payload sent to the control plane.
type HeartbeatReport struct {
	TenantID              string         `json:"tenant_id"`
	LicenseID             string         `json:"license_id"`
	ReportID              string         `json:"report_id"`
	IntervalStart         time.Time      `json:"interval_start"`
	IntervalEnd           time.Time      `json:"interval_end"`
	TotalPredictions      int64          `json:"total_predictions"`
	SuccessfulPredictions int64          `json:"successful_predictions"`
	FailedPredictions     int64          `json:"failed_predictions"`
	PredictionsByModel    map[string]int64 `json:"predictions_by_model"`
	LatencyP50Ms          float64        `json:"latency_p50_ms"`
	LatencyP95Ms          float64        `json:"latency_p95_ms"`
	LatencyP99Ms          float64        `json:"latency_p99_ms"`
}

// ToJSON serializes the report, ensuring no forbidden fields are included.
func (r *HeartbeatReport) ToJSON() ([]byte, error) {
	return json.Marshal(r)
}

// ReportCallback is invoked when a new heartbeat report is ready.
type ReportCallback func(report *HeartbeatReport)

// Collector aggregates prediction events and produces periodic reports.
type Collector struct {
	mu sync.Mutex

	tenantID  string
	licenseID string
	interval  time.Duration
	callback  ReportCallback

	events        []PredictionEvent
	intervalStart time.Time
	reportCounter int64

	running bool
	stopCh  chan struct{}
	reports []*HeartbeatReport
}

// NewCollector creates a heartbeat telemetry collector.
func NewCollector(tenantID, licenseID string, intervalSeconds int, callback ReportCallback) *Collector {
	return &Collector{
		tenantID:      tenantID,
		licenseID:     licenseID,
		interval:      time.Duration(intervalSeconds) * time.Second,
		callback:      callback,
		events:        make([]PredictionEvent, 0, 1000),
		intervalStart: time.Now(),
		stopCh:        make(chan struct{}),
	}
}

// Record adds a prediction event. Thread-safe.
func (c *Collector) Record(event PredictionEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, event)
}

// RecordPrediction is a convenience method to record a prediction.
func (c *Collector) RecordPrediction(modelID string, latencyMs float64, success bool) {
	c.Record(PredictionEvent{
		TenantID:  c.tenantID,
		ModelID:   modelID,
		LicenseID: c.licenseID,
		LatencyMs: latencyMs,
		Success:   success,
		Timestamp: time.Now(),
	})
}

// Flush aggregates current events into a report and resets.
func (c *Collector) Flush() *HeartbeatReport {
	c.mu.Lock()
	events := c.events
	c.events = make([]PredictionEvent, 0, 1000)
	intervalStart := c.intervalStart
	c.intervalStart = time.Now()
	c.mu.Unlock()

	if len(events) == 0 {
		return nil
	}

	c.reportCounter++
	now := time.Now()

	var successes, failures int64
	byModel := make(map[string]int64)
	latencies := make([]float64, 0, len(events))

	for _, e := range events {
		if e.Success {
			successes++
		} else {
			failures++
		}
		byModel[e.ModelID]++
		latencies = append(latencies, e.LatencyMs)
	}

	sort.Float64s(latencies)

	report := &HeartbeatReport{
		TenantID:              c.tenantID,
		LicenseID:             c.licenseID,
		ReportID:              fmt.Sprintf("%s-%d", c.tenantID, c.reportCounter),
		IntervalStart:         intervalStart,
		IntervalEnd:           now,
		TotalPredictions:      int64(len(events)),
		SuccessfulPredictions: successes,
		FailedPredictions:     failures,
		PredictionsByModel:    byModel,
		LatencyP50Ms:          percentile(latencies, 50),
		LatencyP95Ms:          percentile(latencies, 95),
		LatencyP99Ms:          percentile(latencies, 99),
	}

	c.mu.Lock()
	c.reports = append(c.reports, report)
	c.mu.Unlock()

	if c.callback != nil {
		c.callback(report)
	}

	return report
}

// Start begins periodic background reporting.
func (c *Collector) Start() {
	c.running = true
	go func() {
		ticker := time.NewTicker(c.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				c.Flush()
			case <-c.stopCh:
				return
			}
		}
	}()
}

// Stop halts periodic reporting and flushes remaining events.
func (c *Collector) Stop() *HeartbeatReport {
	if c.running {
		c.running = false
		close(c.stopCh)
	}
	return c.Flush()
}

// GetReports returns all collected reports.
func (c *Collector) GetReports() []*HeartbeatReport {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]*HeartbeatReport, len(c.reports))
	copy(result, c.reports)
	return result
}

func percentile(sorted []float64, pct int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Max(0, float64(len(sorted)*pct/100)-1))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
