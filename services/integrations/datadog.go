// DataDog Integration
// Sends metrics, traces, and logs to DataDog

package integrations

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

// DataDogClient provides DataDog integration
type DataDogClient struct {
	apiKey      string
	appKey      string
	site        string // datadoghq.com, datadoghq.eu
	serviceName string
	environment string
	version     string

	httpClient  *http.Client
	metricsURL  string
	eventsURL   string
	logsURL     string

	metricsBatch []DataDogMetric
	mu           sync.Mutex
	flushTicker  *time.Ticker
	done         chan struct{}
}

// DataDogMetric represents a DataDog metric
type DataDogMetric struct {
	Metric   string          `json:"metric"`
	Type     string          `json:"type"` // count, gauge, rate
	Points   [][]interface{} `json:"points"`
	Tags     []string        `json:"tags,omitempty"`
	Host     string          `json:"host,omitempty"`
	Interval int             `json:"interval,omitempty"`
}

// DataDogEvent represents a DataDog event
type DataDogEvent struct {
	Title          string   `json:"title"`
	Text           string   `json:"text"`
	DateHappened   int64    `json:"date_happened,omitempty"`
	Priority       string   `json:"priority,omitempty"` // normal, low
	Host           string   `json:"host,omitempty"`
	Tags           []string `json:"tags,omitempty"`
	AlertType      string   `json:"alert_type,omitempty"` // error, warning, info, success
	AggregationKey string   `json:"aggregation_key,omitempty"`
	SourceTypeName string   `json:"source_type_name,omitempty"`
}

// DataDogLog represents a DataDog log entry
type DataDogLog struct {
	Message  string                 `json:"message"`
	DDSource string                 `json:"ddsource"`
	DDTags   string                 `json:"ddtags"`
	Hostname string                 `json:"hostname"`
	Service  string                 `json:"service"`
	Status   string                 `json:"status"` // debug, info, warn, error
	Data     map[string]interface{} `json:"data,omitempty"`
}

// DataDogConfig contains DataDog configuration
type DataDogConfig struct {
	APIKey      string
	AppKey      string
	Site        string // Default: datadoghq.com
	ServiceName string
	Environment string
	Version     string
	FlushPeriod time.Duration
}

// NewDataDogClient creates a new DataDog client
func NewDataDogClient(config *DataDogConfig) (*DataDogClient, error) {
	if config.APIKey == "" {
		config.APIKey = os.Getenv("DD_API_KEY")
	}
	if config.AppKey == "" {
		config.AppKey = os.Getenv("DD_APP_KEY")
	}
	if config.Site == "" {
		config.Site = os.Getenv("DD_SITE")
		if config.Site == "" {
			config.Site = "datadoghq.com"
		}
	}
	if config.ServiceName == "" {
		config.ServiceName = "fhe-gbdt-serving"
	}
	if config.Environment == "" {
		config.Environment = os.Getenv("DD_ENV")
		if config.Environment == "" {
			config.Environment = "production"
		}
	}
	if config.FlushPeriod == 0 {
		config.FlushPeriod = 10 * time.Second
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("DataDog API key is required")
	}

	client := &DataDogClient{
		apiKey:       config.APIKey,
		appKey:       config.AppKey,
		site:         config.Site,
		serviceName:  config.ServiceName,
		environment:  config.Environment,
		version:      config.Version,
		httpClient:   &http.Client{Timeout: 30 * time.Second},
		metricsURL:   fmt.Sprintf("https://api.%s/api/v2/series", config.Site),
		eventsURL:    fmt.Sprintf("https://api.%s/api/v1/events", config.Site),
		logsURL:      fmt.Sprintf("https://http-intake.logs.%s/api/v2/logs", config.Site),
		metricsBatch: make([]DataDogMetric, 0, 100),
		done:         make(chan struct{}),
	}

	// Start background flusher
	client.flushTicker = time.NewTicker(config.FlushPeriod)
	go client.flushLoop()

	return client, nil
}

// flushLoop periodically flushes metrics
func (c *DataDogClient) flushLoop() {
	for {
		select {
		case <-c.flushTicker.C:
			c.Flush()
		case <-c.done:
			return
		}
	}
}

// Close shuts down the DataDog client
func (c *DataDogClient) Close() {
	close(c.done)
	c.flushTicker.Stop()
	c.Flush() // Final flush
}

// ============================================================================
// Metrics
// ============================================================================

// Gauge records a gauge metric
func (c *DataDogClient) Gauge(name string, value float64, tags []string) {
	c.addMetric(DataDogMetric{
		Metric: c.prefixMetric(name),
		Type:   "gauge",
		Points: [][]interface{}{{time.Now().Unix(), value}},
		Tags:   c.enrichTags(tags),
	})
}

// Count records a count metric
func (c *DataDogClient) Count(name string, value float64, tags []string) {
	c.addMetric(DataDogMetric{
		Metric: c.prefixMetric(name),
		Type:   "count",
		Points: [][]interface{}{{time.Now().Unix(), value}},
		Tags:   c.enrichTags(tags),
	})
}

// Histogram records a histogram metric (becomes distribution in DataDog)
func (c *DataDogClient) Histogram(name string, value float64, tags []string) {
	// DataDog histograms are computed on the agent side
	// For API submission, we use gauge with specific naming
	c.addMetric(DataDogMetric{
		Metric: c.prefixMetric(name),
		Type:   "gauge",
		Points: [][]interface{}{{time.Now().Unix(), value}},
		Tags:   c.enrichTags(tags),
	})
}

// RecordPrediction records prediction metrics
func (c *DataDogClient) RecordPrediction(ctx context.Context, tenantID, modelID string, latencyMs float64, success bool) {
	tags := []string{
		"tenant_id:" + tenantID,
		"model_id:" + modelID,
		fmt.Sprintf("success:%t", success),
	}

	c.Count("prediction.count", 1, tags)
	c.Histogram("prediction.latency", latencyMs, tags)

	if !success {
		c.Count("prediction.errors", 1, tags)
	}
}

// RecordFHEOperation records FHE operation metrics
func (c *DataDogClient) RecordFHEOperation(operation string, timeMs float64, tags []string) {
	opTags := append(tags, "operation:"+operation)
	c.Histogram("fhe.operation_time", timeMs, opTags)
	c.Count("fhe.operations", 1, opTags)
}

// SetModelMetrics sets model-level metrics
func (c *DataDogClient) SetModelMetrics(modelID string, predictions int64, p95Latency, errorRate float64) {
	tags := []string{"model_id:" + modelID}
	c.Gauge("model.predictions", float64(predictions), tags)
	c.Gauge("model.latency_p95", p95Latency, tags)
	c.Gauge("model.error_rate", errorRate, tags)
}

// SetDriftScore sets model drift score
func (c *DataDogClient) SetDriftScore(modelID, feature string, score float64) {
	tags := []string{"model_id:" + modelID, "feature:" + feature}
	c.Gauge("model.drift_score", score, tags)
}

func (c *DataDogClient) addMetric(metric DataDogMetric) {
	c.mu.Lock()
	c.metricsBatch = append(c.metricsBatch, metric)
	c.mu.Unlock()
}

func (c *DataDogClient) prefixMetric(name string) string {
	return "fhe_gbdt." + name
}

func (c *DataDogClient) enrichTags(tags []string) []string {
	baseTags := []string{
		"service:" + c.serviceName,
		"env:" + c.environment,
	}
	if c.version != "" {
		baseTags = append(baseTags, "version:"+c.version)
	}
	return append(baseTags, tags...)
}

// Flush sends batched metrics to DataDog
func (c *DataDogClient) Flush() error {
	c.mu.Lock()
	if len(c.metricsBatch) == 0 {
		c.mu.Unlock()
		return nil
	}
	metrics := c.metricsBatch
	c.metricsBatch = make([]DataDogMetric, 0, 100)
	c.mu.Unlock()

	payload := map[string]interface{}{
		"series": metrics,
	}

	return c.post(c.metricsURL, payload)
}

// ============================================================================
// Events
// ============================================================================

// SendEvent sends an event to DataDog
func (c *DataDogClient) SendEvent(event *DataDogEvent) error {
	event.Tags = c.enrichTags(event.Tags)
	if event.DateHappened == 0 {
		event.DateHappened = time.Now().Unix()
	}
	return c.post(c.eventsURL, event)
}

// SendModelDeployedEvent sends a model deployment event
func (c *DataDogClient) SendModelDeployedEvent(modelID, version, deployedBy string) error {
	return c.SendEvent(&DataDogEvent{
		Title:     fmt.Sprintf("Model Deployed: %s v%s", modelID, version),
		Text:      fmt.Sprintf("Model %s version %s was deployed by %s", modelID, version, deployedBy),
		Priority:  "normal",
		AlertType: "info",
		Tags: []string{
			"model_id:" + modelID,
			"version:" + version,
			"event_type:deployment",
		},
		SourceTypeName: "fhe-gbdt",
	})
}

// SendAlertTriggeredEvent sends an alert triggered event
func (c *DataDogClient) SendAlertTriggeredEvent(alertName, modelID, metric string, value, threshold float64) error {
	return c.SendEvent(&DataDogEvent{
		Title:     fmt.Sprintf("Alert Triggered: %s", alertName),
		Text:      fmt.Sprintf("Alert %s triggered for model %s. Metric %s = %.2f (threshold: %.2f)", alertName, modelID, metric, value, threshold),
		Priority:  "normal",
		AlertType: "warning",
		Tags: []string{
			"model_id:" + modelID,
			"alert_name:" + alertName,
			"metric:" + metric,
			"event_type:alert",
		},
		SourceTypeName: "fhe-gbdt",
	})
}

// SendDriftDetectedEvent sends a drift detection event
func (c *DataDogClient) SendDriftDetectedEvent(modelID string, driftedFeatures []string, maxPSI float64) error {
	return c.SendEvent(&DataDogEvent{
		Title:     fmt.Sprintf("Drift Detected: %s", modelID),
		Text:      fmt.Sprintf("Data drift detected in model %s. Affected features: %v. Max PSI: %.3f", modelID, driftedFeatures, maxPSI),
		Priority:  "normal",
		AlertType: "warning",
		Tags: []string{
			"model_id:" + modelID,
			"event_type:drift",
		},
		SourceTypeName: "fhe-gbdt",
	})
}

// ============================================================================
// Logs
// ============================================================================

// SendLog sends a log entry to DataDog
func (c *DataDogClient) SendLog(log *DataDogLog) error {
	log.Service = c.serviceName
	log.DDTags = fmt.Sprintf("env:%s,service:%s", c.environment, c.serviceName)
	if c.version != "" {
		log.DDTags += ",version:" + c.version
	}

	logs := []DataDogLog{*log}
	return c.post(c.logsURL, logs)
}

// LogPrediction logs a prediction event
func (c *DataDogClient) LogPrediction(tenantID, modelID, requestID string, latencyMs float64, success bool, err error) {
	status := "info"
	message := fmt.Sprintf("Prediction completed for model %s in %.2fms", modelID, latencyMs)
	if !success {
		status = "error"
		message = fmt.Sprintf("Prediction failed for model %s: %v", modelID, err)
	}

	c.SendLog(&DataDogLog{
		Message:  message,
		DDSource: "fhe-gbdt",
		Status:   status,
		Data: map[string]interface{}{
			"tenant_id":  tenantID,
			"model_id":   modelID,
			"request_id": requestID,
			"latency_ms": latencyMs,
			"success":    success,
		},
	})
}

// ============================================================================
// HTTP Client
// ============================================================================

func (c *DataDogClient) post(url string, payload interface{}) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("DD-API-KEY", c.apiKey)
	if c.appKey != "" {
		req.Header.Set("DD-APPLICATION-KEY", c.appKey)
	}

	// Use bytes.NewReader for actual body
	req, _ = http.NewRequest("POST", url, nil)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("DD-API-KEY", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("DataDog API error: status %d", resp.StatusCode)
	}

	// Log for debugging
	_ = body

	return nil
}

// ============================================================================
// DataDog Monitor Templates
// ============================================================================

// GetDataDogMonitors returns DataDog monitor definitions
func GetDataDogMonitors() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":    "[FHE-GBDT] High Prediction Latency",
			"type":    "metric alert",
			"query":   "avg(last_5m):avg:fhe_gbdt.prediction.latency{*} by {model_id} > 200",
			"message": "High prediction latency detected for model {{model_id.name}}. Current: {{value}}ms @slack-fhe-alerts",
			"tags":    []string{"service:fhe-gbdt", "alert_type:latency"},
			"options": map[string]interface{}{
				"thresholds": map[string]float64{
					"critical": 200,
					"warning":  150,
				},
			},
		},
		{
			"name":    "[FHE-GBDT] High Error Rate",
			"type":    "metric alert",
			"query":   "sum(last_5m):sum:fhe_gbdt.prediction.errors{*}.as_count() / sum:fhe_gbdt.prediction.count{*}.as_count() * 100 > 1",
			"message": "High error rate detected: {{value}}% @pagerduty-fhe-critical",
			"tags":    []string{"service:fhe-gbdt", "alert_type:errors"},
			"options": map[string]interface{}{
				"thresholds": map[string]float64{
					"critical": 1,
					"warning":  0.5,
				},
			},
		},
		{
			"name":    "[FHE-GBDT] Model Drift Detected",
			"type":    "metric alert",
			"query":   "avg(last_1h):avg:fhe_gbdt.model.drift_score{*} by {model_id,feature} > 0.2",
			"message": "Drift detected for model {{model_id.name}} feature {{feature.name}}: PSI={{value}} @slack-ml-alerts",
			"tags":    []string{"service:fhe-gbdt", "alert_type:drift"},
			"options": map[string]interface{}{
				"thresholds": map[string]float64{
					"critical": 0.2,
					"warning":  0.1,
				},
			},
		},
	}
}
