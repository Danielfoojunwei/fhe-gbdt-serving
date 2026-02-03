// Prometheus Metrics Exporter
// Exports FHE-GBDT metrics for Prometheus scraping

package integrations

import (
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// PrometheusExporter exports metrics to Prometheus
type PrometheusExporter struct {
	registry *prometheus.Registry

	// Prediction metrics
	predictionTotal      *prometheus.CounterVec
	predictionLatency    *prometheus.HistogramVec
	predictionErrors     *prometheus.CounterVec
	activePredictions    *prometheus.GaugeVec

	// Model metrics
	modelsTotal          *prometheus.GaugeVec
	modelPredictions     *prometheus.CounterVec
	modelLatencyP95      *prometheus.GaugeVec
	modelDriftScore      *prometheus.GaugeVec

	// FHE metrics
	fheEncryptionTime    *prometheus.HistogramVec
	fheDecryptionTime    *prometheus.HistogramVec
	fheEvalTime          *prometheus.HistogramVec
	fheKeySize           *prometheus.GaugeVec

	// Resource metrics
	gpuMemoryUsed        *prometheus.GaugeVec
	gpuUtilization       *prometheus.GaugeVec
	cacheHitRate         *prometheus.GaugeVec

	// Billing metrics
	usagePredictions     *prometheus.GaugeVec
	usageComputeHours    *prometheus.GaugeVec
	usageStorageBytes    *prometheus.GaugeVec

	mu sync.RWMutex
}

// NewPrometheusExporter creates a new Prometheus exporter
func NewPrometheusExporter() *PrometheusExporter {
	registry := prometheus.NewRegistry()

	exporter := &PrometheusExporter{
		registry: registry,

		// Prediction metrics
		predictionTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "fhe_gbdt",
				Name:      "predictions_total",
				Help:      "Total number of predictions",
			},
			[]string{"tenant_id", "model_id", "version_id", "status"},
		),

		predictionLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "fhe_gbdt",
				Name:      "prediction_latency_ms",
				Help:      "Prediction latency in milliseconds",
				Buckets:   []float64{10, 25, 50, 75, 100, 150, 200, 300, 500, 1000},
			},
			[]string{"tenant_id", "model_id", "version_id"},
		),

		predictionErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "fhe_gbdt",
				Name:      "prediction_errors_total",
				Help:      "Total number of prediction errors",
			},
			[]string{"tenant_id", "model_id", "error_type"},
		),

		activePredictions: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "active_predictions",
				Help:      "Number of active predictions in progress",
			},
			[]string{"tenant_id", "model_id"},
		),

		// Model metrics
		modelsTotal: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "models_total",
				Help:      "Total number of models",
			},
			[]string{"tenant_id", "status"},
		),

		modelPredictions: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "fhe_gbdt",
				Name:      "model_predictions_total",
				Help:      "Total predictions per model",
			},
			[]string{"tenant_id", "model_id", "model_name"},
		),

		modelLatencyP95: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "model_latency_p95_ms",
				Help:      "95th percentile latency per model",
			},
			[]string{"tenant_id", "model_id"},
		),

		modelDriftScore: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "model_drift_score",
				Help:      "Drift score (PSI) for model",
			},
			[]string{"tenant_id", "model_id", "feature"},
		),

		// FHE metrics
		fheEncryptionTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "fhe_gbdt",
				Name:      "fhe_encryption_time_ms",
				Help:      "Time to encrypt features",
				Buckets:   []float64{1, 5, 10, 25, 50, 100, 200},
			},
			[]string{"tenant_id", "algorithm"},
		),

		fheDecryptionTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "fhe_gbdt",
				Name:      "fhe_decryption_time_ms",
				Help:      "Time to decrypt results",
				Buckets:   []float64{1, 5, 10, 25, 50, 100, 200},
			},
			[]string{"tenant_id", "algorithm"},
		),

		fheEvalTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "fhe_gbdt",
				Name:      "fhe_evaluation_time_ms",
				Help:      "Time for FHE tree evaluation",
				Buckets:   []float64{10, 25, 50, 100, 200, 500, 1000, 2000},
			},
			[]string{"tenant_id", "model_id", "num_trees"},
		),

		fheKeySize: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "fhe_key_size_bytes",
				Help:      "Size of FHE keys in bytes",
			},
			[]string{"tenant_id", "key_type"},
		),

		// Resource metrics
		gpuMemoryUsed: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "gpu_memory_used_bytes",
				Help:      "GPU memory used in bytes",
			},
			[]string{"device_id", "device_name"},
		),

		gpuUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "gpu_utilization_percent",
				Help:      "GPU utilization percentage",
			},
			[]string{"device_id", "device_name"},
		),

		cacheHitRate: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "cache_hit_rate",
				Help:      "Cache hit rate for evaluation keys",
			},
			[]string{"cache_type"},
		),

		// Billing metrics
		usagePredictions: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "usage_predictions_total",
				Help:      "Total predictions in billing period",
			},
			[]string{"tenant_id"},
		),

		usageComputeHours: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "usage_compute_hours",
				Help:      "Compute hours used in billing period",
			},
			[]string{"tenant_id"},
		),

		usageStorageBytes: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "fhe_gbdt",
				Name:      "usage_storage_bytes",
				Help:      "Storage used in bytes",
			},
			[]string{"tenant_id"},
		),
	}

	// Register all metrics
	registry.MustRegister(
		exporter.predictionTotal,
		exporter.predictionLatency,
		exporter.predictionErrors,
		exporter.activePredictions,
		exporter.modelsTotal,
		exporter.modelPredictions,
		exporter.modelLatencyP95,
		exporter.modelDriftScore,
		exporter.fheEncryptionTime,
		exporter.fheDecryptionTime,
		exporter.fheEvalTime,
		exporter.fheKeySize,
		exporter.gpuMemoryUsed,
		exporter.gpuUtilization,
		exporter.cacheHitRate,
		exporter.usagePredictions,
		exporter.usageComputeHours,
		exporter.usageStorageBytes,
	)

	return exporter
}

// Handler returns the HTTP handler for Prometheus scraping
func (e *PrometheusExporter) Handler() http.Handler {
	return promhttp.HandlerFor(e.registry, promhttp.HandlerOpts{})
}

// ============================================================================
// Metric Recording Methods
// ============================================================================

// RecordPrediction records a prediction event
func (e *PrometheusExporter) RecordPrediction(tenantID, modelID, versionID string, latencyMs float64, success bool) {
	status := "success"
	if !success {
		status = "error"
	}

	e.predictionTotal.WithLabelValues(tenantID, modelID, versionID, status).Inc()
	e.predictionLatency.WithLabelValues(tenantID, modelID, versionID).Observe(latencyMs)
	e.modelPredictions.WithLabelValues(tenantID, modelID, "").Inc()
}

// RecordPredictionError records a prediction error
func (e *PrometheusExporter) RecordPredictionError(tenantID, modelID, errorType string) {
	e.predictionErrors.WithLabelValues(tenantID, modelID, errorType).Inc()
}

// SetActivePredictions sets the number of active predictions
func (e *PrometheusExporter) SetActivePredictions(tenantID, modelID string, count float64) {
	e.activePredictions.WithLabelValues(tenantID, modelID).Set(count)
}

// SetModelCount sets the total model count
func (e *PrometheusExporter) SetModelCount(tenantID, status string, count float64) {
	e.modelsTotal.WithLabelValues(tenantID, status).Set(count)
}

// SetModelLatencyP95 sets the p95 latency for a model
func (e *PrometheusExporter) SetModelLatencyP95(tenantID, modelID string, latencyMs float64) {
	e.modelLatencyP95.WithLabelValues(tenantID, modelID).Set(latencyMs)
}

// SetModelDriftScore sets the drift score for a model feature
func (e *PrometheusExporter) SetModelDriftScore(tenantID, modelID, feature string, score float64) {
	e.modelDriftScore.WithLabelValues(tenantID, modelID, feature).Set(score)
}

// RecordFHEEncryption records FHE encryption time
func (e *PrometheusExporter) RecordFHEEncryption(tenantID, algorithm string, timeMs float64) {
	e.fheEncryptionTime.WithLabelValues(tenantID, algorithm).Observe(timeMs)
}

// RecordFHEDecryption records FHE decryption time
func (e *PrometheusExporter) RecordFHEDecryption(tenantID, algorithm string, timeMs float64) {
	e.fheDecryptionTime.WithLabelValues(tenantID, algorithm).Observe(timeMs)
}

// RecordFHEEvaluation records FHE evaluation time
func (e *PrometheusExporter) RecordFHEEvaluation(tenantID, modelID string, numTrees int, timeMs float64) {
	e.fheEvalTime.WithLabelValues(tenantID, modelID, string(rune(numTrees))).Observe(timeMs)
}

// SetFHEKeySize sets the FHE key size
func (e *PrometheusExporter) SetFHEKeySize(tenantID, keyType string, sizeBytes float64) {
	e.fheKeySize.WithLabelValues(tenantID, keyType).Set(sizeBytes)
}

// SetGPUMetrics sets GPU metrics
func (e *PrometheusExporter) SetGPUMetrics(deviceID, deviceName string, memoryUsed, utilization float64) {
	e.gpuMemoryUsed.WithLabelValues(deviceID, deviceName).Set(memoryUsed)
	e.gpuUtilization.WithLabelValues(deviceID, deviceName).Set(utilization)
}

// SetCacheHitRate sets the cache hit rate
func (e *PrometheusExporter) SetCacheHitRate(cacheType string, rate float64) {
	e.cacheHitRate.WithLabelValues(cacheType).Set(rate)
}

// SetUsageMetrics sets billing usage metrics
func (e *PrometheusExporter) SetUsageMetrics(tenantID string, predictions, computeHours, storageBytes float64) {
	e.usagePredictions.WithLabelValues(tenantID).Set(predictions)
	e.usageComputeHours.WithLabelValues(tenantID).Set(computeHours)
	e.usageStorageBytes.WithLabelValues(tenantID).Set(storageBytes)
}

// ============================================================================
// Grafana Dashboard Template
// ============================================================================

// GetGrafanaDashboard returns a Grafana dashboard JSON template
func GetGrafanaDashboard() string {
	return `{
  "dashboard": {
    "title": "FHE-GBDT Monitoring",
    "tags": ["fhe", "gbdt", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Predictions/sec",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "rate(fhe_gbdt_predictions_total[1m])",
            "legendFormat": "{{model_id}}"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(fhe_gbdt_prediction_latency_ms_bucket[5m]))",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
        "targets": [
          {
            "expr": "rate(fhe_gbdt_prediction_errors_total[5m]) / rate(fhe_gbdt_predictions_total[5m]) * 100",
            "legendFormat": "Error %"
          }
        ]
      },
      {
        "title": "Active Predictions",
        "type": "stat",
        "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(fhe_gbdt_active_predictions)",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Prediction Latency Distribution",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
          {
            "expr": "rate(fhe_gbdt_prediction_latency_ms_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "title": "Model Drift Scores",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
        "targets": [
          {
            "expr": "fhe_gbdt_model_drift_score",
            "legendFormat": "{{model_id}} - {{feature}}"
          }
        ]
      },
      {
        "title": "FHE Evaluation Time",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(fhe_gbdt_fhe_evaluation_time_ms_bucket[5m]))",
            "legendFormat": "p95 {{model_id}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        "targets": [
          {
            "expr": "fhe_gbdt_gpu_utilization_percent",
            "legendFormat": "{{device_name}}"
          }
        ]
      }
    ]
  }
}`
}

// ============================================================================
// Alert Rules Template
// ============================================================================

// GetPrometheusAlertRules returns Prometheus alert rules
func GetPrometheusAlertRules() string {
	return `groups:
- name: fhe_gbdt_alerts
  rules:
  - alert: HighPredictionLatency
    expr: histogram_quantile(0.95, rate(fhe_gbdt_prediction_latency_ms_bucket[5m])) > 200
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High prediction latency detected"
      description: "P95 latency is {{ $value }}ms (threshold: 200ms)"

  - alert: HighErrorRate
    expr: rate(fhe_gbdt_prediction_errors_total[5m]) / rate(fhe_gbdt_predictions_total[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"

  - alert: ModelDriftDetected
    expr: fhe_gbdt_model_drift_score > 0.2
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Model drift detected"
      description: "Model {{ $labels.model_id }} has drift score {{ $value }} for feature {{ $labels.feature }}"

  - alert: GPUMemoryHigh
    expr: fhe_gbdt_gpu_memory_used_bytes / 1024 / 1024 / 1024 > 70
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High GPU memory usage"
      description: "GPU {{ $labels.device_id }} is using {{ $value }}GB memory"

  - alert: NoPredictions
    expr: rate(fhe_gbdt_predictions_total[10m]) == 0
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "No predictions received"
      description: "No predictions have been received in the last 10 minutes"
`
}

// MetricsServer starts the Prometheus metrics server
type MetricsServer struct {
	exporter *PrometheusExporter
	server   *http.Server
}

// NewMetricsServer creates a new metrics server
func NewMetricsServer(addr string) *MetricsServer {
	exporter := NewPrometheusExporter()

	mux := http.NewServeMux()
	mux.Handle("/metrics", exporter.Handler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	return &MetricsServer{
		exporter: exporter,
		server: &http.Server{
			Addr:         addr,
			Handler:      mux,
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 10 * time.Second,
		},
	}
}

// Start starts the metrics server
func (s *MetricsServer) Start() error {
	return s.server.ListenAndServe()
}

// Shutdown gracefully shuts down the metrics server
func (s *MetricsServer) Shutdown() error {
	return s.server.Shutdown(nil)
}

// Exporter returns the Prometheus exporter
func (s *MetricsServer) Exporter() *PrometheusExporter {
	return s.exporter
}
