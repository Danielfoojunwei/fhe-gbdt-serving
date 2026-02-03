// Model Performance Monitoring Service
// Tracks model performance, drift detection, and alerts

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/monitoring"
)

type monitoringServer struct {
	pb.UnimplementedMonitoringServiceServer
	db *sql.DB
}

func newMonitoringServer() (*monitoringServer, error) {
	dbURL := os.Getenv("TIMESCALE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/monitoring?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	return &monitoringServer{db: db}, nil
}

// ============================================================================
// Prediction Logging
// ============================================================================

func (s *monitoringServer) LogPrediction(ctx context.Context, req *pb.LogPredictionRequest) (*pb.LogPredictionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	predictionID := uuid.New().String()
	now := time.Now()

	// Store prediction log (encrypted outputs only - no plaintext)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO prediction_logs (
			id, tenant_id, model_id, version_id, request_id,
			latency_ms, input_size_bytes, output_size_bytes,
			feature_stats, timestamp
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`, predictionID, req.TenantId, req.ModelId, req.VersionId, req.RequestId,
		req.LatencyMs, req.InputSizeBytes, req.OutputSizeBytes,
		marshalJSON(req.FeatureStats), now)

	if err != nil {
		log.Printf("ERROR: Failed to log prediction: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to log prediction: %v", err)
	}

	return &pb.LogPredictionResponse{
		PredictionId: predictionID,
		Timestamp:    timestamppb.New(now),
	}, nil
}

// ============================================================================
// Performance Metrics
// ============================================================================

func (s *monitoringServer) GetModelMetrics(ctx context.Context, req *pb.GetModelMetricsRequest) (*pb.GetModelMetricsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	startTime := time.Now().Add(-24 * time.Hour)
	endTime := time.Now()
	if req.StartTime != nil {
		startTime = req.StartTime.AsTime()
	}
	if req.EndTime != nil {
		endTime = req.EndTime.AsTime()
	}

	var metrics pb.ModelMetrics

	// Get latency metrics
	err := s.db.QueryRowContext(ctx, `
		SELECT
			COUNT(*) as total_predictions,
			AVG(latency_ms) as avg_latency,
			percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
			percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
			percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
			MIN(latency_ms) as min_latency,
			MAX(latency_ms) as max_latency,
			STDDEV(latency_ms) as stddev_latency
		FROM prediction_logs
		WHERE model_id = $1 AND timestamp >= $2 AND timestamp < $3
	`, req.ModelId, startTime, endTime).Scan(
		&metrics.TotalPredictions,
		&metrics.AvgLatencyMs,
		&metrics.P50LatencyMs,
		&metrics.P95LatencyMs,
		&metrics.P99LatencyMs,
		&metrics.MinLatencyMs,
		&metrics.MaxLatencyMs,
		&metrics.StddevLatencyMs,
	)
	if err != nil {
		log.Printf("WARN: Failed to get latency metrics: %v", err)
	}

	// Get throughput
	err = s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) / EXTRACT(EPOCH FROM ($2::timestamp - $1::timestamp)) * 60 as rpm
		FROM prediction_logs
		WHERE model_id = $3 AND timestamp >= $1 AND timestamp < $2
	`, startTime, endTime, req.ModelId).Scan(&metrics.RequestsPerMinute)
	if err != nil {
		log.Printf("WARN: Failed to get throughput: %v", err)
	}

	// Get error rate
	err = s.db.QueryRowContext(ctx, `
		SELECT
			COUNT(*) FILTER (WHERE error = true) * 100.0 / NULLIF(COUNT(*), 0) as error_rate
		FROM prediction_logs
		WHERE model_id = $1 AND timestamp >= $2 AND timestamp < $3
	`, req.ModelId, startTime, endTime).Scan(&metrics.ErrorRatePercent)
	if err != nil {
		log.Printf("WARN: Failed to get error rate: %v", err)
	}

	metrics.ModelId = req.ModelId
	metrics.PeriodStart = timestamppb.New(startTime)
	metrics.PeriodEnd = timestamppb.New(endTime)

	return &pb.GetModelMetricsResponse{Metrics: &metrics}, nil
}

func (s *monitoringServer) GetMetricsTimeSeries(ctx context.Context, req *pb.GetMetricsTimeSeriesRequest) (*pb.GetMetricsTimeSeriesResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	startTime := time.Now().Add(-24 * time.Hour)
	endTime := time.Now()
	if req.StartTime != nil {
		startTime = req.StartTime.AsTime()
	}
	if req.EndTime != nil {
		endTime = req.EndTime.AsTime()
	}

	granularity := "1 hour"
	switch req.Granularity {
	case "minute":
		granularity = "1 minute"
	case "hour":
		granularity = "1 hour"
	case "day":
		granularity = "1 day"
	}

	query := fmt.Sprintf(`
		SELECT
			time_bucket('%s', timestamp) as bucket,
			COUNT(*) as count,
			AVG(latency_ms) as avg_latency,
			percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
			COUNT(*) FILTER (WHERE error = true) as error_count
		FROM prediction_logs
		WHERE model_id = $1 AND timestamp >= $2 AND timestamp < $3
		GROUP BY bucket
		ORDER BY bucket ASC
	`, granularity)

	rows, err := s.db.QueryContext(ctx, query, req.ModelId, startTime, endTime)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get time series: %v", err)
	}
	defer rows.Close()

	var dataPoints []*pb.MetricsDataPoint
	for rows.Next() {
		var dp pb.MetricsDataPoint
		var bucket time.Time

		if err := rows.Scan(&bucket, &dp.Count, &dp.AvgLatencyMs, &dp.P95LatencyMs, &dp.ErrorCount); err != nil {
			continue
		}
		dp.Timestamp = timestamppb.New(bucket)
		dataPoints = append(dataPoints, &dp)
	}

	return &pb.GetMetricsTimeSeriesResponse{
		DataPoints:  dataPoints,
		Granularity: req.Granularity,
	}, nil
}

// ============================================================================
// Drift Detection
// ============================================================================

func (s *monitoringServer) DetectDrift(ctx context.Context, req *pb.DetectDriftRequest) (*pb.DetectDriftResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get baseline statistics (from training period)
	baselineStats, err := s.getBaselineStats(ctx, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get baseline stats: %v", err)
	}

	// Get current statistics
	currentStats, err := s.getCurrentStats(ctx, req.ModelId, req.WindowHours)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get current stats: %v", err)
	}

	// Calculate drift metrics
	var driftResults []*pb.DriftResult

	for feature, baseline := range baselineStats {
		current, ok := currentStats[feature]
		if !ok {
			continue
		}

		// Calculate PSI (Population Stability Index)
		psi := calculatePSI(baseline, current)

		// Calculate KS statistic
		ks := calculateKSStatistic(baseline, current)

		driftDetected := psi > 0.2 || ks > 0.1 // Thresholds

		driftResults = append(driftResults, &pb.DriftResult{
			FeatureName:    feature,
			Psi:            psi,
			KsStatistic:    ks,
			DriftDetected:  driftDetected,
			BaselineMean:   baseline.Mean,
			CurrentMean:    current.Mean,
			BaselineStddev: baseline.Stddev,
			CurrentStddev:  current.Stddev,
		})
	}

	// Overall drift detection
	overallDrift := false
	for _, r := range driftResults {
		if r.DriftDetected {
			overallDrift = true
			break
		}
	}

	return &pb.DetectDriftResponse{
		ModelId:       req.ModelId,
		DriftDetected: overallDrift,
		Results:       driftResults,
		CheckedAt:     timestamppb.Now(),
	}, nil
}

type featureStats struct {
	Mean      float64
	Stddev    float64
	Histogram []float64
}

func (s *monitoringServer) getBaselineStats(ctx context.Context, modelID string) (map[string]featureStats, error) {
	var statsJSON []byte
	err := s.db.QueryRowContext(ctx, `
		SELECT baseline_stats FROM model_baselines WHERE model_id = $1
	`, modelID).Scan(&statsJSON)
	if err != nil {
		return nil, err
	}

	var stats map[string]featureStats
	if err := json.Unmarshal(statsJSON, &stats); err != nil {
		return nil, err
	}

	return stats, nil
}

func (s *monitoringServer) getCurrentStats(ctx context.Context, modelID string, windowHours int32) (map[string]featureStats, error) {
	if windowHours == 0 {
		windowHours = 24
	}

	// Aggregate feature statistics from prediction logs
	rows, err := s.db.QueryContext(ctx, `
		SELECT feature_name, AVG(value), STDDEV(value)
		FROM prediction_log_features
		WHERE model_id = $1 AND timestamp >= NOW() - INTERVAL '1 hour' * $2
		GROUP BY feature_name
	`, modelID, windowHours)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	stats := make(map[string]featureStats)
	for rows.Next() {
		var name string
		var s featureStats
		if err := rows.Scan(&name, &s.Mean, &s.Stddev); err != nil {
			continue
		}
		stats[name] = s
	}

	return stats, nil
}

func calculatePSI(baseline, current featureStats) float64 {
	// Simplified PSI calculation
	// In production, use proper histogram binning
	if baseline.Stddev == 0 || current.Stddev == 0 {
		return 0
	}

	meanShift := math.Abs(baseline.Mean-current.Mean) / baseline.Stddev
	stddevRatio := current.Stddev / baseline.Stddev

	// Approximate PSI
	psi := 0.1*meanShift + 0.1*math.Abs(1-stddevRatio)
	return psi
}

func calculateKSStatistic(baseline, current featureStats) float64 {
	// Simplified KS statistic
	// In production, use proper CDF comparison
	if baseline.Stddev == 0 {
		return 0
	}

	// Approximate KS based on distribution shift
	shift := math.Abs(baseline.Mean-current.Mean) / baseline.Stddev
	ks := 1 - math.Exp(-shift*shift/2)
	return ks
}

// ============================================================================
// Alerting
// ============================================================================

func (s *monitoringServer) CreateAlert(ctx context.Context, req *pb.CreateAlertRequest) (*pb.CreateAlertResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	alertID := uuid.New().String()
	now := time.Now()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO alerts (id, tenant_id, model_id, name, metric, condition, threshold,
		                   window_minutes, notification_channels, enabled, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, true, $10)
	`, alertID, req.TenantId, req.ModelId, req.Name, req.Metric, req.Condition,
		req.Threshold, req.WindowMinutes, marshalJSON(req.NotificationChannels), now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create alert: %v", err)
	}

	return &pb.CreateAlertResponse{
		Alert: &pb.Alert{
			Id:                   alertID,
			TenantId:             req.TenantId,
			ModelId:              req.ModelId,
			Name:                 req.Name,
			Metric:               req.Metric,
			Condition:            req.Condition,
			Threshold:            req.Threshold,
			WindowMinutes:        req.WindowMinutes,
			NotificationChannels: req.NotificationChannels,
			Enabled:              true,
			CreatedAt:            timestamppb.New(now),
		},
	}, nil
}

func (s *monitoringServer) ListAlerts(ctx context.Context, req *pb.ListAlertsRequest) (*pb.ListAlertsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	query := `
		SELECT id, tenant_id, model_id, name, metric, condition, threshold,
		       window_minutes, notification_channels, enabled, created_at, last_triggered_at
		FROM alerts
		WHERE tenant_id = $1
	`
	args := []interface{}{req.TenantId}

	if req.ModelId != "" {
		query += " AND model_id = $2"
		args = append(args, req.ModelId)
	}

	query += " ORDER BY created_at DESC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list alerts: %v", err)
	}
	defer rows.Close()

	var alerts []*pb.Alert
	for rows.Next() {
		var a pb.Alert
		var createdAt time.Time
		var lastTriggeredAt sql.NullTime
		var channelsJSON []byte

		if err := rows.Scan(&a.Id, &a.TenantId, &a.ModelId, &a.Name, &a.Metric,
			&a.Condition, &a.Threshold, &a.WindowMinutes, &channelsJSON,
			&a.Enabled, &createdAt, &lastTriggeredAt); err != nil {
			continue
		}

		a.CreatedAt = timestamppb.New(createdAt)
		if lastTriggeredAt.Valid {
			a.LastTriggeredAt = timestamppb.New(lastTriggeredAt.Time)
		}
		json.Unmarshal(channelsJSON, &a.NotificationChannels)
		alerts = append(alerts, &a)
	}

	return &pb.ListAlertsResponse{Alerts: alerts}, nil
}

func (s *monitoringServer) EvaluateAlerts(ctx context.Context, req *pb.EvaluateAlertsRequest) (*pb.EvaluateAlertsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get all enabled alerts
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, model_id, metric, condition, threshold, window_minutes
		FROM alerts
		WHERE tenant_id = $1 AND enabled = true
	`, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get alerts: %v", err)
	}
	defer rows.Close()

	var triggeredAlerts []*pb.TriggeredAlert
	now := time.Now()

	for rows.Next() {
		var alertID, modelID, metric, condition string
		var threshold float64
		var windowMinutes int32

		if err := rows.Scan(&alertID, &modelID, &metric, &condition, &threshold, &windowMinutes); err != nil {
			continue
		}

		// Get current metric value
		currentValue, err := s.getMetricValue(ctx, modelID, metric, windowMinutes)
		if err != nil {
			continue
		}

		// Check if alert should trigger
		shouldTrigger := evaluateCondition(currentValue, condition, threshold)
		if shouldTrigger {
			// Record triggered alert
			incidentID := uuid.New().String()
			_, _ = s.db.ExecContext(ctx, `
				INSERT INTO alert_incidents (id, alert_id, current_value, threshold, triggered_at)
				VALUES ($1, $2, $3, $4, $5)
			`, incidentID, alertID, currentValue, threshold, now)

			// Update last triggered time
			_, _ = s.db.ExecContext(ctx, `
				UPDATE alerts SET last_triggered_at = $1 WHERE id = $2
			`, now, alertID)

			triggeredAlerts = append(triggeredAlerts, &pb.TriggeredAlert{
				AlertId:      alertID,
				ModelId:      modelID,
				Metric:       metric,
				CurrentValue: currentValue,
				Threshold:    threshold,
				TriggeredAt:  timestamppb.New(now),
			})
		}
	}

	return &pb.EvaluateAlertsResponse{
		TriggeredAlerts: triggeredAlerts,
		EvaluatedAt:     timestamppb.New(now),
	}, nil
}

func (s *monitoringServer) getMetricValue(ctx context.Context, modelID, metric string, windowMinutes int32) (float64, error) {
	var value float64
	var query string

	switch metric {
	case "latency_p95":
		query = `
			SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
			FROM prediction_logs
			WHERE model_id = $1 AND timestamp >= NOW() - INTERVAL '1 minute' * $2
		`
	case "latency_avg":
		query = `
			SELECT AVG(latency_ms)
			FROM prediction_logs
			WHERE model_id = $1 AND timestamp >= NOW() - INTERVAL '1 minute' * $2
		`
	case "error_rate":
		query = `
			SELECT COUNT(*) FILTER (WHERE error = true) * 100.0 / NULLIF(COUNT(*), 0)
			FROM prediction_logs
			WHERE model_id = $1 AND timestamp >= NOW() - INTERVAL '1 minute' * $2
		`
	case "throughput":
		query = `
			SELECT COUNT(*) * 60.0 / $2
			FROM prediction_logs
			WHERE model_id = $1 AND timestamp >= NOW() - INTERVAL '1 minute' * $2
		`
	default:
		return 0, fmt.Errorf("unknown metric: %s", metric)
	}

	err := s.db.QueryRowContext(ctx, query, modelID, windowMinutes).Scan(&value)
	return value, err
}

func evaluateCondition(value float64, condition string, threshold float64) bool {
	switch condition {
	case "gt", ">":
		return value > threshold
	case "gte", ">=":
		return value >= threshold
	case "lt", "<":
		return value < threshold
	case "lte", "<=":
		return value <= threshold
	case "eq", "==":
		return value == threshold
	default:
		return false
	}
}

// ============================================================================
// Helpers
// ============================================================================

func marshalJSON(v interface{}) []byte {
	if v == nil {
		return []byte("{}")
	}
	b, _ := json.Marshal(v)
	return b
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8088"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newMonitoringServer()
	if err != nil {
		log.Fatalf("failed to create monitoring server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterMonitoringServiceServer(s, server)

	log.Printf("Monitoring Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
