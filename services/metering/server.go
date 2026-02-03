package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/metering"
)

type meteringServer struct {
	pb.UnimplementedMeteringServiceServer
	db *sql.DB
}

func newMeteringServer() (*meteringServer, error) {
	// Connect to TimescaleDB
	dbURL := os.Getenv("TIMESCALE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/metering?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	} else {
		log.Printf("Connected to TimescaleDB metering database")
	}

	return &meteringServer{db: db}, nil
}

// RecordUsage records a usage event for billing and analytics
func (s *meteringServer) RecordUsage(ctx context.Context, req *pb.RecordUsageRequest) (*pb.RecordUsageResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	event := req.Event
	if event == nil {
		return nil, status.Error(codes.InvalidArgument, "event is required")
	}

	// Generate event ID if not provided
	eventID := event.EventId
	if eventID == "" {
		eventID = uuid.New().String()
	}

	// Set timestamp if not provided
	timestamp := time.Now()
	if event.Timestamp != nil {
		timestamp = event.Timestamp.AsTime()
	}

	// Insert usage event
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO usage_events (event_id, tenant_id, event_type, model_id, timestamp,
		                         predictions_count, compute_ms, input_bytes, output_bytes, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`, eventID, event.TenantId, event.EventType.String(), event.ModelId, timestamp,
		event.PredictionsCount, event.ComputeMs, event.InputBytes, event.OutputBytes,
		mapToJSON(event.Metadata))
	if err != nil {
		log.Printf("ERROR: Failed to record usage event: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to record usage: %v", err)
	}

	log.Printf("AUDIT: Recorded usage event %s for tenant %s: %d predictions",
		eventID, event.TenantId, event.PredictionsCount)

	// Check quota status and return warning if needed
	quotaWarning := s.checkQuotaWarning(ctx, event.TenantId)

	return &pb.RecordUsageResponse{
		Success:      true,
		EventId:      eventID,
		QuotaWarning: quotaWarning,
	}, nil
}

// GetUsageSummary returns aggregated usage statistics for a tenant
func (s *meteringServer) GetUsageSummary(ctx context.Context, req *pb.GetUsageSummaryRequest) (*pb.GetUsageSummaryResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	startTime := time.Now().AddDate(0, -1, 0) // Default: last 30 days
	endTime := time.Now()
	if req.StartTime != nil {
		startTime = req.StartTime.AsTime()
	}
	if req.EndTime != nil {
		endTime = req.EndTime.AsTime()
	}

	// Get overall summary
	summary, err := s.getUsageSummaryFromDB(ctx, req.TenantId, "", "", startTime, endTime)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get usage summary: %v", err)
	}

	// Get breakdown by model
	byModel, err := s.getUsageSummariesByModel(ctx, req.TenantId, startTime, endTime)
	if err != nil {
		log.Printf("WARN: Failed to get model breakdown: %v", err)
	}

	// Get breakdown by event type
	byEventType, err := s.getUsageSummariesByEventType(ctx, req.TenantId, startTime, endTime)
	if err != nil {
		log.Printf("WARN: Failed to get event type breakdown: %v", err)
	}

	return &pb.GetUsageSummaryResponse{
		Summary:     summary,
		ByModel:     byModel,
		ByEventType: byEventType,
	}, nil
}

func (s *meteringServer) getUsageSummaryFromDB(ctx context.Context, tenantID, modelID, eventType string, startTime, endTime time.Time) (*pb.UsageSummary, error) {
	query := `
		SELECT
			COUNT(*) as total_events,
			COALESCE(SUM(predictions_count), 0) as total_predictions,
			COALESCE(SUM(compute_ms), 0) as total_compute_ms,
			COALESCE(AVG(compute_ms), 0) as avg_latency_ms,
			COALESCE(percentile_cont(0.50) WITHIN GROUP (ORDER BY compute_ms), 0) as p50_latency_ms,
			COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY compute_ms), 0) as p95_latency_ms,
			COALESCE(percentile_cont(0.99) WITHIN GROUP (ORDER BY compute_ms), 0) as p99_latency_ms,
			COALESCE(SUM(input_bytes), 0) as total_input_bytes,
			COALESCE(SUM(output_bytes), 0) as total_output_bytes
		FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
	`
	args := []interface{}{tenantID, startTime, endTime}

	if modelID != "" {
		query += " AND model_id = $4"
		args = append(args, modelID)
	}
	if eventType != "" {
		query += fmt.Sprintf(" AND event_type = $%d", len(args)+1)
		args = append(args, eventType)
	}

	var summary pb.UsageSummary
	err := s.db.QueryRowContext(ctx, query, args...).Scan(
		&summary.TotalEvents, &summary.TotalPredictions, &summary.TotalComputeMs,
		&summary.AvgLatencyMs, &summary.P50LatencyMs, &summary.P95LatencyMs,
		&summary.P99LatencyMs, &summary.TotalInputBytes, &summary.TotalOutputBytes,
	)
	if err != nil {
		return nil, err
	}

	summary.TenantId = tenantID
	summary.ModelId = modelID
	summary.PeriodStart = timestamppb.New(startTime)
	summary.PeriodEnd = timestamppb.New(endTime)

	return &summary, nil
}

func (s *meteringServer) getUsageSummariesByModel(ctx context.Context, tenantID string, startTime, endTime time.Time) ([]*pb.UsageSummary, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT model_id, COUNT(*), COALESCE(SUM(predictions_count), 0),
		       COALESCE(SUM(compute_ms), 0), COALESCE(AVG(compute_ms), 0)
		FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3 AND model_id IS NOT NULL
		GROUP BY model_id
		ORDER BY SUM(predictions_count) DESC
		LIMIT 20
	`, tenantID, startTime, endTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var summaries []*pb.UsageSummary
	for rows.Next() {
		var summary pb.UsageSummary
		summary.TenantId = tenantID
		summary.PeriodStart = timestamppb.New(startTime)
		summary.PeriodEnd = timestamppb.New(endTime)

		if err := rows.Scan(&summary.ModelId, &summary.TotalEvents,
			&summary.TotalPredictions, &summary.TotalComputeMs, &summary.AvgLatencyMs); err != nil {
			continue
		}
		summaries = append(summaries, &summary)
	}
	return summaries, nil
}

func (s *meteringServer) getUsageSummariesByEventType(ctx context.Context, tenantID string, startTime, endTime time.Time) ([]*pb.UsageSummary, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT event_type, COUNT(*), COALESCE(SUM(predictions_count), 0),
		       COALESCE(SUM(compute_ms), 0), COALESCE(AVG(compute_ms), 0)
		FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
		GROUP BY event_type
		ORDER BY COUNT(*) DESC
	`, tenantID, startTime, endTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var summaries []*pb.UsageSummary
	for rows.Next() {
		var summary pb.UsageSummary
		var eventType string
		summary.TenantId = tenantID
		summary.PeriodStart = timestamppb.New(startTime)
		summary.PeriodEnd = timestamppb.New(endTime)

		if err := rows.Scan(&eventType, &summary.TotalEvents,
			&summary.TotalPredictions, &summary.TotalComputeMs, &summary.AvgLatencyMs); err != nil {
			continue
		}
		summary.EventType = pb.EventType(pb.EventType_value[eventType])
		summaries = append(summaries, &summary)
	}
	return summaries, nil
}

// GetUsageTimeSeries returns time-series usage data for visualization
func (s *meteringServer) GetUsageTimeSeries(ctx context.Context, req *pb.GetUsageTimeSeriesRequest) (*pb.GetUsageTimeSeriesResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	startTime := time.Now().AddDate(0, 0, -7) // Default: last 7 days
	endTime := time.Now()
	if req.StartTime != nil {
		startTime = req.StartTime.AsTime()
	}
	if req.EndTime != nil {
		endTime = req.EndTime.AsTime()
	}

	granularity := req.Granularity
	if granularity == "" {
		granularity = "hour"
	}

	// Map granularity to TimescaleDB time_bucket interval
	var interval string
	switch granularity {
	case "minute":
		interval = "1 minute"
	case "hour":
		interval = "1 hour"
	case "day":
		interval = "1 day"
	case "week":
		interval = "1 week"
	case "month":
		interval = "1 month"
	default:
		interval = "1 hour"
	}

	query := fmt.Sprintf(`
		SELECT
			time_bucket('%s', timestamp) as bucket,
			COUNT(*) as event_count,
			COALESCE(SUM(predictions_count), 0) as predictions_count,
			COALESCE(SUM(compute_ms), 0) as compute_ms,
			COALESCE(SUM(input_bytes + output_bytes), 0) as data_bytes
		FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
	`, interval)

	args := []interface{}{req.TenantId, startTime, endTime}
	argIdx := 4

	if req.ModelId != "" {
		query += fmt.Sprintf(" AND model_id = $%d", argIdx)
		args = append(args, req.ModelId)
		argIdx++
	}
	if req.EventType != pb.EventType_EVENT_TYPE_UNSPECIFIED {
		query += fmt.Sprintf(" AND event_type = $%d", argIdx)
		args = append(args, req.EventType.String())
	}

	query += " GROUP BY bucket ORDER BY bucket ASC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get time series: %v", err)
	}
	defer rows.Close()

	var dataPoints []*pb.TimeSeriesDataPoint
	for rows.Next() {
		var dp pb.TimeSeriesDataPoint
		var bucket time.Time

		if err := rows.Scan(&bucket, &dp.EventCount, &dp.PredictionsCount,
			&dp.ComputeMs, &dp.DataBytes); err != nil {
			continue
		}
		dp.Timestamp = timestamppb.New(bucket)
		dataPoints = append(dataPoints, &dp)
	}

	return &pb.GetUsageTimeSeriesResponse{
		DataPoints:  dataPoints,
		Granularity: granularity,
	}, nil
}

// GetQuotaStatus returns current quota status for a tenant
func (s *meteringServer) GetQuotaStatus(ctx context.Context, req *pb.GetQuotaStatusRequest) (*pb.GetQuotaStatusResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	query := `
		SELECT quota_type, limit_value, warning_threshold_percent, period_start, period_end
		FROM quotas
		WHERE tenant_id = $1
	`
	args := []interface{}{req.TenantId}

	if req.QuotaType != pb.QuotaType_QUOTA_TYPE_UNSPECIFIED {
		query += " AND quota_type = $2"
		args = append(args, req.QuotaType.String())
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get quota status: %v", err)
	}
	defer rows.Close()

	var quotas []*pb.QuotaStatus
	for rows.Next() {
		var qs pb.QuotaStatus
		var quotaType string
		var periodStart, periodEnd time.Time

		if err := rows.Scan(&quotaType, &qs.Limit, &qs.WarningThresholdPercent,
			&periodStart, &periodEnd); err != nil {
			continue
		}

		qs.TenantId = req.TenantId
		qs.QuotaType = pb.QuotaType(pb.QuotaType_value[quotaType])
		qs.ResetAt = timestamppb.New(periodEnd)

		// Get current usage
		qs.Used = s.getCurrentUsageForQuota(ctx, req.TenantId, quotaType, periodStart, periodEnd)
		if qs.Limit > 0 {
			qs.Remaining = qs.Limit - qs.Used
			if qs.Remaining < 0 {
				qs.Remaining = 0
			}
			qs.PercentageUsed = int32(float64(qs.Used) / float64(qs.Limit) * 100)
			qs.WarningTriggered = qs.PercentageUsed >= qs.WarningThresholdPercent
			qs.Exceeded = qs.Used >= qs.Limit
		}

		quotas = append(quotas, &qs)
	}

	return &pb.GetQuotaStatusResponse{Quotas: quotas}, nil
}

func (s *meteringServer) getCurrentUsageForQuota(ctx context.Context, tenantID, quotaType string, periodStart, periodEnd time.Time) int64 {
	var usage int64

	switch quotaType {
	case "QUOTA_TYPE_PREDICTIONS_PER_MONTH":
		s.db.QueryRowContext(ctx, `
			SELECT COALESCE(SUM(predictions_count), 0)
			FROM usage_events
			WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3 AND event_type = 'EVENT_TYPE_PREDICTION'
		`, tenantID, periodStart, periodEnd).Scan(&usage)

	case "QUOTA_TYPE_COMPUTE_MS_PER_MONTH":
		s.db.QueryRowContext(ctx, `
			SELECT COALESCE(SUM(compute_ms), 0)
			FROM usage_events
			WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
		`, tenantID, periodStart, periodEnd).Scan(&usage)

	case "QUOTA_TYPE_DATA_BYTES_PER_MONTH":
		s.db.QueryRowContext(ctx, `
			SELECT COALESCE(SUM(input_bytes + output_bytes), 0)
			FROM usage_events
			WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
		`, tenantID, periodStart, periodEnd).Scan(&usage)

	case "QUOTA_TYPE_MODELS_TOTAL":
		s.db.QueryRowContext(ctx, `
			SELECT COUNT(DISTINCT model_id)
			FROM usage_events
			WHERE tenant_id = $1 AND model_id IS NOT NULL
		`, tenantID).Scan(&usage)
	}

	return usage
}

// SetQuota creates or updates quota limits for a tenant
func (s *meteringServer) SetQuota(ctx context.Context, req *pb.SetQuotaRequest) (*pb.SetQuotaResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Setting quota %s for tenant %s: limit=%d",
		req.QuotaType.String(), req.TenantId, req.Limit)

	warningThreshold := req.WarningThresholdPercent
	if warningThreshold == 0 {
		warningThreshold = 80 // Default 80%
	}

	periodStart := time.Now().Truncate(24 * time.Hour).AddDate(0, 0, -time.Now().Day()+1) // Start of month
	periodEnd := periodStart.AddDate(0, 1, 0)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO quotas (tenant_id, quota_type, limit_value, warning_threshold_percent,
		                   period_start, period_end, webhook_url)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (tenant_id, quota_type) DO UPDATE SET
			limit_value = EXCLUDED.limit_value,
			warning_threshold_percent = EXCLUDED.warning_threshold_percent,
			webhook_url = EXCLUDED.webhook_url,
			updated_at = NOW()
	`, req.TenantId, req.QuotaType.String(), req.Limit, warningThreshold,
		periodStart, periodEnd, nullString(req.WebhookUrl))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to set quota: %v", err)
	}

	// Get updated quota status
	statusResp, err := s.GetQuotaStatus(ctx, &pb.GetQuotaStatusRequest{
		TenantId:  req.TenantId,
		QuotaType: req.QuotaType,
	})
	if err != nil {
		return nil, err
	}

	var quota *pb.QuotaStatus
	if len(statusResp.Quotas) > 0 {
		quota = statusResp.Quotas[0]
	}

	return &pb.SetQuotaResponse{
		Success: true,
		Quota:   quota,
	}, nil
}

// ListUsageEvents returns detailed usage events for audit/debugging
func (s *meteringServer) ListUsageEvents(ctx context.Context, req *pb.ListUsageEventsRequest) (*pb.ListUsageEventsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	startTime := time.Now().AddDate(0, 0, -1) // Default: last 24 hours
	endTime := time.Now()
	if req.StartTime != nil {
		startTime = req.StartTime.AsTime()
	}
	if req.EndTime != nil {
		endTime = req.EndTime.AsTime()
	}

	pageSize := req.PageSize
	if pageSize <= 0 || pageSize > 1000 {
		pageSize = 100
	}

	query := `
		SELECT event_id, tenant_id, event_type, model_id, timestamp,
		       predictions_count, compute_ms, input_bytes, output_bytes, metadata
		FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
	`
	args := []interface{}{req.TenantId, startTime, endTime}
	argIdx := 4

	if req.ModelId != "" {
		query += fmt.Sprintf(" AND model_id = $%d", argIdx)
		args = append(args, req.ModelId)
		argIdx++
	}
	if req.EventType != pb.EventType_EVENT_TYPE_UNSPECIFIED {
		query += fmt.Sprintf(" AND event_type = $%d", argIdx)
		args = append(args, req.EventType.String())
		argIdx++
	}

	query += fmt.Sprintf(" ORDER BY timestamp DESC LIMIT $%d", argIdx)
	args = append(args, pageSize+1) // Fetch one extra to check for next page

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list usage events: %v", err)
	}
	defer rows.Close()

	var events []*pb.UsageEvent
	for rows.Next() {
		var event pb.UsageEvent
		var timestamp time.Time
		var eventType string
		var metadata []byte

		if err := rows.Scan(&event.EventId, &event.TenantId, &eventType, &event.ModelId,
			&timestamp, &event.PredictionsCount, &event.ComputeMs,
			&event.InputBytes, &event.OutputBytes, &metadata); err != nil {
			continue
		}

		event.EventType = pb.EventType(pb.EventType_value[eventType])
		event.Timestamp = timestamppb.New(timestamp)
		event.Metadata = jsonToMap(metadata)

		events = append(events, &event)
	}

	// Handle pagination
	var nextPageToken string
	if len(events) > int(pageSize) {
		events = events[:pageSize]
		lastEvent := events[len(events)-1]
		nextPageToken = lastEvent.EventId
	}

	// Get total count
	var totalCount int64
	s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM usage_events
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp < $3
	`, req.TenantId, startTime, endTime).Scan(&totalCount)

	return &pb.ListUsageEventsResponse{
		Events:        events,
		NextPageToken: nextPageToken,
		TotalCount:    totalCount,
	}, nil
}

// checkQuotaWarning checks if any quota threshold is exceeded
func (s *meteringServer) checkQuotaWarning(ctx context.Context, tenantID string) *pb.QuotaWarning {
	statusResp, err := s.GetQuotaStatus(ctx, &pb.GetQuotaStatusRequest{TenantId: tenantID})
	if err != nil {
		return nil
	}

	for _, quota := range statusResp.Quotas {
		if quota.Exceeded {
			return &pb.QuotaWarning{
				QuotaType:      quota.QuotaType,
				CurrentUsage:   quota.Used,
				Limit:          quota.Limit,
				PercentageUsed: quota.PercentageUsed,
				Message:        fmt.Sprintf("Quota exceeded: %s (%d/%d)", quota.QuotaType.String(), quota.Used, quota.Limit),
			}
		}
		if quota.WarningTriggered {
			return &pb.QuotaWarning{
				QuotaType:      quota.QuotaType,
				CurrentUsage:   quota.Used,
				Limit:          quota.Limit,
				PercentageUsed: quota.PercentageUsed,
				Message:        fmt.Sprintf("Warning: %d%% of %s quota used", quota.PercentageUsed, quota.QuotaType.String()),
			}
		}
	}

	return nil
}

// Helper functions
func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

func mapToJSON(m map[string]string) []byte {
	if m == nil {
		return []byte("{}")
	}
	b, _ := json.Marshal(m)
	return b
}

func jsonToMap(b []byte) map[string]string {
	if b == nil {
		return nil
	}
	var m map[string]string
	json.Unmarshal(b, &m)
	return m
}

import "encoding/json"

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8085"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newMeteringServer()
	if err != nil {
		log.Fatalf("failed to create metering server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterMeteringServiceServer(s, server)

	log.Printf("Metering Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
