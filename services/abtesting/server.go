// A/B Testing Service for Model Experiments
// Manages experiments, traffic allocation, and statistical analysis

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
	pb "github.com/fhe-gbdt-serving/proto/abtesting"
)

type abTestingServer struct {
	pb.UnimplementedABTestingServiceServer
	db *sql.DB
}

func newABTestingServer() (*abTestingServer, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/abtesting?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	return &abTestingServer{db: db}, nil
}

// ============================================================================
// Experiment Management
// ============================================================================

func (s *abTestingServer) CreateExperiment(ctx context.Context, req *pb.CreateExperimentRequest) (*pb.CreateExperimentResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating experiment %s for tenant %s", req.Name, req.TenantId)

	// Validate variants
	if len(req.Variants) < 2 {
		return nil, status.Error(codes.InvalidArgument, "at least 2 variants required")
	}

	var totalTraffic int32
	for _, v := range req.Variants {
		totalTraffic += v.TrafficPercent
	}
	if totalTraffic != 100 {
		return nil, status.Errorf(codes.InvalidArgument, "traffic percentages must sum to 100, got %d", totalTraffic)
	}

	experimentID := uuid.New().String()
	now := time.Now()

	// Create experiment
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO experiments (id, tenant_id, name, description, model_id, status,
		                        hypothesis, primary_metric, secondary_metrics,
		                        min_sample_size, significance_level, created_at)
		VALUES ($1, $2, $3, $4, $5, 'draft', $6, $7, $8, $9, $10, $11)
	`, experimentID, req.TenantId, req.Name, req.Description, req.ModelId,
		req.Hypothesis, req.PrimaryMetric, marshalJSON(req.SecondaryMetrics),
		req.MinSampleSize, req.SignificanceLevel, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create experiment: %v", err)
	}

	// Create variants
	for _, v := range req.Variants {
		variantID := uuid.New().String()
		_, err := s.db.ExecContext(ctx, `
			INSERT INTO experiment_variants (id, experiment_id, name, model_version_id,
			                                traffic_percent, is_control)
			VALUES ($1, $2, $3, $4, $5, $6)
		`, variantID, experimentID, v.Name, v.ModelVersionId, v.TrafficPercent, v.IsControl)
		if err != nil {
			log.Printf("WARN: Failed to create variant: %v", err)
		}
	}

	return &pb.CreateExperimentResponse{
		Experiment: &pb.Experiment{
			Id:               experimentID,
			TenantId:         req.TenantId,
			Name:             req.Name,
			Description:      req.Description,
			ModelId:          req.ModelId,
			Status:           "draft",
			Hypothesis:       req.Hypothesis,
			PrimaryMetric:    req.PrimaryMetric,
			SecondaryMetrics: req.SecondaryMetrics,
			MinSampleSize:    req.MinSampleSize,
			SignificanceLevel: req.SignificanceLevel,
			CreatedAt:        timestamppb.New(now),
		},
	}, nil
}

func (s *abTestingServer) GetExperiment(ctx context.Context, req *pb.GetExperimentRequest) (*pb.GetExperimentResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var exp pb.Experiment
	var createdAt time.Time
	var startedAt, endedAt sql.NullTime
	var secondaryMetricsJSON []byte

	err := s.db.QueryRowContext(ctx, `
		SELECT id, tenant_id, name, description, model_id, status, hypothesis,
		       primary_metric, secondary_metrics, min_sample_size, significance_level,
		       created_at, started_at, ended_at
		FROM experiments WHERE id = $1
	`, req.ExperimentId).Scan(&exp.Id, &exp.TenantId, &exp.Name, &exp.Description,
		&exp.ModelId, &exp.Status, &exp.Hypothesis, &exp.PrimaryMetric,
		&secondaryMetricsJSON, &exp.MinSampleSize, &exp.SignificanceLevel,
		&createdAt, &startedAt, &endedAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "experiment not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get experiment: %v", err)
	}

	json.Unmarshal(secondaryMetricsJSON, &exp.SecondaryMetrics)
	exp.CreatedAt = timestamppb.New(createdAt)
	if startedAt.Valid {
		exp.StartedAt = timestamppb.New(startedAt.Time)
	}
	if endedAt.Valid {
		exp.EndedAt = timestamppb.New(endedAt.Time)
	}

	// Get variants
	variants, _ := s.getExperimentVariants(ctx, req.ExperimentId)
	exp.Variants = variants

	return &pb.GetExperimentResponse{Experiment: &exp}, nil
}

func (s *abTestingServer) ListExperiments(ctx context.Context, req *pb.ListExperimentsRequest) (*pb.ListExperimentsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	query := `
		SELECT id, name, model_id, status, primary_metric, created_at, started_at, ended_at
		FROM experiments WHERE tenant_id = $1
	`
	args := []interface{}{req.TenantId}
	argNum := 2

	if req.Status != "" {
		query += fmt.Sprintf(" AND status = $%d", argNum)
		args = append(args, req.Status)
		argNum++
	}
	if req.ModelId != "" {
		query += fmt.Sprintf(" AND model_id = $%d", argNum)
		args = append(args, req.ModelId)
	}

	query += " ORDER BY created_at DESC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list experiments: %v", err)
	}
	defer rows.Close()

	var experiments []*pb.ExperimentSummary
	for rows.Next() {
		var exp pb.ExperimentSummary
		var createdAt time.Time
		var startedAt, endedAt sql.NullTime

		if err := rows.Scan(&exp.Id, &exp.Name, &exp.ModelId, &exp.Status,
			&exp.PrimaryMetric, &createdAt, &startedAt, &endedAt); err != nil {
			continue
		}

		exp.CreatedAt = timestamppb.New(createdAt)
		if startedAt.Valid {
			exp.StartedAt = timestamppb.New(startedAt.Time)
		}
		if endedAt.Valid {
			exp.EndedAt = timestamppb.New(endedAt.Time)
		}

		experiments = append(experiments, &exp)
	}

	return &pb.ListExperimentsResponse{Experiments: experiments}, nil
}

func (s *abTestingServer) StartExperiment(ctx context.Context, req *pb.StartExperimentRequest) (*pb.StartExperimentResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Starting experiment %s", req.ExperimentId)

	now := time.Now()
	result, err := s.db.ExecContext(ctx, `
		UPDATE experiments SET status = 'running', started_at = $1
		WHERE id = $2 AND status = 'draft'
	`, now, req.ExperimentId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to start experiment: %v", err)
	}

	rows, _ := result.RowsAffected()
	if rows == 0 {
		return nil, status.Error(codes.FailedPrecondition, "experiment not in draft status")
	}

	return &pb.StartExperimentResponse{
		Success:   true,
		StartedAt: timestamppb.New(now),
	}, nil
}

func (s *abTestingServer) StopExperiment(ctx context.Context, req *pb.StopExperimentRequest) (*pb.StopExperimentResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Stopping experiment %s, reason: %s", req.ExperimentId, req.Reason)

	now := time.Now()
	_, err := s.db.ExecContext(ctx, `
		UPDATE experiments SET status = 'stopped', ended_at = $1, stop_reason = $2
		WHERE id = $3 AND status = 'running'
	`, now, req.Reason, req.ExperimentId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to stop experiment: %v", err)
	}

	return &pb.StopExperimentResponse{
		Success: true,
		EndedAt: timestamppb.New(now),
	}, nil
}

// ============================================================================
// Traffic Assignment
// ============================================================================

func (s *abTestingServer) AssignVariant(ctx context.Context, req *pb.AssignVariantRequest) (*pb.AssignVariantResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Check if user already has an assignment
	var existingVariantID string
	err := s.db.QueryRowContext(ctx, `
		SELECT variant_id FROM experiment_assignments
		WHERE experiment_id = $1 AND user_id = $2
	`, req.ExperimentId, req.UserId).Scan(&existingVariantID)
	if err == nil {
		// Return existing assignment
		return s.getVariantAssignment(ctx, existingVariantID)
	}

	// Get variants with traffic percentages
	variants, err := s.getExperimentVariants(ctx, req.ExperimentId)
	if err != nil {
		return nil, err
	}

	// Deterministic assignment based on user hash
	variantID := s.assignVariantByHash(req.UserId, variants)

	// Store assignment
	assignmentID := uuid.New().String()
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO experiment_assignments (id, experiment_id, user_id, variant_id, assigned_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT (experiment_id, user_id) DO NOTHING
	`, assignmentID, req.ExperimentId, req.UserId, variantID)
	if err != nil {
		log.Printf("WARN: Failed to store assignment: %v", err)
	}

	return s.getVariantAssignment(ctx, variantID)
}

func (s *abTestingServer) assignVariantByHash(userID string, variants []*pb.Variant) string {
	// Consistent hashing for deterministic assignment
	hash := fnvHash(userID)
	bucket := hash % 100

	var cumulative int32
	for _, v := range variants {
		cumulative += v.TrafficPercent
		if bucket < int(cumulative) {
			return v.Id
		}
	}

	// Fallback to first variant
	if len(variants) > 0 {
		return variants[0].Id
	}
	return ""
}

func (s *abTestingServer) getVariantAssignment(ctx context.Context, variantID string) (*pb.AssignVariantResponse, error) {
	var v pb.Variant
	err := s.db.QueryRowContext(ctx, `
		SELECT id, experiment_id, name, model_version_id, traffic_percent, is_control
		FROM experiment_variants WHERE id = $1
	`, variantID).Scan(&v.Id, &v.ExperimentId, &v.Name, &v.ModelVersionId, &v.TrafficPercent, &v.IsControl)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get variant: %v", err)
	}

	return &pb.AssignVariantResponse{
		VariantId:      v.Id,
		VariantName:    v.Name,
		ModelVersionId: v.ModelVersionId,
		IsControl:      v.IsControl,
	}, nil
}

func (s *abTestingServer) getExperimentVariants(ctx context.Context, experimentID string) ([]*pb.Variant, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, experiment_id, name, model_version_id, traffic_percent, is_control
		FROM experiment_variants WHERE experiment_id = $1
		ORDER BY is_control DESC, name ASC
	`, experimentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var variants []*pb.Variant
	for rows.Next() {
		var v pb.Variant
		if err := rows.Scan(&v.Id, &v.ExperimentId, &v.Name, &v.ModelVersionId,
			&v.TrafficPercent, &v.IsControl); err != nil {
			continue
		}
		variants = append(variants, &v)
	}

	return variants, nil
}

// ============================================================================
// Event Tracking
// ============================================================================

func (s *abTestingServer) TrackEvent(ctx context.Context, req *pb.TrackEventRequest) (*pb.TrackEventResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	eventID := uuid.New().String()
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO experiment_events (id, experiment_id, variant_id, user_id, event_type,
		                              metric_name, metric_value, metadata, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
	`, eventID, req.ExperimentId, req.VariantId, req.UserId, req.EventType,
		req.MetricName, req.MetricValue, marshalJSON(req.Metadata))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to track event: %v", err)
	}

	return &pb.TrackEventResponse{EventId: eventID}, nil
}

// ============================================================================
// Statistical Analysis
// ============================================================================

func (s *abTestingServer) GetExperimentResults(ctx context.Context, req *pb.GetExperimentResultsRequest) (*pb.GetExperimentResultsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get variants
	variants, err := s.getExperimentVariants(ctx, req.ExperimentId)
	if err != nil {
		return nil, err
	}

	// Get experiment significance level
	var significanceLevel float64
	s.db.QueryRowContext(ctx, `SELECT significance_level FROM experiments WHERE id = $1`,
		req.ExperimentId).Scan(&significanceLevel)
	if significanceLevel == 0 {
		significanceLevel = 0.05
	}

	var results []*pb.VariantResult
	var controlResult *pb.VariantResult

	for _, variant := range variants {
		result, err := s.calculateVariantMetrics(ctx, req.ExperimentId, variant.Id, req.MetricName)
		if err != nil {
			continue
		}
		result.VariantId = variant.Id
		result.VariantName = variant.Name
		result.IsControl = variant.IsControl

		if variant.IsControl {
			controlResult = result
		}
		results = append(results, result)
	}

	// Calculate statistical significance
	if controlResult != nil {
		for _, result := range results {
			if !result.IsControl && controlResult.SampleSize > 0 && result.SampleSize > 0 {
				// Two-sample z-test for means
				zScore := (result.Mean - controlResult.Mean) /
					math.Sqrt((controlResult.Variance/float64(controlResult.SampleSize))+
						(result.Variance/float64(result.SampleSize)))

				pValue := 2 * (1 - normalCDF(math.Abs(zScore)))
				result.PValue = pValue
				result.Significant = pValue < significanceLevel
				result.Lift = (result.Mean - controlResult.Mean) / controlResult.Mean * 100

				// Confidence interval (95%)
				marginOfError := 1.96 * math.Sqrt(result.Variance/float64(result.SampleSize))
				result.ConfidenceLower = result.Mean - marginOfError
				result.ConfidenceUpper = result.Mean + marginOfError
			}
		}
	}

	// Determine winner
	var winner string
	var conclusive bool
	if controlResult != nil {
		bestLift := float64(0)
		for _, result := range results {
			if result.Significant && result.Lift > bestLift {
				bestLift = result.Lift
				winner = result.VariantId
				conclusive = true
			}
		}
	}

	return &pb.GetExperimentResultsResponse{
		ExperimentId: req.ExperimentId,
		MetricName:   req.MetricName,
		Results:      results,
		Winner:       winner,
		Conclusive:   conclusive,
	}, nil
}

func (s *abTestingServer) calculateVariantMetrics(ctx context.Context, experimentID, variantID, metricName string) (*pb.VariantResult, error) {
	var result pb.VariantResult

	err := s.db.QueryRowContext(ctx, `
		SELECT COUNT(*), AVG(metric_value), VARIANCE(metric_value)
		FROM experiment_events
		WHERE experiment_id = $1 AND variant_id = $2 AND metric_name = $3
	`, experimentID, variantID, metricName).Scan(&result.SampleSize, &result.Mean, &result.Variance)
	if err != nil {
		return nil, err
	}

	return &result, nil
}

// ============================================================================
// Helpers
// ============================================================================

func fnvHash(s string) int {
	hash := 2166136261
	for i := 0; i < len(s); i++ {
		hash ^= int(s[i])
		hash *= 16777619
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}

func normalCDF(x float64) float64 {
	// Approximation of normal CDF
	a1 := 0.254829592
	a2 := -0.284496736
	a3 := 1.421413741
	a4 := -1.453152027
	a5 := 1.061405429
	p := 0.3275911

	sign := 1.0
	if x < 0 {
		sign = -1.0
	}
	x = math.Abs(x) / math.Sqrt(2)

	t := 1.0 / (1.0 + p*x)
	y := 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*math.Exp(-x*x)

	return 0.5 * (1.0 + sign*y)
}

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
		port = "8093"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newABTestingServer()
	if err != nil {
		log.Fatalf("failed to create A/B testing server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterABTestingServiceServer(s, server)

	log.Printf("A/B Testing Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
