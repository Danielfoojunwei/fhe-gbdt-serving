// Model Versioning and Deployment Management
// Handles model versions, rollbacks, and traffic routing

package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/fhe-gbdt-serving/proto/control"
)

// ModelVersion represents a version of a model
type ModelVersion struct {
	ID              string
	ModelID         string
	Version         string
	CompiledModelID string
	Status          string
	TrafficPercent  int32
	Description     string
	CreatedBy       string
	CreatedAt       time.Time
	DeployedAt      *time.Time
	RetiredAt       *time.Time
}

// VersioningService handles model versioning operations
type VersioningService struct {
	db *sql.DB
}

// NewVersioningService creates a new versioning service
func NewVersioningService(db *sql.DB) *VersioningService {
	return &VersioningService{db: db}
}

// CreateVersion creates a new model version
func (s *VersioningService) CreateVersion(ctx context.Context, req *pb.CreateVersionRequest) (*pb.CreateVersionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating version %s for model %s", req.Version, req.ModelId)

	versionID := uuid.New().String()
	now := time.Now()

	// Check if version already exists
	var existingID string
	err := s.db.QueryRowContext(ctx, `
		SELECT id FROM model_versions WHERE model_id = $1 AND version = $2
	`, req.ModelId, req.Version).Scan(&existingID)
	if err == nil {
		return nil, status.Errorf(codes.AlreadyExists, "version %s already exists", req.Version)
	}

	// Create version
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO model_versions (id, model_id, version, compiled_model_id, status,
		                           traffic_percent, description, created_by, created_at)
		VALUES ($1, $2, $3, $4, 'draft', 0, $5, $6, $7)
	`, versionID, req.ModelId, req.Version, req.CompiledModelId, req.Description, req.CreatedBy, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create version: %v", err)
	}

	return &pb.CreateVersionResponse{
		Version: &pb.ModelVersion{
			Id:              versionID,
			ModelId:         req.ModelId,
			Version:         req.Version,
			CompiledModelId: req.CompiledModelId,
			Status:          "draft",
			TrafficPercent:  0,
			Description:     req.Description,
			CreatedAt:       timestamppb.New(now),
		},
	}, nil
}

// ListVersions lists all versions of a model
func (s *VersioningService) ListVersions(ctx context.Context, req *pb.ListVersionsRequest) (*pb.ListVersionsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, model_id, version, compiled_model_id, status, traffic_percent,
		       description, created_by, created_at, deployed_at, retired_at
		FROM model_versions
		WHERE model_id = $1
		ORDER BY created_at DESC
	`, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list versions: %v", err)
	}
	defer rows.Close()

	var versions []*pb.ModelVersion
	for rows.Next() {
		var v pb.ModelVersion
		var createdAt time.Time
		var deployedAt, retiredAt sql.NullTime

		if err := rows.Scan(&v.Id, &v.ModelId, &v.Version, &v.CompiledModelId, &v.Status,
			&v.TrafficPercent, &v.Description, &v.CreatedBy, &createdAt, &deployedAt, &retiredAt); err != nil {
			continue
		}

		v.CreatedAt = timestamppb.New(createdAt)
		if deployedAt.Valid {
			v.DeployedAt = timestamppb.New(deployedAt.Time)
		}
		if retiredAt.Valid {
			v.RetiredAt = timestamppb.New(retiredAt.Time)
		}
		versions = append(versions, &v)
	}

	return &pb.ListVersionsResponse{Versions: versions}, nil
}

// DeployVersion deploys a version with specified traffic percentage
func (s *VersioningService) DeployVersion(ctx context.Context, req *pb.DeployVersionRequest) (*pb.DeployVersionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Deploying version %s with %d%% traffic", req.VersionId, req.TrafficPercent)

	// Validate traffic percentage
	if req.TrafficPercent < 0 || req.TrafficPercent > 100 {
		return nil, status.Error(codes.InvalidArgument, "traffic_percent must be between 0 and 100")
	}

	// Get version info
	var modelID, currentStatus string
	err := s.db.QueryRowContext(ctx, `
		SELECT model_id, status FROM model_versions WHERE id = $1
	`, req.VersionId).Scan(&modelID, &currentStatus)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "version not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get version: %v", err)
	}

	now := time.Now()

	// Start transaction
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to start transaction: %v", err)
	}
	defer tx.Rollback()

	// If deploying to 100%, set other versions to 0%
	if req.TrafficPercent == 100 {
		_, err = tx.ExecContext(ctx, `
			UPDATE model_versions SET traffic_percent = 0, status = 'retired', retired_at = $1
			WHERE model_id = $2 AND id != $3 AND status = 'deployed'
		`, now, modelID, req.VersionId)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to update other versions: %v", err)
		}
	}

	// Update version
	deployedAt := now
	_, err = tx.ExecContext(ctx, `
		UPDATE model_versions SET status = 'deployed', traffic_percent = $1, deployed_at = $2
		WHERE id = $3
	`, req.TrafficPercent, deployedAt, req.VersionId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to deploy version: %v", err)
	}

	// Create deployment record
	deploymentID := uuid.New().String()
	_, err = tx.ExecContext(ctx, `
		INSERT INTO deployments (id, model_id, version_id, traffic_percent, deployed_by, deployed_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`, deploymentID, modelID, req.VersionId, req.TrafficPercent, req.DeployedBy, now)
	if err != nil {
		log.Printf("WARN: Failed to create deployment record: %v", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to commit transaction: %v", err)
	}

	return &pb.DeployVersionResponse{
		Success:      true,
		DeploymentId: deploymentID,
	}, nil
}

// Rollback rolls back to a previous version
func (s *VersioningService) Rollback(ctx context.Context, req *pb.RollbackRequest) (*pb.RollbackResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Rolling back model %s to version %s", req.ModelId, req.TargetVersionId)

	// Get target version
	var targetVersion, targetStatus string
	err := s.db.QueryRowContext(ctx, `
		SELECT version, status FROM model_versions WHERE id = $1 AND model_id = $2
	`, req.TargetVersionId, req.ModelId).Scan(&targetVersion, &targetStatus)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "target version not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get target version: %v", err)
	}

	// Deploy target version with 100% traffic
	deployResp, err := s.DeployVersion(ctx, &pb.DeployVersionRequest{
		VersionId:      req.TargetVersionId,
		TrafficPercent: 100,
		DeployedBy:     req.RolledBackBy,
	})
	if err != nil {
		return nil, err
	}

	// Create rollback record
	rollbackID := uuid.New().String()
	_, _ = s.db.ExecContext(ctx, `
		INSERT INTO rollbacks (id, model_id, from_version_id, to_version_id, reason, rolled_back_by, created_at)
		VALUES ($1, $2, (SELECT id FROM model_versions WHERE model_id = $2 AND status = 'deployed' AND id != $3 LIMIT 1),
		        $3, $4, $5, NOW())
	`, rollbackID, req.ModelId, req.TargetVersionId, req.Reason, req.RolledBackBy)

	return &pb.RollbackResponse{
		Success:         true,
		RollbackId:      rollbackID,
		DeploymentId:    deployResp.DeploymentId,
		RestoredVersion: targetVersion,
	}, nil
}

// GetActiveVersions gets versions receiving traffic for a model
func (s *VersioningService) GetActiveVersions(ctx context.Context, req *pb.GetActiveVersionsRequest) (*pb.GetActiveVersionsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, version, compiled_model_id, traffic_percent
		FROM model_versions
		WHERE model_id = $1 AND status = 'deployed' AND traffic_percent > 0
		ORDER BY traffic_percent DESC
	`, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get active versions: %v", err)
	}
	defer rows.Close()

	var versions []*pb.ActiveVersion
	for rows.Next() {
		var v pb.ActiveVersion
		if err := rows.Scan(&v.VersionId, &v.Version, &v.CompiledModelId, &v.TrafficPercent); err != nil {
			continue
		}
		versions = append(versions, &v)
	}

	return &pb.GetActiveVersionsResponse{Versions: versions}, nil
}

// SetTrafficSplit sets traffic split between versions (canary/blue-green)
func (s *VersioningService) SetTrafficSplit(ctx context.Context, req *pb.SetTrafficSplitRequest) (*pb.SetTrafficSplitResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Setting traffic split for model %s", req.ModelId)

	// Validate total traffic is 100%
	var total int32
	for _, split := range req.Splits {
		total += split.TrafficPercent
	}
	if total != 100 {
		return nil, status.Errorf(codes.InvalidArgument, "traffic percentages must sum to 100, got %d", total)
	}

	// Start transaction
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to start transaction: %v", err)
	}
	defer tx.Rollback()

	now := time.Now()

	// Reset all traffic to 0 first
	_, err = tx.ExecContext(ctx, `
		UPDATE model_versions SET traffic_percent = 0
		WHERE model_id = $1 AND status = 'deployed'
	`, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to reset traffic: %v", err)
	}

	// Set new traffic splits
	for _, split := range req.Splits {
		_, err = tx.ExecContext(ctx, `
			UPDATE model_versions SET traffic_percent = $1, status = 'deployed', deployed_at = COALESCE(deployed_at, $2)
			WHERE id = $3 AND model_id = $4
		`, split.TrafficPercent, now, split.VersionId, req.ModelId)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to set traffic for version %s: %v", split.VersionId, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to commit transaction: %v", err)
	}

	return &pb.SetTrafficSplitResponse{Success: true}, nil
}

// GetDeploymentHistory gets deployment history for a model
func (s *VersioningService) GetDeploymentHistory(ctx context.Context, req *pb.GetDeploymentHistoryRequest) (*pb.GetDeploymentHistoryResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	limit := req.Limit
	if limit <= 0 || limit > 100 {
		limit = 20
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT d.id, d.version_id, v.version, d.traffic_percent, d.deployed_by, d.deployed_at
		FROM deployments d
		JOIN model_versions v ON d.version_id = v.id
		WHERE d.model_id = $1
		ORDER BY d.deployed_at DESC
		LIMIT $2
	`, req.ModelId, limit)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get deployment history: %v", err)
	}
	defer rows.Close()

	var deployments []*pb.Deployment
	for rows.Next() {
		var d pb.Deployment
		var deployedAt time.Time

		if err := rows.Scan(&d.Id, &d.VersionId, &d.Version, &d.TrafficPercent, &d.DeployedBy, &deployedAt); err != nil {
			continue
		}
		d.DeployedAt = timestamppb.New(deployedAt)
		deployments = append(deployments, &d)
	}

	return &pb.GetDeploymentHistoryResponse{Deployments: deployments}, nil
}

// RouteRequest routes a request to the appropriate model version based on traffic split
func (s *VersioningService) RouteRequest(ctx context.Context, modelID string) (string, error) {
	if s.db == nil {
		return "", fmt.Errorf("database not available")
	}

	// Get active versions with traffic
	rows, err := s.db.QueryContext(ctx, `
		SELECT compiled_model_id, traffic_percent
		FROM model_versions
		WHERE model_id = $1 AND status = 'deployed' AND traffic_percent > 0
		ORDER BY traffic_percent DESC
	`, modelID)
	if err != nil {
		return "", fmt.Errorf("failed to get active versions: %v", err)
	}
	defer rows.Close()

	type versionTraffic struct {
		compiledModelID string
		trafficPercent  int32
	}
	var versions []versionTraffic

	for rows.Next() {
		var v versionTraffic
		if err := rows.Scan(&v.compiledModelID, &v.trafficPercent); err != nil {
			continue
		}
		versions = append(versions, v)
	}

	if len(versions) == 0 {
		return "", fmt.Errorf("no active versions found")
	}

	// Simple weighted random selection
	// In production, use consistent hashing for request stickiness
	total := int32(0)
	for _, v := range versions {
		total += v.trafficPercent
	}

	r := rand.Int31n(total)
	cumulative := int32(0)
	for _, v := range versions {
		cumulative += v.trafficPercent
		if r < cumulative {
			return v.compiledModelID, nil
		}
	}

	return versions[0].compiledModelID, nil
}
