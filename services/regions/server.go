// Multi-Region and Data Residency Service
// Manages region configuration, data residency, and geo-routing

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
	"sort"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/regions"
)

type regionsServer struct {
	pb.UnimplementedRegionsServiceServer
	db *sql.DB
}

func newRegionsServer() (*regionsServer, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/regions?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	return &regionsServer{db: db}, nil
}

// Region represents a deployment region
type Region struct {
	ID          string
	Code        string // us-east-1, eu-west-1, ap-southeast-1
	Name        string
	Provider    string // aws, gcp, azure
	Country     string
	Continent   string
	Latitude    float64
	Longitude   float64
	Status      string // active, maintenance, disabled
	Tier        string // primary, secondary, edge
	Endpoints   map[string]string
	Features    []string
	Compliance  []string
	MaxCapacity int32
	CurrentLoad int32
	CreatedAt   time.Time
}

// ============================================================================
// Region Management
// ============================================================================

func (s *regionsServer) ListRegions(ctx context.Context, req *pb.ListRegionsRequest) (*pb.ListRegionsResponse, error) {
	if s.db == nil {
		// Return hardcoded regions for demo
		return &pb.ListRegionsResponse{
			Regions: getDefaultRegions(),
		}, nil
	}

	query := `
		SELECT id, code, name, provider, country, continent, latitude, longitude,
		       status, tier, endpoints, features, compliance, max_capacity, current_load, created_at
		FROM regions
		WHERE 1=1
	`
	args := []interface{}{}
	argNum := 1

	if req.Provider != "" {
		query += fmt.Sprintf(" AND provider = $%d", argNum)
		args = append(args, req.Provider)
		argNum++
	}
	if req.Continent != "" {
		query += fmt.Sprintf(" AND continent = $%d", argNum)
		args = append(args, req.Continent)
		argNum++
	}
	if req.Status != "" {
		query += fmt.Sprintf(" AND status = $%d", argNum)
		args = append(args, req.Status)
		argNum++
	}

	query += " ORDER BY tier ASC, code ASC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list regions: %v", err)
	}
	defer rows.Close()

	var regions []*pb.Region
	for rows.Next() {
		var r pb.Region
		var createdAt time.Time
		var endpointsJSON, featuresJSON, complianceJSON []byte

		if err := rows.Scan(&r.Id, &r.Code, &r.Name, &r.Provider, &r.Country, &r.Continent,
			&r.Latitude, &r.Longitude, &r.Status, &r.Tier, &endpointsJSON, &featuresJSON,
			&complianceJSON, &r.MaxCapacity, &r.CurrentLoad, &createdAt); err != nil {
			continue
		}

		json.Unmarshal(endpointsJSON, &r.Endpoints)
		json.Unmarshal(featuresJSON, &r.Features)
		json.Unmarshal(complianceJSON, &r.Compliance)
		r.CreatedAt = timestamppb.New(createdAt)

		regions = append(regions, &r)
	}

	return &pb.ListRegionsResponse{Regions: regions}, nil
}

func (s *regionsServer) GetRegion(ctx context.Context, req *pb.GetRegionRequest) (*pb.GetRegionResponse, error) {
	if s.db == nil {
		for _, r := range getDefaultRegions() {
			if r.Code == req.RegionCode {
				return &pb.GetRegionResponse{Region: r}, nil
			}
		}
		return nil, status.Error(codes.NotFound, "region not found")
	}

	var r pb.Region
	var createdAt time.Time
	var endpointsJSON, featuresJSON, complianceJSON []byte

	err := s.db.QueryRowContext(ctx, `
		SELECT id, code, name, provider, country, continent, latitude, longitude,
		       status, tier, endpoints, features, compliance, max_capacity, current_load, created_at
		FROM regions WHERE code = $1
	`, req.RegionCode).Scan(&r.Id, &r.Code, &r.Name, &r.Provider, &r.Country, &r.Continent,
		&r.Latitude, &r.Longitude, &r.Status, &r.Tier, &endpointsJSON, &featuresJSON,
		&complianceJSON, &r.MaxCapacity, &r.CurrentLoad, &createdAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "region not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get region: %v", err)
	}

	json.Unmarshal(endpointsJSON, &r.Endpoints)
	json.Unmarshal(featuresJSON, &r.Features)
	json.Unmarshal(complianceJSON, &r.Compliance)
	r.CreatedAt = timestamppb.New(createdAt)

	return &pb.GetRegionResponse{Region: &r}, nil
}

// ============================================================================
// Tenant Region Configuration
// ============================================================================

func (s *regionsServer) SetTenantRegions(ctx context.Context, req *pb.SetTenantRegionsRequest) (*pb.SetTenantRegionsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Setting regions for tenant %s: %v", req.TenantId, req.RegionCodes)

	// Validate regions
	for _, code := range req.RegionCodes {
		var exists bool
		err := s.db.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM regions WHERE code = $1 AND status = 'active')`, code).Scan(&exists)
		if err != nil || !exists {
			return nil, status.Errorf(codes.InvalidArgument, "invalid or inactive region: %s", code)
		}
	}

	// Start transaction
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to start transaction: %v", err)
	}
	defer tx.Rollback()

	// Delete existing region assignments
	_, err = tx.ExecContext(ctx, `DELETE FROM tenant_regions WHERE tenant_id = $1`, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to clear existing regions: %v", err)
	}

	// Insert new region assignments
	for _, code := range req.RegionCodes {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO tenant_regions (id, tenant_id, region_code, primary_region, created_at)
			VALUES ($1, $2, $3, $4, NOW())
		`, uuid.New().String(), req.TenantId, code, code == req.PrimaryRegion)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to assign region: %v", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to commit transaction: %v", err)
	}

	return &pb.SetTenantRegionsResponse{Success: true}, nil
}

func (s *regionsServer) GetTenantRegions(ctx context.Context, req *pb.GetTenantRegionsRequest) (*pb.GetTenantRegionsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT tr.region_code, tr.primary_region, r.name, r.provider, r.country, r.compliance
		FROM tenant_regions tr
		JOIN regions r ON tr.region_code = r.code
		WHERE tr.tenant_id = $1
	`, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get tenant regions: %v", err)
	}
	defer rows.Close()

	var regions []*pb.TenantRegion
	var primaryRegion string
	for rows.Next() {
		var tr pb.TenantRegion
		var complianceJSON []byte
		var isPrimary bool

		if err := rows.Scan(&tr.RegionCode, &isPrimary, &tr.Name, &tr.Provider, &tr.Country, &complianceJSON); err != nil {
			continue
		}
		json.Unmarshal(complianceJSON, &tr.Compliance)
		regions = append(regions, &tr)
		if isPrimary {
			primaryRegion = tr.RegionCode
		}
	}

	return &pb.GetTenantRegionsResponse{
		Regions:       regions,
		PrimaryRegion: primaryRegion,
	}, nil
}

// ============================================================================
// Data Residency
// ============================================================================

func (s *regionsServer) SetDataResidency(ctx context.Context, req *pb.SetDataResidencyRequest) (*pb.SetDataResidencyResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Setting data residency for tenant %s: allowed=%v, blocked=%v",
		req.TenantId, req.AllowedCountries, req.BlockedCountries)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO data_residency (id, tenant_id, allowed_countries, blocked_countries,
		                           compliance_requirements, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
		ON CONFLICT (tenant_id) DO UPDATE SET
			allowed_countries = $3,
			blocked_countries = $4,
			compliance_requirements = $5,
			updated_at = NOW()
	`, uuid.New().String(), req.TenantId, marshalJSON(req.AllowedCountries),
		marshalJSON(req.BlockedCountries), marshalJSON(req.ComplianceRequirements))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to set data residency: %v", err)
	}

	return &pb.SetDataResidencyResponse{Success: true}, nil
}

func (s *regionsServer) GetDataResidency(ctx context.Context, req *pb.GetDataResidencyRequest) (*pb.GetDataResidencyResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var allowedJSON, blockedJSON, complianceJSON []byte
	var updatedAt time.Time

	err := s.db.QueryRowContext(ctx, `
		SELECT allowed_countries, blocked_countries, compliance_requirements, updated_at
		FROM data_residency WHERE tenant_id = $1
	`, req.TenantId).Scan(&allowedJSON, &blockedJSON, &complianceJSON, &updatedAt)
	if err == sql.ErrNoRows {
		return &pb.GetDataResidencyResponse{
			Policy: &pb.DataResidencyPolicy{
				TenantId: req.TenantId,
			},
		}, nil
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get data residency: %v", err)
	}

	var policy pb.DataResidencyPolicy
	policy.TenantId = req.TenantId
	json.Unmarshal(allowedJSON, &policy.AllowedCountries)
	json.Unmarshal(blockedJSON, &policy.BlockedCountries)
	json.Unmarshal(complianceJSON, &policy.ComplianceRequirements)
	policy.UpdatedAt = timestamppb.New(updatedAt)

	return &pb.GetDataResidencyResponse{Policy: &policy}, nil
}

func (s *regionsServer) CheckDataResidency(ctx context.Context, req *pb.CheckDataResidencyRequest) (*pb.CheckDataResidencyResponse, error) {
	if s.db == nil {
		// Allow all if no database
		return &pb.CheckDataResidencyResponse{Allowed: true}, nil
	}

	var allowedJSON, blockedJSON []byte
	err := s.db.QueryRowContext(ctx, `
		SELECT allowed_countries, blocked_countries FROM data_residency WHERE tenant_id = $1
	`, req.TenantId).Scan(&allowedJSON, &blockedJSON)
	if err == sql.ErrNoRows {
		// No restrictions
		return &pb.CheckDataResidencyResponse{Allowed: true}, nil
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to check data residency: %v", err)
	}

	var allowed, blocked []string
	json.Unmarshal(allowedJSON, &allowed)
	json.Unmarshal(blockedJSON, &blocked)

	// Check if country is blocked
	for _, c := range blocked {
		if c == req.Country {
			return &pb.CheckDataResidencyResponse{
				Allowed: false,
				Reason:  fmt.Sprintf("data processing in %s is blocked by policy", req.Country),
			}, nil
		}
	}

	// Check if country is in allowed list (if list is not empty)
	if len(allowed) > 0 {
		found := false
		for _, c := range allowed {
			if c == req.Country {
				found = true
				break
			}
		}
		if !found {
			return &pb.CheckDataResidencyResponse{
				Allowed: false,
				Reason:  fmt.Sprintf("data processing in %s is not in allowed countries list", req.Country),
			}, nil
		}
	}

	return &pb.CheckDataResidencyResponse{Allowed: true}, nil
}

// ============================================================================
// Geo-Routing
// ============================================================================

func (s *regionsServer) GetOptimalRegion(ctx context.Context, req *pb.GetOptimalRegionRequest) (*pb.GetOptimalRegionResponse, error) {
	// Get tenant's allowed regions
	tenantRegions, err := s.GetTenantRegions(ctx, &pb.GetTenantRegionsRequest{TenantId: req.TenantId})
	if err != nil {
		return nil, err
	}

	if len(tenantRegions.Regions) == 0 {
		// Default to closest active region
		return s.getClosestRegion(ctx, req.ClientLatitude, req.ClientLongitude)
	}

	// Check data residency
	residency, err := s.GetDataResidency(ctx, &pb.GetDataResidencyRequest{TenantId: req.TenantId})
	if err != nil {
		return nil, err
	}

	// Filter regions by data residency policy
	var eligibleRegions []*pb.TenantRegion
	for _, r := range tenantRegions.Regions {
		allowed := true
		if residency.Policy != nil {
			// Check blocked countries
			for _, c := range residency.Policy.BlockedCountries {
				if c == r.Country {
					allowed = false
					break
				}
			}
			// Check allowed countries
			if len(residency.Policy.AllowedCountries) > 0 && allowed {
				allowed = false
				for _, c := range residency.Policy.AllowedCountries {
					if c == r.Country {
						allowed = true
						break
					}
				}
			}
		}
		if allowed {
			eligibleRegions = append(eligibleRegions, r)
		}
	}

	if len(eligibleRegions) == 0 {
		return nil, status.Error(codes.FailedPrecondition, "no eligible regions available")
	}

	// Sort by distance from client
	sort.Slice(eligibleRegions, func(i, j int) bool {
		// Get region coordinates
		ri, _ := s.GetRegion(ctx, &pb.GetRegionRequest{RegionCode: eligibleRegions[i].RegionCode})
		rj, _ := s.GetRegion(ctx, &pb.GetRegionRequest{RegionCode: eligibleRegions[j].RegionCode})
		if ri == nil || rj == nil {
			return false
		}
		distI := distance(req.ClientLatitude, req.ClientLongitude, ri.Region.Latitude, ri.Region.Longitude)
		distJ := distance(req.ClientLatitude, req.ClientLongitude, rj.Region.Latitude, rj.Region.Longitude)
		return distI < distJ
	})

	return &pb.GetOptimalRegionResponse{
		RegionCode: eligibleRegions[0].RegionCode,
		Reason:     "closest eligible region based on geo-location and data residency policy",
	}, nil
}

func (s *regionsServer) getClosestRegion(ctx context.Context, lat, lon float64) (*pb.GetOptimalRegionResponse, error) {
	regions := getDefaultRegions()
	if s.db != nil {
		resp, err := s.ListRegions(ctx, &pb.ListRegionsRequest{Status: "active"})
		if err == nil && len(resp.Regions) > 0 {
			regions = resp.Regions
		}
	}

	if len(regions) == 0 {
		return nil, status.Error(codes.Internal, "no regions available")
	}

	sort.Slice(regions, func(i, j int) bool {
		distI := distance(lat, lon, regions[i].Latitude, regions[i].Longitude)
		distJ := distance(lat, lon, regions[j].Latitude, regions[j].Longitude)
		return distI < distJ
	})

	return &pb.GetOptimalRegionResponse{
		RegionCode: regions[0].Code,
		Reason:     "closest active region based on geo-location",
	}, nil
}

// ============================================================================
// Replication
// ============================================================================

func (s *regionsServer) SetupReplication(ctx context.Context, req *pb.SetupReplicationRequest) (*pb.SetupReplicationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Setting up replication for tenant %s model %s from %s to %v",
		req.TenantId, req.ModelId, req.SourceRegion, req.TargetRegions)

	replicationID := uuid.New().String()
	now := time.Now()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO model_replications (id, tenant_id, model_id, source_region, target_regions,
		                               replication_mode, status, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7)
	`, replicationID, req.TenantId, req.ModelId, req.SourceRegion,
		marshalJSON(req.TargetRegions), req.ReplicationMode, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to setup replication: %v", err)
	}

	// Trigger async replication (in production, this would use a message queue)
	go s.performReplication(context.Background(), replicationID)

	return &pb.SetupReplicationResponse{
		ReplicationId: replicationID,
		Status:        "pending",
	}, nil
}

func (s *regionsServer) GetReplicationStatus(ctx context.Context, req *pb.GetReplicationStatusRequest) (*pb.GetReplicationStatusResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var status, sourceRegion string
	var targetRegionsJSON []byte
	var progress float32
	var lastSyncAt sql.NullTime
	var errorMsg sql.NullString

	err := s.db.QueryRowContext(ctx, `
		SELECT status, source_region, target_regions, progress, last_sync_at, error_message
		FROM model_replications WHERE id = $1
	`, req.ReplicationId).Scan(&status, &sourceRegion, &targetRegionsJSON, &progress, &lastSyncAt, &errorMsg)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "replication not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get replication status: %v", err)
	}

	resp := &pb.GetReplicationStatusResponse{
		ReplicationId: req.ReplicationId,
		Status:        status,
		SourceRegion:  sourceRegion,
		Progress:      progress,
	}
	json.Unmarshal(targetRegionsJSON, &resp.TargetRegions)
	if lastSyncAt.Valid {
		resp.LastSyncAt = timestamppb.New(lastSyncAt.Time)
	}
	if errorMsg.Valid {
		resp.ErrorMessage = errorMsg.String
	}

	return resp, nil
}

func (s *regionsServer) performReplication(ctx context.Context, replicationID string) {
	// In production, this would:
	// 1. Connect to source region storage
	// 2. Stream model data to target regions
	// 3. Verify checksums
	// 4. Update replication status
	log.Printf("INFO: Starting replication %s", replicationID)

	// Simulate replication progress
	for progress := 0; progress <= 100; progress += 25 {
		time.Sleep(time.Second)
		s.db.ExecContext(ctx, `
			UPDATE model_replications SET progress = $1 WHERE id = $2
		`, progress, replicationID)
	}

	s.db.ExecContext(ctx, `
		UPDATE model_replications SET status = 'completed', last_sync_at = NOW() WHERE id = $1
	`, replicationID)

	log.Printf("INFO: Completed replication %s", replicationID)
}

// ============================================================================
// Helpers
// ============================================================================

func getDefaultRegions() []*pb.Region {
	return []*pb.Region{
		{
			Id: "1", Code: "us-east-1", Name: "US East (N. Virginia)", Provider: "aws",
			Country: "US", Continent: "North America", Latitude: 37.478, Longitude: -76.453,
			Status: "active", Tier: "primary",
			Endpoints: map[string]string{
				"grpc": "us-east-1.fhe-gbdt.dev:443",
				"rest": "https://us-east-1.fhe-gbdt.dev",
			},
			Features:   []string{"fhe-inference", "model-compilation", "key-management"},
			Compliance: []string{"SOC2", "HIPAA", "GDPR"},
			MaxCapacity: 10000, CurrentLoad: 3500,
		},
		{
			Id: "2", Code: "eu-west-1", Name: "EU West (Ireland)", Provider: "aws",
			Country: "IE", Continent: "Europe", Latitude: 53.349, Longitude: -6.260,
			Status: "active", Tier: "primary",
			Endpoints: map[string]string{
				"grpc": "eu-west-1.fhe-gbdt.dev:443",
				"rest": "https://eu-west-1.fhe-gbdt.dev",
			},
			Features:   []string{"fhe-inference", "model-compilation", "key-management"},
			Compliance: []string{"SOC2", "GDPR", "ISO27001"},
			MaxCapacity: 8000, CurrentLoad: 2800,
		},
		{
			Id: "3", Code: "eu-central-1", Name: "EU Central (Frankfurt)", Provider: "aws",
			Country: "DE", Continent: "Europe", Latitude: 50.110, Longitude: 8.682,
			Status: "active", Tier: "primary",
			Endpoints: map[string]string{
				"grpc": "eu-central-1.fhe-gbdt.dev:443",
				"rest": "https://eu-central-1.fhe-gbdt.dev",
			},
			Features:   []string{"fhe-inference", "model-compilation", "key-management"},
			Compliance: []string{"SOC2", "GDPR", "ISO27001", "C5"},
			MaxCapacity: 8000, CurrentLoad: 3200,
		},
		{
			Id: "4", Code: "ap-southeast-1", Name: "Asia Pacific (Singapore)", Provider: "aws",
			Country: "SG", Continent: "Asia", Latitude: 1.352, Longitude: 103.820,
			Status: "active", Tier: "primary",
			Endpoints: map[string]string{
				"grpc": "ap-southeast-1.fhe-gbdt.dev:443",
				"rest": "https://ap-southeast-1.fhe-gbdt.dev",
			},
			Features:   []string{"fhe-inference", "model-compilation", "key-management"},
			Compliance: []string{"SOC2", "MTCS", "PDPA"},
			MaxCapacity: 6000, CurrentLoad: 2100,
		},
		{
			Id: "5", Code: "ap-northeast-1", Name: "Asia Pacific (Tokyo)", Provider: "aws",
			Country: "JP", Continent: "Asia", Latitude: 35.682, Longitude: 139.759,
			Status: "active", Tier: "secondary",
			Endpoints: map[string]string{
				"grpc": "ap-northeast-1.fhe-gbdt.dev:443",
				"rest": "https://ap-northeast-1.fhe-gbdt.dev",
			},
			Features:   []string{"fhe-inference", "model-compilation"},
			Compliance: []string{"SOC2", "ISMAP"},
			MaxCapacity: 5000, CurrentLoad: 1800,
		},
	}
}

func distance(lat1, lon1, lat2, lon2 float64) float64 {
	// Haversine formula (simplified)
	const R = 6371 // Earth's radius in km
	dLat := (lat2 - lat1) * math.Pi / 180
	dLon := (lon2 - lon1) * math.Pi / 180
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180)*math.Cos(lat2*math.Pi/180)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

func marshalJSON(v interface{}) []byte {
	if v == nil {
		return []byte("[]")
	}
	b, _ := json.Marshal(v)
	return b
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8090"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newRegionsServer()
	if err != nil {
		log.Fatalf("failed to create regions server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterRegionsServiceServer(s, server)

	log.Printf("Regions Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
