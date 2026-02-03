// Batch Inference Service
// Handles large-scale batch predictions with FHE

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/batch"
)

type batchServer struct {
	pb.UnimplementedBatchServiceServer
	db      *sql.DB
	workers int
	jobs    sync.Map
}

func newBatchServer() (*batchServer, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/batch?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	workers := 4
	if w := os.Getenv("BATCH_WORKERS"); w != "" {
		fmt.Sscanf(w, "%d", &workers)
	}

	return &batchServer{
		db:      db,
		workers: workers,
	}, nil
}

// BatchJob represents a batch prediction job
type BatchJob struct {
	ID           string
	TenantID     string
	ModelID      string
	VersionID    string
	InputURL     string
	OutputURL    string
	InputFormat  string
	OutputFormat string
	Status       string
	Progress     float32
	TotalRecords int64
	Processed    int64
	Failed       int64
	Error        string
	Config       *pb.BatchConfig
	CreatedAt    time.Time
	StartedAt    *time.Time
	CompletedAt  *time.Time
}

// ============================================================================
// Job Management
// ============================================================================

func (s *batchServer) CreateBatchJob(ctx context.Context, req *pb.CreateBatchJobRequest) (*pb.CreateBatchJobResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating batch job for tenant %s, model %s", req.TenantId, req.ModelId)

	jobID := uuid.New().String()
	now := time.Now()

	// Validate input URL/data
	if req.InputUrl == "" && len(req.InputData) == 0 {
		return nil, status.Error(codes.InvalidArgument, "either input_url or input_data must be provided")
	}

	// Create job record
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO batch_jobs (id, tenant_id, model_id, version_id, input_url, output_url,
		                       input_format, output_format, status, config, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', $9, $10)
	`, jobID, req.TenantId, req.ModelId, req.VersionId, req.InputUrl, req.OutputUrl,
		req.InputFormat, req.OutputFormat, marshalJSON(req.Config), now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create batch job: %v", err)
	}

	// If input data is provided directly, store it
	if len(req.InputData) > 0 {
		_, err = s.db.ExecContext(ctx, `
			INSERT INTO batch_job_input (job_id, data) VALUES ($1, $2)
		`, jobID, req.InputData)
		if err != nil {
			log.Printf("WARN: Failed to store input data: %v", err)
		}
	}

	// Start processing asynchronously
	go s.processJob(jobID)

	return &pb.CreateBatchJobResponse{
		JobId:     jobID,
		Status:    "pending",
		CreatedAt: timestamppb.New(now),
	}, nil
}

func (s *batchServer) GetBatchJob(ctx context.Context, req *pb.GetBatchJobRequest) (*pb.GetBatchJobResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var job pb.BatchJob
	var createdAt time.Time
	var startedAt, completedAt sql.NullTime
	var configJSON []byte
	var errorMsg sql.NullString

	err := s.db.QueryRowContext(ctx, `
		SELECT id, tenant_id, model_id, version_id, input_url, output_url, input_format,
		       output_format, status, progress, total_records, processed_records,
		       failed_records, error_message, config, created_at, started_at, completed_at
		FROM batch_jobs WHERE id = $1
	`, req.JobId).Scan(&job.Id, &job.TenantId, &job.ModelId, &job.VersionId, &job.InputUrl,
		&job.OutputUrl, &job.InputFormat, &job.OutputFormat, &job.Status, &job.Progress,
		&job.TotalRecords, &job.ProcessedRecords, &job.FailedRecords, &errorMsg,
		&configJSON, &createdAt, &startedAt, &completedAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "job not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get job: %v", err)
	}

	job.CreatedAt = timestamppb.New(createdAt)
	if startedAt.Valid {
		job.StartedAt = timestamppb.New(startedAt.Time)
	}
	if completedAt.Valid {
		job.CompletedAt = timestamppb.New(completedAt.Time)
	}
	if errorMsg.Valid {
		job.ErrorMessage = errorMsg.String
	}
	json.Unmarshal(configJSON, &job.Config)

	return &pb.GetBatchJobResponse{Job: &job}, nil
}

func (s *batchServer) ListBatchJobs(ctx context.Context, req *pb.ListBatchJobsRequest) (*pb.ListBatchJobsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	query := `
		SELECT id, tenant_id, model_id, version_id, status, progress,
		       total_records, processed_records, failed_records, created_at, completed_at
		FROM batch_jobs
		WHERE tenant_id = $1
	`
	args := []interface{}{req.TenantId}
	argNum := 2

	if req.ModelId != "" {
		query += fmt.Sprintf(" AND model_id = $%d", argNum)
		args = append(args, req.ModelId)
		argNum++
	}
	if req.Status != "" {
		query += fmt.Sprintf(" AND status = $%d", argNum)
		args = append(args, req.Status)
		argNum++
	}

	query += " ORDER BY created_at DESC"

	limit := req.Limit
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	query += fmt.Sprintf(" LIMIT %d", limit)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list jobs: %v", err)
	}
	defer rows.Close()

	var jobs []*pb.BatchJobSummary
	for rows.Next() {
		var j pb.BatchJobSummary
		var createdAt time.Time
		var completedAt sql.NullTime

		if err := rows.Scan(&j.Id, &j.TenantId, &j.ModelId, &j.VersionId, &j.Status,
			&j.Progress, &j.TotalRecords, &j.ProcessedRecords, &j.FailedRecords,
			&createdAt, &completedAt); err != nil {
			continue
		}

		j.CreatedAt = timestamppb.New(createdAt)
		if completedAt.Valid {
			j.CompletedAt = timestamppb.New(completedAt.Time)
		}
		jobs = append(jobs, &j)
	}

	return &pb.ListBatchJobsResponse{Jobs: jobs}, nil
}

func (s *batchServer) CancelBatchJob(ctx context.Context, req *pb.CancelBatchJobRequest) (*pb.CancelBatchJobResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Cancelling batch job %s", req.JobId)

	// Check current status
	var currentStatus string
	err := s.db.QueryRowContext(ctx, `SELECT status FROM batch_jobs WHERE id = $1`, req.JobId).Scan(&currentStatus)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "job not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get job: %v", err)
	}

	if currentStatus == "completed" || currentStatus == "failed" || currentStatus == "cancelled" {
		return nil, status.Errorf(codes.FailedPrecondition, "job is already %s", currentStatus)
	}

	// Update status
	_, err = s.db.ExecContext(ctx, `
		UPDATE batch_jobs SET status = 'cancelled', completed_at = NOW() WHERE id = $1
	`, req.JobId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to cancel job: %v", err)
	}

	return &pb.CancelBatchJobResponse{Success: true}, nil
}

func (s *batchServer) GetBatchResults(ctx context.Context, req *pb.GetBatchResultsRequest) (*pb.GetBatchResultsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Check job status
	var jobStatus, outputURL string
	err := s.db.QueryRowContext(ctx, `SELECT status, output_url FROM batch_jobs WHERE id = $1`, req.JobId).Scan(&jobStatus, &outputURL)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "job not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get job: %v", err)
	}

	if jobStatus != "completed" {
		return nil, status.Errorf(codes.FailedPrecondition, "job is not completed (status: %s)", jobStatus)
	}

	// Get results
	limit := req.Limit
	if limit <= 0 || limit > 1000 {
		limit = 100
	}
	offset := req.Offset

	rows, err := s.db.QueryContext(ctx, `
		SELECT record_index, encrypted_output, error_message
		FROM batch_results
		WHERE job_id = $1
		ORDER BY record_index
		LIMIT $2 OFFSET $3
	`, req.JobId, limit, offset)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get results: %v", err)
	}
	defer rows.Close()

	var results []*pb.BatchResult
	for rows.Next() {
		var r pb.BatchResult
		var errorMsg sql.NullString

		if err := rows.Scan(&r.RecordIndex, &r.EncryptedOutput, &errorMsg); err != nil {
			continue
		}
		if errorMsg.Valid {
			r.Error = errorMsg.String
		}
		results = append(results, &r)
	}

	return &pb.GetBatchResultsResponse{
		Results:   results,
		OutputUrl: outputURL,
	}, nil
}

// ============================================================================
// Job Processing
// ============================================================================

func (s *batchServer) processJob(jobID string) {
	ctx := context.Background()

	log.Printf("INFO: Starting batch job %s", jobID)

	// Update status to running
	_, err := s.db.ExecContext(ctx, `
		UPDATE batch_jobs SET status = 'running', started_at = NOW() WHERE id = $1
	`, jobID)
	if err != nil {
		log.Printf("ERROR: Failed to update job status: %v", err)
		return
	}

	// Get job details
	var inputURL, modelID, versionID string
	var configJSON []byte
	err = s.db.QueryRowContext(ctx, `
		SELECT input_url, model_id, version_id, config FROM batch_jobs WHERE id = $1
	`, jobID).Scan(&inputURL, &modelID, &versionID, &configJSON)
	if err != nil {
		s.failJob(ctx, jobID, fmt.Sprintf("failed to get job details: %v", err))
		return
	}

	var config pb.BatchConfig
	json.Unmarshal(configJSON, &config)

	// Load input data
	records, err := s.loadInputData(ctx, jobID, inputURL)
	if err != nil {
		s.failJob(ctx, jobID, fmt.Sprintf("failed to load input data: %v", err))
		return
	}

	totalRecords := int64(len(records))
	s.db.ExecContext(ctx, `UPDATE batch_jobs SET total_records = $1 WHERE id = $2`, totalRecords, jobID)

	// Process records in batches
	batchSize := int(config.BatchSize)
	if batchSize <= 0 {
		batchSize = 100
	}

	var processed, failed int64
	var mu sync.Mutex
	var wg sync.WaitGroup

	workerChan := make(chan int, s.workers)

	for i := 0; i < len(records); i += batchSize {
		// Check if job was cancelled
		var status string
		s.db.QueryRowContext(ctx, `SELECT status FROM batch_jobs WHERE id = $1`, jobID).Scan(&status)
		if status == "cancelled" {
			log.Printf("INFO: Job %s was cancelled", jobID)
			return
		}

		end := i + batchSize
		if end > len(records) {
			end = len(records)
		}
		batch := records[i:end]

		wg.Add(1)
		workerChan <- 1

		go func(batchRecords [][]byte, startIdx int) {
			defer wg.Done()
			defer func() { <-workerChan }()

			for idx, record := range batchRecords {
				recordIdx := int64(startIdx + idx)
				result, err := s.processSingleRecord(ctx, modelID, versionID, record)

				mu.Lock()
				if err != nil {
					failed++
					s.db.ExecContext(ctx, `
						INSERT INTO batch_results (job_id, record_index, error_message)
						VALUES ($1, $2, $3)
					`, jobID, recordIdx, err.Error())
				} else {
					processed++
					s.db.ExecContext(ctx, `
						INSERT INTO batch_results (job_id, record_index, encrypted_output)
						VALUES ($1, $2, $3)
					`, jobID, recordIdx, result)
				}

				// Update progress
				progress := float32(processed+failed) / float32(totalRecords) * 100
				s.db.ExecContext(ctx, `
					UPDATE batch_jobs SET processed_records = $1, failed_records = $2, progress = $3
					WHERE id = $4
				`, processed, failed, progress, jobID)
				mu.Unlock()
			}
		}(batch, i)
	}

	wg.Wait()

	// Update final status
	completedStatus := "completed"
	if failed > 0 && processed == 0 {
		completedStatus = "failed"
	} else if failed > 0 {
		completedStatus = "completed_with_errors"
	}

	s.db.ExecContext(ctx, `
		UPDATE batch_jobs SET status = $1, progress = 100, completed_at = NOW()
		WHERE id = $2
	`, completedStatus, jobID)

	log.Printf("INFO: Completed batch job %s: %d processed, %d failed", jobID, processed, failed)
}

func (s *batchServer) loadInputData(ctx context.Context, jobID, inputURL string) ([][]byte, error) {
	// First check for inline data
	var inlineData []byte
	err := s.db.QueryRowContext(ctx, `SELECT data FROM batch_job_input WHERE job_id = $1`, jobID).Scan(&inlineData)
	if err == nil && len(inlineData) > 0 {
		// Parse inline data (assumed to be JSON array)
		var records []json.RawMessage
		if err := json.Unmarshal(inlineData, &records); err != nil {
			return nil, fmt.Errorf("failed to parse inline data: %w", err)
		}
		result := make([][]byte, len(records))
		for i, r := range records {
			result[i] = []byte(r)
		}
		return result, nil
	}

	// Otherwise load from URL
	if inputURL != "" {
		// In production, this would:
		// 1. Download from S3/GCS/HTTP URL
		// 2. Parse CSV/JSON/Parquet format
		// 3. Return records
		log.Printf("INFO: Loading data from URL: %s", inputURL)

		// For demo, return empty
		return [][]byte{}, fmt.Errorf("URL loading not implemented in demo")
	}

	return nil, fmt.Errorf("no input data provided")
}

func (s *batchServer) processSingleRecord(ctx context.Context, modelID, versionID string, record []byte) ([]byte, error) {
	// In production, this would:
	// 1. Parse the record
	// 2. Encrypt features using client's public key
	// 3. Call the FHE runtime for inference
	// 4. Return encrypted output

	// Simulate processing
	time.Sleep(10 * time.Millisecond)

	// Return mock encrypted result
	return []byte(fmt.Sprintf(`{"encrypted_output": "base64..._%s", "model_version": "%s"}`, modelID, versionID)), nil
}

func (s *batchServer) failJob(ctx context.Context, jobID, errorMsg string) {
	log.Printf("ERROR: Batch job %s failed: %s", jobID, errorMsg)
	s.db.ExecContext(ctx, `
		UPDATE batch_jobs SET status = 'failed', error_message = $1, completed_at = NOW()
		WHERE id = $2
	`, errorMsg, jobID)
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
		port = "8091"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newBatchServer()
	if err != nil {
		log.Fatalf("failed to create batch server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterBatchServiceServer(s, server)

	log.Printf("Batch Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
