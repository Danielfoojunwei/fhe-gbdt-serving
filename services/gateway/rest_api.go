// Package gateway provides REST API endpoints for FHE-GBDT
// Aligned with TenSafe's OpenAI-compatible API design
package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// RESTServer provides HTTP REST API for FHE-GBDT
type RESTServer struct {
	router      *mux.Router
	logger      *zap.Logger
	tracer      trace.Tracer
	grpcGateway *Server // Underlying gRPC gateway
	config      *RESTConfig
}

// RESTConfig holds REST API configuration
type RESTConfig struct {
	ListenAddr      string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	MaxRequestSize  int64
	EnableCORS      bool
	AllowedOrigins  []string
	RateLimitPerSec int
}

// DefaultRESTConfig returns default REST configuration
func DefaultRESTConfig() *RESTConfig {
	return &RESTConfig{
		ListenAddr:      ":8080",
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    120 * time.Second,
		MaxRequestSize:  100 * 1024 * 1024, // 100MB
		EnableCORS:      true,
		AllowedOrigins:  []string{"*"},
		RateLimitPerSec: 100,
	}
}

// NewRESTServer creates a new REST API server
func NewRESTServer(grpcGateway *Server, config *RESTConfig, logger *zap.Logger) *RESTServer {
	if config == nil {
		config = DefaultRESTConfig()
	}

	s := &RESTServer{
		router:      mux.NewRouter(),
		logger:      logger,
		tracer:      otel.Tracer("fhe-gbdt-rest"),
		grpcGateway: grpcGateway,
		config:      config,
	}

	s.setupRoutes()
	return s
}

// setupRoutes configures all REST API routes
func (s *RESTServer) setupRoutes() {
	// Middleware
	s.router.Use(s.loggingMiddleware)
	s.router.Use(s.tracingMiddleware)
	s.router.Use(s.recoveryMiddleware)
	if s.config.EnableCORS {
		s.router.Use(s.corsMiddleware)
	}

	// API v1 routes
	v1 := s.router.PathPrefix("/v1").Subrouter()
	v1.Use(s.authMiddleware)
	v1.Use(s.rateLimitMiddleware)

	// Health & Ready (no auth)
	s.router.HandleFunc("/health", s.handleHealth).Methods("GET")
	s.router.HandleFunc("/ready", s.handleReady).Methods("GET")
	s.router.HandleFunc("/v1/health", s.handleHealth).Methods("GET")
	s.router.HandleFunc("/v1/ready", s.handleReady).Methods("GET")

	// Inference endpoints
	v1.HandleFunc("/predict", s.handlePredict).Methods("POST")
	v1.HandleFunc("/batch/predict", s.handleBatchPredict).Methods("POST")

	// Model management
	v1.HandleFunc("/models", s.handleListModels).Methods("GET")
	v1.HandleFunc("/models", s.handleRegisterModel).Methods("POST")
	v1.HandleFunc("/models/{id}", s.handleGetModel).Methods("GET")
	v1.HandleFunc("/models/{id}", s.handleDeleteModel).Methods("DELETE")
	v1.HandleFunc("/models/{id}/compile", s.handleCompileModel).Methods("POST")
	v1.HandleFunc("/models/{id}/compile/status", s.handleCompileStatus).Methods("GET")

	// Key management
	v1.HandleFunc("/keys", s.handleUploadKeys).Methods("POST")
	v1.HandleFunc("/keys/{id}", s.handleGetKeyStatus).Methods("GET")
	v1.HandleFunc("/keys/{id}", s.handleRevokeKeys).Methods("DELETE")
	v1.HandleFunc("/keys/{id}/rotate", s.handleRotateKeys).Methods("POST")

	// Training endpoints
	v1.HandleFunc("/training/jobs", s.handleStartTraining).Methods("POST")
	v1.HandleFunc("/training/jobs", s.handleListTrainingJobs).Methods("GET")
	v1.HandleFunc("/training/jobs/{id}", s.handleGetTrainingJob).Methods("GET")
	v1.HandleFunc("/training/jobs/{id}", s.handleStopTraining).Methods("DELETE")
	v1.HandleFunc("/training/jobs/{id}/checkpoint", s.handleGetCheckpoint).Methods("GET")

	// Package management
	v1.HandleFunc("/packages", s.handleCreatePackage).Methods("POST")
	v1.HandleFunc("/packages/{id}", s.handleGetPackage).Methods("GET")
	v1.HandleFunc("/packages/{id}/verify", s.handleVerifyPackage).Methods("POST")
	v1.HandleFunc("/packages/{id}/extract", s.handleExtractPackage).Methods("POST")

	// Audit endpoints
	v1.HandleFunc("/audit/logs", s.handleGetAuditLogs).Methods("GET")
	v1.HandleFunc("/audit/export", s.handleExportAuditLogs).Methods("POST")
}

// Response types
type ErrorResponse struct {
	Error struct {
		Code    string            `json:"code"`
		Message string            `json:"message"`
		Details map[string]string `json:"details,omitempty"`
	} `json:"error"`
}

type SuccessResponse struct {
	Data interface{} `json:"data"`
}

// Prediction types
type PredictRequest struct {
	ModelID          string            `json:"model_id"`
	CompiledModelID  string            `json:"compiled_model_id,omitempty"`
	Profile          string            `json:"profile,omitempty"` // "latency" or "throughput"
	Ciphertext       string            `json:"ciphertext"`        // Base64 encoded
	CiphertextFormat string            `json:"ciphertext_format,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

type PredictResponse struct {
	RequestID     string         `json:"request_id"`
	ModelID       string         `json:"model_id"`
	Ciphertext    string         `json:"ciphertext"` // Base64 encoded result
	Stats         *RuntimeStats  `json:"stats,omitempty"`
	ProcessedAt   string         `json:"processed_at"`
}

type RuntimeStats struct {
	Comparisons    uint64  `json:"comparisons"`
	SchemeSwitches uint64  `json:"scheme_switches"`
	Bootstraps     uint64  `json:"bootstraps"`
	Rotations      uint64  `json:"rotations"`
	RuntimeMs      float64 `json:"runtime_ms"`
}

// Model types
type Model struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	TenantID        string            `json:"tenant_id"`
	Library         string            `json:"library"` // xgboost, lightgbm, catboost
	Status          string            `json:"status"`
	CompiledModelID string            `json:"compiled_model_id,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
	CreatedAt       string            `json:"created_at"`
	UpdatedAt       string            `json:"updated_at"`
}

type RegisterModelRequest struct {
	Name     string            `json:"name"`
	Library  string            `json:"library"`
	Model    string            `json:"model"`    // Base64 encoded model
	Metadata map[string]string `json:"metadata,omitempty"`
}

type CompileModelRequest struct {
	Profile      string            `json:"profile,omitempty"` // "latency" or "throughput"
	CryptoParams *CryptoParams     `json:"crypto_params,omitempty"`
	Options      map[string]string `json:"options,omitempty"`
}

type CryptoParams struct {
	RingDimension     int    `json:"ring_dimension,omitempty"`
	CiphertextModulus string `json:"ciphertext_modulus,omitempty"`
	SecurityLevel     string `json:"security_level,omitempty"`
}

// Training types
type TrainingJobRequest struct {
	Name            string              `json:"name"`
	DatasetPath     string              `json:"dataset_path"`
	Library         string              `json:"library"`
	Hyperparameters map[string]interface{} `json:"hyperparameters"`
	DPConfig        *DPConfig           `json:"dp_config,omitempty"`
	OutputPath      string              `json:"output_path,omitempty"`
}

type DPConfig struct {
	Enabled       bool    `json:"enabled"`
	Epsilon       float64 `json:"epsilon"`
	Delta         float64 `json:"delta"`
	NoiseType     string  `json:"noise_type,omitempty"`
	MaxGradNorm   float64 `json:"max_grad_norm,omitempty"`
}

type TrainingJob struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Status          string                 `json:"status"`
	Progress        float64                `json:"progress"`
	Metrics         map[string]float64     `json:"metrics,omitempty"`
	DPSpent         *DPSpent               `json:"dp_spent,omitempty"`
	StartedAt       string                 `json:"started_at"`
	CompletedAt     string                 `json:"completed_at,omitempty"`
	Error           string                 `json:"error,omitempty"`
}

type DPSpent struct {
	Epsilon float64 `json:"epsilon"`
	Delta   float64 `json:"delta"`
}

// Package types
type CreatePackageRequest struct {
	ModelID     string   `json:"model_id"`
	Recipients  []string `json:"recipients,omitempty"`
	DPCertificate bool   `json:"dp_certificate,omitempty"`
}

type Package struct {
	ID           string `json:"id"`
	ModelID      string `json:"model_id"`
	Status       string `json:"status"`
	DownloadURL  string `json:"download_url,omitempty"`
	Hash         string `json:"hash,omitempty"`
	CreatedAt    string `json:"created_at"`
}

// Handler implementations

func (s *RESTServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]string{
		"status": "healthy",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *RESTServer) handleReady(w http.ResponseWriter, r *http.Request) {
	// Check dependencies
	ready := true // TODO: Check gRPC gateway, registry, etc.

	if ready {
		s.writeJSON(w, http.StatusOK, map[string]string{
			"status": "ready",
			"time":   time.Now().UTC().Format(time.RFC3339),
		})
	} else {
		s.writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "not_ready",
			"time":   time.Now().UTC().Format(time.RFC3339),
		})
	}
}

func (s *RESTServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.Predict")
	defer span.End()

	var req PredictRequest
	if err := s.readJSON(r, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate request
	if req.Ciphertext == "" {
		s.writeError(w, http.StatusBadRequest, "MISSING_CIPHERTEXT", "ciphertext is required")
		return
	}

	if req.ModelID == "" && req.CompiledModelID == "" {
		s.writeError(w, http.StatusBadRequest, "MISSING_MODEL", "model_id or compiled_model_id is required")
		return
	}

	// Get tenant from context
	tenantID := getTenantFromContext(ctx)
	span.SetAttributes(attribute.String("tenant_id", tenantID))

	// TODO: Forward to gRPC gateway
	// For now, return mock response
	response := PredictResponse{
		RequestID:   getRequestIDFromContext(ctx),
		ModelID:     req.ModelID,
		Ciphertext:  "base64_encrypted_result...",
		ProcessedAt: time.Now().UTC().Format(time.RFC3339),
		Stats: &RuntimeStats{
			Comparisons:    6400,
			SchemeSwitches: 200,
			Bootstraps:     0,
			Rotations:      12,
			RuntimeMs:      62.5,
		},
	}

	s.writeJSON(w, http.StatusOK, SuccessResponse{Data: response})
}

func (s *RESTServer) handleBatchPredict(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.BatchPredict")
	defer span.End()

	// TODO: Implement batch prediction
	s.writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "batch prediction not yet implemented")
	_ = ctx
}

func (s *RESTServer) handleListModels(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.ListModels")
	defer span.End()

	// Parse pagination
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	// TODO: Fetch from registry
	models := []Model{}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"data":   models,
		"limit":  limit,
		"offset": offset,
		"total":  0,
	})
	_ = ctx
}

func (s *RESTServer) handleRegisterModel(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.RegisterModel")
	defer span.End()

	var req RegisterModelRequest
	if err := s.readJSON(r, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate
	if req.Name == "" {
		s.writeError(w, http.StatusBadRequest, "MISSING_NAME", "name is required")
		return
	}
	if req.Library == "" {
		s.writeError(w, http.StatusBadRequest, "MISSING_LIBRARY", "library is required")
		return
	}
	validLibraries := map[string]bool{"xgboost": true, "lightgbm": true, "catboost": true}
	if !validLibraries[strings.ToLower(req.Library)] {
		s.writeError(w, http.StatusBadRequest, "INVALID_LIBRARY", "library must be xgboost, lightgbm, or catboost")
		return
	}

	// TODO: Forward to registry
	model := Model{
		ID:        "model-" + generateID(),
		Name:      req.Name,
		TenantID:  getTenantFromContext(ctx),
		Library:   req.Library,
		Status:    "registered",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSON(w, http.StatusCreated, SuccessResponse{Data: model})
}

func (s *RESTServer) handleGetModel(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.GetModel")
	defer span.End()

	vars := mux.Vars(r)
	modelID := vars["id"]

	// TODO: Fetch from registry
	_ = ctx
	_ = modelID

	s.writeError(w, http.StatusNotFound, "MODEL_NOT_FOUND", "model not found")
}

func (s *RESTServer) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.DeleteModel")
	defer span.End()

	vars := mux.Vars(r)
	modelID := vars["id"]

	// TODO: Delete from registry
	_ = ctx
	_ = modelID

	w.WriteHeader(http.StatusNoContent)
}

func (s *RESTServer) handleCompileModel(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.CompileModel")
	defer span.End()

	vars := mux.Vars(r)
	modelID := vars["id"]

	var req CompileModelRequest
	if err := s.readJSON(r, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// TODO: Forward to compiler service
	_ = ctx

	s.writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"model_id":   modelID,
		"job_id":     "compile-" + generateID(),
		"status":     "pending",
		"started_at": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *RESTServer) handleCompileStatus(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.CompileStatus")
	defer span.End()

	vars := mux.Vars(r)
	modelID := vars["id"]

	// TODO: Get status from compiler
	_ = ctx
	_ = modelID

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"model_id": modelID,
		"status":   "completed",
		"progress": 100,
	})
}

func (s *RESTServer) handleUploadKeys(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.UploadKeys")
	defer span.End()

	// TODO: Forward to keystore
	_ = ctx

	s.writeJSON(w, http.StatusCreated, map[string]interface{}{
		"key_id":     "key-" + generateID(),
		"status":     "uploaded",
		"expires_at": time.Now().Add(90 * 24 * time.Hour).UTC().Format(time.RFC3339),
	})
}

func (s *RESTServer) handleGetKeyStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	keyID := vars["id"]

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"key_id": keyID,
		"status": "active",
	})
}

func (s *RESTServer) handleRevokeKeys(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	keyID := vars["id"]
	_ = keyID

	w.WriteHeader(http.StatusNoContent)
}

func (s *RESTServer) handleRotateKeys(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	keyID := vars["id"]

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"old_key_id": keyID,
		"new_key_id": "key-" + generateID(),
		"status":     "rotated",
	})
}

func (s *RESTServer) handleStartTraining(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.StartTraining")
	defer span.End()

	var req TrainingJobRequest
	if err := s.readJSON(r, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// TODO: Forward to training service
	job := TrainingJob{
		ID:        "train-" + generateID(),
		Name:      req.Name,
		Status:    "pending",
		Progress:  0,
		StartedAt: time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSON(w, http.StatusAccepted, SuccessResponse{Data: job})
	_ = ctx
}

func (s *RESTServer) handleListTrainingJobs(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"data":  []TrainingJob{},
		"total": 0,
	})
}

func (s *RESTServer) handleGetTrainingJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]
	_ = jobID

	s.writeError(w, http.StatusNotFound, "JOB_NOT_FOUND", "training job not found")
}

func (s *RESTServer) handleStopTraining(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]
	_ = jobID

	w.WriteHeader(http.StatusNoContent)
}

func (s *RESTServer) handleGetCheckpoint(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]
	_ = jobID

	s.writeError(w, http.StatusNotFound, "CHECKPOINT_NOT_FOUND", "checkpoint not found")
}

func (s *RESTServer) handleCreatePackage(w http.ResponseWriter, r *http.Request) {
	ctx, span := s.tracer.Start(r.Context(), "REST.CreatePackage")
	defer span.End()

	var req CreatePackageRequest
	if err := s.readJSON(r, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// TODO: Forward to packaging service
	pkg := Package{
		ID:        "pkg-" + generateID(),
		ModelID:   req.ModelID,
		Status:    "creating",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSON(w, http.StatusAccepted, SuccessResponse{Data: pkg})
	_ = ctx
}

func (s *RESTServer) handleGetPackage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	pkgID := vars["id"]
	_ = pkgID

	s.writeError(w, http.StatusNotFound, "PACKAGE_NOT_FOUND", "package not found")
}

func (s *RESTServer) handleVerifyPackage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	pkgID := vars["id"]

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"package_id": pkgID,
		"valid":      true,
		"checks": map[string]bool{
			"signature":      true,
			"integrity":      true,
			"policy":         true,
			"dp_certificate": true,
		},
	})
}

func (s *RESTServer) handleExtractPackage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	pkgID := vars["id"]
	_ = pkgID

	s.writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "package extraction not yet implemented")
}

func (s *RESTServer) handleGetAuditLogs(w http.ResponseWriter, r *http.Request) {
	// Parse query params
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 || limit > 1000 {
		limit = 100
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"data":  []map[string]interface{}{},
		"limit": limit,
		"total": 0,
	})
}

func (s *RESTServer) handleExportAuditLogs(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"export_id": "export-" + generateID(),
		"status":    "processing",
	})
}

// Middleware implementations

func (s *RESTServer) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		s.logger.Info("HTTP request",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Int("status", wrapped.statusCode),
			zap.Duration("duration", time.Since(start)),
			zap.String("remote_addr", r.RemoteAddr),
		)
	})
}

func (s *RESTServer) tracingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, span := s.tracer.Start(r.Context(), fmt.Sprintf("HTTP %s %s", r.Method, r.URL.Path))
		defer span.End()

		// Add request ID
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = "req-" + generateID()
		}
		ctx = context.WithValue(ctx, requestIDKey, requestID)
		w.Header().Set("X-Request-ID", requestID)

		span.SetAttributes(
			attribute.String("http.method", r.Method),
			attribute.String("http.url", r.URL.String()),
			attribute.String("request_id", requestID),
		)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *RESTServer) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				s.logger.Error("Panic recovered", zap.Any("error", err))
				s.writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "internal server error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *RESTServer) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Check if origin is allowed
		allowed := false
		for _, o := range s.config.AllowedOrigins {
			if o == "*" || o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Tenant-ID, X-Request-ID")
			w.Header().Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *RESTServer) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for health endpoints
		if strings.HasPrefix(r.URL.Path, "/health") || strings.HasPrefix(r.URL.Path, "/ready") {
			next.ServeHTTP(w, r)
			return
		}

		apiKey := r.Header.Get("X-API-Key")
		if apiKey == "" {
			apiKey = r.Header.Get("Authorization")
			if strings.HasPrefix(apiKey, "Bearer ") {
				apiKey = strings.TrimPrefix(apiKey, "Bearer ")
			}
		}

		if apiKey == "" {
			s.writeError(w, http.StatusUnauthorized, "UNAUTHORIZED", "API key required")
			return
		}

		// TODO: Validate API key and extract tenant
		tenantID := r.Header.Get("X-Tenant-ID")
		if tenantID == "" {
			tenantID = "default"
		}

		ctx := context.WithValue(r.Context(), tenantIDKey, tenantID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *RESTServer) rateLimitMiddleware(next http.Handler) http.Handler {
	// TODO: Implement proper rate limiting with token bucket
	return next
}

// Helper methods

func (s *RESTServer) readJSON(r *http.Request, v interface{}) error {
	// Limit request size
	r.Body = http.MaxBytesReader(nil, r.Body, s.config.MaxRequestSize)

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	return decoder.Decode(v)
}

func (s *RESTServer) writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (s *RESTServer) writeError(w http.ResponseWriter, status int, code, message string) {
	resp := ErrorResponse{}
	resp.Error.Code = code
	resp.Error.Message = message
	s.writeJSON(w, status, resp)
}

// ServeHTTP implements http.Handler
func (s *RESTServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

// Start starts the REST server
func (s *RESTServer) Start() error {
	server := &http.Server{
		Addr:         s.config.ListenAddr,
		Handler:      s,
		ReadTimeout:  s.config.ReadTimeout,
		WriteTimeout: s.config.WriteTimeout,
	}

	s.logger.Info("Starting REST API server", zap.String("addr", s.config.ListenAddr))
	return server.ListenAndServe()
}

// Helper types and functions

type contextKey string

const (
	requestIDKey contextKey = "request_id"
	tenantIDKey  contextKey = "tenant_id"
)

func getRequestIDFromContext(ctx context.Context) string {
	if v := ctx.Value(requestIDKey); v != nil {
		return v.(string)
	}
	return ""
}

func getTenantFromContext(ctx context.Context) string {
	if v := ctx.Value(tenantIDKey); v != nil {
		return v.(string)
	}
	return ""
}

func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}
