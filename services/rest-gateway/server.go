// REST Gateway Server
// Provides REST API endpoints via grpc-gateway for the FHE-GBDT platform

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// Config holds the REST gateway configuration
type Config struct {
	Port            string
	GatewayEndpoint string
	RegistryEndpoint string
	BillingEndpoint string
	MeteringEndpoint string
	AllowedOrigins  []string
}

// Server represents the REST gateway server
type Server struct {
	config         Config
	gatewayConn    *grpc.ClientConn
	registryConn   *grpc.ClientConn
	billingConn    *grpc.ClientConn
	meteringConn   *grpc.ClientConn
	router         *mux.Router
}

// NewServer creates a new REST gateway server
func NewServer(config Config) (*Server, error) {
	s := &Server{
		config: config,
		router: mux.NewRouter(),
	}

	// Connect to gRPC services
	var err error

	s.gatewayConn, err = grpc.Dial(config.GatewayEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("WARN: Could not connect to gateway: %v", err)
	}

	s.registryConn, err = grpc.Dial(config.RegistryEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("WARN: Could not connect to registry: %v", err)
	}

	s.billingConn, err = grpc.Dial(config.BillingEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("WARN: Could not connect to billing: %v", err)
	}

	s.meteringConn, err = grpc.Dial(config.MeteringEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("WARN: Could not connect to metering: %v", err)
	}

	s.setupRoutes()
	return s, nil
}

// setupRoutes configures all REST API routes
func (s *Server) setupRoutes() {
	// API versioning
	api := s.router.PathPrefix("/api/v1").Subrouter()
	api.Use(s.authMiddleware)
	api.Use(s.loggingMiddleware)
	api.Use(s.rateLimitMiddleware)

	// Health endpoints (no auth required)
	s.router.HandleFunc("/health", s.healthCheck).Methods("GET")
	s.router.HandleFunc("/ready", s.readinessCheck).Methods("GET")

	// OpenAPI spec
	s.router.HandleFunc("/openapi.json", s.serveOpenAPISpec).Methods("GET")
	s.router.HandleFunc("/openapi.yaml", s.serveOpenAPISpec).Methods("GET")

	// Models API
	api.HandleFunc("/models", s.listModels).Methods("GET")
	api.HandleFunc("/models", s.registerModel).Methods("POST")
	api.HandleFunc("/models/{modelId}", s.getModel).Methods("GET")
	api.HandleFunc("/models/{modelId}", s.deleteModel).Methods("DELETE")
	api.HandleFunc("/models/{modelId}/compile", s.compileModel).Methods("POST")
	api.HandleFunc("/models/{modelId}/compile/status", s.getCompileStatus).Methods("GET")

	// Compiled Models API
	api.HandleFunc("/compiled-models", s.listCompiledModels).Methods("GET")
	api.HandleFunc("/compiled-models/{compiledModelId}", s.getCompiledModel).Methods("GET")

	// Predictions API
	api.HandleFunc("/predict", s.predict).Methods("POST")
	api.HandleFunc("/predict/batch", s.batchPredict).Methods("POST")

	// Keys API
	api.HandleFunc("/keys", s.getKeyStatus).Methods("GET")
	api.HandleFunc("/keys/eval", s.uploadEvalKeys).Methods("POST")
	api.HandleFunc("/keys/rotate", s.rotateKeys).Methods("POST")
	api.HandleFunc("/keys/revoke", s.revokeKeys).Methods("DELETE")

	// Billing API
	api.HandleFunc("/billing/plans", s.listPlans).Methods("GET")
	api.HandleFunc("/billing/subscription", s.getSubscription).Methods("GET")
	api.HandleFunc("/billing/subscription", s.createSubscription).Methods("POST")
	api.HandleFunc("/billing/subscription", s.updateSubscription).Methods("PATCH")
	api.HandleFunc("/billing/subscription/cancel", s.cancelSubscription).Methods("POST")
	api.HandleFunc("/billing/usage", s.getUsage).Methods("GET")
	api.HandleFunc("/billing/invoices", s.listInvoices).Methods("GET")
	api.HandleFunc("/billing/invoices/{invoiceId}", s.getInvoice).Methods("GET")
	api.HandleFunc("/billing/checkout", s.createCheckoutSession).Methods("POST")

	// Metering API
	api.HandleFunc("/metering/usage", s.getUsageSummary).Methods("GET")
	api.HandleFunc("/metering/usage/timeseries", s.getUsageTimeSeries).Methods("GET")
	api.HandleFunc("/metering/quota", s.getQuotaStatus).Methods("GET")

	// Webhooks (no auth - uses webhook signature)
	s.router.HandleFunc("/webhooks/stripe", s.handleStripeWebhook).Methods("POST")
}

// Middleware

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey := r.Header.Get("X-API-Key")
		if apiKey == "" {
			apiKey = r.Header.Get("Authorization")
			if strings.HasPrefix(apiKey, "Bearer ") {
				apiKey = strings.TrimPrefix(apiKey, "Bearer ")
			}
		}

		if apiKey == "" {
			s.errorResponse(w, http.StatusUnauthorized, "API key required", "UNAUTHORIZED")
			return
		}

		// Add API key to context for downstream services
		ctx := metadata.AppendToOutgoingContext(r.Context(), "x-api-key", apiKey)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		log.Printf("REST %s %s %d %v", r.Method, r.URL.Path, wrapped.statusCode, time.Since(start))
	})
}

func (s *Server) rateLimitMiddleware(next http.Handler) http.Handler {
	// Simple rate limiting - in production use Redis-based limiter
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// TODO: Implement proper rate limiting
		next.ServeHTTP(w, r)
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Health endpoints

func (s *Server) healthCheck(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]string{
		"status": "healthy",
		"service": "rest-gateway",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) readinessCheck(w http.ResponseWriter, r *http.Request) {
	ready := true
	components := make(map[string]string)

	if s.gatewayConn != nil {
		components["gateway"] = "connected"
	} else {
		components["gateway"] = "disconnected"
		ready = false
	}

	if s.registryConn != nil {
		components["registry"] = "connected"
	} else {
		components["registry"] = "disconnected"
		ready = false
	}

	status := http.StatusOK
	statusText := "ready"
	if !ready {
		status = http.StatusServiceUnavailable
		statusText = "not_ready"
	}

	s.jsonResponse(w, status, map[string]interface{}{
		"status": statusText,
		"components": components,
	})
}

func (s *Server) serveOpenAPISpec(w http.ResponseWriter, r *http.Request) {
	// Serve OpenAPI specification
	spec, err := os.ReadFile("openapi.yaml")
	if err != nil {
		s.errorResponse(w, http.StatusNotFound, "OpenAPI spec not found", "NOT_FOUND")
		return
	}

	if strings.HasSuffix(r.URL.Path, ".json") {
		w.Header().Set("Content-Type", "application/json")
		// Convert YAML to JSON if needed
	} else {
		w.Header().Set("Content-Type", "application/x-yaml")
	}
	w.Write(spec)
}

// Models API handlers

func (s *Server) listModels(w http.ResponseWriter, r *http.Request) {
	limit := r.URL.Query().Get("limit")
	if limit == "" {
		limit = "20"
	}

	// TODO: Call registry service
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"models": []interface{}{},
		"total": 0,
	})
}

func (s *Server) registerModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name        string `json:"name"`
		LibraryType string `json:"library_type"`
		Content     string `json:"content"` // Base64 encoded
		Description string `json:"description,omitempty"`
		Tags        []string `json:"tags,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.errorResponse(w, http.StatusBadRequest, "Invalid request body", "INVALID_REQUEST")
		return
	}

	if req.Name == "" || req.LibraryType == "" || req.Content == "" {
		s.errorResponse(w, http.StatusBadRequest, "name, library_type, and content are required", "MISSING_FIELDS")
		return
	}

	// TODO: Call registry service
	s.jsonResponse(w, http.StatusCreated, map[string]interface{}{
		"model_id": "model-" + time.Now().Format("20060102150405"),
		"name": req.Name,
		"library_type": req.LibraryType,
		"status": "registered",
		"created_at": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) getModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelId := vars["modelId"]

	// TODO: Call registry service
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"model_id": modelId,
		"status": "registered",
	})
}

func (s *Server) deleteModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelId := vars["modelId"]

	// TODO: Call registry service
	log.Printf("AUDIT: Deleting model %s", modelId)
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) compileModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelId := vars["modelId"]

	var req struct {
		Profile string `json:"profile"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		req.Profile = "balanced"
	}

	// TODO: Call registry service
	s.jsonResponse(w, http.StatusAccepted, map[string]interface{}{
		"compiled_model_id": "cm-" + time.Now().Format("20060102150405"),
		"model_id": modelId,
		"profile": req.Profile,
		"status": "compiling",
	})
}

func (s *Server) getCompileStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelId := vars["modelId"]

	// TODO: Call registry service
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"model_id": modelId,
		"status": "successful",
		"plan_id": "plan-abc123",
	})
}

func (s *Server) listCompiledModels(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"compiled_models": []interface{}{},
		"total": 0,
	})
}

func (s *Server) getCompiledModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	compiledModelId := vars["compiledModelId"]

	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"compiled_model_id": compiledModelId,
		"status": "ready",
	})
}

// Predictions API handlers

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CompiledModelId string `json:"compiled_model_id"`
		Payload         string `json:"payload"` // Base64 encoded ciphertext
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.errorResponse(w, http.StatusBadRequest, "Invalid request body", "INVALID_REQUEST")
		return
	}

	start := time.Now()

	// TODO: Call gateway service

	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"result": map[string]interface{}{
			"payload": req.Payload, // Echo back for now
		},
		"latency_ms": time.Since(start).Milliseconds(),
		"request_id": fmt.Sprintf("req-%d", time.Now().UnixNano()),
	})
}

func (s *Server) batchPredict(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CompiledModelId string   `json:"compiled_model_id"`
		Payloads        []string `json:"payloads"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.errorResponse(w, http.StatusBadRequest, "Invalid request body", "INVALID_REQUEST")
		return
	}

	// TODO: Implement batch prediction
	s.jsonResponse(w, http.StatusAccepted, map[string]interface{}{
		"job_id": "batch-" + time.Now().Format("20060102150405"),
		"status": "processing",
		"total_samples": len(req.Payloads),
	})
}

// Keys API handlers

func (s *Server) getKeyStatus(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"status": "active",
		"key_id": "key-abc123",
		"uploaded_at": time.Now().Add(-24 * time.Hour).UTC().Format(time.RFC3339),
		"expires_at": time.Now().Add(30 * 24 * time.Hour).UTC().Format(time.RFC3339),
	})
}

func (s *Server) uploadEvalKeys(w http.ResponseWriter, r *http.Request) {
	// Read multipart form or JSON body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.errorResponse(w, http.StatusBadRequest, "Failed to read request body", "INVALID_REQUEST")
		return
	}

	log.Printf("AUDIT: Uploading eval keys, size: %d bytes", len(body))

	s.jsonResponse(w, http.StatusCreated, map[string]interface{}{
		"key_id": "key-" + time.Now().Format("20060102150405"),
		"status": "active",
		"expires_at": time.Now().Add(30 * 24 * time.Hour).UTC().Format(time.RFC3339),
	})
}

func (s *Server) rotateKeys(w http.ResponseWriter, r *http.Request) {
	log.Printf("AUDIT: Rotating keys")
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"new_key_id": "key-" + time.Now().Format("20060102150405"),
		"status": "active",
	})
}

func (s *Server) revokeKeys(w http.ResponseWriter, r *http.Request) {
	log.Printf("AUDIT: Revoking keys")
	w.WriteHeader(http.StatusNoContent)
}

// Billing API handlers

func (s *Server) listPlans(w http.ResponseWriter, r *http.Request) {
	plans := []map[string]interface{}{
		{
			"id": "free",
			"name": "Free",
			"price_cents": 0,
			"prediction_limit": 1000,
			"description": "For evaluation and development",
		},
		{
			"id": "pro",
			"name": "Pro",
			"price_cents": 9900,
			"prediction_limit": 100000,
			"description": "For production workloads",
		},
		{
			"id": "enterprise",
			"name": "Enterprise",
			"price_cents": -1,
			"prediction_limit": -1,
			"description": "Custom pricing for large deployments",
		},
	}
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{"plans": plans})
}

func (s *Server) getSubscription(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"subscription_id": "sub-abc123",
		"plan_id": "pro",
		"status": "active",
		"current_period_start": time.Now().AddDate(0, -1, 0).UTC().Format(time.RFC3339),
		"current_period_end": time.Now().AddDate(0, 0, 0).UTC().Format(time.RFC3339),
	})
}

func (s *Server) createSubscription(w http.ResponseWriter, r *http.Request) {
	var req struct {
		PlanId string `json:"plan_id"`
		Email  string `json:"email"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	s.jsonResponse(w, http.StatusCreated, map[string]interface{}{
		"subscription_id": "sub-" + time.Now().Format("20060102150405"),
		"plan_id": req.PlanId,
		"status": "active",
	})
}

func (s *Server) updateSubscription(w http.ResponseWriter, r *http.Request) {
	var req struct {
		PlanId string `json:"plan_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"subscription_id": "sub-abc123",
		"plan_id": req.PlanId,
		"status": "active",
	})
}

func (s *Server) cancelSubscription(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"subscription_id": "sub-abc123",
		"status": "canceled",
		"cancel_at_period_end": true,
	})
}

func (s *Server) getUsage(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"predictions_count": 42567,
		"predictions_limit": 100000,
		"usage_percentage": 42.567,
		"period_start": time.Now().AddDate(0, -1, 0).UTC().Format(time.RFC3339),
		"period_end": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) listInvoices(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"invoices": []interface{}{},
	})
}

func (s *Server) getInvoice(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	invoiceId := vars["invoiceId"]

	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"invoice_id": invoiceId,
		"status": "paid",
	})
}

func (s *Server) createCheckoutSession(w http.ResponseWriter, r *http.Request) {
	var req struct {
		PlanId     string `json:"plan_id"`
		SuccessUrl string `json:"success_url"`
		CancelUrl  string `json:"cancel_url"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	s.jsonResponse(w, http.StatusCreated, map[string]interface{}{
		"session_id": "cs_" + time.Now().Format("20060102150405"),
		"checkout_url": "https://checkout.stripe.com/pay/cs_test_xxx",
	})
}

// Metering API handlers

func (s *Server) getUsageSummary(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"summary": map[string]interface{}{
			"total_predictions": 42567,
			"total_compute_ms": 2567890,
			"avg_latency_ms": 60.3,
		},
	})
}

func (s *Server) getUsageTimeSeries(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"data_points": []interface{}{},
		"granularity": "hour",
	})
}

func (s *Server) getQuotaStatus(w http.ResponseWriter, r *http.Request) {
	s.jsonResponse(w, http.StatusOK, map[string]interface{}{
		"quotas": []map[string]interface{}{
			{
				"quota_type": "predictions_per_month",
				"limit": 100000,
				"used": 42567,
				"remaining": 57433,
				"percentage_used": 42.567,
			},
		},
	})
}

// Webhooks

func (s *Server) handleStripeWebhook(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	signature := r.Header.Get("Stripe-Signature")

	log.Printf("AUDIT: Received Stripe webhook, signature: %s", signature[:20]+"...")

	// TODO: Verify signature and process webhook
	_ = body

	w.WriteHeader(http.StatusOK)
}

// Helper methods

func (s *Server) jsonResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (s *Server) errorResponse(w http.ResponseWriter, status int, message string, code string) {
	s.jsonResponse(w, status, map[string]interface{}{
		"error": map[string]interface{}{
			"code": code,
			"message": message,
		},
	})
}

func (s *Server) Run() error {
	// Setup CORS
	c := cors.New(cors.Options{
		AllowedOrigins:   s.config.AllowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Authorization", "Content-Type", "X-API-Key"},
		AllowCredentials: true,
		MaxAge:           86400,
	})

	handler := c.Handler(s.router)

	log.Printf("REST Gateway listening on :%s", s.config.Port)
	return http.ListenAndServe(":"+s.config.Port, handler)
}

func main() {
	config := Config{
		Port:             getEnv("PORT", "8090"),
		GatewayEndpoint:  getEnv("GATEWAY_ENDPOINT", "localhost:8080"),
		RegistryEndpoint: getEnv("REGISTRY_ENDPOINT", "localhost:8081"),
		BillingEndpoint:  getEnv("BILLING_ENDPOINT", "localhost:8084"),
		MeteringEndpoint: getEnv("METERING_ENDPOINT", "localhost:8085"),
		AllowedOrigins:   strings.Split(getEnv("ALLOWED_ORIGINS", "http://localhost:3000"), ","),
	}

	server, err := NewServer(config)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	if err := server.Run(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
