// Webhooks Service
// Event-driven notifications for system events

package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/webhooks"
)

type webhooksServer struct {
	pb.UnimplementedWebhooksServiceServer
	db         *sql.DB
	httpClient *http.Client
	eventQueue chan *Event
	wg         sync.WaitGroup
}

type Event struct {
	ID        string
	Type      string
	TenantID  string
	Payload   map[string]interface{}
	CreatedAt time.Time
}

func newWebhooksServer() (*webhooksServer, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/webhooks?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	server := &webhooksServer{
		db: db,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		eventQueue: make(chan *Event, 1000),
	}

	// Start event processor
	server.wg.Add(1)
	go server.processEvents()

	return server, nil
}

// ============================================================================
// Webhook Management
// ============================================================================

func (s *webhooksServer) CreateWebhook(ctx context.Context, req *pb.CreateWebhookRequest) (*pb.CreateWebhookResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating webhook for tenant %s: %s", req.TenantId, req.Url)

	webhookID := uuid.New().String()
	secret := generateSecret()
	now := time.Now()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO webhooks (id, tenant_id, name, url, secret, events, headers, enabled, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, true, $8)
	`, webhookID, req.TenantId, req.Name, req.Url, secret,
		marshalJSON(req.Events), marshalJSON(req.Headers), now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create webhook: %v", err)
	}

	return &pb.CreateWebhookResponse{
		Webhook: &pb.Webhook{
			Id:        webhookID,
			TenantId:  req.TenantId,
			Name:      req.Name,
			Url:       req.Url,
			Secret:    secret,
			Events:    req.Events,
			Headers:   req.Headers,
			Enabled:   true,
			CreatedAt: timestamppb.New(now),
		},
	}, nil
}

func (s *webhooksServer) GetWebhook(ctx context.Context, req *pb.GetWebhookRequest) (*pb.GetWebhookResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var w pb.Webhook
	var createdAt time.Time
	var eventsJSON, headersJSON []byte

	err := s.db.QueryRowContext(ctx, `
		SELECT id, tenant_id, name, url, secret, events, headers, enabled, created_at
		FROM webhooks WHERE id = $1
	`, req.WebhookId).Scan(&w.Id, &w.TenantId, &w.Name, &w.Url, &w.Secret,
		&eventsJSON, &headersJSON, &w.Enabled, &createdAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "webhook not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get webhook: %v", err)
	}

	json.Unmarshal(eventsJSON, &w.Events)
	json.Unmarshal(headersJSON, &w.Headers)
	w.CreatedAt = timestamppb.New(createdAt)

	return &pb.GetWebhookResponse{Webhook: &w}, nil
}

func (s *webhooksServer) ListWebhooks(ctx context.Context, req *pb.ListWebhooksRequest) (*pb.ListWebhooksResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, tenant_id, name, url, events, enabled, created_at, last_triggered_at
		FROM webhooks WHERE tenant_id = $1 ORDER BY created_at DESC
	`, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list webhooks: %v", err)
	}
	defer rows.Close()

	var webhooks []*pb.WebhookSummary
	for rows.Next() {
		var w pb.WebhookSummary
		var createdAt time.Time
		var lastTriggeredAt sql.NullTime
		var eventsJSON []byte

		if err := rows.Scan(&w.Id, &w.TenantId, &w.Name, &w.Url, &eventsJSON,
			&w.Enabled, &createdAt, &lastTriggeredAt); err != nil {
			continue
		}

		json.Unmarshal(eventsJSON, &w.Events)
		w.CreatedAt = timestamppb.New(createdAt)
		if lastTriggeredAt.Valid {
			w.LastTriggeredAt = timestamppb.New(lastTriggeredAt.Time)
		}
		webhooks = append(webhooks, &w)
	}

	return &pb.ListWebhooksResponse{Webhooks: webhooks}, nil
}

func (s *webhooksServer) UpdateWebhook(ctx context.Context, req *pb.UpdateWebhookRequest) (*pb.UpdateWebhookResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Updating webhook %s", req.WebhookId)

	_, err := s.db.ExecContext(ctx, `
		UPDATE webhooks SET name = $1, url = $2, events = $3, headers = $4, enabled = $5, updated_at = NOW()
		WHERE id = $6
	`, req.Name, req.Url, marshalJSON(req.Events), marshalJSON(req.Headers), req.Enabled, req.WebhookId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update webhook: %v", err)
	}

	return s.GetWebhook(ctx, &pb.GetWebhookRequest{WebhookId: req.WebhookId})
}

func (s *webhooksServer) DeleteWebhook(ctx context.Context, req *pb.DeleteWebhookRequest) (*pb.DeleteWebhookResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Deleting webhook %s", req.WebhookId)

	result, err := s.db.ExecContext(ctx, `DELETE FROM webhooks WHERE id = $1`, req.WebhookId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to delete webhook: %v", err)
	}

	rows, _ := result.RowsAffected()
	return &pb.DeleteWebhookResponse{Success: rows > 0}, nil
}

func (s *webhooksServer) RotateSecret(ctx context.Context, req *pb.RotateSecretRequest) (*pb.RotateSecretResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Rotating secret for webhook %s", req.WebhookId)

	newSecret := generateSecret()
	_, err := s.db.ExecContext(ctx, `
		UPDATE webhooks SET secret = $1, updated_at = NOW() WHERE id = $2
	`, newSecret, req.WebhookId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to rotate secret: %v", err)
	}

	return &pb.RotateSecretResponse{NewSecret: newSecret}, nil
}

func (s *webhooksServer) TestWebhook(ctx context.Context, req *pb.TestWebhookRequest) (*pb.TestWebhookResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get webhook details
	resp, err := s.GetWebhook(ctx, &pb.GetWebhookRequest{WebhookId: req.WebhookId})
	if err != nil {
		return nil, err
	}
	webhook := resp.Webhook

	// Send test event
	testPayload := map[string]interface{}{
		"event_type": "test",
		"webhook_id": webhook.Id,
		"tenant_id":  webhook.TenantId,
		"message":    "This is a test webhook event",
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
	}

	statusCode, responseBody, err := s.deliverWebhook(webhook.Url, webhook.Secret, webhook.Headers, testPayload)
	success := err == nil && statusCode >= 200 && statusCode < 300

	return &pb.TestWebhookResponse{
		Success:      success,
		StatusCode:   int32(statusCode),
		ResponseBody: responseBody,
		ErrorMessage: errorString(err),
	}, nil
}

// ============================================================================
// Event Publishing
// ============================================================================

func (s *webhooksServer) PublishEvent(ctx context.Context, req *pb.PublishEventRequest) (*pb.PublishEventResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	eventID := uuid.New().String()
	now := time.Now()

	// Store event
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO webhook_events (id, tenant_id, event_type, payload, created_at)
		VALUES ($1, $2, $3, $4, $5)
	`, eventID, req.TenantId, req.EventType, marshalJSON(req.Payload), now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to store event: %v", err)
	}

	// Queue for delivery
	event := &Event{
		ID:        eventID,
		Type:      req.EventType,
		TenantID:  req.TenantId,
		Payload:   req.Payload,
		CreatedAt: now,
	}

	select {
	case s.eventQueue <- event:
		// Event queued
	default:
		log.Printf("WARN: Event queue full, processing synchronously")
		go s.deliverEvent(event)
	}

	return &pb.PublishEventResponse{
		EventId:     eventID,
		PublishedAt: timestamppb.New(now),
	}, nil
}

func (s *webhooksServer) GetEventDeliveries(ctx context.Context, req *pb.GetEventDeliveriesRequest) (*pb.GetEventDeliveriesResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT d.id, d.webhook_id, d.event_id, d.status, d.status_code,
		       d.response_body, d.error_message, d.attempt, d.delivered_at
		FROM webhook_deliveries d
		JOIN webhook_events e ON d.event_id = e.id
		WHERE e.tenant_id = $1 AND ($2 = '' OR d.event_id = $2)
		ORDER BY d.delivered_at DESC
		LIMIT 100
	`, req.TenantId, req.EventId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get deliveries: %v", err)
	}
	defer rows.Close()

	var deliveries []*pb.EventDelivery
	for rows.Next() {
		var d pb.EventDelivery
		var deliveredAt time.Time
		var responseBody, errorMsg sql.NullString

		if err := rows.Scan(&d.Id, &d.WebhookId, &d.EventId, &d.Status, &d.StatusCode,
			&responseBody, &errorMsg, &d.Attempt, &deliveredAt); err != nil {
			continue
		}

		if responseBody.Valid {
			d.ResponseBody = responseBody.String
		}
		if errorMsg.Valid {
			d.ErrorMessage = errorMsg.String
		}
		d.DeliveredAt = timestamppb.New(deliveredAt)
		deliveries = append(deliveries, &d)
	}

	return &pb.GetEventDeliveriesResponse{Deliveries: deliveries}, nil
}

func (s *webhooksServer) RetryDelivery(ctx context.Context, req *pb.RetryDeliveryRequest) (*pb.RetryDeliveryResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get delivery details
	var webhookID, eventID string
	err := s.db.QueryRowContext(ctx, `
		SELECT webhook_id, event_id FROM webhook_deliveries WHERE id = $1
	`, req.DeliveryId).Scan(&webhookID, &eventID)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "delivery not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get delivery: %v", err)
	}

	// Get event
	var eventType, tenantID string
	var payloadJSON []byte
	err = s.db.QueryRowContext(ctx, `
		SELECT tenant_id, event_type, payload FROM webhook_events WHERE id = $1
	`, eventID).Scan(&tenantID, &eventType, &payloadJSON)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get event: %v", err)
	}

	var payload map[string]interface{}
	json.Unmarshal(payloadJSON, &payload)

	// Retry delivery
	event := &Event{
		ID:        eventID,
		Type:      eventType,
		TenantID:  tenantID,
		Payload:   payload,
		CreatedAt: time.Now(),
	}

	go s.deliverEventToWebhook(event, webhookID)

	return &pb.RetryDeliveryResponse{Queued: true}, nil
}

// ============================================================================
// Event Processing
// ============================================================================

func (s *webhooksServer) processEvents() {
	defer s.wg.Done()

	for event := range s.eventQueue {
		s.deliverEvent(event)
	}
}

func (s *webhooksServer) deliverEvent(event *Event) {
	if s.db == nil {
		return
	}

	// Get all webhooks subscribed to this event type
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, url, secret, headers
		FROM webhooks
		WHERE tenant_id = $1 AND enabled = true AND events @> $2::jsonb
	`, event.TenantID, fmt.Sprintf(`["%s"]`, event.Type))
	if err != nil {
		log.Printf("ERROR: Failed to get webhooks: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var webhookID, url, secret string
		var headersJSON []byte

		if err := rows.Scan(&webhookID, &url, &secret, &headersJSON); err != nil {
			continue
		}

		var headers map[string]string
		json.Unmarshal(headersJSON, &headers)

		s.deliverToWebhook(event, webhookID, url, secret, headers)
	}
}

func (s *webhooksServer) deliverEventToWebhook(event *Event, webhookID string) {
	if s.db == nil {
		return
	}

	var url, secret string
	var headersJSON []byte
	err := s.db.QueryRowContext(context.Background(), `
		SELECT url, secret, headers FROM webhooks WHERE id = $1
	`, webhookID).Scan(&url, &secret, &headersJSON)
	if err != nil {
		log.Printf("ERROR: Failed to get webhook %s: %v", webhookID, err)
		return
	}

	var headers map[string]string
	json.Unmarshal(headersJSON, &headers)

	s.deliverToWebhook(event, webhookID, url, secret, headers)
}

func (s *webhooksServer) deliverToWebhook(event *Event, webhookID, url, secret string, headers map[string]string) {
	payload := map[string]interface{}{
		"event_id":   event.ID,
		"event_type": event.Type,
		"tenant_id":  event.TenantID,
		"data":       event.Payload,
		"created_at": event.CreatedAt.UTC().Format(time.RFC3339),
	}

	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		statusCode, responseBody, err := s.deliverWebhook(url, secret, headers, payload)

		deliveryStatus := "success"
		if err != nil || statusCode < 200 || statusCode >= 300 {
			deliveryStatus = "failed"
		}

		// Record delivery
		deliveryID := uuid.New().String()
		s.db.ExecContext(context.Background(), `
			INSERT INTO webhook_deliveries (id, webhook_id, event_id, status, status_code,
			                               response_body, error_message, attempt, delivered_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
		`, deliveryID, webhookID, event.ID, deliveryStatus, statusCode, responseBody, errorString(err), attempt)

		// Update last triggered
		s.db.ExecContext(context.Background(), `
			UPDATE webhooks SET last_triggered_at = NOW() WHERE id = $1
		`, webhookID)

		if deliveryStatus == "success" {
			log.Printf("INFO: Delivered event %s to webhook %s", event.ID, webhookID)
			return
		}

		// Exponential backoff for retries
		if attempt < maxRetries {
			backoff := time.Duration(attempt*attempt) * time.Second
			log.Printf("WARN: Webhook delivery failed, retrying in %v", backoff)
			time.Sleep(backoff)
		}
	}

	log.Printf("ERROR: Failed to deliver event %s to webhook %s after %d attempts", event.ID, webhookID, maxRetries)
}

func (s *webhooksServer) deliverWebhook(url, secret string, headers map[string]string, payload map[string]interface{}) (int, string, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, "", fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return 0, "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "FHE-GBDT-Webhooks/1.0")

	// Add custom headers
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// Add signature
	signature := computeSignature(body, secret)
	req.Header.Set("X-Webhook-Signature", signature)
	req.Header.Set("X-Webhook-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return 0, "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return resp.StatusCode, string(responseBody), nil
}

// ============================================================================
// Helpers
// ============================================================================

func generateSecret() string {
	b := make([]byte, 32)
	for i := range b {
		b[i] = byte(time.Now().UnixNano() % 256)
	}
	return "whsec_" + hex.EncodeToString(b)
}

func computeSignature(payload []byte, secret string) string {
	h := hmac.New(sha256.New, []byte(secret))
	h.Write(payload)
	return "sha256=" + hex.EncodeToString(h.Sum(nil))
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
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
		port = "8092"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newWebhooksServer()
	if err != nil {
		log.Fatalf("failed to create webhooks server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterWebhooksServiceServer(s, server)

	log.Printf("Webhooks Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
