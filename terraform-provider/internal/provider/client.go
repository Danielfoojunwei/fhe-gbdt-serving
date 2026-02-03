// API Client for FHE-GBDT Provider

package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is the FHE-GBDT API client
type Client struct {
	endpoint   string
	apiKey     string
	tenantID   string
	region     string
	httpClient *http.Client
}

// NewClient creates a new API client
func NewClient(endpoint, apiKey, tenantID, region string) *Client {
	return &Client{
		endpoint: endpoint,
		apiKey:   apiKey,
		tenantID: tenantID,
		region:   region,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Request makes an API request
func (c *Client) Request(ctx context.Context, method, path string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("X-Tenant-ID", c.tenantID)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "terraform-provider-fhegbdt/0.1.0")

	if c.region != "" {
		req.Header.Set("X-Region", c.region)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// ============================================================================
// Model Operations
// ============================================================================

type Model struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	LibraryType   string            `json:"library_type"`
	Status        string            `json:"status"`
	CurrentVersion string           `json:"current_version"`
	Regions       []string          `json:"regions"`
	Labels        map[string]string `json:"labels"`
	CreatedAt     string            `json:"created_at"`
	UpdatedAt     string            `json:"updated_at"`
}

func (c *Client) CreateModel(ctx context.Context, model *Model) (*Model, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/models", model)
	if err != nil {
		return nil, err
	}

	var result Model
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) GetModel(ctx context.Context, id string) (*Model, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/models/"+id, nil)
	if err != nil {
		return nil, err
	}

	var result Model
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) UpdateModel(ctx context.Context, id string, model *Model) (*Model, error) {
	data, err := c.Request(ctx, "PUT", "/api/v1/models/"+id, model)
	if err != nil {
		return nil, err
	}

	var result Model
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) DeleteModel(ctx context.Context, id string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/models/"+id, nil)
	return err
}

func (c *Client) ListModels(ctx context.Context) ([]Model, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/models", nil)
	if err != nil {
		return nil, err
	}

	var result struct {
		Models []Model `json:"models"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return result.Models, nil
}

// ============================================================================
// Key Operations
// ============================================================================

type Key struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	KeyType     string `json:"key_type"`
	Algorithm   string `json:"algorithm"`
	Status      string `json:"status"`
	ExpiresAt   string `json:"expires_at"`
	CreatedAt   string `json:"created_at"`
}

func (c *Client) CreateKey(ctx context.Context, key *Key) (*Key, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/keys", key)
	if err != nil {
		return nil, err
	}

	var result Key
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) GetKey(ctx context.Context, id string) (*Key, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/keys/"+id, nil)
	if err != nil {
		return nil, err
	}

	var result Key
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) DeleteKey(ctx context.Context, id string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/keys/"+id, nil)
	return err
}

// ============================================================================
// Webhook Operations
// ============================================================================

type Webhook struct {
	ID      string            `json:"id"`
	Name    string            `json:"name"`
	URL     string            `json:"url"`
	Secret  string            `json:"secret"`
	Events  []string          `json:"events"`
	Headers map[string]string `json:"headers"`
	Enabled bool              `json:"enabled"`
}

func (c *Client) CreateWebhook(ctx context.Context, webhook *Webhook) (*Webhook, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/webhooks", webhook)
	if err != nil {
		return nil, err
	}

	var result Webhook
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) GetWebhook(ctx context.Context, id string) (*Webhook, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/webhooks/"+id, nil)
	if err != nil {
		return nil, err
	}

	var result Webhook
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) UpdateWebhook(ctx context.Context, id string, webhook *Webhook) (*Webhook, error) {
	data, err := c.Request(ctx, "PUT", "/api/v1/webhooks/"+id, webhook)
	if err != nil {
		return nil, err
	}

	var result Webhook
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) DeleteWebhook(ctx context.Context, id string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/webhooks/"+id, nil)
	return err
}

// ============================================================================
// Alert Operations
// ============================================================================

type Alert struct {
	ID            string   `json:"id"`
	Name          string   `json:"name"`
	ModelID       string   `json:"model_id"`
	Metric        string   `json:"metric"`
	Condition     string   `json:"condition"`
	Threshold     float64  `json:"threshold"`
	WindowMinutes int      `json:"window_minutes"`
	Channels      []string `json:"notification_channels"`
	Enabled       bool     `json:"enabled"`
}

func (c *Client) CreateAlert(ctx context.Context, alert *Alert) (*Alert, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/alerts", alert)
	if err != nil {
		return nil, err
	}

	var result Alert
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) GetAlert(ctx context.Context, id string) (*Alert, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/alerts/"+id, nil)
	if err != nil {
		return nil, err
	}

	var result Alert
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) UpdateAlert(ctx context.Context, id string, alert *Alert) (*Alert, error) {
	data, err := c.Request(ctx, "PUT", "/api/v1/alerts/"+id, alert)
	if err != nil {
		return nil, err
	}

	var result Alert
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) DeleteAlert(ctx context.Context, id string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/alerts/"+id, nil)
	return err
}

// ============================================================================
// Team Operations
// ============================================================================

type Team struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

type TeamMember struct {
	ID     string `json:"id"`
	TeamID string `json:"team_id"`
	UserID string `json:"user_id"`
	Email  string `json:"email"`
	Role   string `json:"role"`
}

func (c *Client) CreateTeam(ctx context.Context, team *Team) (*Team, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/teams", team)
	if err != nil {
		return nil, err
	}

	var result Team
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) GetTeam(ctx context.Context, id string) (*Team, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/teams/"+id, nil)
	if err != nil {
		return nil, err
	}

	var result Team
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) UpdateTeam(ctx context.Context, id string, team *Team) (*Team, error) {
	data, err := c.Request(ctx, "PUT", "/api/v1/teams/"+id, team)
	if err != nil {
		return nil, err
	}

	var result Team
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) DeleteTeam(ctx context.Context, id string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/teams/"+id, nil)
	return err
}

func (c *Client) AddTeamMember(ctx context.Context, member *TeamMember) (*TeamMember, error) {
	data, err := c.Request(ctx, "POST", "/api/v1/teams/"+member.TeamID+"/members", member)
	if err != nil {
		return nil, err
	}

	var result TeamMember
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}

func (c *Client) RemoveTeamMember(ctx context.Context, teamID, memberID string) error {
	_, err := c.Request(ctx, "DELETE", "/api/v1/teams/"+teamID+"/members/"+memberID, nil)
	return err
}

// ============================================================================
// Region Operations
// ============================================================================

type Region struct {
	Code       string   `json:"code"`
	Name       string   `json:"name"`
	Provider   string   `json:"provider"`
	Country    string   `json:"country"`
	Status     string   `json:"status"`
	Compliance []string `json:"compliance"`
}

func (c *Client) ListRegions(ctx context.Context) ([]Region, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/regions", nil)
	if err != nil {
		return nil, err
	}

	var result struct {
		Regions []Region `json:"regions"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return result.Regions, nil
}

// ============================================================================
// Usage Operations
// ============================================================================

type Usage struct {
	TotalPredictions int64   `json:"total_predictions"`
	TotalCompute     float64 `json:"total_compute_hours"`
	TotalStorage     float64 `json:"total_storage_gb"`
	PeriodStart      string  `json:"period_start"`
	PeriodEnd        string  `json:"period_end"`
}

func (c *Client) GetUsage(ctx context.Context) (*Usage, error) {
	data, err := c.Request(ctx, "GET", "/api/v1/usage", nil)
	if err != nil {
		return nil, err
	}

	var result Usage
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &result, nil
}
