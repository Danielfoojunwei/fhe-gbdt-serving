package db

import (
	"context"
	"database/sql"
	"fmt"
	"os"

	_ "github.com/lib/pq"
)

// Store provides database operations for the registry
type Store struct {
	db *sql.DB
}

// Model represents a registered model
type Model struct {
	ID               string
	TenantID         string
	Name             string
	LibraryType      string
	ModelContentPath string
}

// CompiledModel represents a compiled model
type CompiledModel struct {
	ID              string
	ModelID         string
	Profile         string
	PlanID          string
	PlanContentPath string
	Status          string
	ErrorMessage    sql.NullString
}

// NewStore creates a new database store
func NewStore() (*Store, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		return nil, fmt.Errorf("DATABASE_URL environment variable not set")
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &Store{db: db}, nil
}

// Close closes the database connection
func (s *Store) Close() error {
	return s.db.Close()
}

// CreateModel inserts a new model record
func (s *Store) CreateModel(ctx context.Context, tenantID, name, libraryType, contentPath string) (string, error) {
	var id string
	err := s.db.QueryRowContext(ctx,
		`INSERT INTO models (tenant_id, name, library_type, model_content_path) 
		 VALUES ($1, $2, $3, $4) RETURNING id`,
		tenantID, name, libraryType, contentPath,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("failed to create model: %w", err)
	}
	return id, nil
}

// GetModel retrieves a model by ID
func (s *Store) GetModel(ctx context.Context, modelID string) (*Model, error) {
	var m Model
	err := s.db.QueryRowContext(ctx,
		`SELECT id, tenant_id, name, library_type, model_content_path 
		 FROM models WHERE id = $1`,
		modelID,
	).Scan(&m.ID, &m.TenantID, &m.Name, &m.LibraryType, &m.ModelContentPath)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get model: %w", err)
	}
	return &m, nil
}

// CreateCompiledModel inserts a new compiled model record
func (s *Store) CreateCompiledModel(ctx context.Context, modelID, profile, planID, planPath string) (string, error) {
	var id string
	err := s.db.QueryRowContext(ctx,
		`INSERT INTO compiled_models (model_id, profile, plan_id, plan_content_path, status) 
		 VALUES ($1, $2, $3, $4, 'pending') RETURNING id`,
		modelID, profile, planID, planPath,
	).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("failed to create compiled model: %w", err)
	}
	return id, nil
}

// UpdateCompiledModelStatus updates the compilation status
func (s *Store) UpdateCompiledModelStatus(ctx context.Context, compiledID, status, errorMsg string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE compiled_models SET status = $1, error_message = $2 WHERE id = $3`,
		status, sql.NullString{String: errorMsg, Valid: errorMsg != ""}, compiledID,
	)
	return err
}

// GetCompiledModel retrieves a compiled model by ID
func (s *Store) GetCompiledModel(ctx context.Context, compiledID string) (*CompiledModel, error) {
	var cm CompiledModel
	err := s.db.QueryRowContext(ctx,
		`SELECT id, model_id, profile, plan_id, plan_content_path, status, error_message 
		 FROM compiled_models WHERE id = $1`,
		compiledID,
	).Scan(&cm.ID, &cm.ModelID, &cm.Profile, &cm.PlanID, &cm.PlanContentPath, &cm.Status, &cm.ErrorMessage)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get compiled model: %w", err)
	}
	return &cm, nil
}

// EnsureTenant creates tenant if not exists
func (s *Store) EnsureTenant(ctx context.Context, tenantID string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO tenants (tenant_id) VALUES ($1) ON CONFLICT (tenant_id) DO NOTHING`,
		tenantID,
	)
	return err
}
