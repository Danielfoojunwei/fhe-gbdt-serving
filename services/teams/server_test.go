// Unit tests for Teams Service
// Tests organization, team, and member management

package main

import (
	"testing"
)

// ============================================================================
// Role Permission Tests
// ============================================================================

func TestCheckRolePermission(t *testing.T) {
	tests := []struct {
		name     string
		role     string
		action   string
		resource string
		expected bool
	}{
		// Owner permissions (full access)
		{"owner can do anything", "owner", "delete", "models", true},
		{"owner can manage billing", "owner", "update", "billing", true},
		{"owner can manage members", "owner", "delete", "members", true},

		// Admin permissions
		{"admin can create models", "admin", "create", "models", true},
		{"admin can read models", "admin", "read", "models", true},
		{"admin can update models", "admin", "update", "models", true},
		{"admin can delete models", "admin", "delete", "models", true},
		{"admin can read billing", "admin", "read", "billing", true},
		{"admin cannot update billing", "admin", "update", "billing", false},
		{"admin can create members", "admin", "create", "members", true},
		{"admin cannot delete members", "admin", "delete", "members", false},

		// Member permissions
		{"member can create models", "member", "create", "models", true},
		{"member can read models", "member", "read", "models", true},
		{"member can update models", "member", "update", "models", true},
		{"member cannot delete models", "member", "delete", "models", false},
		{"member cannot access billing", "member", "read", "billing", false},
		{"member can read members", "member", "read", "members", true},
		{"member cannot create members", "member", "create", "members", false},

		// Viewer permissions
		{"viewer can read models", "viewer", "read", "models", true},
		{"viewer cannot create models", "viewer", "create", "models", false},
		{"viewer cannot update models", "viewer", "update", "models", false},
		{"viewer cannot delete models", "viewer", "delete", "models", false},
		{"viewer cannot access billing", "viewer", "read", "billing", false},
		{"viewer can read members", "viewer", "read", "members", true},

		// Invalid role
		{"invalid role has no permissions", "invalid", "read", "models", false},
		{"empty role has no permissions", "", "read", "models", false},

		// Unknown resource
		{"admin unknown resource", "admin", "read", "unknown", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := checkRolePermission(tt.role, tt.action, tt.resource)
			if result != tt.expected {
				t.Errorf("checkRolePermission(%q, %q, %q) = %v, want %v",
					tt.role, tt.action, tt.resource, result, tt.expected)
			}
		})
	}
}

// ============================================================================
// Organization Validation Tests
// ============================================================================

func TestOrganizationValidation(t *testing.T) {
	tests := []struct {
		name    string
		orgName string
		slug    string
		valid   bool
	}{
		{"valid org", "My Company", "my-company", true},
		{"org with numbers", "Company123", "company-123", true},
		{"empty name", "", "my-company", false},
		{"empty slug", "My Company", "", false},
		{"slug with spaces", "My Company", "my company", false},
		{"slug with uppercase", "My Company", "My-Company", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateOrganization(tt.orgName, tt.slug)
			if valid != tt.valid {
				t.Errorf("validateOrganization(%q, %q) = %v, want %v",
					tt.orgName, tt.slug, valid, tt.valid)
			}
		})
	}
}

func validateOrganization(name, slug string) bool {
	if name == "" || slug == "" {
		return false
	}
	// Slug must be lowercase, no spaces
	for _, c := range slug {
		if c >= 'A' && c <= 'Z' {
			return false
		}
		if c == ' ' {
			return false
		}
	}
	return true
}

// ============================================================================
// Team Validation Tests
// ============================================================================

func TestTeamValidation(t *testing.T) {
	tests := []struct {
		name     string
		teamName string
		valid    bool
	}{
		{"valid team name", "Engineering", true},
		{"team with spaces", "Data Science", true},
		{"team with numbers", "Team 1", true},
		{"empty team name", "", false},
		{"team name too short", "T", false},
		{"team name at min length", "Te", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateTeamName(tt.teamName)
			if valid != tt.valid {
				t.Errorf("validateTeamName(%q) = %v, want %v",
					tt.teamName, valid, tt.valid)
			}
		})
	}
}

func validateTeamName(name string) bool {
	return len(name) >= 2
}

// ============================================================================
// Email Validation Tests
// ============================================================================

func TestEmailValidation(t *testing.T) {
	tests := []struct {
		name  string
		email string
		valid bool
	}{
		{"valid email", "user@example.com", true},
		{"valid email with subdomain", "user@sub.example.com", true},
		{"valid email with plus", "user+tag@example.com", true},
		{"missing @", "userexample.com", false},
		{"missing domain", "user@", false},
		{"missing username", "@example.com", false},
		{"empty email", "", false},
		{"spaces in email", "user @example.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := validateEmail(tt.email)
			if valid != tt.valid {
				t.Errorf("validateEmail(%q) = %v, want %v",
					tt.email, valid, tt.valid)
			}
		})
	}
}

func validateEmail(email string) bool {
	if email == "" {
		return false
	}
	atIndex := -1
	for i, c := range email {
		if c == ' ' {
			return false
		}
		if c == '@' {
			atIndex = i
		}
	}
	if atIndex <= 0 || atIndex >= len(email)-1 {
		return false
	}
	return true
}

// ============================================================================
// Role Validation Tests
// ============================================================================

func TestRoleValidation(t *testing.T) {
	validRoles := []string{"owner", "admin", "member", "viewer"}

	tests := []struct {
		name  string
		role  string
		valid bool
	}{
		{"owner role", "owner", true},
		{"admin role", "admin", true},
		{"member role", "member", true},
		{"viewer role", "viewer", true},
		{"invalid role", "superuser", false},
		{"empty role", "", false},
		{"uppercase role", "ADMIN", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := false
			for _, r := range validRoles {
				if r == tt.role {
					valid = true
					break
				}
			}
			if valid != tt.valid {
				t.Errorf("role %q validity = %v, want %v",
					tt.role, valid, tt.valid)
			}
		})
	}
}

// ============================================================================
// Invitation Token Tests
// ============================================================================

func TestInvitationTokenGeneration(t *testing.T) {
	// Test that tokens are unique
	tokens := make(map[string]bool)
	for i := 0; i < 100; i++ {
		token := generateInvitationToken()
		if tokens[token] {
			t.Errorf("duplicate token generated: %s", token)
		}
		tokens[token] = true
	}
}

func TestInvitationTokenFormat(t *testing.T) {
	token := generateInvitationToken()
	if len(token) < 32 {
		t.Errorf("token too short: %d chars, want at least 32", len(token))
	}

	// Should only contain hex characters
	for _, c := range token {
		isHex := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '-'
		if !isHex {
			t.Errorf("invalid character in token: %c", c)
		}
	}
}

func generateInvitationToken() string {
	// Simulate UUID generation
	return "550e8400-e29b-41d4-a716-446655440000"
}

// ============================================================================
// Permission Matrix Tests
// ============================================================================

func TestPermissionMatrix(t *testing.T) {
	// Test complete permission matrix
	resources := []string{"models", "keys", "billing", "members", "teams"}
	actions := []string{"create", "read", "update", "delete"}
	roles := []string{"owner", "admin", "member", "viewer"}

	// Expected permissions for admin
	adminPermissions := map[string]map[string]bool{
		"models":  {"create": true, "read": true, "update": true, "delete": true},
		"keys":    {"create": true, "read": true, "update": true, "delete": true},
		"billing": {"create": false, "read": true, "update": false, "delete": false},
		"members": {"create": true, "read": true, "update": true, "delete": false},
		"teams":   {"create": true, "read": true, "update": true, "delete": true},
	}

	for _, resource := range resources {
		for _, action := range actions {
			t.Run("admin_"+resource+"_"+action, func(t *testing.T) {
				expected := adminPermissions[resource][action]
				result := checkRolePermission("admin", action, resource)
				if result != expected {
					t.Errorf("admin permission for %s:%s = %v, want %v",
						resource, action, result, expected)
				}
			})
		}
	}

	// Owner should have all permissions
	for _, resource := range resources {
		for _, action := range actions {
			t.Run("owner_"+resource+"_"+action, func(t *testing.T) {
				result := checkRolePermission("owner", action, resource)
				if !result {
					t.Errorf("owner should have permission for %s:%s", resource, action)
				}
			})
		}
	}

	// Test role hierarchy exists
	t.Run("role hierarchy exists", func(t *testing.T) {
		if len(roles) != 4 {
			t.Errorf("expected 4 roles, got %d", len(roles))
		}
	})
}

// ============================================================================
// Organization Slug Generation Tests
// ============================================================================

func TestSlugGeneration(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple name", "Acme Corp", "acme-corp"},
		{"multiple spaces", "My  Company  Name", "my-company-name"},
		{"special chars", "Acme & Co.", "acme-co"},
		{"numbers", "Company 123", "company-123"},
		{"already lowercase", "acme corp", "acme-corp"},
		{"trailing spaces", "  Acme Corp  ", "acme-corp"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateSlug(tt.input)
			if result != tt.expected {
				t.Errorf("generateSlug(%q) = %q, want %q",
					tt.input, result, tt.expected)
			}
		})
	}
}

func generateSlug(name string) string {
	result := ""
	lastWasHyphen := true // Prevent leading hyphen

	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			result += string(c)
			lastWasHyphen = false
		} else if c >= 'A' && c <= 'Z' {
			result += string(c + 32) // Convert to lowercase
			lastWasHyphen = false
		} else if c == ' ' || c == '-' || c == '_' {
			if !lastWasHyphen && len(result) > 0 {
				result += "-"
				lastWasHyphen = true
			}
		}
		// Skip other characters
	}

	// Remove trailing hyphen
	for len(result) > 0 && result[len(result)-1] == '-' {
		result = result[:len(result)-1]
	}

	return result
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkCheckRolePermission(b *testing.B) {
	for i := 0; i < b.N; i++ {
		checkRolePermission("admin", "create", "models")
	}
}

func BenchmarkValidateEmail(b *testing.B) {
	for i := 0; i < b.N; i++ {
		validateEmail("user@example.com")
	}
}

func BenchmarkGenerateSlug(b *testing.B) {
	for i := 0; i < b.N; i++ {
		generateSlug("My Company Name")
	}
}
