// Team and Organization Management Service
// Handles multi-tenant team structures, roles, and permissions

package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/teams"
)

type teamsServer struct {
	pb.UnimplementedTeamsServiceServer
	db *sql.DB
}

func newTeamsServer() (*teamsServer, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/teams?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	}

	return &teamsServer{db: db}, nil
}

// ============================================================================
// Organization Management
// ============================================================================

func (s *teamsServer) CreateOrganization(ctx context.Context, req *pb.CreateOrganizationRequest) (*pb.CreateOrganizationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating organization %s", req.Name)

	orgID := uuid.New().String()
	now := time.Now()

	// Create organization
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO organizations (id, name, slug, plan_id, settings, created_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`, orgID, req.Name, req.Slug, req.PlanId, "{}", now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create organization: %v", err)
	}

	// Add creator as owner
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO organization_members (organization_id, user_id, role, invited_by, joined_at)
		VALUES ($1, $2, 'owner', $2, $3)
	`, orgID, req.OwnerId, now)
	if err != nil {
		log.Printf("WARN: Failed to add owner: %v", err)
	}

	// Create default team
	defaultTeamID := uuid.New().String()
	_, _ = s.db.ExecContext(ctx, `
		INSERT INTO teams (id, organization_id, name, description, created_at)
		VALUES ($1, $2, 'Default', 'Default team for all members', $3)
	`, defaultTeamID, orgID, now)

	return &pb.CreateOrganizationResponse{
		Organization: &pb.Organization{
			Id:        orgID,
			Name:      req.Name,
			Slug:      req.Slug,
			PlanId:    req.PlanId,
			CreatedAt: timestamppb.New(now),
		},
	}, nil
}

func (s *teamsServer) GetOrganization(ctx context.Context, req *pb.GetOrganizationRequest) (*pb.GetOrganizationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var org pb.Organization
	var createdAt, updatedAt time.Time
	var settings []byte

	err := s.db.QueryRowContext(ctx, `
		SELECT id, name, slug, plan_id, settings, created_at, updated_at
		FROM organizations WHERE id = $1
	`, req.OrganizationId).Scan(
		&org.Id, &org.Name, &org.Slug, &org.PlanId, &settings, &createdAt, &updatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "organization not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get organization: %v", err)
	}

	org.CreatedAt = timestamppb.New(createdAt)
	org.UpdatedAt = timestamppb.New(updatedAt)

	// Get member count
	s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM organization_members WHERE organization_id = $1
	`, req.OrganizationId).Scan(&org.MemberCount)

	// Get team count
	s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM teams WHERE organization_id = $1
	`, req.OrganizationId).Scan(&org.TeamCount)

	return &pb.GetOrganizationResponse{Organization: &org}, nil
}

func (s *teamsServer) UpdateOrganization(ctx context.Context, req *pb.UpdateOrganizationRequest) (*pb.UpdateOrganizationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Updating organization %s", req.OrganizationId)

	_, err := s.db.ExecContext(ctx, `
		UPDATE organizations SET name = $1, settings = $2, updated_at = NOW()
		WHERE id = $3
	`, req.Name, req.Settings, req.OrganizationId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update organization: %v", err)
	}

	return s.GetOrganization(ctx, &pb.GetOrganizationRequest{OrganizationId: req.OrganizationId})
}

func (s *teamsServer) DeleteOrganization(ctx context.Context, req *pb.DeleteOrganizationRequest) (*pb.DeleteOrganizationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Deleting organization %s", req.OrganizationId)

	// Soft delete
	_, err := s.db.ExecContext(ctx, `
		UPDATE organizations SET deleted_at = NOW() WHERE id = $1
	`, req.OrganizationId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to delete organization: %v", err)
	}

	return &pb.DeleteOrganizationResponse{Success: true}, nil
}

// ============================================================================
// Team Management
// ============================================================================

func (s *teamsServer) CreateTeam(ctx context.Context, req *pb.CreateTeamRequest) (*pb.CreateTeamResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating team %s in org %s", req.Name, req.OrganizationId)

	teamID := uuid.New().String()
	now := time.Now()

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO teams (id, organization_id, name, description, created_by, created_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`, teamID, req.OrganizationId, req.Name, req.Description, req.CreatedBy, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create team: %v", err)
	}

	return &pb.CreateTeamResponse{
		Team: &pb.Team{
			Id:             teamID,
			OrganizationId: req.OrganizationId,
			Name:           req.Name,
			Description:    req.Description,
			CreatedAt:      timestamppb.New(now),
		},
	}, nil
}

func (s *teamsServer) GetTeam(ctx context.Context, req *pb.GetTeamRequest) (*pb.GetTeamResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var team pb.Team
	var createdAt time.Time

	err := s.db.QueryRowContext(ctx, `
		SELECT id, organization_id, name, description, created_at
		FROM teams WHERE id = $1
	`, req.TeamId).Scan(&team.Id, &team.OrganizationId, &team.Name, &team.Description, &createdAt)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "team not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get team: %v", err)
	}

	team.CreatedAt = timestamppb.New(createdAt)

	// Get member count
	s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM team_members WHERE team_id = $1
	`, req.TeamId).Scan(&team.MemberCount)

	return &pb.GetTeamResponse{Team: &team}, nil
}

func (s *teamsServer) ListTeams(ctx context.Context, req *pb.ListTeamsRequest) (*pb.ListTeamsResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT t.id, t.organization_id, t.name, t.description, t.created_at,
		       (SELECT COUNT(*) FROM team_members WHERE team_id = t.id) as member_count
		FROM teams t
		WHERE t.organization_id = $1 AND t.deleted_at IS NULL
		ORDER BY t.name
	`, req.OrganizationId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list teams: %v", err)
	}
	defer rows.Close()

	var teams []*pb.Team
	for rows.Next() {
		var team pb.Team
		var createdAt time.Time

		if err := rows.Scan(&team.Id, &team.OrganizationId, &team.Name, &team.Description,
			&createdAt, &team.MemberCount); err != nil {
			continue
		}
		team.CreatedAt = timestamppb.New(createdAt)
		teams = append(teams, &team)
	}

	return &pb.ListTeamsResponse{Teams: teams}, nil
}

func (s *teamsServer) UpdateTeam(ctx context.Context, req *pb.UpdateTeamRequest) (*pb.UpdateTeamResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Updating team %s", req.TeamId)

	_, err := s.db.ExecContext(ctx, `
		UPDATE teams SET name = $1, description = $2, updated_at = NOW()
		WHERE id = $3
	`, req.Name, req.Description, req.TeamId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update team: %v", err)
	}

	return s.GetTeam(ctx, &pb.GetTeamRequest{TeamId: req.TeamId})
}

func (s *teamsServer) DeleteTeam(ctx context.Context, req *pb.DeleteTeamRequest) (*pb.DeleteTeamResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Deleting team %s", req.TeamId)

	_, err := s.db.ExecContext(ctx, `
		UPDATE teams SET deleted_at = NOW() WHERE id = $1
	`, req.TeamId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to delete team: %v", err)
	}

	return &pb.DeleteTeamResponse{Success: true}, nil
}

// ============================================================================
// Member Management
// ============================================================================

func (s *teamsServer) InviteMember(ctx context.Context, req *pb.InviteMemberRequest) (*pb.InviteMemberResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Inviting %s to org %s", req.Email, req.OrganizationId)

	inviteID := uuid.New().String()
	inviteToken := uuid.New().String()
	now := time.Now()
	expiresAt := now.Add(7 * 24 * time.Hour)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO invitations (id, organization_id, email, role, teams, token, invited_by, created_at, expires_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`, inviteID, req.OrganizationId, req.Email, req.Role, req.TeamIds, inviteToken, req.InvitedBy, now, expiresAt)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create invitation: %v", err)
	}

	// TODO: Send invitation email

	return &pb.InviteMemberResponse{
		Invitation: &pb.Invitation{
			Id:             inviteID,
			OrganizationId: req.OrganizationId,
			Email:          req.Email,
			Role:           req.Role,
			Status:         "pending",
			CreatedAt:      timestamppb.New(now),
			ExpiresAt:      timestamppb.New(expiresAt),
		},
	}, nil
}

func (s *teamsServer) AcceptInvitation(ctx context.Context, req *pb.AcceptInvitationRequest) (*pb.AcceptInvitationResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get invitation
	var orgID, email, role string
	var teamIds []string

	err := s.db.QueryRowContext(ctx, `
		SELECT organization_id, email, role, teams FROM invitations
		WHERE token = $1 AND status = 'pending' AND expires_at > NOW()
	`, req.Token).Scan(&orgID, &email, &role, &teamIds)
	if err == sql.ErrNoRows {
		return nil, status.Error(codes.NotFound, "invalid or expired invitation")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get invitation: %v", err)
	}

	now := time.Now()

	// Add user to organization
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO organization_members (organization_id, user_id, role, joined_at)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (organization_id, user_id) DO NOTHING
	`, orgID, req.UserId, role, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to add member: %v", err)
	}

	// Add user to teams
	for _, teamID := range teamIds {
		_, _ = s.db.ExecContext(ctx, `
			INSERT INTO team_members (team_id, user_id, joined_at)
			VALUES ($1, $2, $3)
			ON CONFLICT (team_id, user_id) DO NOTHING
		`, teamID, req.UserId, now)
	}

	// Mark invitation as accepted
	_, _ = s.db.ExecContext(ctx, `
		UPDATE invitations SET status = 'accepted', accepted_at = NOW() WHERE token = $1
	`, req.Token)

	log.Printf("AUDIT: User %s joined org %s", req.UserId, orgID)

	return &pb.AcceptInvitationResponse{
		Success:        true,
		OrganizationId: orgID,
	}, nil
}

func (s *teamsServer) RemoveMember(ctx context.Context, req *pb.RemoveMemberRequest) (*pb.RemoveMemberResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Removing user %s from org %s", req.UserId, req.OrganizationId)

	// Remove from organization
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM organization_members WHERE organization_id = $1 AND user_id = $2
	`, req.OrganizationId, req.UserId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to remove member: %v", err)
	}

	// Remove from all teams in the organization
	_, _ = s.db.ExecContext(ctx, `
		DELETE FROM team_members WHERE user_id = $1 AND team_id IN (
			SELECT id FROM teams WHERE organization_id = $2
		)
	`, req.UserId, req.OrganizationId)

	return &pb.RemoveMemberResponse{Success: true}, nil
}

func (s *teamsServer) UpdateMemberRole(ctx context.Context, req *pb.UpdateMemberRoleRequest) (*pb.UpdateMemberRoleResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Updating role for user %s in org %s to %s", req.UserId, req.OrganizationId, req.Role)

	_, err := s.db.ExecContext(ctx, `
		UPDATE organization_members SET role = $1 WHERE organization_id = $2 AND user_id = $3
	`, req.Role, req.OrganizationId, req.UserId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update role: %v", err)
	}

	return &pb.UpdateMemberRoleResponse{Success: true}, nil
}

func (s *teamsServer) ListMembers(ctx context.Context, req *pb.ListMembersRequest) (*pb.ListMembersResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT om.user_id, u.email, u.name, om.role, om.joined_at
		FROM organization_members om
		JOIN users u ON om.user_id = u.id
		WHERE om.organization_id = $1
		ORDER BY om.joined_at DESC
	`, req.OrganizationId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list members: %v", err)
	}
	defer rows.Close()

	var members []*pb.Member
	for rows.Next() {
		var member pb.Member
		var joinedAt time.Time

		if err := rows.Scan(&member.UserId, &member.Email, &member.Name, &member.Role, &joinedAt); err != nil {
			continue
		}
		member.JoinedAt = timestamppb.New(joinedAt)
		members = append(members, &member)
	}

	return &pb.ListMembersResponse{Members: members}, nil
}

// ============================================================================
// Team Member Management
// ============================================================================

func (s *teamsServer) AddTeamMember(ctx context.Context, req *pb.AddTeamMemberRequest) (*pb.AddTeamMemberResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Adding user %s to team %s", req.UserId, req.TeamId)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO team_members (team_id, user_id, role, joined_at)
		VALUES ($1, $2, $3, NOW())
		ON CONFLICT (team_id, user_id) DO UPDATE SET role = $3
	`, req.TeamId, req.UserId, req.Role)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to add team member: %v", err)
	}

	return &pb.AddTeamMemberResponse{Success: true}, nil
}

func (s *teamsServer) RemoveTeamMember(ctx context.Context, req *pb.RemoveTeamMemberRequest) (*pb.RemoveTeamMemberResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Removing user %s from team %s", req.UserId, req.TeamId)

	_, err := s.db.ExecContext(ctx, `
		DELETE FROM team_members WHERE team_id = $1 AND user_id = $2
	`, req.TeamId, req.UserId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to remove team member: %v", err)
	}

	return &pb.RemoveTeamMemberResponse{Success: true}, nil
}

func (s *teamsServer) ListTeamMembers(ctx context.Context, req *pb.ListTeamMembersRequest) (*pb.ListTeamMembersResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT tm.user_id, u.email, u.name, tm.role, tm.joined_at
		FROM team_members tm
		JOIN users u ON tm.user_id = u.id
		WHERE tm.team_id = $1
		ORDER BY tm.joined_at DESC
	`, req.TeamId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list team members: %v", err)
	}
	defer rows.Close()

	var members []*pb.TeamMember
	for rows.Next() {
		var member pb.TeamMember
		var joinedAt time.Time

		if err := rows.Scan(&member.UserId, &member.Email, &member.Name, &member.Role, &joinedAt); err != nil {
			continue
		}
		member.JoinedAt = timestamppb.New(joinedAt)
		members = append(members, &member)
	}

	return &pb.ListTeamMembersResponse{Members: members}, nil
}

// ============================================================================
// Permission Checks
// ============================================================================

func (s *teamsServer) CheckPermission(ctx context.Context, req *pb.CheckPermissionRequest) (*pb.CheckPermissionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Get user's role in the organization
	var role string
	err := s.db.QueryRowContext(ctx, `
		SELECT role FROM organization_members
		WHERE organization_id = $1 AND user_id = $2
	`, req.OrganizationId, req.UserId).Scan(&role)

	if err == sql.ErrNoRows {
		return &pb.CheckPermissionResponse{Allowed: false}, nil
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to check permission: %v", err)
	}

	// Check permission based on role
	allowed := checkRolePermission(role, req.Action, req.Resource)

	return &pb.CheckPermissionResponse{
		Allowed: allowed,
		Role:    role,
	}, nil
}

func checkRolePermission(role, action, resource string) bool {
	// Role hierarchy: owner > admin > member > viewer
	permissions := map[string]map[string][]string{
		"owner": {
			"*": {"*"}, // Owner can do everything
		},
		"admin": {
			"models":  {"create", "read", "update", "delete"},
			"keys":    {"create", "read", "update", "delete"},
			"billing": {"read"},
			"members": {"create", "read", "update"},
			"teams":   {"create", "read", "update", "delete"},
		},
		"member": {
			"models":  {"create", "read", "update"},
			"keys":    {"create", "read"},
			"billing": {},
			"members": {"read"},
			"teams":   {"read"},
		},
		"viewer": {
			"models":  {"read"},
			"keys":    {"read"},
			"billing": {},
			"members": {"read"},
			"teams":   {"read"},
		},
	}

	rolePerms, ok := permissions[role]
	if !ok {
		return false
	}

	// Check wildcard
	if wildcardPerms, ok := rolePerms["*"]; ok {
		for _, p := range wildcardPerms {
			if p == "*" || p == action {
				return true
			}
		}
	}

	// Check specific resource
	resourcePerms, ok := rolePerms[resource]
	if !ok {
		return false
	}

	for _, p := range resourcePerms {
		if p == action {
			return true
		}
	}

	return false
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8087"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newTeamsServer()
	if err != nil {
		log.Fatalf("failed to create teams server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterTeamsServiceServer(s, server)

	log.Printf("Teams Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
