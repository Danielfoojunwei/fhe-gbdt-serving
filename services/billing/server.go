package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/stripe/stripe-go/v76"
	"github.com/stripe/stripe-go/v76/checkout/session"
	"github.com/stripe/stripe-go/v76/customer"
	"github.com/stripe/stripe-go/v76/invoice"
	"github.com/stripe/stripe-go/v76/subscription"
	"github.com/stripe/stripe-go/v76/webhook"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	_ "github.com/lib/pq"
	pb "github.com/fhe-gbdt-serving/proto/billing"
)

type billingServer struct {
	pb.UnimplementedBillingServiceServer
	db            *sql.DB
	stripeKey     string
	webhookSecret string
}

func newBillingServer() (*billingServer, error) {
	// Initialize Stripe
	stripeKey := os.Getenv("STRIPE_API_KEY")
	if stripeKey == "" {
		log.Printf("WARN: STRIPE_API_KEY not set, Stripe features will be disabled")
	}
	stripe.Key = stripeKey

	webhookSecret := os.Getenv("STRIPE_WEBHOOK_SECRET")

	// Connect to database
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/billing?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		log.Printf("WARN: Database not available: %v", err)
		db = nil
	} else {
		log.Printf("Connected to billing database")
	}

	return &billingServer{
		db:            db,
		stripeKey:     stripeKey,
		webhookSecret: webhookSecret,
	}, nil
}

// ============================================================================
// Plan Operations
// ============================================================================

func (s *billingServer) GetPlan(ctx context.Context, req *pb.GetPlanRequest) (*pb.GetPlanResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var plan pb.Plan
	var features json.RawMessage
	var createdAt time.Time

	err := s.db.QueryRowContext(ctx, `
		SELECT id, name, description, price_cents, currency, prediction_limit,
		       overage_price_micros, features, stripe_price_id, is_active, created_at
		FROM plans WHERE id = $1 OR name = $1
	`, req.PlanId).Scan(
		&plan.Id, &plan.Name, &plan.Description, &plan.PriceCents, &plan.Currency,
		&plan.PredictionLimit, &plan.OveragePriceMicros, &features, &plan.StripePriceId,
		&plan.IsActive, &createdAt,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "plan %s not found", req.PlanId)
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get plan: %v", err)
	}

	plan.CreatedAt = timestamppb.New(createdAt)
	// Parse features into PlanFeature list
	plan.Features = parsePlanFeatures(features)

	return &pb.GetPlanResponse{Plan: &plan}, nil
}

func (s *billingServer) ListPlans(ctx context.Context, req *pb.ListPlansRequest) (*pb.ListPlansResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	query := `
		SELECT id, name, description, price_cents, currency, prediction_limit,
		       overage_price_micros, features, stripe_price_id, is_active, created_at
		FROM plans
	`
	if !req.IncludeInactive {
		query += " WHERE is_active = true"
	}
	query += " ORDER BY price_cents ASC"

	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list plans: %v", err)
	}
	defer rows.Close()

	var plans []*pb.Plan
	for rows.Next() {
		var plan pb.Plan
		var features json.RawMessage
		var createdAt time.Time

		if err := rows.Scan(
			&plan.Id, &plan.Name, &plan.Description, &plan.PriceCents, &plan.Currency,
			&plan.PredictionLimit, &plan.OveragePriceMicros, &features, &plan.StripePriceId,
			&plan.IsActive, &createdAt,
		); err != nil {
			continue
		}
		plan.CreatedAt = timestamppb.New(createdAt)
		plan.Features = parsePlanFeatures(features)
		plans = append(plans, &plan)
	}

	return &pb.ListPlansResponse{Plans: plans}, nil
}

// ============================================================================
// Subscription Operations
// ============================================================================

func (s *billingServer) CreateSubscription(ctx context.Context, req *pb.CreateSubscriptionRequest) (*pb.CreateSubscriptionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating subscription for tenant %s, plan %s", req.TenantId, req.PlanId)

	// Check if tenant already has a subscription
	var existingID string
	err := s.db.QueryRowContext(ctx, "SELECT id FROM subscriptions WHERE tenant_id = $1", req.TenantId).Scan(&existingID)
	if err == nil {
		return nil, status.Errorf(codes.AlreadyExists, "tenant %s already has subscription %s", req.TenantId, existingID)
	}

	// Get plan details
	planResp, err := s.GetPlan(ctx, &pb.GetPlanRequest{PlanId: req.PlanId})
	if err != nil {
		return nil, err
	}
	plan := planResp.Plan

	var stripeSubID, stripeCustomerID string
	var clientSecret string

	// Create Stripe customer and subscription if Stripe is configured and plan has price
	if s.stripeKey != "" && plan.PriceCents > 0 && plan.StripePriceId != "" {
		// Create Stripe customer
		customerParams := &stripe.CustomerParams{
			Email: stripe.String(req.Email),
			Metadata: map[string]string{
				"tenant_id": req.TenantId,
			},
		}
		cust, err := customer.New(customerParams)
		if err != nil {
			log.Printf("ERROR: Failed to create Stripe customer: %v", err)
		} else {
			stripeCustomerID = cust.ID

			// Create subscription
			subParams := &stripe.SubscriptionParams{
				Customer: stripe.String(cust.ID),
				Items: []*stripe.SubscriptionItemsParams{
					{Price: stripe.String(plan.StripePriceId)},
				},
				PaymentBehavior: stripe.String("default_incomplete"),
			}
			subParams.AddExpand("latest_invoice.payment_intent")

			sub, err := subscription.New(subParams)
			if err != nil {
				log.Printf("ERROR: Failed to create Stripe subscription: %v", err)
			} else {
				stripeSubID = sub.ID
				if sub.LatestInvoice != nil && sub.LatestInvoice.PaymentIntent != nil {
					clientSecret = sub.LatestInvoice.PaymentIntent.ClientSecret
				}
			}
		}
	}

	// Create subscription in database
	subID := uuid.New().String()
	now := time.Now()
	periodEnd := now.AddDate(0, 1, 0) // 1 month from now

	subscriptionStatus := "active"
	if plan.PriceCents > 0 && clientSecret != "" {
		subscriptionStatus = "pending" // Awaiting payment
	}

	metadataJSON, _ := json.Marshal(req.Metadata)

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO subscriptions (id, tenant_id, plan_id, status, stripe_subscription_id,
		                          stripe_customer_id, current_period_start, current_period_end, metadata)
		VALUES ($1, $2, (SELECT id FROM plans WHERE id = $3 OR name = $3), $4, $5, $6, $7, $8, $9)
	`, subID, req.TenantId, req.PlanId, subscriptionStatus, nullString(stripeSubID),
		nullString(stripeCustomerID), now, periodEnd, metadataJSON)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create subscription: %v", err)
	}

	// Initialize usage record for this period
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO usage_records (tenant_id, subscription_id, period_start, period_end, predictions_count)
		VALUES ($1, $2, $3, $4, 0)
	`, req.TenantId, subID, now, periodEnd)
	if err != nil {
		log.Printf("WARN: Failed to create initial usage record: %v", err)
	}

	log.Printf("AUDIT: Created subscription %s for tenant %s", subID, req.TenantId)

	return &pb.CreateSubscriptionResponse{
		Subscription: &pb.Subscription{
			Id:                   subID,
			TenantId:             req.TenantId,
			PlanId:               req.PlanId,
			Status:               pb.SubscriptionStatus(pb.SubscriptionStatus_value["SUBSCRIPTION_STATUS_"+toUpperStatus(subscriptionStatus)]),
			StripeSubscriptionId: stripeSubID,
			StripeCustomerId:     stripeCustomerID,
			CurrentPeriodStart:   timestamppb.New(now),
			CurrentPeriodEnd:     timestamppb.New(periodEnd),
			CreatedAt:            timestamppb.New(now),
			Metadata:             req.Metadata,
		},
		ClientSecret: clientSecret,
	}, nil
}

func (s *billingServer) GetSubscription(ctx context.Context, req *pb.GetSubscriptionRequest) (*pb.GetSubscriptionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var query string
	var arg string

	switch id := req.Identifier.(type) {
	case *pb.GetSubscriptionRequest_SubscriptionId:
		query = "SELECT * FROM subscription_details WHERE id = $1"
		arg = id.SubscriptionId
	case *pb.GetSubscriptionRequest_TenantId:
		query = "SELECT * FROM subscription_details WHERE tenant_id = $1"
		arg = id.TenantId
	default:
		return nil, status.Error(codes.InvalidArgument, "must provide subscription_id or tenant_id")
	}

	var sub pb.Subscription
	var periodStart, periodEnd, canceledAt, createdAt, updatedAt sql.NullTime
	var planFeatures json.RawMessage

	err := s.db.QueryRowContext(ctx, query, arg).Scan(
		&sub.Id, &sub.TenantId, &sub.Status, &sub.StripeSubscriptionId, &sub.StripeCustomerId,
		&periodStart, &periodEnd, &canceledAt, nil, &createdAt, &updatedAt,
		&sub.PlanId, nil, nil, nil, nil, &planFeatures,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "subscription not found")
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get subscription: %v", err)
	}

	if periodStart.Valid {
		sub.CurrentPeriodStart = timestamppb.New(periodStart.Time)
	}
	if periodEnd.Valid {
		sub.CurrentPeriodEnd = timestamppb.New(periodEnd.Time)
	}
	if canceledAt.Valid {
		sub.CanceledAt = timestamppb.New(canceledAt.Time)
	}
	if createdAt.Valid {
		sub.CreatedAt = timestamppb.New(createdAt.Time)
	}
	if updatedAt.Valid {
		sub.UpdatedAt = timestamppb.New(updatedAt.Time)
	}

	return &pb.GetSubscriptionResponse{Subscription: &sub}, nil
}

func (s *billingServer) UpdateSubscription(ctx context.Context, req *pb.UpdateSubscriptionRequest) (*pb.UpdateSubscriptionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Updating subscription %s to plan %s", req.SubscriptionId, req.NewPlanId)

	// Get current subscription
	subResp, err := s.GetSubscription(ctx, &pb.GetSubscriptionRequest{
		Identifier: &pb.GetSubscriptionRequest_SubscriptionId{SubscriptionId: req.SubscriptionId},
	})
	if err != nil {
		return nil, err
	}

	// Update in Stripe if applicable
	if s.stripeKey != "" && subResp.Subscription.StripeSubscriptionId != "" {
		// Get new plan's Stripe price ID
		planResp, err := s.GetPlan(ctx, &pb.GetPlanRequest{PlanId: req.NewPlanId})
		if err != nil {
			return nil, err
		}

		if planResp.Plan.StripePriceId != "" {
			params := &stripe.SubscriptionParams{
				ProrationBehavior: stripe.String("create_prorations"),
			}
			if !req.Prorate {
				params.ProrationBehavior = stripe.String("none")
			}
			_, err := subscription.Update(subResp.Subscription.StripeSubscriptionId, params)
			if err != nil {
				log.Printf("ERROR: Failed to update Stripe subscription: %v", err)
			}
		}
	}

	// Update in database
	_, err = s.db.ExecContext(ctx, `
		UPDATE subscriptions SET plan_id = (SELECT id FROM plans WHERE id = $1 OR name = $1)
		WHERE id = $2
	`, req.NewPlanId, req.SubscriptionId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update subscription: %v", err)
	}

	return s.GetSubscription(ctx, &pb.GetSubscriptionRequest{
		Identifier: &pb.GetSubscriptionRequest_SubscriptionId{SubscriptionId: req.SubscriptionId},
	}).(*pb.GetSubscriptionResponse), nil
}

func (s *billingServer) CancelSubscription(ctx context.Context, req *pb.CancelSubscriptionRequest) (*pb.CancelSubscriptionResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Canceling subscription %s, reason: %s", req.SubscriptionId, req.Reason)

	// Get subscription
	subResp, err := s.GetSubscription(ctx, &pb.GetSubscriptionRequest{
		Identifier: &pb.GetSubscriptionRequest_SubscriptionId{SubscriptionId: req.SubscriptionId},
	})
	if err != nil {
		return nil, err
	}

	// Cancel in Stripe
	if s.stripeKey != "" && subResp.Subscription.StripeSubscriptionId != "" {
		params := &stripe.SubscriptionParams{
			CancelAtPeriodEnd: stripe.Bool(req.CancelAtPeriodEnd),
		}
		_, err := subscription.Update(subResp.Subscription.StripeSubscriptionId, params)
		if err != nil {
			log.Printf("ERROR: Failed to cancel Stripe subscription: %v", err)
		}
	}

	// Update database
	newStatus := "canceled"
	if req.CancelAtPeriodEnd {
		newStatus = "active" // Still active until period end
	}

	_, err = s.db.ExecContext(ctx, `
		UPDATE subscriptions SET status = $1, cancel_at_period_end = $2, canceled_at = NOW()
		WHERE id = $3
	`, newStatus, req.CancelAtPeriodEnd, req.SubscriptionId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to cancel subscription: %v", err)
	}

	// Audit log
	_, _ = s.db.ExecContext(ctx, `
		INSERT INTO billing_audit_log (tenant_id, action, resource_type, resource_id, new_value, actor)
		VALUES ($1, 'cancel', 'subscription', $2, $3, 'api')
	`, subResp.Subscription.TenantId, req.SubscriptionId, fmt.Sprintf(`{"reason": "%s"}`, req.Reason))

	return &pb.CancelSubscriptionResponse{
		Subscription: subResp.Subscription,
	}, nil
}

// ============================================================================
// Usage Operations
// ============================================================================

func (s *billingServer) GetUsage(ctx context.Context, req *pb.GetUsageRequest) (*pb.GetUsageResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var usage pb.Usage
	var periodStart, periodEnd, updatedAt time.Time
	var predLimit int64
	var overagePriceMicros int64

	err := s.db.QueryRowContext(ctx, `
		SELECT u.id, u.tenant_id, u.subscription_id, u.period_start, u.period_end,
		       u.predictions_count, p.prediction_limit, p.overage_price_micros,
		       u.compute_time_ms, u.data_transfer_bytes, u.updated_at
		FROM usage_records u
		JOIN subscriptions s ON u.subscription_id = s.id
		JOIN plans p ON s.plan_id = p.id
		WHERE u.tenant_id = $1 AND NOW() BETWEEN u.period_start AND u.period_end
	`, req.TenantId).Scan(
		&usage.Id, &usage.TenantId, &usage.SubscriptionId, &periodStart, &periodEnd,
		&usage.PredictionsCount, &predLimit, &overagePriceMicros,
		&usage.ComputeTimeMs, &usage.DataTransferBytes, &updatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "no usage found for tenant %s", req.TenantId)
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get usage: %v", err)
	}

	usage.PeriodStart = timestamppb.New(periodStart)
	usage.PeriodEnd = timestamppb.New(periodEnd)
	usage.PredictionsLimit = predLimit
	usage.UpdatedAt = timestamppb.New(updatedAt)

	// Calculate overage
	if predLimit > 0 && usage.PredictionsCount > predLimit {
		usage.OverageCount = usage.PredictionsCount - predLimit
		usage.OverageCostCents = (usage.OverageCount * overagePriceMicros) / 10000
	}

	// Calculate usage percentage
	var usagePercentage float64
	if predLimit > 0 {
		usagePercentage = float64(usage.PredictionsCount) / float64(predLimit) * 100
	}

	// Get current plan
	planResp, _ := s.GetPlan(ctx, &pb.GetPlanRequest{PlanId: usage.SubscriptionId})

	return &pb.GetUsageResponse{
		Usage:           &usage,
		CurrentPlan:     planResp.GetPlan(),
		UsagePercentage: usagePercentage,
	}, nil
}

func (s *billingServer) RecordUsage(ctx context.Context, req *pb.RecordUsageRequest) (*pb.RecordUsageResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	// Idempotency check
	if req.IdempotencyKey != "" {
		var exists bool
		err := s.db.QueryRowContext(ctx, `
			SELECT EXISTS(SELECT 1 FROM usage_events WHERE tenant_id = $1 AND idempotency_key = $2)
		`, req.TenantId, req.IdempotencyKey).Scan(&exists)
		if err == nil && exists {
			// Return current usage
			return s.getUpdatedUsageResponse(ctx, req.TenantId)
		}
	}

	// Record the event
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO usage_events (tenant_id, idempotency_key, predictions_count, compute_time_ms, data_transfer_bytes)
		VALUES ($1, $2, $3, $4, $5)
	`, req.TenantId, nullString(req.IdempotencyKey), req.PredictionsCount, req.ComputeTimeMs, req.DataTransferBytes)
	if err != nil {
		log.Printf("ERROR: Failed to record usage event: %v", err)
	}

	// Update aggregate usage
	_, err = s.db.ExecContext(ctx, `
		UPDATE usage_records SET
			predictions_count = predictions_count + $1,
			compute_time_ms = compute_time_ms + $2,
			data_transfer_bytes = data_transfer_bytes + $3
		WHERE tenant_id = $4 AND NOW() BETWEEN period_start AND period_end
	`, req.PredictionsCount, req.ComputeTimeMs, req.DataTransferBytes, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update usage: %v", err)
	}

	return s.getUpdatedUsageResponse(ctx, req.TenantId)
}

func (s *billingServer) getUpdatedUsageResponse(ctx context.Context, tenantID string) (*pb.RecordUsageResponse, error) {
	usageResp, err := s.GetUsage(ctx, &pb.GetUsageRequest{TenantId: tenantID})
	if err != nil {
		return nil, err
	}

	var warningMsg string
	var limitExceeded bool

	if usageResp.UsagePercentage >= 100 {
		limitExceeded = true
		warningMsg = "Usage limit exceeded. Overage charges apply."
	} else if usageResp.UsagePercentage >= 80 {
		warningMsg = fmt.Sprintf("Warning: %.1f%% of usage limit consumed", usageResp.UsagePercentage)
	}

	return &pb.RecordUsageResponse{
		UpdatedUsage:   usageResp.Usage,
		LimitExceeded:  limitExceeded,
		WarningMessage: warningMsg,
	}, nil
}

// ============================================================================
// Invoice Operations
// ============================================================================

func (s *billingServer) GetInvoice(ctx context.Context, req *pb.GetInvoiceRequest) (*pb.GetInvoiceResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	var inv pb.Invoice
	var periodStart, periodEnd, dueDate, paidAt, createdAt sql.NullTime
	var lineItemsJSON json.RawMessage

	err := s.db.QueryRowContext(ctx, `
		SELECT id, tenant_id, subscription_id, stripe_invoice_id, status, currency,
		       subtotal_cents, tax_cents, total_cents, amount_paid_cents, amount_due_cents,
		       line_items, period_start, period_end, due_date, paid_at,
		       hosted_invoice_url, pdf_url, created_at
		FROM invoices WHERE id = $1
	`, req.InvoiceId).Scan(
		&inv.Id, &inv.TenantId, &inv.SubscriptionId, &inv.StripeInvoiceId, &inv.Status,
		&inv.Currency, &inv.SubtotalCents, &inv.TaxCents, &inv.TotalCents,
		&inv.AmountPaidCents, &inv.AmountDueCents, &lineItemsJSON, &periodStart, &periodEnd,
		&dueDate, &paidAt, &inv.HostedInvoiceUrl, &inv.PdfUrl, &createdAt,
	)
	if err == sql.ErrNoRows {
		return nil, status.Errorf(codes.NotFound, "invoice %s not found", req.InvoiceId)
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get invoice: %v", err)
	}

	if periodStart.Valid {
		inv.PeriodStart = timestamppb.New(periodStart.Time)
	}
	if periodEnd.Valid {
		inv.PeriodEnd = timestamppb.New(periodEnd.Time)
	}
	if dueDate.Valid {
		inv.DueDate = timestamppb.New(dueDate.Time)
	}
	if paidAt.Valid {
		inv.PaidAt = timestamppb.New(paidAt.Time)
	}
	if createdAt.Valid {
		inv.CreatedAt = timestamppb.New(createdAt.Time)
	}

	// Parse line items
	var lineItems []pb.InvoiceLineItem
	if err := json.Unmarshal(lineItemsJSON, &lineItems); err == nil {
		for i := range lineItems {
			inv.LineItems = append(inv.LineItems, &lineItems[i])
		}
	}

	return &pb.GetInvoiceResponse{Invoice: &inv}, nil
}

func (s *billingServer) ListInvoices(ctx context.Context, req *pb.ListInvoicesRequest) (*pb.ListInvoicesResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	limit := int(req.Limit)
	if limit <= 0 || limit > 100 {
		limit = 20
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, tenant_id, subscription_id, stripe_invoice_id, status, currency,
		       total_cents, amount_due_cents, period_start, period_end, created_at
		FROM invoices
		WHERE tenant_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`, req.TenantId, limit)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list invoices: %v", err)
	}
	defer rows.Close()

	var invoices []*pb.Invoice
	for rows.Next() {
		var inv pb.Invoice
		var periodStart, periodEnd, createdAt sql.NullTime

		if err := rows.Scan(
			&inv.Id, &inv.TenantId, &inv.SubscriptionId, &inv.StripeInvoiceId, &inv.Status,
			&inv.Currency, &inv.TotalCents, &inv.AmountDueCents, &periodStart, &periodEnd, &createdAt,
		); err != nil {
			continue
		}

		if periodStart.Valid {
			inv.PeriodStart = timestamppb.New(periodStart.Time)
		}
		if periodEnd.Valid {
			inv.PeriodEnd = timestamppb.New(periodEnd.Time)
		}
		if createdAt.Valid {
			inv.CreatedAt = timestamppb.New(createdAt.Time)
		}
		invoices = append(invoices, &inv)
	}

	return &pb.ListInvoicesResponse{Invoices: invoices}, nil
}

func (s *billingServer) CreateInvoice(ctx context.Context, req *pb.CreateInvoiceRequest) (*pb.CreateInvoiceResponse, error) {
	if s.db == nil {
		return nil, status.Error(codes.Unavailable, "database not available")
	}

	log.Printf("AUDIT: Creating invoice for tenant %s", req.TenantId)

	// Get usage for the period
	usageResp, err := s.GetUsage(ctx, &pb.GetUsageRequest{TenantId: req.TenantId})
	if err != nil {
		return nil, err
	}

	usage := usageResp.Usage
	plan := usageResp.CurrentPlan

	// Build line items
	lineItems := []map[string]interface{}{
		{
			"description":      fmt.Sprintf("%s Plan - Monthly", plan.Name),
			"quantity":         1,
			"unit_price_cents": plan.PriceCents,
			"amount_cents":     plan.PriceCents,
		},
	}

	subtotal := plan.PriceCents

	// Add overage if applicable
	if req.IncludeOverage && usage.OverageCount > 0 {
		overageItem := map[string]interface{}{
			"description":      fmt.Sprintf("Overage: %d predictions", usage.OverageCount),
			"quantity":         usage.OverageCount,
			"unit_price_cents": plan.OveragePriceMicros / 10000, // Convert microdollars to cents
			"amount_cents":     usage.OverageCostCents,
		}
		lineItems = append(lineItems, overageItem)
		subtotal += usage.OverageCostCents
	}

	lineItemsJSON, _ := json.Marshal(lineItems)

	// Create invoice in database
	invID := uuid.New().String()
	invStatus := "draft"
	if req.AutoFinalize {
		invStatus = "open"
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO invoices (id, tenant_id, subscription_id, status, currency, subtotal_cents,
		                     total_cents, amount_due_cents, line_items, period_start, period_end)
		VALUES ($1, $2, $3, $4, 'USD', $5, $5, $5, $6, $7, $8)
	`, invID, req.TenantId, req.SubscriptionId, invStatus, subtotal, lineItemsJSON,
		req.PeriodStart.AsTime(), req.PeriodEnd.AsTime())
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create invoice: %v", err)
	}

	// Create in Stripe if configured
	if s.stripeKey != "" {
		subResp, _ := s.GetSubscription(ctx, &pb.GetSubscriptionRequest{
			Identifier: &pb.GetSubscriptionRequest_TenantId{TenantId: req.TenantId},
		})
		if subResp != nil && subResp.Subscription.StripeCustomerId != "" {
			params := &stripe.InvoiceParams{
				Customer: stripe.String(subResp.Subscription.StripeCustomerId),
			}
			stripeInv, err := invoice.New(params)
			if err != nil {
				log.Printf("ERROR: Failed to create Stripe invoice: %v", err)
			} else {
				_, _ = s.db.ExecContext(ctx, `
					UPDATE invoices SET stripe_invoice_id = $1, hosted_invoice_url = $2
					WHERE id = $3
				`, stripeInv.ID, stripeInv.HostedInvoiceURL, invID)
			}
		}
	}

	return s.GetInvoice(ctx, &pb.GetInvoiceRequest{InvoiceId: invID})
}

// ============================================================================
// Checkout Operations
// ============================================================================

func (s *billingServer) CreateCheckoutSession(ctx context.Context, req *pb.CreateCheckoutSessionRequest) (*pb.CreateCheckoutSessionResponse, error) {
	if s.stripeKey == "" {
		return nil, status.Error(codes.FailedPrecondition, "Stripe not configured")
	}

	// Get plan
	planResp, err := s.GetPlan(ctx, &pb.GetPlanRequest{PlanId: req.PlanId})
	if err != nil {
		return nil, err
	}

	if planResp.Plan.StripePriceId == "" {
		return nil, status.Errorf(codes.InvalidArgument, "plan %s not available for checkout", req.PlanId)
	}

	params := &stripe.CheckoutSessionParams{
		Mode: stripe.String(string(stripe.CheckoutSessionModeSubscription)),
		LineItems: []*stripe.CheckoutSessionLineItemParams{
			{
				Price:    stripe.String(planResp.Plan.StripePriceId),
				Quantity: stripe.Int64(1),
			},
		},
		SuccessURL: stripe.String(req.SuccessUrl),
		CancelURL:  stripe.String(req.CancelUrl),
		Metadata: map[string]string{
			"tenant_id": req.TenantId,
			"plan_id":   req.PlanId,
		},
	}

	if req.CustomerEmail != "" {
		params.CustomerEmail = stripe.String(req.CustomerEmail)
	}

	sess, err := session.New(params)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create checkout session: %v", err)
	}

	return &pb.CreateCheckoutSessionResponse{
		SessionId:   sess.ID,
		CheckoutUrl: sess.URL,
		ExpiresAt:   timestamppb.New(time.Unix(sess.ExpiresAt, 0)),
	}, nil
}

// ============================================================================
// Webhook Operations
// ============================================================================

func (s *billingServer) HandleWebhook(ctx context.Context, req *pb.HandleWebhookRequest) (*pb.HandleWebhookResponse, error) {
	if s.webhookSecret == "" {
		return nil, status.Error(codes.FailedPrecondition, "webhook secret not configured")
	}

	event, err := webhook.ConstructEvent(req.Payload, req.Signature, s.webhookSecret)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid webhook signature: %v", err)
	}

	log.Printf("AUDIT: Processing webhook event %s of type %s", event.ID, event.Type)

	// Idempotency check
	if s.db != nil {
		var exists bool
		err := s.db.QueryRowContext(ctx, `
			SELECT EXISTS(SELECT 1 FROM webhook_events WHERE stripe_event_id = $1)
		`, event.ID).Scan(&exists)
		if err == nil && exists {
			return &pb.HandleWebhookResponse{
				Processed: true,
				EventType: string(event.Type),
				EventId:   event.ID,
				Message:   "Event already processed",
			}, nil
		}
	}

	// Process event
	switch event.Type {
	case "checkout.session.completed":
		// Create subscription after successful checkout
		var sess stripe.CheckoutSession
		if err := json.Unmarshal(event.Data.Raw, &sess); err == nil {
			tenantID := sess.Metadata["tenant_id"]
			planID := sess.Metadata["plan_id"]
			if tenantID != "" && planID != "" {
				_, _ = s.CreateSubscription(ctx, &pb.CreateSubscriptionRequest{
					TenantId: tenantID,
					PlanId:   planID,
					Email:    sess.CustomerEmail,
				})
			}
		}

	case "invoice.paid":
		var inv stripe.Invoice
		if err := json.Unmarshal(event.Data.Raw, &inv); err == nil {
			_, _ = s.db.ExecContext(ctx, `
				UPDATE invoices SET status = 'paid', paid_at = NOW(), amount_paid_cents = total_cents
				WHERE stripe_invoice_id = $1
			`, inv.ID)
		}

	case "customer.subscription.deleted":
		var sub stripe.Subscription
		if err := json.Unmarshal(event.Data.Raw, &sub); err == nil {
			_, _ = s.db.ExecContext(ctx, `
				UPDATE subscriptions SET status = 'canceled', canceled_at = NOW()
				WHERE stripe_subscription_id = $1
			`, sub.ID)
		}
	}

	// Record webhook event
	if s.db != nil {
		_, _ = s.db.ExecContext(ctx, `
			INSERT INTO webhook_events (stripe_event_id, event_type, payload, processed, processed_at)
			VALUES ($1, $2, $3, true, NOW())
		`, event.ID, event.Type, event.Data.Raw)
	}

	return &pb.HandleWebhookResponse{
		Processed: true,
		EventType: string(event.Type),
		EventId:   event.ID,
		Message:   "Event processed successfully",
	}, nil
}

// ============================================================================
// Helper Functions
// ============================================================================

func parsePlanFeatures(featuresJSON json.RawMessage) []*pb.PlanFeature {
	var features map[string]interface{}
	if err := json.Unmarshal(featuresJSON, &features); err != nil {
		return nil
	}

	var result []*pb.PlanFeature
	for name, value := range features {
		pf := &pb.PlanFeature{Name: name}
		switch v := value.(type) {
		case bool:
			pf.Enabled = v
		case string:
			pf.Value = v
			pf.Enabled = true
		case float64:
			pf.Value = fmt.Sprintf("%v", v)
			pf.Enabled = true
		}
		result = append(result, pf)
	}
	return result
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

func toUpperStatus(s string) string {
	switch s {
	case "active":
		return "ACTIVE"
	case "canceled":
		return "CANCELED"
	case "past_due":
		return "PAST_DUE"
	case "trialing":
		return "TRIALING"
	case "paused":
		return "PAUSED"
	default:
		return "UNSPECIFIED"
	}
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8084"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	server, err := newBillingServer()
	if err != nil {
		log.Fatalf("failed to create billing server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterBillingServiceServer(s, server)

	log.Printf("Billing Service listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
