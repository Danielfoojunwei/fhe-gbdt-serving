package main

import (
	"context"
	"time"

	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	MaxPayloadBytes     = 64 * 1024 * 1024 // 64MB
	DefaultQPSLimit     = 100
	DefaultBurstLimit   = 200
	RequestTimeout      = 30 * time.Second
)

// Per-tenant rate limiters
var tenantLimiters = make(map[string]*rate.Limiter)

func GetTenantLimiter(tenantID string) *rate.Limiter {
	if limiter, ok := tenantLimiters[tenantID]; ok {
		return limiter
	}
	limiter := rate.NewLimiter(DefaultQPSLimit, DefaultBurstLimit)
	tenantLimiters[tenantID] = limiter
	return limiter
}

func RateLimitInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		tenant := ctx.Value("tenant")
		if tenant == nil {
			return handler(ctx, req)
		}
		
		tenantID := tenant.(*TenantContext).TenantID
		limiter := GetTenantLimiter(tenantID)
		
		if !limiter.Allow() {
			return nil, status.Errorf(codes.ResourceExhausted, "rate limit exceeded for tenant %s", tenantID)
		}
		
		return handler(ctx, req)
	}
}

func TimeoutInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		ctx, cancel := context.WithTimeout(ctx, RequestTimeout)
		defer cancel()
		return handler(ctx, req)
	}
}
