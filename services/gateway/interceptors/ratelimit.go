package interceptors

import (
	"context"
	"log"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// RateLimiter implements a token bucket rate limiter per tenant
type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*tokenBucket
	
	// Configuration
	rate       float64 // tokens per second
	bucketSize int     // max tokens
}

type tokenBucket struct {
	tokens     float64
	lastUpdate time.Time
}

// NewRateLimiter creates a new rate limiter
// rate: requests per second allowed
// bucketSize: burst capacity
func NewRateLimiter(rate float64, bucketSize int) *RateLimiter {
	return &RateLimiter{
		buckets:    make(map[string]*tokenBucket),
		rate:       rate,
		bucketSize: bucketSize,
	}
}

// Allow checks if a request should be allowed for the given tenant
func (rl *RateLimiter) Allow(tenantID string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	bucket, ok := rl.buckets[tenantID]
	if !ok {
		bucket = &tokenBucket{
			tokens:     float64(rl.bucketSize),
			lastUpdate: now,
		}
		rl.buckets[tenantID] = bucket
	}

	// Refill tokens based on elapsed time
	elapsed := now.Sub(bucket.lastUpdate).Seconds()
	bucket.tokens += elapsed * rl.rate
	if bucket.tokens > float64(rl.bucketSize) {
		bucket.tokens = float64(rl.bucketSize)
	}
	bucket.lastUpdate = now

	// Check if we have tokens available
	if bucket.tokens >= 1 {
		bucket.tokens--
		return true
	}

	return false
}

// RateLimitInterceptor creates a gRPC interceptor that enforces rate limits
func RateLimitInterceptor(rl *RateLimiter) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Extract tenant ID from metadata
		tenantID := "unknown"
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if keys := md.Get("x-api-key"); len(keys) > 0 {
				// API key format: <tenant_id>.<secret>
				key := keys[0]
				for i, c := range key {
					if c == '.' {
						tenantID = key[:i]
						break
					}
				}
			}
		}

		// Check rate limit
		if !rl.Allow(tenantID) {
			log.Printf("RATE_LIMIT: Tenant %s exceeded rate limit", tenantID)
			return nil, status.Error(codes.ResourceExhausted, "rate limit exceeded")
		}

		return handler(ctx, req)
	}
}

// ChainUnaryInterceptors chains multiple unary interceptors
func ChainUnaryInterceptors(interceptors ...grpc.UnaryServerInterceptor) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Build the chain from the end
		chain := handler
		for i := len(interceptors) - 1; i >= 0; i-- {
			interceptor := interceptors[i]
			next := chain
			chain = func(ctx context.Context, req interface{}) (interface{}, error) {
				return interceptor(ctx, req, info, next)
			}
		}
		return chain(ctx, req)
	}
}
