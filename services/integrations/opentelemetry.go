// OpenTelemetry Integration
// Distributed tracing and observability for FHE-GBDT

package integrations

import (
	"context"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// OTelConfig contains OpenTelemetry configuration
type OTelConfig struct {
	ServiceName    string
	ServiceVersion string
	Environment    string
	Endpoint       string // OTLP endpoint (e.g., localhost:4317)
	SampleRate     float64
	Insecure       bool
}

// OTelProvider wraps OpenTelemetry functionality
type OTelProvider struct {
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer
	config         *OTelConfig
}

// NewOTelProvider creates a new OpenTelemetry provider
func NewOTelProvider(config *OTelConfig) (*OTelProvider, error) {
	if config.ServiceName == "" {
		config.ServiceName = "fhe-gbdt-serving"
	}
	if config.SampleRate == 0 {
		config.SampleRate = 1.0 // Sample all traces by default
	}
	if config.Endpoint == "" {
		config.Endpoint = "localhost:4317"
	}

	// Create OTLP exporter
	opts := []otlptracegrpc.Option{
		otlptracegrpc.WithEndpoint(config.Endpoint),
	}
	if config.Insecure {
		opts = append(opts, otlptracegrpc.WithInsecure())
	}

	exporter, err := otlptrace.New(
		context.Background(),
		otlptracegrpc.NewClient(opts...),
	)
	if err != nil {
		return nil, err
	}

	// Create resource
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(config.ServiceName),
			semconv.ServiceVersion(config.ServiceVersion),
			attribute.String("deployment.environment", config.Environment),
		),
	)
	if err != nil {
		return nil, err
	}

	// Create trace provider
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(config.SampleRate)),
	)

	// Set global trace provider
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return &OTelProvider{
		tracerProvider: tp,
		tracer:         tp.Tracer(config.ServiceName),
		config:         config,
	}, nil
}

// Shutdown gracefully shuts down the provider
func (p *OTelProvider) Shutdown(ctx context.Context) error {
	return p.tracerProvider.Shutdown(ctx)
}

// Tracer returns the tracer instance
func (p *OTelProvider) Tracer() trace.Tracer {
	return p.tracer
}

// ============================================================================
// Span Helpers
// ============================================================================

// StartSpan starts a new span
func (p *OTelProvider) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	return p.tracer.Start(ctx, name, opts...)
}

// SpanFromContext retrieves the span from context
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// ============================================================================
// FHE-GBDT Specific Tracing
// ============================================================================

// PredictionSpan creates a span for a prediction request
type PredictionSpan struct {
	span      trace.Span
	ctx       context.Context
	startTime time.Time
}

// StartPredictionSpan starts a span for prediction
func (p *OTelProvider) StartPredictionSpan(ctx context.Context, tenantID, modelID, requestID string) (*PredictionSpan, context.Context) {
	ctx, span := p.tracer.Start(ctx, "fhe.predict",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("fhe.tenant_id", tenantID),
			attribute.String("fhe.model_id", modelID),
			attribute.String("fhe.request_id", requestID),
		),
	)

	return &PredictionSpan{
		span:      span,
		ctx:       ctx,
		startTime: time.Now(),
	}, ctx
}

// AddEvent adds an event to the prediction span
func (ps *PredictionSpan) AddEvent(name string, attrs ...attribute.KeyValue) {
	ps.span.AddEvent(name, trace.WithAttributes(attrs...))
}

// SetModelVersion sets the model version attribute
func (ps *PredictionSpan) SetModelVersion(version string) {
	ps.span.SetAttributes(attribute.String("fhe.model_version", version))
}

// SetFeatureCount sets the number of features
func (ps *PredictionSpan) SetFeatureCount(count int) {
	ps.span.SetAttributes(attribute.Int("fhe.feature_count", count))
}

// SetTreeCount sets the number of trees
func (ps *PredictionSpan) SetTreeCount(count int) {
	ps.span.SetAttributes(attribute.Int("fhe.tree_count", count))
}

// End ends the prediction span
func (ps *PredictionSpan) End(success bool, err error) {
	duration := time.Since(ps.startTime)
	ps.span.SetAttributes(
		attribute.Bool("fhe.success", success),
		attribute.Int64("fhe.duration_ms", duration.Milliseconds()),
	)

	if err != nil {
		ps.span.RecordError(err)
		ps.span.SetStatus(codes.Error, err.Error())
	} else {
		ps.span.SetStatus(codes.Ok, "")
	}

	ps.span.End()
}

// StartChildSpan starts a child span for sub-operations
func (p *OTelProvider) StartChildSpan(ctx context.Context, name string) (context.Context, trace.Span) {
	return p.tracer.Start(ctx, name)
}

// ============================================================================
// FHE Operation Tracing
// ============================================================================

// TraceFHEEncryption traces FHE encryption operation
func (p *OTelProvider) TraceFHEEncryption(ctx context.Context, tenantID string, dataSize int) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "fhe.encrypt",
		trace.WithAttributes(
			attribute.String("fhe.tenant_id", tenantID),
			attribute.Int("fhe.data_size_bytes", dataSize),
			attribute.String("fhe.operation", "encryption"),
		),
	)
	return ctx, span
}

// TraceFHEDecryption traces FHE decryption operation
func (p *OTelProvider) TraceFHEDecryption(ctx context.Context, tenantID string, dataSize int) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "fhe.decrypt",
		trace.WithAttributes(
			attribute.String("fhe.tenant_id", tenantID),
			attribute.Int("fhe.data_size_bytes", dataSize),
			attribute.String("fhe.operation", "decryption"),
		),
	)
	return ctx, span
}

// TraceFHEEvaluation traces FHE tree evaluation
func (p *OTelProvider) TraceFHEEvaluation(ctx context.Context, modelID string, numTrees int) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "fhe.evaluate",
		trace.WithAttributes(
			attribute.String("fhe.model_id", modelID),
			attribute.Int("fhe.tree_count", numTrees),
			attribute.String("fhe.operation", "tree_evaluation"),
		),
	)
	return ctx, span
}

// TraceKeyLoad traces evaluation key loading
func (p *OTelProvider) TraceKeyLoad(ctx context.Context, tenantID, keyID string, keySize int64) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "fhe.load_keys",
		trace.WithAttributes(
			attribute.String("fhe.tenant_id", tenantID),
			attribute.String("fhe.key_id", keyID),
			attribute.Int64("fhe.key_size_bytes", keySize),
		),
	)
	return ctx, span
}

// TraceModelLoad traces model loading
func (p *OTelProvider) TraceModelLoad(ctx context.Context, modelID, version string) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "fhe.load_model",
		trace.WithAttributes(
			attribute.String("fhe.model_id", modelID),
			attribute.String("fhe.model_version", version),
		),
	)
	return ctx, span
}

// ============================================================================
// Database and External Service Tracing
// ============================================================================

// TraceDBQuery traces a database query
func (p *OTelProvider) TraceDBQuery(ctx context.Context, operation, table string) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "db.query",
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			attribute.String("db.system", "postgresql"),
			attribute.String("db.operation", operation),
			attribute.String("db.sql.table", table),
		),
	)
	return ctx, span
}

// TraceExternalCall traces an external service call
func (p *OTelProvider) TraceExternalCall(ctx context.Context, service, method, endpoint string) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "external.call",
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			attribute.String("peer.service", service),
			attribute.String("http.method", method),
			attribute.String("http.url", endpoint),
		),
	)
	return ctx, span
}

// TraceCacheOperation traces a cache operation
func (p *OTelProvider) TraceCacheOperation(ctx context.Context, operation, key string, hit bool) (context.Context, trace.Span) {
	ctx, span := p.tracer.Start(ctx, "cache."+operation,
		trace.WithAttributes(
			attribute.String("cache.operation", operation),
			attribute.String("cache.key", key),
			attribute.Bool("cache.hit", hit),
		),
	)
	return ctx, span
}

// ============================================================================
// Span Decorators
// ============================================================================

// WithError records an error on the span
func WithError(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
}

// WithSuccess marks the span as successful
func WithSuccess(span trace.Span) {
	span.SetStatus(codes.Ok, "")
}

// WithAttributes adds attributes to the span
func WithAttributes(span trace.Span, attrs ...attribute.KeyValue) {
	span.SetAttributes(attrs...)
}

// ============================================================================
// Context Propagation Helpers
// ============================================================================

// ExtractContext extracts trace context from carrier (e.g., HTTP headers)
func ExtractContext(ctx context.Context, carrier propagation.TextMapCarrier) context.Context {
	return otel.GetTextMapPropagator().Extract(ctx, carrier)
}

// InjectContext injects trace context into carrier
func InjectContext(ctx context.Context, carrier propagation.TextMapCarrier) {
	otel.GetTextMapPropagator().Inject(ctx, carrier)
}

// ============================================================================
// Middleware Helpers
// ============================================================================

// GRPCServerInterceptorAttrs returns common attributes for gRPC server interceptor
func GRPCServerInterceptorAttrs(method, peerAddr string) []attribute.KeyValue {
	return []attribute.KeyValue{
		semconv.RPCSystemGRPC,
		semconv.RPCMethod(method),
		semconv.NetPeerName(peerAddr),
	}
}

// HTTPServerAttrs returns common attributes for HTTP server
func HTTPServerAttrs(method, path string, statusCode int) []attribute.KeyValue {
	return []attribute.KeyValue{
		semconv.HTTPMethod(method),
		semconv.HTTPTarget(path),
		semconv.HTTPStatusCode(statusCode),
	}
}
