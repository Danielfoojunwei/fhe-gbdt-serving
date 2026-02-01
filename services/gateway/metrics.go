package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	requestTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "gateway_request_total",
		Help: "Total number of requests to the gateway",
	}, []string{"method", "status"})

	requestLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "gateway_request_latency_ms",
		Help:    "Request latency in milliseconds",
		Buckets: prometheus.ExponentialBuckets(1, 2, 15), // 1ms to 16s
	}, []string{"method", "profile"})

	requestErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "gateway_request_errors_total",
		Help: "Total number of request errors",
	}, []string{"method", "code"})
)

func RecordRequest(method string, status string, latencyMs float64, profile string) {
	requestTotal.WithLabelValues(method, status).Inc()
	requestLatency.WithLabelValues(method, profile).Observe(latencyMs)
}

func RecordError(method string, code string) {
	requestErrors.WithLabelValues(method, code).Inc()
}
