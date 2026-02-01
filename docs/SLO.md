# Service Level Objectives (SLOs)

This document defines the Service Level Indicators (SLIs), Objectives (SLOs), and Error Budget policies for the FHE-GBDT Inference API.

## Primary SLIs

### Predict Latency
| Profile     | Metric                          | p50 Target | p95 Target | p99 Target |
|-------------|----------------------------------|------------|------------|------------|
| `latency`   | `runtime_predict_latency_ms`     | < 50ms     | < 100ms    | < 200ms    |
| `throughput`| `runtime_predict_latency_ms`     | < 500ms    | < 1000ms   | < 2000ms   |

### Availability
- **SLI**: `1 - (gateway_request_errors_total / gateway_request_total)`
- **Target**: 99.9% monthly

### Error Rate
- **SLI**: `gateway_request_errors_total{code=~"5.."} / gateway_request_total`
- **Target**: < 0.1%

## Secondary SLIs

### Tail Latency
- **SLI**: `histogram_quantile(0.999, runtime_predict_latency_ms_bucket)`
- **Target**: < 5x p95

### Saturation
- **SLI**: CPU/GPU/Memory utilization on runtime pods
- **Target**: < 80% sustained

### Queue Depth
- **SLI**: `runtime_queue_depth_gauge`
- **Target**: < 100 pending requests

## Crypto SLIs

| Metric                          | Target (per request) |
|---------------------------------|----------------------|
| `runtime_rotations_total`       | Minimize (tracked)   |
| `runtime_scheme_switches_total` | < 2 per tree         |
| `runtime_bootstraps_total`      | < 1 per request      |

## Error Budget Policy

- **Monthly Budget**: 0.1% of requests can fail or miss latency SLO.
- **Burn Rate Alert**: If >50% of budget consumed in 1 hour, page on-call.
- **Freeze Policy**: If budget exhausted, halt non-critical deployments until recovered.

## Alert Thresholds

| Alert Name             | Condition                                      | Severity |
|------------------------|------------------------------------------------|----------|
| `HighErrorRate`        | error_rate > 0.5% for 5m                       | Critical |
| `LatencyP95Breach`     | p95 > target for 10m                           | Warning  |
| `LatencyP99Breach`     | p99 > target for 5m                            | Critical |
| `HighQueueDepth`       | queue_depth > 100 for 5m                       | Warning  |
| `CPUSaturation`        | cpu_usage > 90% for 10m                        | Warning  |
