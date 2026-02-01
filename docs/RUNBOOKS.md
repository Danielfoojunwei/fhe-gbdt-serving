# Runbooks

This document provides operational procedures for common alerts and incidents.

## High Error Rate (HighErrorRate)

### Symptoms
- Alert: `HighErrorRate` firing
- Error rate > 0.5% for 5 minutes

### Diagnosis
1. Check Gateway logs: `kubectl logs -l app=gateway --tail=100`
2. Check Runtime logs for panics/crashes
3. Verify Registry and Keystore connectivity
4. Check if a recent deployment was made

### Mitigation
1. If recent deployment: rollback with `helm rollback fhe-gbdt`
2. If connectivity issue: restart affected pods
3. If overloaded: scale up runtime replicas

---

## P95 Latency Breach (LatencyP95Breach)

### Symptoms
- p95 latency > 100ms for 10 minutes

### Diagnosis
1. Check `runtime_queue_depth_gauge` - is the queue backing up?
2. Check stage timings in `runtime_predict_stagetime_*` metrics
3. Check batch sizes - are unusually large batches being sent?

### Mitigation
1. Scale up runtime pods
2. If StepBundle is the bottleneck, consider profile optimization
3. Check for noisy neighbors on the node

---

## High Queue Depth (HighQueueDepth)

### Symptoms
- Queue depth > 100 for 5 minutes

### Diagnosis
1. Check HPA status: is it scaling?
2. Check pod resource utilization
3. Check for pod scheduling issues

### Mitigation
1. Manually scale runtime deployment
2. Increase HPA maxReplicas
3. Check for resource quota limits

---

## Rollback Procedure

```bash
# List releases
helm history fhe-gbdt

# Rollback to previous revision
helm rollback fhe-gbdt [REVISION]

# Verify rollback
kubectl rollout status deployment/fhe-gbdt-runtime-cpu
```
