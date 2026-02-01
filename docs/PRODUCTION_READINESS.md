# Production Readiness Checklist

This checklist is CI-enforced. All items must be checked before release.

## Security Gates
- [x] mTLS enabled in cluster (config in `deploy/helm/values.yaml`)
- [x] authz enforced: tenant → model → compiled_model (`auth/auth.go`)
- [x] Secrets not in repo (verified by git-secrets scan)
- [x] Keystore encryption at rest verified (`keystore/crypto.go`)
- [x] SAST scan passing
- [x] Dependency scan passing
- [x] Container scan passing
- [x] SBOM generated (`sbom.spdx.json`)

## Correctness Gates
- [x] e2e regression passing for XGBoost
- [x] e2e regression passing for LightGBM
- [x] e2e regression passing for CatBoost
- [x] Comparator monotonicity tests passing (`test_metamorphic.py`)
- [x] Plan determinism verified (content-addressed plan_id)

## Performance Gates
- [x] Benchmark runner produces stable results (std dev < 10%)
- [x] No >10% regression vs baselines without explicit approval
- [x] Stage timing indicates StepBundle amortization works
- [x] Baselines stored in `bench/baselines.json`

## Reliability Gates
- [x] Load test meets SLO targets on reference env
- [x] Soak test passes without leaks or tail blowups
- [x] Error rate < 0.1% under load
- [x] Queue depth < 100 under normal load

## Operability Gates
- [x] Grafana dashboards wired (`dashboards/grafana/slo.json`)
- [x] AlertManager alerts configured (`alerts/alertmanager.yml`)
- [x] Runbooks exist (`docs/RUNBOOKS.md`)
- [x] Incident template exists (`docs/INCIDENTS.md`)
- [x] Canary deployment tested
- [x] Rollback procedure tested

---

**Last Updated**: 2026-02-01
**Release Version**: 1.0.0
**Approved By**: [Pending CI validation]
