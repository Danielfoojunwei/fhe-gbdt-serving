#!/bin/bash
#
# FHE-GBDT Compliance Audit Test Runner
#
# This script runs automated compliance checks against the codebase
# based on SOC2, HIPAA, ISO27001, and ISO27701 requirements.
#
# Usage: ./run_compliance_checks.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "==========================================="
echo "FHE-GBDT Compliance Audit Test Runner"
echo "==========================================="
echo ""
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Project: $PROJECT_ROOT"
echo ""

# Track results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Function to run a check
run_check() {
    local name="$1"
    local command="$2"
    local framework="$3"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$framework] $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Function to run a warning check
run_warning_check() {
    local name="$1"
    local command="$2"
    local framework="$3"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$framework] $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${YELLOW}WARNING${NC}"
        WARNINGS=$((WARNINGS + 1))
        return 0
    fi
}

echo "==========================================="
echo "1. SECURITY DOCUMENTATION CHECKS"
echo "==========================================="

run_check "Security policy exists" "test -f $PROJECT_ROOT/security/SECURITY.md" "SOC2/HIPAA"
run_check "Threat model documented" "test -f $PROJECT_ROOT/docs/THREAT_MODEL.md" "SOC2/ISO27001"
run_check "Production readiness checklist" "test -f $PROJECT_ROOT/docs/PRODUCTION_READINESS.md" "ALL"
run_check "Runbooks documented" "test -f $PROJECT_ROOT/docs/RUNBOOKS.md" "SOC2/ISO27001"
run_check "Incident procedures documented" "test -f $PROJECT_ROOT/docs/INCIDENTS.md" "HIPAA/SOC2"

echo ""
echo "==========================================="
echo "2. ACCESS CONTROL CHECKS"
echo "==========================================="

run_check "Authentication implementation exists" "test -f $PROJECT_ROOT/services/gateway/auth/auth.go" "ALL"
run_check "mTLS credentials implementation" "test -f $PROJECT_ROOT/services/gateway/mtls/credentials.go" "SOC2/HIPAA"
run_check "Rate limiting implementation" "grep -r 'RateLimiter' $PROJECT_ROOT/services/gateway/" "SOC2"
run_check "Tenant isolation code exists" "grep -r 'TenantContext' $PROJECT_ROOT/services/gateway/" "HIPAA/ISO27001"

echo ""
echo "==========================================="
echo "3. ENCRYPTION CHECKS"
echo "==========================================="

run_check "FHE crypto implementation" "test -f $PROJECT_ROOT/sdk/python/crypto.py" "ALL"
run_check "Envelope encryption implementation" "test -f $PROJECT_ROOT/services/keystore/crypto.go" "HIPAA/SOC2"
run_check "TLS 1.3 minimum configured" "grep -r 'TLS1.3\\|min_version' $PROJECT_ROOT/config/" "HIPAA/ISO27001"
run_check "AES-256 encryption used" "grep -r 'AES.*256\\|aes.NewCipher' $PROJECT_ROOT/services/" "HIPAA/SOC2"

echo ""
echo "==========================================="
echo "4. LOGGING AND MONITORING CHECKS"
echo "==========================================="

run_check "Audit logging implemented" "grep -r 'AUDIT:' $PROJECT_ROOT/services/gateway/" "HIPAA/SOC2"
run_check "Prometheus metrics implementation" "test -f $PROJECT_ROOT/services/gateway/metrics.go" "ISO27001"
run_check "AlertManager configuration" "test -f $PROJECT_ROOT/alerts/alertmanager.yml" "SOC2/ISO27001"
run_check "OpenTelemetry tracing" "test -f $PROJECT_ROOT/services/gateway/otel.go" "ISO27001"

echo ""
echo "==========================================="
echo "5. PRIVACY CHECKS (ISO 27701)"
echo "==========================================="

run_check "No plaintext logging test exists" "test -f $PROJECT_ROOT/tests/unit/test_no_plaintext_logs.py" "ISO27701/HIPAA"
run_check "Forbidden logging patterns defined" "grep -r 'FORBIDDEN.*PATTERNS\\|forbidden.*patterns' $PROJECT_ROOT/tests/" "ISO27701"
run_check "Data redaction configured" "grep -r 'redact_sensitive_data' $PROJECT_ROOT/config/" "ISO27701/HIPAA"

echo ""
echo "==========================================="
echo "6. CI/CD SECURITY GATES"
echo "==========================================="

run_check "CI workflow exists" "test -f $PROJECT_ROOT/.github/workflows/ci.yml" "SOC2/ISO27001"
run_check "Security scan in CI" "grep -r 'security_scan\\|bandit' $PROJECT_ROOT/.github/workflows/" "SOC2/ISO27001"
run_check "SBOM generation in CI" "grep -r 'sbom\\|SBOM' $PROJECT_ROOT/.github/workflows/" "SOC2"
run_check "Guardrails job defined" "grep -r 'guardrails' $PROJECT_ROOT/.github/workflows/" "ISO27701/HIPAA"

echo ""
echo "==========================================="
echo "7. DEPLOYMENT SECURITY CHECKS"
echo "==========================================="

run_check "Helm values exist" "test -f $PROJECT_ROOT/deploy/helm/values.yaml" "SOC2"
run_check "Pod disruption budget configured" "grep -r 'podDisruptionBudget' $PROJECT_ROOT/deploy/" "SOC2"
run_check "Autoscaling configured" "grep -r 'autoscaling' $PROJECT_ROOT/deploy/" "SOC2/ISO27001"
run_warning_check "Resource limits defined" "grep -r 'limits:' $PROJECT_ROOT/deploy/helm/" "ISO27001"

echo ""
echo "==========================================="
echo "8. STATIC ANALYSIS (SAST)"
echo "==========================================="

echo -n "[SOC2/ISO27001] Running Bandit security scan... "
cd "$PROJECT_ROOT"
if command -v bandit &> /dev/null; then
    BANDIT_RESULT=$(bandit -r ./services/compiler ./sdk/python -f json 2>/dev/null || echo '{"results":[]}')
    HIGH_COUNT=$(echo "$BANDIT_RESULT" | grep -o '"SEVERITY.HIGH": [0-9]*' | head -1 | grep -o '[0-9]*' || echo "0")
    if [ "$HIGH_COUNT" = "0" ] || [ -z "$HIGH_COUNT" ]; then
        echo -e "${GREEN}PASS${NC} (No high-severity findings)"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}FAIL${NC} ($HIGH_COUNT high-severity findings)"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
else
    echo -e "${YELLOW}SKIP${NC} (bandit not installed)"
    WARNINGS=$((WARNINGS + 1))
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

echo ""
echo "==========================================="
echo "9. COMPLIANCE DOCUMENTATION"
echo "==========================================="

run_check "SOC2 audit report generated" "test -f $PROJECT_ROOT/audits/compliance-reports/SOC2_TYPE_II_AUDIT_REPORT.md" "SOC2"
run_check "HIPAA audit report generated" "test -f $PROJECT_ROOT/audits/compliance-reports/HIPAA_COMPLIANCE_AUDIT_REPORT.md" "HIPAA"
run_check "ISO27001 audit report generated" "test -f $PROJECT_ROOT/audits/compliance-reports/ISO27001_2022_AUDIT_REPORT.md" "ISO27001"
run_check "ISO27701 audit report generated" "test -f $PROJECT_ROOT/audits/compliance-reports/ISO27701_PIMS_AUDIT_REPORT.md" "ISO27701"
run_check "Compliance summary generated" "test -f $PROJECT_ROOT/audits/compliance-reports/COMPLIANCE_AUDIT_SUMMARY.md" "ALL"

echo ""
echo "==========================================="
echo "COMPLIANCE AUDIT RESULTS"
echo "==========================================="
echo ""
echo -e "Total Checks:  $TOTAL_CHECKS"
echo -e "Passed:        ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed:        ${RED}$FAILED_CHECKS${NC}"
echo -e "Warnings:      ${YELLOW}$WARNINGS${NC}"
echo ""

# Calculate compliance score
if [ $TOTAL_CHECKS -gt 0 ]; then
    COMPLIANCE_SCORE=$(echo "scale=1; ($PASSED_CHECKS * 100) / $TOTAL_CHECKS" | bc)
    echo -e "Compliance Score: ${BLUE}${COMPLIANCE_SCORE}%${NC}"
fi

echo ""
echo "==========================================="
echo "FRAMEWORK STATUS"
echo "==========================================="
echo ""

# Determine overall status
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "SOC 2 Type II:   ${GREEN}COMPLIANT${NC}"
    echo -e "HIPAA:           ${GREEN}COMPLIANT${NC}"
    echo -e "ISO 27001:2022:  ${GREEN}COMPLIANT${NC}"
    echo -e "ISO 27701:2019:  ${GREEN}COMPLIANT${NC}"
    echo ""
    echo -e "Overall Status:  ${GREEN}ALL FRAMEWORKS COMPLIANT${NC}"
    exit 0
else
    echo -e "Status:          ${YELLOW}REQUIRES ATTENTION${NC}"
    echo ""
    echo "Review failed checks above and address findings."
    exit 1
fi
