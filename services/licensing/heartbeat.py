"""
Heartbeat Telemetry Module for FHE-GBDT Serving

Collects counter-only usage telemetry from the on-prem runtime and
periodically reports it to the vendor's cloud control plane.

Telemetry includes ONLY:
- Prediction counts per model
- Latency percentiles
- Error counts
- License ID and tenant ID

Telemetry NEVER includes:
- Ciphertext payloads
- Feature values (encrypted or plaintext)
- Model weights or parameters
- Evaluation keys
"""

import json
import time
import threading
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

# Fields that must NEVER appear in telemetry
FORBIDDEN_FIELDS = frozenset({
    "ciphertext", "plaintext", "secret_key", "eval_key",
    "features", "predictions", "payload", "api_key",
    "password", "token", "model_weights",
})


@dataclass
class PredictionEvent:
    """A single prediction event for telemetry aggregation."""
    tenant_id: str
    model_id: str
    license_id: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class HeartbeatReport:
    """Aggregated telemetry report sent to the control plane."""
    tenant_id: str
    license_id: str
    report_id: str
    interval_start: float
    interval_end: float
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    predictions_by_model: Dict[str, int]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Safety: ensure no forbidden fields leak
        for key in list(d.keys()):
            if key in FORBIDDEN_FIELDS:
                del d[key]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class HeartbeatCollector:
    """
    Collects prediction events and produces periodic heartbeat reports.

    Runs on the customer's on-prem infrastructure. Reports are counters only --
    no sensitive data is ever included.
    """

    def __init__(
        self,
        tenant_id: str,
        license_id: str,
        report_interval_seconds: int = 60,
        report_callback: Optional[Callable[[HeartbeatReport], None]] = None,
    ):
        self._tenant_id = tenant_id
        self._license_id = license_id
        self._interval = report_interval_seconds
        self._callback = report_callback

        self._lock = threading.Lock()
        self._events: List[PredictionEvent] = []
        self._interval_start = time.time()
        self._report_counter = 0

        self._running = False
        self._timer: Optional[threading.Timer] = None
        self._reports: List[HeartbeatReport] = []

    def record(self, event: PredictionEvent) -> None:
        """Record a prediction event. Thread-safe."""
        with self._lock:
            self._events.append(event)

    def record_prediction(
        self,
        model_id: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Convenience method to record a prediction."""
        self.record(PredictionEvent(
            tenant_id=self._tenant_id,
            model_id=model_id,
            license_id=self._license_id,
            latency_ms=latency_ms,
            success=success,
        ))

    def flush(self) -> Optional[HeartbeatReport]:
        """Flush current events and produce a report."""
        with self._lock:
            if not self._events:
                return None

            events = self._events
            self._events = []
            interval_start = self._interval_start
            self._interval_start = time.time()

        now = time.time()
        self._report_counter += 1

        # Aggregate
        total = len(events)
        successes = sum(1 for e in events if e.success)
        failures = total - successes

        by_model: Dict[str, int] = defaultdict(int)
        for e in events:
            by_model[e.model_id] += 1

        latencies = sorted(e.latency_ms for e in events)
        p50 = self._percentile(latencies, 50)
        p95 = self._percentile(latencies, 95)
        p99 = self._percentile(latencies, 99)

        report = HeartbeatReport(
            tenant_id=self._tenant_id,
            license_id=self._license_id,
            report_id=f"{self._tenant_id}-{self._report_counter}",
            interval_start=interval_start,
            interval_end=now,
            total_predictions=total,
            successful_predictions=successes,
            failed_predictions=failures,
            predictions_by_model=dict(by_model),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
        )

        self._reports.append(report)

        if self._callback:
            try:
                self._callback(report)
            except Exception as exc:
                logger.warning("Heartbeat callback failed: %s", exc)

        return report

    def start(self) -> None:
        """Start periodic reporting in a background thread."""
        self._running = True
        self._schedule_next()

    def stop(self) -> Optional[HeartbeatReport]:
        """Stop periodic reporting and flush remaining events."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        return self.flush()

    def get_reports(self) -> List[HeartbeatReport]:
        """Return all collected reports."""
        return list(self._reports)

    def _schedule_next(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        self.flush()
        self._schedule_next()

    @staticmethod
    def _percentile(sorted_values: List[float], pct: int) -> float:
        if not sorted_values:
            return 0.0
        idx = max(0, int(len(sorted_values) * pct / 100) - 1)
        return sorted_values[idx]
