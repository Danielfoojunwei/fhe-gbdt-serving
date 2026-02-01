#pragma once
#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>

namespace fhe_gbdt::metrics {

class RuntimeMetrics {
public:
    static RuntimeMetrics& Instance() {
        static RuntimeMetrics instance;
        return instance;
    }

    void RecordPredictLatency(double ms, const std::string& profile) {
        // predict_latency_->Observe(ms);
    }

    void RecordRotations(uint64_t count) {
        // rotations_total_->Increment(count);
    }

    void RecordSchemeSwitches(uint64_t count) {
        // scheme_switches_total_->Increment(count);
    }

    void RecordBootstraps(uint64_t count) {
        // bootstraps_total_->Increment(count);
    }

    void RecordBatchSize(uint32_t size) {
        // batch_size_histogram_->Observe(size);
    }

    void SetQueueDepth(int depth) {
        // queue_depth_->Set(depth);
    }

private:
    RuntimeMetrics() {
        // Initialize Prometheus registry and metrics
    }
};

} // namespace fhe_gbdt::metrics
