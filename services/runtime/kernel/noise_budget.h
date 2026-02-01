#pragma once
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace fhe_gbdt::kernel {

/**
 * NoiseBudget: Tracks accumulated noise in FHE ciphertexts
 *
 * In FHE, each operation adds noise to the ciphertext. When noise exceeds
 * the modulus budget, decryption fails. This class tracks noise accumulation
 * to detect when bootstrapping is needed.
 *
 * Noise model (simplified):
 * - Initial encryption: ~log2(sigma) bits
 * - Addition: max(noise_a, noise_b) + epsilon
 * - Multiplication: noise_a + noise_b + constant
 * - Step (LUT): Significant noise increase
 */
class NoiseBudget {
public:
    // Initialize with N2HE default parameters
    // log2(q) = 32, so initial budget is ~32 bits
    explicit NoiseBudget(int64_t log_q = 32, double initial_noise_bits = 3.2)
        : log_q_(log_q)
        , max_noise_bits_(static_cast<double>(log_q) - 1.0)  // Leave 1 bit margin
        , current_noise_bits_(initial_noise_bits)
        , initial_noise_bits_(initial_noise_bits) {
    }

    // Get remaining noise budget in bits
    double remaining_bits() const {
        return max_noise_bits_ - current_noise_bits_;
    }

    // Get current noise level in bits
    double current_noise() const {
        return current_noise_bits_;
    }

    // Check if ciphertext is still valid (has positive budget)
    bool is_valid() const {
        return current_noise_bits_ < max_noise_bits_;
    }

    // Check if bootstrapping is recommended (less than threshold budget remaining)
    bool needs_bootstrap(double threshold_bits = 5.0) const {
        return remaining_bits() < threshold_bits;
    }

    // Consume noise for addition operation
    // Addition barely increases noise (max of operands)
    void consume_add(const NoiseBudget& other) {
        current_noise_bits_ = std::max(current_noise_bits_, other.current_noise_bits_) + 0.1;
        check_overflow();
    }

    // Consume noise for subtraction (same as addition)
    void consume_sub(const NoiseBudget& other) {
        consume_add(other);
    }

    // Consume noise for plaintext multiplication
    // Multiplies noise by the plaintext value's bit size
    void consume_mul_plain(double plaintext_bits = 10.0) {
        current_noise_bits_ += plaintext_bits;
        check_overflow();
    }

    // Consume noise for ciphertext multiplication (external product)
    // This is expensive: roughly doubles the noise
    void consume_mul_cipher(const NoiseBudget& other) {
        current_noise_bits_ = current_noise_bits_ + other.current_noise_bits_ + 1.0;
        check_overflow();
    }

    // Consume noise for Step function (comparison via LUT)
    // LUT operations have significant noise cost
    void consume_step() {
        current_noise_bits_ += 8.0;  // LUT lookup adds ~8 bits of noise
        check_overflow();
    }

    // Consume noise for rotation (automorphism)
    // Rotation itself is nearly noise-free in RLWE
    void consume_rotation() {
        current_noise_bits_ += 0.5;
        check_overflow();
    }

    // Reset noise after bootstrapping
    void refresh() {
        current_noise_bits_ = initial_noise_bits_;
    }

    // Create a copy with same noise level
    NoiseBudget clone() const {
        NoiseBudget copy(log_q_, initial_noise_bits_);
        copy.current_noise_bits_ = current_noise_bits_;
        return copy;
    }

    // Print status for debugging
    void print_status(const std::string& label = "") const {
        std::cout << "NoiseBudget";
        if (!label.empty()) {
            std::cout << " [" << label << "]";
        }
        std::cout << ": " << current_noise_bits_ << "/" << max_noise_bits_
                  << " bits (remaining: " << remaining_bits() << ")"
                  << (needs_bootstrap() ? " [BOOTSTRAP RECOMMENDED]" : "")
                  << (!is_valid() ? " [OVERFLOW!]" : "")
                  << std::endl;
    }

private:
    int64_t log_q_;
    double max_noise_bits_;
    double current_noise_bits_;
    double initial_noise_bits_;

    void check_overflow() {
        if (!is_valid()) {
            std::cerr << "WARNING: Noise budget exhausted! Decryption will fail." << std::endl;
            print_status("OVERFLOW");
        }
    }
};

/**
 * NoiseTracker: Manages noise budgets for multiple ciphertexts
 */
class NoiseTracker {
public:
    NoiseTracker() = default;

    // Create a new noise budget for a fresh ciphertext
    size_t create_budget(int64_t log_q = 32) {
        budgets_.emplace_back(log_q);
        return budgets_.size() - 1;
    }

    // Get budget by ID
    NoiseBudget& get(size_t id) {
        if (id >= budgets_.size()) {
            throw std::out_of_range("Invalid noise budget ID");
        }
        return budgets_[id];
    }

    // Check if any budget needs bootstrapping
    bool any_needs_bootstrap() const {
        for (const auto& b : budgets_) {
            if (b.needs_bootstrap()) {
                return true;
            }
        }
        return false;
    }

    // Get statistics
    struct Stats {
        size_t total_ciphertexts = 0;
        size_t needs_bootstrap = 0;
        size_t overflowed = 0;
        double min_remaining_bits = 0.0;
        double avg_remaining_bits = 0.0;
    };

    Stats get_stats() const {
        Stats s;
        s.total_ciphertexts = budgets_.size();

        if (budgets_.empty()) {
            return s;
        }

        double total_remaining = 0.0;
        s.min_remaining_bits = budgets_[0].remaining_bits();

        for (const auto& b : budgets_) {
            if (b.needs_bootstrap()) s.needs_bootstrap++;
            if (!b.is_valid()) s.overflowed++;

            double remaining = b.remaining_bits();
            total_remaining += remaining;
            if (remaining < s.min_remaining_bits) {
                s.min_remaining_bits = remaining;
            }
        }

        s.avg_remaining_bits = total_remaining / budgets_.size();
        return s;
    }

    void print_summary() const {
        auto s = get_stats();
        std::cout << "NoiseTracker Summary:" << std::endl
                  << "  Total ciphertexts: " << s.total_ciphertexts << std::endl
                  << "  Need bootstrap: " << s.needs_bootstrap << std::endl
                  << "  Overflowed: " << s.overflowed << std::endl
                  << "  Min remaining bits: " << s.min_remaining_bits << std::endl
                  << "  Avg remaining bits: " << s.avg_remaining_bits << std::endl;
    }

private:
    std::vector<NoiseBudget> budgets_;
};

} // namespace fhe_gbdt::kernel
