#pragma once
#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <chrono>
#include <memory>
#include <vector>
#include <iostream>

namespace fhe_gbdt::engine {

/**
 * EvalKeyEntry: Cached evaluation key with metadata
 */
struct EvalKeyEntry {
    std::string tenant_model_id;
    std::vector<uint8_t> serialized_key;
    std::chrono::steady_clock::time_point last_access;
    std::chrono::steady_clock::time_point created_at;
    size_t access_count = 0;

    size_t size_bytes() const {
        return serialized_key.size() + tenant_model_id.size() + sizeof(*this);
    }
};

/**
 * EvalKeyCache: LRU cache for deserialized evaluation keys
 *
 * Evaluation keys are large (~100KB-1MB each) and expensive to deserialize.
 * This cache stores deserialized keys to avoid repeated parsing overhead.
 *
 * Features:
 * - LRU eviction policy
 * - Configurable max size in bytes
 * - TTL-based expiration
 * - Thread-safe access
 * - Hit/miss statistics
 */
class EvalKeyCache {
public:
    struct Config {
        size_t max_size_bytes = 1ULL << 30;  // 1GB default
        std::chrono::seconds ttl = std::chrono::seconds(300);  // 5 minutes
        bool enable_stats = true;
    };

    struct Stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t expirations = 0;
        size_t current_entries = 0;
        size_t current_size_bytes = 0;

        double hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    explicit EvalKeyCache(const Config& config = Config{})
        : config_(config), current_size_bytes_(0) {}

    /**
     * Get cached key, returns nullptr if not found or expired
     */
    std::shared_ptr<EvalKeyEntry> get(const std::string& tenant_model_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_map_.find(tenant_model_id);
        if (it == cache_map_.end()) {
            stats_.misses++;
            return nullptr;
        }

        // Check expiration
        auto& entry = it->second;
        auto now = std::chrono::steady_clock::now();
        if (now - entry->created_at > config_.ttl) {
            // Expired - remove and return miss
            remove_entry(it);
            stats_.expirations++;
            stats_.misses++;
            return nullptr;
        }

        // Update LRU order
        lru_list_.erase(lru_map_[tenant_model_id]);
        lru_list_.push_front(tenant_model_id);
        lru_map_[tenant_model_id] = lru_list_.begin();

        // Update access metadata
        entry->last_access = now;
        entry->access_count++;

        stats_.hits++;
        return entry;
    }

    /**
     * Put key into cache, evicting if necessary
     */
    void put(const std::string& tenant_model_id,
             const std::vector<uint8_t>& serialized_key) {

        std::lock_guard<std::mutex> lock(mutex_);

        // Create entry
        auto entry = std::make_shared<EvalKeyEntry>();
        entry->tenant_model_id = tenant_model_id;
        entry->serialized_key = serialized_key;
        entry->created_at = std::chrono::steady_clock::now();
        entry->last_access = entry->created_at;
        entry->access_count = 1;

        size_t entry_size = entry->size_bytes();

        // Check if entry itself is too large
        if (entry_size > config_.max_size_bytes) {
            std::cerr << "EvalKeyCache: Entry too large (" << entry_size
                      << " bytes), not caching" << std::endl;
            return;
        }

        // Remove existing entry if present
        auto existing = cache_map_.find(tenant_model_id);
        if (existing != cache_map_.end()) {
            remove_entry(existing);
        }

        // Evict until we have space
        while (current_size_bytes_ + entry_size > config_.max_size_bytes && !lru_list_.empty()) {
            evict_lru();
        }

        // Insert new entry
        cache_map_[tenant_model_id] = entry;
        lru_list_.push_front(tenant_model_id);
        lru_map_[tenant_model_id] = lru_list_.begin();
        current_size_bytes_ += entry_size;

        stats_.current_entries = cache_map_.size();
        stats_.current_size_bytes = current_size_bytes_;
    }

    /**
     * Remove specific key from cache
     */
    void invalidate(const std::string& tenant_model_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_map_.find(tenant_model_id);
        if (it != cache_map_.end()) {
            remove_entry(it);
        }
    }

    /**
     * Clear all entries
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);

        cache_map_.clear();
        lru_list_.clear();
        lru_map_.clear();
        current_size_bytes_ = 0;

        stats_.current_entries = 0;
        stats_.current_size_bytes = 0;
    }

    /**
     * Remove expired entries
     */
    void cleanup_expired() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();
        auto it = cache_map_.begin();

        while (it != cache_map_.end()) {
            if (now - it->second->created_at > config_.ttl) {
                auto to_remove = it++;
                remove_entry(to_remove);
                stats_.expirations++;
            } else {
                ++it;
            }
        }
    }

    /**
     * Get cache statistics
     */
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    /**
     * Print cache status
     */
    void print_status() const {
        auto s = get_stats();
        std::cout << "EvalKeyCache Status:" << std::endl
                  << "  Entries: " << s.current_entries << std::endl
                  << "  Size: " << s.current_size_bytes / (1024.0 * 1024.0) << " MB" << std::endl
                  << "  Hit rate: " << (s.hit_rate() * 100.0) << "%" << std::endl
                  << "  Hits: " << s.hits << ", Misses: " << s.misses << std::endl
                  << "  Evictions: " << s.evictions << ", Expirations: " << s.expirations << std::endl;
    }

private:
    Config config_;
    mutable std::mutex mutex_;

    // Main storage: tenant_model_id -> entry
    std::unordered_map<std::string, std::shared_ptr<EvalKeyEntry>> cache_map_;

    // LRU tracking
    std::list<std::string> lru_list_;  // Front = most recently used
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;

    size_t current_size_bytes_;
    Stats stats_;

    void remove_entry(std::unordered_map<std::string, std::shared_ptr<EvalKeyEntry>>::iterator it) {
        const std::string& key = it->first;
        current_size_bytes_ -= it->second->size_bytes();

        lru_list_.erase(lru_map_[key]);
        lru_map_.erase(key);
        cache_map_.erase(it);

        stats_.current_entries = cache_map_.size();
        stats_.current_size_bytes = current_size_bytes_;
    }

    void evict_lru() {
        if (lru_list_.empty()) return;

        const std::string& lru_key = lru_list_.back();
        auto it = cache_map_.find(lru_key);
        if (it != cache_map_.end()) {
            remove_entry(it);
            stats_.evictions++;
        }
    }
};

} // namespace fhe_gbdt::engine
