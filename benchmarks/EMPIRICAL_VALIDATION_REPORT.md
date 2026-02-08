# Empirical Validation Report: FHE-GBDT Innovations

Generated: 2026-02-08 16:34:22 UTC
Environment: Python 3.11.14, NumPy 2.4.2
XGBoost 3.1.3, LightGBM 4.6.0

## 1. Accuracy Benchmarks

Measures accuracy impact of each innovation on real trained models.


### 1a. MOAI Oblivious Conversion

| dataset | library | num_trees | baseline_auc | moai_oblivious_auc | moai_auc_delta | moai_rotation_savings |
|---|---|---|---|---|---|---|
| adult | xgboost | 100 | 0.9319 | 0.6030 |  |  |
| adult | lightgbm | 100 | 0.9317 | 0.6155 |  |  |
| higgs_synth | xgboost | 100 | 0.9908 | 0.5804 |  |  |
| higgs_synth | lightgbm | 100 | 0.9912 | 0.6076 |  |  |
| criteo_synth | xgboost | 100 | 0.9781 | 0.5889 |  |  |
| criteo_synth | lightgbm | 100 | 0.9789 | 0.5092 |  |  |

### 1b. Polynomial Leaf Functions

| dataset | library | baseline_auc | poly_deg1_auc | poly_deg1_auc_delta | poly_deg1_coverage | poly_deg2_auc | poly_deg2_auc_delta | poly_deg2_coverage | poly_deg3_auc | poly_deg3_auc_delta | poly_deg3_coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | xgboost | 0.9319 | 0.1186 | -0.8134 | 0.2914 | 0.4853 | -0.4466 | 0.3824 | 0.4522 | -0.4798 | 0.4208 |
| adult | lightgbm | 0.9317 | 0.1313 | -0.8003 | 0.3091 | 0.4864 | -0.4453 | 0.4020 | 0.5930 | -0.3387 | 0.4434 |
| higgs_synth | xgboost | 0.9908 | 0.0377 | -0.9531 | 0.4166 | 0.0660 | -0.9248 | 0.4874 | 0.0679 | -0.9229 | 0.5356 |
| higgs_synth | lightgbm | 0.9912 | 0.0407 | -0.9505 | 0.4721 | 0.0779 | -0.9133 | 0.5533 | 0.0760 | -0.9153 | 0.6038 |
| criteo_synth | xgboost | 0.9781 | 0.1025 | -0.8756 | 0.3467 | 0.1751 | -0.8030 | 0.4194 | 0.1742 | -0.8039 | 0.4752 |
| criteo_synth | lightgbm | 0.9789 | 0.1163 | -0.8626 | 0.4023 | 0.1863 | -0.7926 | 0.4977 | 0.1748 | -0.8041 | 0.5620 |

### 1c. Gradient-Aware Noise Allocation (Adaptive vs Uniform)

| dataset | library | baseline_auc | noise_low_adaptive_auc | noise_low_uniform_auc | noise_low_gain | noise_low_avg_bits | noise_mid_adaptive_auc | noise_mid_uniform_auc | noise_mid_gain | noise_mid_avg_bits | noise_high_adaptive_auc | noise_high_uniform_auc | noise_high_gain | noise_high_avg_bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | xgboost | 0.9319 | 0.9319 | 0.9319 | 0.000000 | 6.2100 | 0.9319 | 0.9319 | 0.000000 | 8.2100 | 0.9319 | 0.9319 | 0.000000 | 13.8600 |
| adult | lightgbm | 0.9317 | 0.9317 | 0.9317 | 0.000000 | 6.5000 | 0.9317 | 0.9317 | 0.000000 | 8.5000 | 0.9317 | 0.9317 | 0.000000 | 14.0000 |
| higgs_synth | xgboost | 0.9908 | 0.9908 | 0.9908 | 0.000015 | 6.8200 | 0.9908 | 0.9908 | 0.000000 | 8.8200 | 0.9908 | 0.9908 | 0.000000 | 14.5400 |
| higgs_synth | lightgbm | 0.9912 | 0.9912 | 0.9912 | -0.000011 | 6.9300 | 0.9912 | 0.9912 | 0.000006 | 8.9300 | 0.9912 | 0.9912 | 0.000000 | 14.5000 |
| criteo_synth | xgboost | 0.9781 | 0.9782 | 0.9782 | 0.000020 | 6.4900 | 0.9781 | 0.9782 | -0.000021 | 8.4900 | 0.9781 | 0.9781 | -0.000004 | 13.9500 |
| criteo_synth | lightgbm | 0.9789 | 0.9790 | 0.9789 | 0.000033 | 6.4600 | 0.9789 | 0.9789 | -0.000013 | 8.4600 | 0.9789 | 0.9789 | -0.000000 | 14.0000 |

### 1d. Homomorphic Ensemble Pruning

| dataset | library | baseline_auc | prune_keep90_auc | prune_keep90_auc_delta | prune_keep90_active_trees | prune_keep90_ratio | prune_keep75_auc | prune_keep75_auc_delta | prune_keep75_active_trees | prune_keep75_ratio | prune_keep50_auc | prune_keep50_auc_delta | prune_keep50_active_trees | prune_keep50_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | xgboost | 0.9319 | 0.9219 | -0.0101 | 30 | 0.7000 | 0.9219 | -0.0101 | 30 | 0.7000 | 0.9219 | -0.0101 | 30 | 0.7000 |
| adult | lightgbm | 0.9317 | 0.9214 | -0.0103 | 30 | 0.7000 | 0.9214 | -0.0103 | 30 | 0.7000 | 0.9214 | -0.0103 | 30 | 0.7000 |
| higgs_synth | xgboost | 0.9908 | 0.9821 | -0.008732 | 30 | 0.7000 | 0.9821 | -0.008732 | 30 | 0.7000 | 0.9821 | -0.008732 | 30 | 0.7000 |
| higgs_synth | lightgbm | 0.9912 | 0.9832 | -0.008017 | 30 | 0.7000 | 0.9832 | -0.008017 | 30 | 0.7000 | 0.9832 | -0.008017 | 30 | 0.7000 |
| criteo_synth | xgboost | 0.9781 | 0.9589 | -0.0193 | 30 | 0.7000 | 0.9589 | -0.0193 | 30 | 0.7000 | 0.9589 | -0.0193 | 30 | 0.7000 |
| criteo_synth | lightgbm | 0.9789 | 0.9611 | -0.0179 | 30 | 0.7000 | 0.9611 | -0.0179 | 30 | 0.7000 | 0.9611 | -0.0179 | 30 | 0.7000 |

### 1e. Bootstrap-Aligned Chunking

| dataset | library | baseline_auc | bootstrap_auc | bootstrap_auc_delta | bootstrap_chunks | bootstrap_points | bootstrap_noise_utilization |
|---|---|---|---|---|---|---|---|
| adult | xgboost | 0.9319 | 0.9319 | 0.000000 | 100 | 98 | 1.6935 |
| adult | lightgbm | 0.9317 | 0.9317 | 0.000000 | 100 | 98 | 1.6935 |
| higgs_synth | xgboost | 0.9908 | 0.9908 | 0.000000 | 100 | 98 | 1.6935 |
| higgs_synth | lightgbm | 0.9912 | 0.9912 | 0.000000 | 100 | 98 | 1.6935 |
| criteo_synth | xgboost | 0.9781 | 0.9781 | 0.000000 | 100 | 98 | 1.6935 |
| criteo_synth | lightgbm | 0.9789 | 0.9789 | 0.000000 | 100 | 98 | 1.6935 |

### 1f. Streaming Gradient Updates

| dataset | library | streaming_updates | streaming_final_lr | streaming_avg_grad_norm |
|---|---|---|---|---|
| adult | xgboost | 700 | 0.000496 | 1.0000 |
| adult | lightgbm | 700 | 0.000496 | 1.0000 |
| higgs_synth | xgboost | 700 | 0.000496 | 1.0000 |
| higgs_synth | lightgbm | 700 | 0.000496 | 1.0000 |
| criteo_synth | xgboost | 700 | 0.000496 | 1.0000 |
| criteo_synth | lightgbm | 700 | 0.000496 | 1.0000 |

## 2. FHE Simulation Benchmarks


### 2a. Polynomial Evaluation: Horner vs Naive

| link | degree | max_error | horner_us | naive_us | fhe_horner_ms | fhe_naive_ms | fhe_speedup | fhe_horner_muls | fhe_naive_muls |
|---|---|---|---|---|---|---|---|---|---|
| logit | 3 | 0.1132 | 26.3500 | 640.9700 | 30.3000 | 60.3000 | 1.9900 | 3 | 6 |
| logit | 5 | 0.0601 | 47.0100 | 1843.4100 | 50.5000 | 150.5000 | 2.9800 | 5 | 15 |
| logit | 7 | 0.0311 | 59.0400 | 3019.7600 | 70.7000 | 280.7000 | 3.9700 | 7 | 28 |
| logit | 9 | 0.0156 | 87.8600 | 4247.7900 | 90.9000 | 450.9000 | 4.9600 | 9 | 45 |
| logit | 11 | 0.007786 | 90.2800 | 5418.5600 | 111.1000 | 661.1000 | 5.9500 | 11 | 66 |
| log | 3 | 7.6865 | 29.5800 | 648.4200 | 30.3000 | 60.3000 | 1.9900 | 3 | 6 |
| log | 5 | 0.9011 | 48.3400 | 1844.9300 | 50.5000 | 150.5000 | 2.9800 | 5 | 15 |
| log | 7 | 0.0609 | 65.8500 | 3093.4400 | 70.7000 | 280.7000 | 3.9700 | 7 | 28 |
| log | 9 | 0.002634 | 75.5300 | 4192.9400 | 90.9000 | 450.9000 | 4.9600 | 9 | 45 |
| log | 11 | 0.000079 | 98.1700 | 5408.5300 | 111.1000 | 661.1000 | 5.9500 | 11 | 66 |
| probit | 3 | 0.0949 | 28.9600 | 636.1900 | 30.3000 | 60.3000 | 1.9900 | 3 | 6 |
| probit | 5 | 0.0394 | 45.9400 | 1871.4900 | 50.5000 | 150.5000 | 2.9800 | 5 | 15 |
| probit | 7 | 0.0140 | 65.3700 | 3114.5900 | 70.7000 | 280.7000 | 3.9700 | 7 | 28 |
| probit | 9 | 0.004338 | 78.0300 | 4159.5400 | 90.9000 | 450.9000 | 4.9600 | 9 | 45 |
| probit | 11 | 0.001180 | 97.3300 | 5418.1000 | 111.1000 | 661.1000 | 5.9500 | 11 | 66 |
| reciprocal | 3 | 6.8157 | 28.3300 | 63.8900 | 30.3000 | 60.3000 | 1.9900 | 3 | 6 |
| reciprocal | 5 | 5.0974 | 44.6600 | 133.0400 | 50.5000 | 150.5000 | 2.9800 | 5 | 15 |
| reciprocal | 7 | 3.7050 | 61.9000 | 191.1600 | 70.7000 | 280.7000 | 3.9700 | 7 | 28 |
| reciprocal | 9 | 2.6390 | 85.4800 | 278.3000 | 90.9000 | 450.9000 | 4.9600 | 9 | 45 |
| reciprocal | 11 | 1.8634 | 106.5500 | 318.5100 | 111.1000 | 661.1000 | 5.9500 | 11 | 66 |

### 2b. Adaptive Precision Encoding

| n_features | encode_us | decode_us | mean_quant_error | max_quant_error | avg_precision_bits | min_precision_bits | max_precision_bits |
|---|---|---|---|---|---|---|---|
| 14 | 91.6200 | 22.6200 | 0.000044 | 0.000122 | 13.0000 | 12 | 16 |
| 28 | 186.7700 | 41.6800 | 0.000048 | 0.000122 | 12.7500 | 12 | 16 |
| 39 | 242.5100 | 60.2400 | 0.000050 | 0.000122 | 12.6200 | 12 | 16 |
| 100 | 634.4900 | 145.7900 | 0.000054 | 0.000122 | 12.4200 | 12 | 16 |
| 500 | 4295.5400 | 749.1900 | 0.000058 | 0.000122 | 12.1900 | 12 | 16 |

## 3. Noise Budget Validation


### 3a. Per-Operation Noise Cost

| operation | initial_log_noise | after_log_noise | noise_growth_bits | model_predicted_bits |
|---|---|---|---|---|
| fresh_encryption | 7.6800 | 7.6800 | 0.000000 | 3.2000 |
| addition | 7.6800 | 8.6800 | 1.0000 | 0.1000 |
| plain_mult_x10 | 7.6800 | 11.0000 | 3.3200 | 3.3200 |
| plain_mult_x100 | 7.6800 | 14.3200 | 6.6400 | 6.6400 |
| ct_ct_mult | 7.6800 | 0.000000 | -7.6800 | 10.0000 |
| rotation | 7.6800 | 8.2600 | 0.5800 | 0.5000 |

### 3b. Step Function Chain Noise Growth (Leveled FHE Model)

| num_levels | total_multiplications | simulated_log_noise | predicted_noise_bits | budget_remaining_simulated | budget_remaining_predicted | levels_consumed | model_conservative |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 14.7500 | 11.2000 | 45.2500 | 48.8000 | 7 | False |
| 2 | 14 | 21.8200 | 19.2000 | 38.1800 | 40.8000 | 14 | False |
| 3 | 21 | 28.8900 | 27.2000 | 31.1100 | 32.8000 | 21 | False |
| 4 | 28 | 35.9600 | 35.2000 | 24.0400 | 24.8000 | 28 | False |
| 5 | 35 | 43.0300 | 43.2000 | 16.9700 | 16.8000 | 35 | True |
| 6 | 42 | 50.1000 | 51.2000 | 9.9000 | 8.8000 | 42 | True |
| 7 | 49 | 57.1700 | 59.2000 | 2.8300 | 0.8000 | 49 | True |
| 8 | 56 | 64.2400 | 67.2000 | -4.2400 | -7.2000 | 56 | True |
| 9 | 63 | 71.3100 | 75.2000 | -11.3100 | -15.2000 | 63 | True |
| 10 | 70 | 78.3800 | 83.2000 | -18.3800 | -23.2000 | 70 | True |
| 11 | 77 | 85.4500 | 91.2000 | -25.4500 | -31.2000 | 77 | True |

### 3c. Bootstrap Chunking Validation

| num_trees | tree_depth | total_noise_bits | noise_budget | needs_bootstrap | num_chunks | budget_utilization |
|---|---|---|---|---|---|---|
| 50 | 5 | 44.3000 | 31.0000 | True | 50 | 1.4290 |
| 50 | 7 | 60.5000 | 31.0000 | True | 50 | 1.9516 |
| 100 | 5 | 44.4000 | 31.0000 | True | 100 | 1.4323 |
| 100 | 7 | 60.6000 | 31.0000 | True | 100 | 1.9548 |
| 200 | 5 | 44.5000 | 31.0000 | True | 200 | 1.4355 |
| 200 | 7 | 60.7000 | 31.0000 | True | 200 | 1.9581 |
| 500 | 5 | 44.6000 | 31.0000 | True | 500 | 1.4387 |
| 500 | 7 | 60.8000 | 31.0000 | True | 500 | 1.9613 |

## 4. SilentWood Comparison


### 4a. Rotation Count & Latency Comparison

| num_trees | tree_depth | total_nodes | traditional_rotations | moai_rotations | silentwood_rotations | rotation_reduction_vs_trad | rotation_reduction_vs_sw | traditional_latency_ms | our_latency_ms | silentwood_latency_ms | speedup_vs_traditional | speedup_vs_silentwood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 4 | 750 | 750 | 6 | 28 | 0.9920 | 0.7857 | 1877.5000 | 455.5000 | 890 | 4.1200 | 1.9500 |
| 50 | 6 | 3150 | 3150 | 6 | 42 | 0.9981 | 0.8571 | 7877.5000 | 655.5000 | 1680 | 12.0200 | 2.5600 |
| 50 | 8 | 12750 | 12750 | 6 | 56 | 0.9995 | 0.8929 | 31877.5000 | 855.5000 | 1680 | 37.2600 | 1.9600 |
| 100 | 4 | 1500 | 1500 | 7 | 52 | 0.9953 | 0.8654 | 3755.0000 | 858.5000 | 1450 | 4.3700 | 1.6900 |
| 100 | 6 | 6300 | 6300 | 7 | 78 | 0.9989 | 0.9103 | 15755.0000 | 1258.5000 | 2400 | 12.5200 | 1.9100 |
| 100 | 8 | 25500 | 25500 | 7 | 104 | 0.9997 | 0.9327 | 63755.0000 | 1658.5000 | 2400 | 38.4400 | 1.4500 |
| 200 | 4 | 3000 | 3000 | 8 | 100 | 0.9973 | 0.9200 | 7510.0000 | 1664.0000 | 262.9000 | 4.5100 | 0.1600 |
| 200 | 6 | 12600 | 12600 | 8 | 150 | 0.9994 | 0.9467 | 31510.0000 | 2464.0000 | 4200 | 12.7900 | 1.7000 |
| 200 | 8 | 51000 | 51000 | 8 | 200 | 0.9998 | 0.9600 | 127510.0000 | 3264.0000 | 4200 | 39.0700 | 1.2900 |
| 500 | 4 | 7500 | 7500 | 9 | 252 | 0.9988 | 0.9643 | 18775.0000 | 4079.5000 | 657.1000 | 4.6000 | 0.1600 |
| 500 | 6 | 31500 | 31500 | 9 | 378 | 0.9997 | 0.9762 | 78775.0000 | 6079.5000 | 9800 | 12.9600 | 1.6100 |
| 500 | 8 | 127500 | 127500 | 9 | 504 | 0.9999 | 0.9821 | 318775.0000 | 8079.5000 | 9800 | 39.4500 | 1.2100 |

## 5. Key Empirical Findings


**Polynomial Leaves (degree 2)**: Average AUC change = -0.7209 (range: -0.9248 to -0.4453)

**Adaptive vs Uniform (low precision, ~7 bits)**: Average AUC gain = +0.000010 (positive = adaptive better)

**Adaptive vs Uniform (mid precision, ~9 bits)**: Average AUC gain = -0.000005 (positive = adaptive better)

**Adaptive vs Uniform (high precision, ~14 bits)**: Average AUC gain = -0.000001 (positive = adaptive better)

**Homomorphic Pruning (keep90)**: Average 70.0% trees pruned, AUC delta = -0.0124

**Homomorphic Pruning (keep75)**: Average 70.0% trees pruned, AUC delta = -0.0124

**Homomorphic Pruning (keep50)**: Average 70.0% trees pruned, AUC delta = -0.0124

**Horner vs Naive FHE Evaluation**: Average 4.0x fewer multiplications in FHE domain

**vs Traditional**: 18.5x average speedup (rotation elimination)

**vs SilentWood**: 1.47x average (faster on rotation count; note: SilentWood has additional ct-compression advantage not modeled here)

**Noise Model Validation**: Per-operation costs:
  - fresh_encryption: simulated=0.00 bits, model=3.20 bits
  - addition: simulated=1.00 bits, model=0.10 bits
  - plain_mult_x10: simulated=3.32 bits, model=3.32 bits
  - plain_mult_x100: simulated=6.64 bits, model=6.64 bits
  - ct_ct_mult: simulated=-7.68 bits, model=10.00 bits
  - rotation: simulated=0.58 bits, model=0.50 bits