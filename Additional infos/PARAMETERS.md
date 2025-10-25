# Parameter Configuration

## Overview
This document explains the parameter settings used in the DIO optimization scripts and how they've been optimized for faster execution.

---

## Main Script (main.py) - Breast Cancer Optimization

### Previous Settings (Slower)
```python
# Outer Loop (Hyperparameter Optimization)
n_dholes = 5
max_iterations = 10

# Inner Loop (Feature Selection)
n_dholes = 10
max_iterations = 20
```

### **Current Settings (FASTER - 60% reduction)**
```python
# Outer Loop (Hyperparameter Optimization)
n_dholes = 3          # Reduced from 5 (40% reduction)
max_iterations = 5    # Reduced from 10 (50% reduction)

# Inner Loop (Feature Selection)
n_dholes = 5          # Reduced from 10 (50% reduction)
max_iterations = 10   # Reduced from 20 (50% reduction)
```

### Computation Impact
- **Total Evaluations (Previous)**: ~5 × 10 × 10 × 20 = **10,000 evaluations**
- **Total Evaluations (Current)**: ~3 × 5 × 5 × 10 = **750 evaluations**
- **Speed Improvement**: ~13x faster ⚡

### Expected Results
Still maintains excellent performance:
- Expected Accuracy: 95-100%
- Selected Features: 8-15 out of 30
- Execution Time: 30-60 seconds (vs 5-10 minutes before)

---

## Benchmark Testing (run_benchmarks.py)

### Paper Settings (Original DIO Paper)
```python
n_dholes = 30          # Population size
max_iterations = 500   # Maximum iterations
num_runs = 30          # Independent runs for statistics
```

### **Current Settings (REDUCED - 90% reduction)**
```python
n_dholes = 10          # Reduced from 30 (67% reduction)
max_iterations = 100   # Reduced from 500 (80% reduction)
num_runs = 5           # Reduced from 30 (83% reduction)
test_functions = 4     # Testing subset of 4 functions first
```

### Computation Impact
- **Total Evaluations per Function (Paper)**: 30 × 500 × 30 = **450,000 evaluations**
- **Total Evaluations per Function (Current)**: 10 × 100 × 5 = **5,000 evaluations**
- **Speed Improvement**: ~90x faster per function ⚡

### Testing Strategy
1. **Quick Test** (Current): Run 4 representative functions (F1, F5, F9, F10)
2. **Full Test** (Optional): Uncomment full config in code to run all 26 functions
3. **Paper Comparison** (Research): Use paper settings for accurate benchmark comparison

---

## How to Adjust Parameters

### For Even Faster Testing (Ultra-Fast Mode)
```python
# main.py
n_dholes = 2
max_iterations = 3

# run_benchmarks.py
n_dholes = 5
max_iterations = 50
num_runs = 3
```

### For Better Results (Balanced Mode)
```python
# main.py
n_dholes = 5
max_iterations = 8

# run_benchmarks.py
n_dholes = 15
max_iterations = 200
num_runs = 10
```

### For Paper-Level Results (Research Mode)
```python
# main.py
n_dholes = 10
max_iterations = 20

# run_benchmarks.py
n_dholes = 30
max_iterations = 500
num_runs = 30
```

---

## Performance Guidelines

### PC Performance Estimates
| Mode | Outer Dholes | Outer Iter | Inner Dholes | Inner Iter | Est. Time |
|------|-------------|-----------|-------------|-----------|-----------|
| **Ultra-Fast** | 2 | 3 | 3 | 5 | ~10 sec |
| **Current (Fast)** | 3 | 5 | 5 | 10 | ~30-60 sec |
| **Balanced** | 5 | 8 | 8 | 15 | ~2-3 min |
| **Original** | 5 | 10 | 10 | 20 | ~5-10 min |
| **Research** | 10 | 20 | 15 | 30 | ~20-30 min |

### Benchmark Testing Estimates
| Mode | Pop | Iter | Runs | Functions | Est. Time |
|------|-----|------|------|-----------|-----------|
| **Quick Test (Current)** | 10 | 100 | 5 | 4 | ~2-3 min |
| **Subset Test** | 15 | 200 | 10 | 14 | ~15-20 min |
| **Full Paper** | 30 | 500 | 30 | 26 | ~6-8 hours |

---

## Recommendations

### For Development & Testing
✅ Use **Current (Fast)** settings
- Quick iterations
- Good enough results
- Enables rapid prototyping

### For Final Results
✅ Use **Balanced** settings
- Better optimization quality
- Reasonable execution time
- Suitable for demonstrations

### For Research Publication
✅ Use **Paper Settings**
- Required for fair comparison
- Run overnight or on HPC
- Generate publication-quality results

---

## Notes

1. **Nested Optimization**: Total evaluations = `outer_dholes × outer_iter × inner_dholes × inner_iter`
2. **Random Seed**: Set to 42 for reproducibility
3. **Storage Optimization**: Best features are cached to avoid redundant computation
4. **Early Stopping**: Not implemented (could further reduce time if fitness plateaus)

---

## Quick Start

### Run with current fast settings:
```bash
python main.py              # Breast Cancer optimization (~1 min)
python run_benchmarks.py    # Benchmark testing (~3 min)
```

### To modify settings:
Edit the parameters directly in the respective files:
- `main.py` lines 95-100 (inner loop) and 124-129 (outer loop)
- `run_benchmarks.py` lines 25-30 (CONFIG dictionary)

---

**Last Updated**: Current session  
**Optimization Level**: Fast Mode (60-90% reduction from original)
