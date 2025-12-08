# Essential Schemas for DIO Multi-Domain Research

## 📐 SCHEMA NAMING CONVENTION & FILE MAPPING

**Standardized naming:** `schema#_descriptive_name.png` (all lowercase)

### 🗂️ Schema Files to Create/Rename:
```
schema1_cross_domain_framework.png          → Cross-domain DIO overview
schema2_algorithm_dependent_overfitting.png → Three approaches comparison
schema3_cross_domain_results_table.png      → Success/failure quantification
schema4_nested_optimization_structure.png   → Two-level hierarchy
schema5_fitness_driven_optimization.png     → Fitness function mechanism (MOST IMPORTANT)
schema6_three_approaches_evolution.png      → Research progression timeline
schema7_cifar10_statistical_failure.png     → Negative result & budget analysis
```

### 📍 Schema Placement Guide:

#### **PRESENTATION (create_presentation_v2.py):**
- **Slide after "1.4 Applications"** → `schema1_cross_domain_framework.png`
- **Slide after "2.2 Methodology"** → `schema4_nested_optimization_structure.png`
- **Slide after "2.2 Methodology"** → `schema5_fitness_driven_optimization.png` ⭐ CRITICAL
- **Slide after "2.3.4 Medical Results"** → `schema6_three_approaches_evolution.png`
- **Slide "2.3.4 Comparison"** → `schema2_algorithm_dependent_overfitting.png`
- **Slide "2.5 Cross-Domain"** → `schema3_cross_domain_results_table.png`
- **Slide "2.4.1 CIFAR-10"** → `schema7_cifar10_statistical_failure.png`

#### **REPORT (report.tex):**
- **Section 2 Introduction** → `schema1_cross_domain_framework.png` (Figure 1)
- **Section 3.5 DIO Benchmark** → Keep existing `dio_flowchart.png` & `comparaison_table...png`
- **Section 4.1 Architecture** → `schema4_nested_optimization_structure.png` (Figure 4)
- **Section 4.2 Fitness Function** → `schema5_fitness_driven_optimization.png` (Figure 5) ⭐
- **Section 5 Medical RF Results** → `schema2_algorithm_dependent_overfitting.png` (Figure 10)
- **Section 8 XGBoost Results** → `schema6_three_approaches_evolution.png` (Figure 16)
- **Section 10 Cross-Domain** → `schema3_cross_domain_results_table.png` (Figure 21)
- **Section 10.3 CIFAR-10 Discussion** → `schema7_cifar10_statistical_failure.png` (Figure 22)

### 🔄 Existing Files to Keep:
- `dio_optimise_snippet.png` → Code snippet (Section 3)
- `dio_flowchart.png` → DIO algorithm flow (Section 3.5)
- `comparaison_table_of_results...png` → Benchmark table (Section 3.5)
- `feature_selection_objective_func_rf.png` → RF feature fitness (Section 4.2)
- `hyperparameter_objective_func_rf.png` → RF hyperparameter fitness (Section 4.2)
- `outer_optimization_and_retreiving_results.png` → Results retrieval (Section 4.3)
- `xgboost_hyperparameters_search_space_cancer.png` → XGBoost search space medical (Section 8)
- `xgboost_hyperparameters_search_space_images.png` → XGBoost search space CIFAR-10 (Section 10)

### ❌ Files to Delete/Archive:
- `shema1 (1).png` → Rename to `schema1_cross_domain_framework.png`
- `Shema2 (1).png` → Rename to `schema2_algorithm_dependent_overfitting.png`
- `shema3 (1).png` → Rename to `schema3_cross_domain_results_table.png`
- `shema4 (1).png` → Rename to `schema4_nested_optimization_structure.png`
- `shema5 (1).PNG` → Rename to `schema5_fitness_driven_optimization.png`

---

## 📐 7 ESSENTIAL SCHEMAS - Detailed Specifications

Based on your complete research (Medical + CIFAR-10), here are the **absolutely essential** schemas:

---

## 1. 🔄 **Cross-Domain DIO Framework Overview** (MOST IMPORTANT)

### Purpose
Show DIO's versatility across medical and vision domains - **Simple diagram**

```
┌──────────────────────────────────────────────────────────────┐
│            DIO OPTIMIZATION FRAMEWORK                        │
│            Multi-Domain Validation                           │
└──────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  DIO ALGORITHM  │
                    │ (Nature-Based)  │
                    └─────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ↓                               ↓
    ┌───────────────┐              ┌───────────────┐
    │ MEDICAL       │              │ VISION        │
    │ Breast Cancer │              │ CIFAR-10      │
    │               │              │               │
    │ • 30 features │              │ • 2048 feat.  │
    │ • Binary      │              │ • 10 classes  │
    └───────────────┘              └───────────────┘
            │                               │
            ↓                               ↓
    ┌───────────────┐              ┌───────────────┐
    │ Nested DIO    │              │ Nested DIO    │
    │ Optimization  │              │ Optimization  │
    └───────────────┘              └───────────────┘
            │                               │
            ↓                               ↓
    ┌───────────────┐              ┌───────────────┐
    │ RESULTS       │              │ RESULTS       │
    │ 96.88% acc    │              │ 81.91% acc    │
    │ 10/30 feat    │              │ 598/2048 feat │
    │ 67% reduction │              │ 70.8% reduct. │
    └───────────────┘              └───────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ↓
                ┌───────────────────────┐
                │ VALIDATED FRAMEWORK   │
                │ • 68× scale-up        │
                │ • Medical: SUCCESS ✅ │
                │ • Vision: FAILURE ❌  │
                └───────────────────────┘
```

**For draw.io:**
- 1 top box (DIO Algorithm)
- 2 parallel paths (Medical | Vision)
- 3 boxes per path (Data → Process → Results)
- 1 bottom box (Validation)
- Simple arrows connecting all

**Why Essential:** Shows complete cross-domain research in one simple diagram

---

## 2. 🎯 **Algorithm-Dependent Optimization Overfitting** (YOUR KEY DISCOVERY)

### Purpose
Explain THE main research contribution - **Simple 3-box comparison**

```
┌─────────────────────────────────────────────────────────────┐
│     OPTIMIZATION OVERFITTING: ALGORITHM-DEPENDENT           │
└─────────────────────────────────────────────────────────────┘

┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ RF Single-Split        │  │ RF Cross-Validation    │  │ XGBoost Single-Split   │
├────────────────────────┤  ├────────────────────────┤  ├────────────────────────┤
│                        │  │                        │  │                        │
│ Configuration:         │  │ Configuration:         │  │ Configuration:         │
│ 5 dholes, 10 iter (O)  │  │ 5 dholes, 10 iter (O)  │  │ 5 dholes, 10 iter (O)  │
│ 10 dholes, 20 iter (I) │  │ 10 dholes, 20 iter (I) │  │ 10 dholes, 20 iter (I) │
│                        │  │                        │  │                        │
│ Optimization:          │  │ Optimization:          │  │ Optimization:          │
│ 99% (overfit!)         │  │ 95.91% (CV avg)        │  │ 98.83% (holdout)       │
│                        │  │                        │  │                        │
│        ↓               │  │        ↓               │  │        ↓               │
│                        │  │                        │  │                        │
│ Validation:            │  │ Validation:            │  │ Validation:            │
│ 94.37% ± 1.82%         │  │ 96.55% ± 1.51%         │  │ 96.88% ± 1.10% 🏆      │
│ Rank: #6               │  │ Rank: #1               │  │ Rank: #1               │
│                        │  │                        │  │                        │
│ Time: ~60 min          │  │ Time: 7.9 hrs          │  │ Time: 54 sec           │
│ (~10,000 evals)        │  │ (~10,000 evals + CV)   │  │ (~10,000 evals)        │
│                        │  │                        │  │                        │
│ ❌ OVERFITTING         │  │ ✅ FIXED               │  │ ✅ NO ISSUE            │
│    (memorized split)   │  │    (but slow)          │  │    (built-in reg.)     │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘

KEY DISCOVERY:
┌──────────────────────────────────────────────────────────────┐
│ XGBoost's multi-layer regularization prevents meta-overfitting│
│ → Single-split is SUFFICIENT and 526× FASTER than RF-CV      │
│ → Algorithm choice determines whether CV is necessary!        │
└──────────────────────────────────────────────────────────────┘
```

**For draw.io:**
- 3 vertical boxes side-by-side (RF-Single | RF-CV | XGBoost)
- Each shows: Configuration → Optimization → Validation → Result
- Color code: Red (bad), Yellow (ok), Green (best)
- Bottom: Key discovery box

**Why Essential:** Your novel contribution - algorithm choice matters!

---

## 3. 📊 **Cross-Domain Results Comparison**

### Purpose
Quantify achievements across both domains in one table

```
┌─────────────────────────────────────────────────────────────┐
│         CROSS-DOMAIN VALIDATION RESULTS                      │
└─────────────────────────────────────────────────────────────┘

Metric                  │  Medical (Breast Cancer) │  Vision (CIFAR-10)
────────────────────────┼──────────────────────────┼──────────────────────
Feature Dimension       │  30-D                    │  2,048-D (68× larger)
Task                    │  Binary (2 classes)      │  Multi-class (10)
Training Samples        │  399 (train)             │  2,000 (subset)
Test Samples            │  171 (test)              │  500 (subset)
────────────────────────┼──────────────────────────┼──────────────────────
Best Algorithm          │  XGBoost                 │  XGBoost (Default!)
Baseline Accuracy       │  94.74% ± 1.55% (XGB)    │  80.8% (single-run)
                        │                          │  83.27% ± 1.25% (30-run)
────────────────────────┼──────────────────────────┼──────────────────────
DIO-Optimized (Single)  │  98.83% (training)       │  83.0% (single-run)
DIO Validation (30-run) │  96.88% ± 1.10% 🏆       │  81.91% ± 1.38%
Accuracy Gain/Loss      │  +2.14% (p=0.0047 **)    │  -1.36% (p<0.0001 ***)
Statistical Rank        │  #1 out of 10            │  #3 out of 9 (WORSE)
────────────────────────┼──────────────────────────┼──────────────────────
Feature Reduction       │  67% (30 → 10)           │  70.8% (2048 → 598)
Optimization Config     │  5/10 outer, 5/10 inner  │  3/8 outer, 3/8 inner
Optimization Budget     │  ~2,500 evaluations      │  ~576 evaluations
Optimization Time       │  54 seconds              │  215.98 min (3.6 hrs)
────────────────────────┼──────────────────────────┼──────────────────────
Outcome                 │  ✅ SUCCESS              │  ❌ FAILURE
Key Finding             │  Best accuracy           │  Insufficient budget
                        │  + Moderate reduction    │  (need 10-50K evals)
────────────────────────┴──────────────────────────┴──────────────────────

✅ MEDICAL SUCCESS: DIO achieves #1 rank with 96.88% (p=0.0047)
❌ VISION FAILURE: DIO ranks #3, worse than defaults (p<0.0001)
⚠️ CRITICAL INSIGHT: Optimization budget must scale with dimensionality²
   • 30-D with 2,500 evals → SUCCESS
   • 2048-D with 576 evals → FAILURE (need ~17-87× more evaluations)
```

**Why Essential:** Quantifies all results, proves cross-domain effectiveness AND limitations

---

## 4. 🔄 **Nested DIO Optimization Structure**

### Purpose
Show the two-level hierarchical optimization - **Simple nested boxes**

```
┌─────────────────────────────────────────────────────────────┐
│           NESTED DIO OPTIMIZATION STRUCTURE                 │
└─────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ OUTER LOOP: Hyperparameter Optimization                      │
│                                                               │
│  Population: 5 dholes × 10 iterations                        │
│  Search: n_estimators, max_depth, learning_rate, etc.       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ INNER LOOP: Feature Selection                          │ │
│  │                                                         │ │
│  │  Population: 5 dholes × 10 iterations                  │ │
│  │  Search: Feature mask [1,0,1,1,0,...] (D features)    │ │
│  │                                                         │ │
│  │  Process:                                               │ │
│  │  • Use fixed θ from outer loop                         │ │
│  │  • Find best features S* for this θ                    │ │
│  │  • Minimize F = 0.99×(1-Acc) + 0.01×(Feat/Total)      │ │
│  │                                                         │ │
│  │  Return: Best S* to outer loop                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Use S* from inner loop to evaluate this θ                  │
│  Find best θ* that minimizes F(θ, S*)                       │
└───────────────────────────────────────────────────────────────┘
                              ↓
                  ┌─────────────────────┐
                  │ OUTPUT:             │
                  │ • θ* (hyperparams)  │
                  │ • S* (features)     │
                  │ • Minimized F       │
                  └─────────────────────┘

Cost: Outer_evals × Inner_evals = Total model trainings
Medical: 50 × 50 = 2,500 → 54 seconds
Vision:  24 × 24 = 576 → 3.6 hours (215.98 min)
```

**For draw.io:**
- 1 large outer box (Outer Loop)
- 1 nested box inside (Inner Loop)
- 1 output box at bottom
- Arrows: Top → Inner → Bottom
- Label showing hierarchical relationship

**Why Essential:** Shows how the two optimization levels work together

---

## 5. 🔄 **Modularization: Fitness Function & Optimization Loop** (THE MOST IMPORTANT!)

### Purpose
**Show EXACTLY how the optimization works** - Simple, visual diagram for draw.io

```
┌─────────────────────────────────────────────────────────────┐
│         DIO OPTIMIZATION: FITNESS-DRIVEN PROCESS            │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │  FITNESS FUNCTION    │
                    │  (The Goal)          │
                    ├──────────────────────┤
                    │ F = 0.99×(1-Acc) +  │
                    │     0.01×(Feat/Tot) │
                    │                      │
                    │ Lower is better ↓   │
                    └──────────────────────┘
                            │
                            │ Drives both loops
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ↓                               │
┌───────────────────────────────────────────┼───────────────┐
│ OUTER LOOP: Hyperparameter Optimization   │               │
├───────────────────────────────────────────┘               │
│                                                            │
│  Input: Random θ (n_estimators, max_depth, lr, etc.)     │
│                                                            │
│        ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ INNER LOOP: Feature Selection                       │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                      │  │
│  │  Input: Fixed θ from outer loop                     │  │
│  │         Random feature mask S: [1,0,1,1,0,...]      │  │
│  │                                                      │  │
│  │  Process:                                            │  │
│  │  • Train model with θ and S                         │  │
│  │  • Calculate F(θ, S)                                │  │
│  │  • DIO updates S to minimize F                      │  │
│  │  • Repeat 10 iterations                             │  │
│  │                                                      │  │
│  │  Output: Best features S* for this θ                │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│        ↓                                                   │
│  Calculate F(θ, S*) for this θ                            │
│  DIO updates θ to minimize F                              │
│  Repeat 10 iterations                                     │
│                                                            │
│  Output: Best hyperparameters θ*                          │
└────────────────────────────────────────────────────────────┘
                            │
                            ↓
                ┌──────────────────────┐
                │   FINAL RESULT       │
                ├──────────────────────┤
                │ • Best θ* (hyper)    │
                │ • Best S* (features) │
                │                      │
                │ Medical: 96.88% acc  │
                │          10/30 feat  │
                │                      │
                │ Vision:  81.91% acc  │
                │          598/2048 ft │
                └──────────────────────┘

KEY CONCEPT:
┌────────────────────────────────────────────────────────────┐
│  Fitness F drives BOTH loops:                             │
│  • Outer: Tests different hyperparameters θ               │
│  • Inner: For each θ, finds best features S               │
│  • Hierarchical: Outer contains Inner                     │
│  • Goal: Minimize F(θ*, S*)                               │
│                                                            │
│  Total evaluations: Outer_iterations × Inner_iterations   │
│  Medical: 10 × 10 = 100 (but 5 dholes) = 2,500 → 54 sec  │
│  Vision:  8 × 8 = 64 (but 3 dholes) = 576 → 3.6 hrs (215.98 min) │
└────────────────────────────────────────────────────────────┘
```

**For draw.io: Create 3 main boxes**
1. **Top**: Fitness Function box (yellow)
2. **Middle-Outer**: Outer Loop box (blue) - contains next box
3. **Middle-Inner**: Inner Loop box (green) - nested inside outer
4. **Bottom**: Final Result box (gold)
5. **Arrows**: Show fitness driving both loops

**Why Essential:** Shows the complete optimization process in one simple diagram - easy to draw and understand!

---

## 6. 📈 **Three-Approach Evolution & Results**

### Purpose
Show research progression and justify final choice - **Simple timeline**

```
┌────────────────────────────────────────────────────────────────┐
│          EVOLUTION OF OPTIMIZATION APPROACHES                  │
└────────────────────────────────────────────────────────────────┘

    ATTEMPT 1              ATTEMPT 2              ATTEMPT 3
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   RF-Single  │  →   │    RF-CV     │  →   │   XGBoost    │
│              │      │              │      │              │
│ Config:      │      │ Config:      │      │ Config:      │
│ 5/10 outer   │      │ 5/10 outer   │      │ 5/10 outer   │
│ 10/20 inner  │      │ 10/20 inner  │      │ 5/10 inner   │
│              │      │              │      │              │
│ ❌ Overfits  │      │ ✅ Fixed     │      │ ✅ BEST      │
│ Opt: 99%     │      │ Uses 5-Fold  │      │ Built-in     │
│ Val: 94.37%  │      │ CV during    │      │ regularize   │
│ Rank: #6     │      │ optimization │      │ Val: 96.88%  │
│              │      │ Val: 96.55%  │      │ Rank: #1     │
│              │      │ Rank: #1     │      │              │
│ Time: ~60min │      │ Time: 7.9 hr │      │ Time: 54 sec │
│ (~10K evals) │      │ (~10K + CV)  │      │ (~2.5K evals)│
└──────────────┘      └──────────────┘      └──────────────┘
   Discovery:            Discovery:            Discovery:
   Single-split          CV fixes              XGBoost doesn't
   causes overfit        overfitting           need CV!

FINAL COMPARISON TABLE (30-Run Statistical Validation):
┌────────────┬──────────┬──────────┬─────────┬──────┬──────────┐
│ Approach   │ Time     │ vs Best  │ Val Acc │ Rank │ Features │
├────────────┼──────────┼──────────┼─────────┼──────┼──────────┤
│ RF-Single  │ ~60 min  │ 1.1×     │ 94.37%  │ #6   │ 8/30     │
│ RF-CV      │ 7.9 hrs  │ 8.8×     │ 96.55%  │ #1   │ 6/30     │
│ XGBoost    │ 54 sec   │ 1×       │ 96.88%🏆│ #1   │ 10/30    │
└────────────┴──────────┴──────────┴─────────┴──────┴──────────┘

KEY INSIGHTS:
• XGBoost achieves BEST accuracy (96.88%) 526× faster than RF-CV!
• Same inner loop config as RF (10/20), but simpler outer (5/10 vs 5/10)
└────────────┴──────────┴──────────┴─────────┴──────┴──────────┘

KEY INSIGHTS:
• XGBoost achieves BEST accuracy (96.88%) 526× faster than RF-CV!
• Built-in regularization (gamma, lambda, learning_rate) prevents overfitting
• RF needs CV for robust optimization, XGBoost doesn't (algorithm-dependent)
• Trade-off: XGBoost (best acc, fast) vs RF-CV (fewest features, slow)
```

**For draw.io:**
- 3 horizontal boxes (timeline left to right)
- Simple table below
- Arrows between boxes showing progression
- Color code: Red → Yellow → Green
- Time labels prominent (~60 min → 7.9 hrs → 54 sec)

**Why Essential:** Justifies final algorithm choice (XGBoost) and shows research rigor

---

## 7. 📉 **CIFAR-10 Statistical Comparison: When Budget Fails** (CRITICAL NEGATIVE RESULT)

### Purpose
Show the importance of optimization budget scaling - **Honest failure analysis**

```
┌─────────────────────────────────────────────────────────────┐
│     CIFAR-10: OPTIMIZATION FAILURE DUE TO BUDGET            │
│     30-Run Statistical Validation (Wilcoxon Signed-Rank)    │
└─────────────────────────────────────────────────────────────┘

SETUP:
• Dataset: CIFAR-10 ResNet50 features (2048-D, 68× larger than medical)
• Subset: 2000 train, 500 test (computational constraints)
• Configuration: 3 dholes/8 iterations (both loops) = ~576 evaluations
• Validation: 30 independent runs with different random seeds

┌──────────────────────────────────────────────────────────────┐
│               STATISTICAL COMPARISON TABLE                   │
├────────────────┬─────────────┬──────────┬──────┬────────────┤
│ Model          │ Mean Acc    │ Std Dev  │ Rank │ vs Default │
├────────────────┼─────────────┼──────────┼──────┼────────────┤
│ XGBoost        │ 83.27%      │ 1.25%    │  #1  │ Baseline   │
│ Default (All)  │             │          │      │            │
│ 2048 features  │             │          │      │            │
├────────────────┼─────────────┼──────────┼──────┼────────────┤
│ RF Default     │ 82.45%      │ 1.29%    │  #2  │ -0.82%     │
│ (All features) │             │          │      │            │
├────────────────┼─────────────┼──────────┼──────┼────────────┤
│ DIO-XGBoost    │ 81.91%      │ 1.38%    │  #3  │ -1.36%     │
│ OPTIMIZED      │             │          │      │ (WORSE!)   │
│ 598 features   │             │          │      │            │
├────────────────┼─────────────┼──────────┼──────┼────────────┤
│ Gradient Boost │ 80.49%      │ 1.51%    │  #4  │ -2.78%     │
└────────────────┴─────────────┴──────────┴──────┴────────────┘

WILCOXON SIGNED-RANK TEST RESULTS:
┌────────────────────────────────────────────────────────────┐
│ DIO-XGBoost (81.91%) vs XGBoost Default (83.27%)         │
│ • p-value: 7.15×10⁻⁵ (***)                               │
│ • Result: HIGHLY SIGNIFICANT WORSE                         │
│ • Mean difference: -1.36% (DIO underperforms!)            │
│ • Conclusion: Optimization FAILED - worse than defaults    │
└────────────────────────────────────────────────────────────┘

WHY DID IT FAIL?
┌────────────────────────────────────────────────────────────┐
│ 1. INSUFFICIENT BUDGET (Critical Issue)                   │
│    • Search space: 2048 features + 3 hyperparams = 2051-D │
│    • Budget provided: ~576 evaluations                     │
│    • Budget needed: ~10,000-50,000 evaluations            │
│    • Ratio: OFF BY 17-87×!                                │
│                                                            │
│ 2. OPTIMIZATION OVERFITTING (Again!)                      │
│    • Single-run result: 83.0% (looked good vs 80.8%)     │
│    • 30-run average: 81.91% (actually worse)              │
│    • Optimizer found config perfect for ONE split         │
│    • Didn't generalize across different partitions        │
│                                                            │
│ 3. DIMENSIONALITY CURSE                                   │
│    • Medical (30-D): 2,500 evals → SUCCESS ✅             │
│    • Vision (2048-D): 576 evals → FAILURE ❌              │
│    • Budget must scale with dimensionality²               │
│    • We underestimated by nearly 2 orders of magnitude    │
└────────────────────────────────────────────────────────────┘

KEY LESSONS:
• ⚠️  Even XGBoost's regularization can't save inadequate budgets
• ⚠️  Single-run results are DECEPTIVE - always validate statistically
• ⚠️  Optimization budget must scale with problem dimensionality
• ⚠️  What works for 30-D (medical) doesn't work for 2048-D (vision)
• ✅  Honest reporting: We discovered the limits of our approach

COMPUTATIONAL REALITY:
┌────────────────────────────────────────────────────────────┐
│ To succeed on CIFAR-10, we'd need:                        │
│ • 10-20 dholes (not 3)                                     │
│ • 20-50 iterations (not 8)                                 │
│ • 5-fold CV (not single-split)                            │
│ • Result: 50,000+ evaluations × ~0.4 hrs/100 = 200+ hours│
│ • Our budget: 576 evaluations = 3.6 hours (215.98 min)   │
│ • Gap: 55× more computation needed                        │
└────────────────────────────────────────────────────────────┘
```

**For draw.io:**
- Top: Statistical comparison table (4 models, clear winner)
- Middle: Wilcoxon test result box (red, emphasize WORSE)
- Bottom-left: "Why it failed" box (3 reasons)
- Bottom-right: "Lessons learned" box (key takeaways)
- Color: Red theme (negative result, but valuable insight)

**Why Essential:** Shows research honesty, explains failure, validates lessons about budget scaling

---

## 📝 FINAL Summary - All 7 Schemas (Draw.io Ready!)

**✅ All schemas are now simplified for quick drawing:**

1. ✅ **Cross-Domain Framework** (~40 lines) - Simple flow: 1 top + 2 parallel paths + 1 bottom
2. ✅ **Optimization Overfitting** (~45 lines) - 3 columns side-by-side comparison with configs
3. ✅ **Results Comparison** (table) - Already clean, includes success/failure analysis
4. ✅ **Nested Structure** (~35 lines) - 2 nested boxes + 1 output box
5. ✅ **Modularization & Fitness** (~50 lines) - 4 boxes with clear flow ⭐ MOST IMPORTANT
6. ✅ **Three Approaches** (~45 lines) - Timeline with 3 boxes + comparison table
7. ✅ **CIFAR-10 Statistical Failure** (~70 lines) - Honest negative result with lessons ⚠️ CRITICAL

**Estimated drawing time in draw.io:**
- Schema 1: 10 minutes
- Schema 2: 12 minutes (added config details)
- Schema 3: 8 minutes (table + notes)
- Schema 4: 8 minutes
- Schema 5: 15 minutes (most important, take time)
- Schema 6: 12 minutes
- Schema 7: 20 minutes (most complex, statistical results)
- **Total: ~85 minutes for all 7 schemas**

**Each schema now includes:**
- Simple box structure (max 4-5 boxes)
- Clear "For draw.io" instructions
- Minimal text, maximum clarity
- Real research numbers (updated with correct results)
- Color coding suggestions

**What each schema explains:**
- Schema 1: Big picture (scope) - Both domains
- Schema 2: Novel finding (contribution) - Algorithm-dependent overfitting
- Schema 3: Evidence (results) - Success AND failure quantified
- Schema 4: Architecture (nested loops) - How it works
- Schema 5: **Mechanism (fitness + optimization)** ← MOST TECHNICAL
- Schema 6: Justification (methodology) - Research evolution
- Schema 7: **Negative result (budget failure)** ← MOST HONEST, shows research integrity

**Updated Results Summary:**
- **Medical Success:** 96.88% ± 1.10% (Rank #1), 10 features, p=0.0047
- **Medical Alternative:** 96.55% ± 1.51% (Rank #1), 6 features, best interpretability
- **Medical Failure:** 94.37% ± 1.82% (Rank #6), discovered optimization overfitting
- **Vision Failure:** 81.91% ± 1.38% (Rank #3), worse than defaults (83.27%), p<0.0001

**Critical Configurations (CORRECTED):**
- **RF Single-Split:** 5/10 outer (dholes/iterations), 10/20 inner → ~10,000 evals
- **RF-CV:** 5/10 outer, 10/20 inner + 5-fold CV → ~10,000 evals × CV
- **XGBoost Medical:** 5/10 outer, 10/20 inner → ~10,000 evals (same as RF!)
- **XGBoost CIFAR-10:** 3/8 outer, 3/8 inner → ~576 evals (INSUFFICIENT!)

---

## 🎨 Quick Draw.io Tips

### Color Scheme
- **Medical Domain**: Blue (#3498db)
- **Vision Domain**: Orange (#e67e22)
- **DIO Components**: Green (#2ecc71)
- **Results**: Gold (#f39c12)
- **Errors/Issues**: Red (#e74c3c)

### Export Settings
- Resolution: 300 DPI minimum
- Format: PNG for paper
- File naming: `Fig1_CrossDomain.png`, `Fig2_Overfitting.png`, etc.
- [ ] Schema 5: Modularization & Fitness Function (optimization mechanism) ⭐ CRITICAL
- [ ] Schema 6: Three Approaches (research evolution)
- [ ] Schema 7: CIFAR-10 Statistical Failure (negative result, budget analysis) ⚠️ CRITICAL

**That's it! 7 schemas - Schema 5 is the MOST IMPORTANT for understanding HOW the optimization works, Schema 7 is CRITICAL for showing research honesty and budget lessons.**

---

**Last Updated**: December 8, 2025  
**Scope**: Medical + Vision (Cross-Domain)  
**Key Results (CORRECTED):**
- Medical Success: 96.88% ± 1.10% (XGBoost, Rank #1, p=0.0047)
- Medical Alternative: 96.55% ± 1.51% (RF-CV, Rank #1, 6 features)
- Vision Failure: 81.91% ± 1.38% (Rank #3, worse than 83.27% defaults, p<0.0001)
- Key Discovery: Algorithm-dependent optimization overfitting
- Key Lesson: Optimization budget must scale with dimensionality²
