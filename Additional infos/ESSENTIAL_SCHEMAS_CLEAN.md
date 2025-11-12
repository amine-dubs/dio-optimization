# Essential Schemas for DIO Multi-Domain Research

## 📐 ONLY 6 Schemas - Maximum Impact, Minimum Space

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
    │ 96.34% acc    │              │ 83.6% acc     │
    │ 17/30 feat    │              │ 853/2048 feat │
    │ 43% reduction │              │ 58% reduction │
    └───────────────┘              └───────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ↓
                ┌───────────────────────┐
                │ VALIDATED FRAMEWORK   │
                │ • 68× scale-up        │
                │ • Both domains work! │
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
│ Optimization:          │  │ Optimization:          │  │ Optimization:          │
│ 100% (overfit!)        │  │ 95.91% (CV avg)        │  │ 98.83% (holdout)       │
│                        │  │                        │  │                        │
│        ↓               │  │        ↓               │  │        ↓               │
│                        │  │                        │  │                        │
│ Validation:            │  │ Validation:            │  │ Validation:            │
│ 94.72% (poor)          │  │ 96.26% (good)          │  │ 96.34% (BEST!) 🏆     │
│ Rank: #7               │  │ Rank: #3               │  │ Rank: #1               │
│                        │  │                        │  │                        │
│ Time: 1 min            │  │ Time: 7.9 hrs          │  │ Time: 54 sec           │
│                        │  │                        │  │                        │
│ ❌ OVERFITTING         │  │ ✅ FIXED               │  │ ✅ NO ISSUE            │
│    (memorized split)   │  │    (but slow)          │  │    (built-in reg.)     │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘

KEY DISCOVERY:
┌──────────────────────────────────────────────────────────────┐
│ XGBoost's multi-layer regularization prevents meta-overfitting│
│ → Single-split is SUFFICIENT and 870× FASTER than RF-CV      │
└──────────────────────────────────────────────────────────────┘
```

**For draw.io:**
- 3 vertical boxes side-by-side (RF-Single | RF-CV | XGBoost)
- Each shows: Optimization → Validation → Result
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
Training Samples        │  455                     │  2,000
────────────────────────┼──────────────────────────┼──────────────────────
Best Algorithm          │  XGBoost                 │  XGBoost
Baseline Accuracy       │  94.74% (defaults)       │  80.8% (subset)
                        │                          │  85.0% (full data)
────────────────────────┼──────────────────────────┼──────────────────────
DIO-Optimized           │  96.34% 🏆               │  83.6%
Accuracy Gain           │  +1.60%                  │  +2.8%
────────────────────────┼──────────────────────────┼──────────────────────
Feature Reduction       │  43% (30 → 17)           │  58.35% (2048 → 853)
Inference Speedup       │  1.8×                    │  2.4×
Optimization Time       │  54 seconds              │  5.4 hours
────────────────────────┼──────────────────────────┼──────────────────────
Statistical Rank        │  #1 out of 10            │  N/A (subset exp.)
────────────────────────┼──────────────────────────┼──────────────────────
Key Advantage           │  Best accuracy           │  Edge deployment
                        │  + Moderate reduction    │  (2.4× faster)
────────────────────────┴──────────────────────────┴──────────────────────

✅ CONSISTENT PATTERN: DIO achieves substantial improvements in BOTH domains
✅ SCALE VALIDATION: 30-D → 2048-D (68× dimensionality increase)
```

**Why Essential:** Quantifies all results, proves cross-domain effectiveness

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
Vision:  24 × 24 = 576 → 5.4 hours
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
                │ Medical: 96.34% acc  │
                │          17/30 feat  │
                │                      │
                │ Vision:  83.6% acc   │
                │          853/2048 feat│
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
│  Vision:  8 × 8 = 64 (but 3 dholes) = 576 → 5.4 hours    │
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
│ ❌ Overfits  │      │ ✅ Fixed     │      │ ✅ BEST      │
│ Opt: 100%    │      │ Uses 5-Fold  │      │ Built-in     │
│ Val: 94.72%  │      │ Val: 96.26%  │      │ regularize   │
│ Rank: #7     │      │ Rank: #3     │      │ Val: 96.34%  │
│              │      │              │      │ Rank: #1     │
│ Time: 1 min  │      │ Time: 7.9 hr │      │ Time: 54 sec │
└──────────────┘      └──────────────┘      └──────────────┘
   Discovery:            Discovery:            Discovery:
   Single-split          CV fixes              XGBoost doesn't
   causes overfit        overfitting           need CV!

FINAL COMPARISON TABLE:
┌────────────┬──────────┬──────────┬─────────┬──────┐
│ Approach   │ Time     │ vs Best  │ Val Acc │ Rank │
├────────────┼──────────┼──────────┼─────────┼──────┤
│ RF-Single  │ 1 min    │ 1×       │ 94.72%  │ #7   │
│ RF-CV      │ 7.9 hrs  │ 474×     │ 96.26%  │ #3   │
│ XGBoost    │ 54 sec   │ 0.9×     │ 96.34%🏆│ #1   │
└────────────┴──────────┴──────────┴─────────┴──────┘

KEY INSIGHT: XGBoost achieves BEST accuracy 870× faster than RF-CV!
Built-in regularization (gamma, lambda) prevents overfitting without CV.
```

**For draw.io:**
- 3 horizontal boxes (timeline left to right)
- Simple table below
- Arrows between boxes showing progression
- Color code: Red → Yellow → Green
- Time labels prominent (1 min → 7.9 hrs → 54 sec)

**Why Essential:** Justifies final algorithm choice (XGBoost)

**Why Essential:** Justifies your methodology choices and shows research rigor

---

## 📝 FINAL Summary - All 6 Schemas (Draw.io Ready!)

**✅ All schemas are now simplified for quick drawing:**

1. ✅ **Cross-Domain Framework** (~40 lines) - Simple flow: 1 top + 2 parallel paths + 1 bottom
2. ✅ **Optimization Overfitting** (~30 lines) - 3 columns side-by-side comparison
3. ✅ **Results Comparison** (table) - Already clean, just draw table
4. ✅ **Nested Structure** (~35 lines) - 2 nested boxes + 1 output box
5. ✅ **Modularization & Fitness** (~50 lines) - 4 boxes with clear flow ⭐ MOST IMPORTANT
6. ✅ **Three Approaches** (~40 lines) - Timeline with 3 boxes + comparison table

**Estimated drawing time in draw.io:**
- Schema 1: 10 minutes
- Schema 2: 10 minutes
- Schema 3: 5 minutes (table)
- Schema 4: 8 minutes
- Schema 5: 15 minutes (most important, take time)
- Schema 6: 10 minutes
- **Total: ~60 minutes for all 6 schemas**

**Each schema now includes:**
- Simple box structure (max 4-5 boxes)
- Clear "For draw.io" instructions
- Minimal text, maximum clarity
- Real research numbers
- Color coding suggestions

**What each schema explains:**
- Schema 1: Big picture (scope)
- Schema 2: Novel finding (contribution)
- Schema 3: Evidence (results)
- Schema 4: Architecture (nested loops)
- Schema 5: **Mechanism (fitness + optimization)** ← MOST TECHNICAL
- Schema 6: Justification (methodology)

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

**That's it! Only 6 schemas needed - Schema 5 is the MOST IMPORTANT for understanding HOW the optimization works.**

---

**Last Updated**: November 11, 2025  
**Scope**: Medical + Vision (Cross-Domain)  
**Key Results**: 96.34% (Medical), 83.6% (Vision)
