# Schema Naming and Placement Guide
**Final Reference for DIO Multi-Domain Research**

## 📋 Schema File Naming Convention

**Standard format:** `schema#_descriptive_name.png` (all lowercase, no spaces in number)

### ✅ Required Schema Files (7 Total)

| File Name | Purpose | Status |
|-----------|---------|--------|
| `schema1_cross_domain_framework.png` | Cross-domain overview (big picture) | **RENAME** from `shema1 (1).png` |
| `schema2_algorithm_dependent_overfitting.png` | Three approaches comparison (key discovery) | **RENAME** from `Shema2 (1).png` |
| `schema3_cross_domain_results_table.png` | Success/failure quantification | **RENAME** from `shema3 (1).png` |
| `schema4_nested_optimization_structure.png` | Two-level hierarchy architecture | **RENAME** from `shema4 (1).png` |
| `schema5_fitness_driven_optimization.png` | Fitness mechanism (MOST IMPORTANT!) | **RENAME** from `shema5 (1).PNG` |
| `schema6_three_approaches_evolution.png` | Research progression timeline | **CREATE NEW** |
| `schema7_cifar10_statistical_failure.png` | Negative result & budget analysis | **CREATE NEW** |

### 📁 Supporting Files (Keep As-Is)

These files have specific technical purposes and don't need renaming:

- `dio_optimise_snippet.png` - Code snippet (Section 3)
- `dio_flowchart.png` - DIO algorithm flowchart (Section 3.5)
- `comparaison_table_of_results_for_pressure_vessel_design_problem_between_dio_and_other_algos.png` - Benchmark comparison (Section 3.5)
- `feature_selection_objective_func_rf.png` - RF feature fitness function (Section 4.2)
- `hyperparameter_objective_func_rf.png` - RF hyperparameter fitness function (Section 4.2)
- `outer_optimization_and_retreiving_results.png` - Results retrieval (Section 4.3)
- `xgboost_hyperparameters_search_space_cancer.png` - XGBoost search space medical (Section 8)
- `xgboost_hyperparameters_search_space_images.png` - XGBoost search space CIFAR-10 (Section 10)

---

## 📍 Schema Placement in Presentation

### Presentation File: `create_presentation_v2.py`

| Slide Location | Schema File | Purpose |
|----------------|-------------|---------|
| **After "1.4 Applications"** | `schema1_cross_domain_framework.png` | Show DIO's cross-domain versatility |
| **After "2.2 Methodology"** | `schema4_nested_optimization_structure.png` | Explain two-level architecture |
| **After "2.2 Methodology"** | `schema5_fitness_driven_optimization.png` | Show fitness mechanism ⭐ |
| **After "2.3.4 Medical Results"** | `schema6_three_approaches_evolution.png` | Timeline of research evolution |
| **Before "Summary: Three Approaches"** | `schema2_algorithm_dependent_overfitting.png` | Key discovery visualization |
| **After "2.4.1 CIFAR-10 Results"** | `schema7_cifar10_statistical_failure.png` | Honest negative result |
| **Slide "2.5 Cross-Domain"** | `schema3_cross_domain_results_table.png` | Complete quantitative comparison |

**All presentation schema references updated with:**
- Standardized file paths
- Descriptive comments explaining purpose
- Clear captions indicating success/failure

---

## 📍 Schema Placement in Report

### Report File: `report.tex`

| Section | Figure Label | Schema File | Caption Summary |
|---------|--------------|-------------|-----------------|
| **Section 2 (Introduction)** | `fig:schema1_crossdomain` | `schema1_cross_domain_framework.png` | Cross-domain DIO framework overview |
| **Section 4.1 (Architecture)** | `fig:schema4_nested` | `schema4_nested_optimization_structure.png` | Nested optimization architecture |
| **Section 4.2 (Fitness)** | `fig:schema5_modularization` | `schema5_fitness_driven_optimization.png` | Fitness-driven mechanism ⭐ |
| **Section 5 (RF Results)** | `fig:schema2_overfitting` | `schema2_algorithm_dependent_overfitting.png` | Algorithm-dependent overfitting discovery |
| **Section 10 (XGBoost Summary)** | `fig:schema6_evolution` | `schema6_three_approaches_evolution.png` | Three-approach research evolution |
| **Section 10 (Cross-Domain)** | `fig:schema3_comparison` | `schema3_cross_domain_results_table.png` | Cross-domain results comparison |
| **Section 10.3 (CIFAR Failure)** | `fig:schema7_cifar_failure` | `schema7_cifar10_statistical_failure.png` | CIFAR-10 statistical failure analysis |

**All report schema references updated with:**
- Corrected file paths (no more `shema` or `Shema`)
- Updated captions with accurate statistics
- Proper figure labels for cross-referencing

---

## 🔄 Renaming Actions Required

### Step 1: Rename Existing Files

In folder: `c:\Users\LENOVO\Desktop\Dio_expose\schemas and snippets\`

```powershell
# PowerShell commands to rename files
Rename-Item "shema1 (1).png" "schema1_cross_domain_framework.png"
Rename-Item "Shema2 (1).png" "schema2_algorithm_dependent_overfitting.png"
Rename-Item "shema3 (1).png" "schema3_cross_domain_results_table.png"
Rename-Item "shema4 (1).png" "schema4_nested_optimization_structure.png"
Rename-Item "shema5 (1).PNG" "schema5_fitness_driven_optimization.png"
```

### Step 2: Create Missing Schemas

**Schema 6: `schema6_three_approaches_evolution.png`**
- **Content:** Timeline showing RF Single-Split → RF-CV → XGBoost
- **Details:** 3 boxes horizontally with arrows, comparison table below
- **Key Info:** Times (~60 min, 7.9 hrs, 54 sec), Results (94.37%, 96.55%, 96.88%), Ranks (#6, #1, #1)
- **Reference:** See ESSENTIAL_SCHEMAS_CLEAN.md Schema 6 specification

**Schema 7: `schema7_cifar10_statistical_failure.png`**
- **Content:** Statistical comparison table + Wilcoxon test results + failure analysis
- **Details:** 4-model table, p-value box, "Why it failed" box (3 reasons)
- **Key Info:** 81.91% vs 83.27% (p<0.0001), Budget 576 vs needed 10K-50K
- **Reference:** See ESSENTIAL_SCHEMAS_CLEAN.md Schema 7 specification

---

## 📊 Schema Purpose Summary

### Research Narrative Arc

1. **Schema 1** (Cross-Domain) → "Here's what we did across both domains"
2. **Schema 4** (Nested Structure) → "Here's how the optimization architecture works"
3. **Schema 5** (Fitness Mechanism) ⭐ → "Here's HOW fitness drives the whole process"
4. **Schema 2** (Algorithm Overfitting) → "Here's our key discovery about algorithms"
5. **Schema 6** (Evolution) → "Here's how we evolved our approach"
6. **Schema 3** (Results Table) → "Here's the complete quantitative evidence"
7. **Schema 7** (CIFAR Failure) → "Here's our honest negative result"

### Technical Depth Levels

- **Beginner-friendly:** Schema 1 (big picture), Schema 6 (timeline)
- **Intermediate:** Schema 4 (nested loops), Schema 2 (overfitting comparison)
- **Advanced:** Schema 5 (complete mechanism) ⭐ MOST TECHNICAL
- **Research honesty:** Schema 7 (failure analysis), Schema 3 (success + failure)

### Key Statistics per Schema

| Schema | Medical Results | CIFAR-10 Results |
|--------|-----------------|------------------|
| Schema 1 | 96.88% acc, 10 feat | 81.91% acc, 598 feat |
| Schema 2 | RF: 94.37% → XGB: 96.88% | N/A (medical only) |
| Schema 3 | 96.88% ± 1.10% (Rank #1) | 81.91% ± 1.38% (Rank #3) |
| Schema 4 | 10K evals, 54 sec | 576 evals, 215.98 min |
| Schema 5 | F = 0.99×(1-Acc) + 0.01×(Feat/Total) | Same formula |
| Schema 6 | 94.37% → 96.55% → 96.88% | N/A (medical only) |
| Schema 7 | N/A (success case) | 81.91% vs 83.27% (p<0.0001) |

---

## ✅ Verification Checklist

### Files Updated ✓
- [x] `ESSENTIAL_SCHEMAS_CLEAN.md` - Added naming convention and placement guide
- [x] `create_presentation_v2.py` - All 7 schema references updated
- [x] `report.tex` - All 7 schema references updated with correct paths

### Schema Status
- [x] Schema 1 - Reference updated (needs file rename)
- [x] Schema 2 - Reference updated (needs file rename)
- [x] Schema 3 - Reference updated (needs file rename)
- [x] Schema 4 - Reference updated (needs file rename)
- [x] Schema 5 - Reference updated (needs file rename)
- [ ] Schema 6 - Reference updated (needs file creation)
- [ ] Schema 7 - Reference updated (needs file creation)

### Next Actions Required
1. **Rename 5 existing schema files** (see Step 1 above)
2. **Create Schema 6** - Three approaches evolution timeline
3. **Create Schema 7** - CIFAR-10 statistical failure analysis
4. **Verify all references** - Check presentation and report compile correctly

---

## 🎨 Design Guidelines for New Schemas

### Schema 6 (Three Approaches Evolution)
- **Layout:** 3 horizontal boxes (timeline left to right)
- **Colors:** Red (RF-Single) → Yellow (RF-CV) → Green (XGBoost)
- **Font:** Clear, large text for times and accuracies
- **Table:** Below timeline, 4 columns (Approach, Time, Val Acc, Rank)

### Schema 7 (CIFAR-10 Failure)
- **Layout:** Top = Table, Middle = Wilcoxon box, Bottom = Two boxes (Why/Lessons)
- **Colors:** Red theme (emphasize negative result)
- **Font:** Bold for p-value and key statistics
- **Emphasis:** Budget insufficiency (576 vs 10K-50K needed)

---

## 📝 Final Notes

**Schema 5 is the MOST IMPORTANT technical schema** - it shows the complete optimization mechanism with fitness driving both loops. Take extra care with this one.

**Schema 7 shows research integrity** - honest reporting of negative results is critical for scientific credibility.

**All schemas now have unique purposes** - no duplication, each tells a specific part of the research story.

**Naming consistency matters** - all lowercase, underscores instead of spaces, sequential numbering.

---

**Last Updated:** December 8, 2025
**Status:** Presentation and Report files updated, awaiting schema file renaming
**Next Step:** Execute PowerShell rename commands, then create Schema 6 and Schema 7
