# Visio Schemas for Research Paper

## 📐 Required Diagrams for DIO Research Paper

Your professor requested Visio for schemas. Here are the essential diagrams to create:

---

## 1. 📊 DIO Algorithm Flowchart

### Purpose
Show the complete DIO optimization process flow

### Elements to Include

```
START
  ↓
[Initialize Population]
- N dholes (search agents)
- Random positions in search space
  ↓
[Evaluate Fitness]
- Calculate objective function
- Identify alpha dhole (best solution)
  ↓
[Main Loop: For each iteration]
  ↓
  [For each dhole]
    ↓
    [Movement Strategy 1: Chase Alpha]
    X1 = alpha_position + r1 × (alpha - current)
    ↓
    [Movement Strategy 2: Random Pack Member]
    X2 = random_dhole + r2 × (random - current)
    ↓
    [Movement Strategy 3: Pack Center]
    X3 = mean(all_dholes) + r3 × (mean - current)
    ↓
    [Update Position]
    X_new = (X1 + X2 + X3) / 3
    ↓
    [Apply Boundary Constraints]
    Clip to [lower_bound, upper_bound]
    ↓
  [End For Each Dhole]
  ↓
  [Evaluate New Fitness]
  ↓
  [Update Alpha (Best Solution)]
  ↓
  [Check Convergence]
  ↓
[End Main Loop]
  ↓
[Return Best Solution (Alpha)]
  ↓
END
```

### Visio Tips
- Use **Flowchart shapes**: Process, Decision, Data
- **Colors**: 
  - Initialization: Light Blue
  - Main loop: Green
  - Movement strategies: Orange
  - Evaluation: Purple
  - Output: Dark Blue
- **Connectors**: Use arrows to show flow
- **Swimlanes**: Consider using 3 lanes for the 3 movement strategies

---

## 2. 🔄 Nested Optimization Structure

### Purpose
Illustrate the hierarchical optimization (hyperparameters → features)

### Diagram Layout

```
┌─────────────────────────────────────────────────────────────┐
│          OUTER LOOP: Hyperparameter Optimization            │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  DIO Optimizer (3 dholes, 5 iterations)               │  │
│  │  Search Space:                                         │  │
│  │    • n_estimators: [10, 200]                          │  │
│  │    • max_depth: [1, 20]                               │  │
│  │    • min_samples_split: [2, 10]                       │  │
│  │    • min_samples_leaf: [1, 10]                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         INNER LOOP: Feature Selection                 │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  DIO Optimizer (5 dholes, 10 iterations)        │  │  │
│  │  │  Search Space:                                   │  │  │
│  │  │    • Feature vector: [0,1]^30                   │  │  │
│  │  │    • Threshold: 0.5                             │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                          ↓                              │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Fitness Evaluation                             │  │  │
│  │  │  F = 0.99×(1-accuracy) + 0.01×(features/total) │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                          ↓                              │  │
│  │           Return: Best Feature Subset                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│      Store: Best features for current hyperparameters        │
│                            ↓                                  │
│            Return: Fitness of best feature subset            │
└─────────────────────────────────────────────────────────────┘
                              ↓
                  FINAL OUTPUT:
                  • Optimized Hyperparameters
                  • Selected Features (8/30)
                  • Best Fitness Score
```

### Visio Tips
- Use **Container shapes** for nested loops
- **Different border styles**: Solid for outer, dashed for inner
- **Arrows**: Show data flow between loops
- **Text boxes**: Label each component clearly

---

## 3. 🧬 DIO Movement Strategies (Pack Hunting Behavior)

### Purpose
Visualize the three movement strategies inspired by dhole hunting

### Diagram Elements

```
┌────────────────────────────────────────────────────────────────┐
│                    Pack Hunting Strategies                      │
└────────────────────────────────────────────────────────────────┘

Strategy 1: CHASING ALPHA (Exploitation)
┌─────────────────────────────────────┐
│      Current                        │
│      Position  ──────→  Alpha       │
│        (Xi)           (Best)        │
│                                     │
│  X1 = Alpha + r1 × (Alpha - Xi)    │
└─────────────────────────────────────┘

Strategy 2: RANDOM PACK MEMBER (Exploration)
┌─────────────────────────────────────┐
│      Current           Random        │
│      Position  ──────→  Dhole       │
│        (Xi)            (Xr)         │
│                                     │
│  X2 = Xr + r2 × (Xr - Xi)          │
└─────────────────────────────────────┘

Strategy 3: PACK CENTER (Cooperation)
┌─────────────────────────────────────┐
│      Current           Mean of       │
│      Position  ──────→  All Pack    │
│        (Xi)            (Xmean)      │
│                                     │
│  X3 = Xmean + r3 × (Xmean - Xi)    │
└─────────────────────────────────────┘

FINAL POSITION:
┌─────────────────────────────────────┐
│    Xnew = (X1 + X2 + X3) / 3       │
│                                     │
│    Average of all three strategies  │
└─────────────────────────────────────┘
```

### Visio Tips
- Use **3D shapes** for dholes (icons or circles)
- **Arrows** with different colors for each strategy
- **Mathematical notation** in text boxes
- **Legend** explaining r1, r2, r3 (random numbers [0,1])

---

## 4. 📈 Experimental Design Flowchart

### Purpose
Show the complete experimental methodology

```
START: Breast Cancer Dataset
         (569 samples, 30 features)
                  ↓
     ┌────────────────────────────┐
     │   Data Preprocessing       │
     │   • Normalization          │
     │   • Train/Test Split       │
     │     (70/30, stratified)    │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   DIO Optimization         │
     │   • Nested structure       │
     │   • Hyperparameter tuning  │
     │   • Feature selection      │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   Statistical Validation   │
     │   • 30 independent runs    │
     │   • Different splits       │
     │   • Random seeds: 42-71    │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   Baseline Comparison      │
     │   • 10 ML algorithms       │
     │   • Same test sets         │
     │   • Performance metrics    │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   Statistical Testing      │
     │   • Wilcoxon signed-rank   │
     │   • p-value analysis       │
     │   • Significance levels    │
     └────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   Results Analysis         │
     │   • Accuracy: 94.72%       │
     │   • Features: 8/30 (73%↓)  │
     │   • Rank: 7/10             │
     └────────────────────────────┘
                  ↓
                 END
```

### Visio Tips
- Use **BPMN notation** (Business Process Model)
- **Swimlanes**: Different phases of experiment
- **Milestones**: Key decision points
- **Data stores**: Show where results are saved

---

## 5. 🎯 Model Comparison Architecture

### Purpose
Visual comparison of model architectures and performance

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: 30 Features                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │                                       │
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│   DIO-Optimized  │                  │   Traditional    │
│       Path       │                  │     Models       │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│ Feature Selection│                  │   All Features   │
│   (DIO: 8/30)   │                  │      (30/30)     │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│  Hyperparameter  │                  │     Default      │
│   Optimization   │                  │   Parameters     │
│ (DIO-optimized)  │                  │                  │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│ Random Forest    │                  │  XGBoost, SVM,   │
│ n_est=193        │                  │  KNN, NB, etc.   │
│ depth=13         │                  │                  │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
        └───────────────────┬───────────────────┘
                            ↓
                  ┌──────────────────┐
                  │  Performance     │
                  │  Evaluation      │
                  │  (30 runs)       │
                  └──────────────────┘
                            ↓
                  ┌──────────────────┐
                  │ Statistical Test │
                  │ (Wilcoxon)       │
                  └──────────────────┘
                            ↓
                    FINAL RESULTS
```

### Visio Tips
- Use **Block diagrams**
- **Two parallel paths**: DIO vs Traditional
- **Color coding**: Green for DIO, Blue for traditional
- **Metrics boxes**: Show accuracy, features, time

---

## 6. 📊 Performance Trade-off Diagram

### Purpose
Visualize Pareto frontier (accuracy vs. complexity)

```
High Accuracy
    ↑
    │                    ● XGBoost (All)
    │                  ● RF (All)
    │                ● Gradient Boosting
    │              ● XGBoost (Selected)
    │            ● Logistic Regression
    │          ● RF Default (Selected)
    │        ● DIO-Optimized RF ←── Pareto Optimal
    │      ● Naive Bayes                  Zone
    │    ● KNN
    │  ● SVM
    │
    └────────────────────────────────────────→
Low Complexity                    High Complexity
(Fewer Features)                (More Features)

Legend:
● = Individual model
Gray area = Pareto-dominated region
Green area = Pareto-optimal region
```

### Visio Tips
- Use **2D chart** background
- **Scatter plot** with labeled points
- **Shaded regions** for Pareto zones
- **Arrows** highlighting DIO position

---

## 7. 🔬 Fitness Function Diagram

### Purpose
Explain the multi-objective fitness function

```
┌────────────────────────────────────────────────────────┐
│              FITNESS FUNCTION COMPONENTS               │
└────────────────────────────────────────────────────────┘

                    Objective 1
                  ┌──────────────┐
                  │  Maximize    │
                  │  Accuracy    │
                  └──────────────┘
                         ↓
                  Weight: 0.99
                         ↓
        Component 1: 0.99 × (1 - accuracy)
                         │
                         ├──────────────┐
                         │              │
                         ↓              ↓
                    Objective 2   Weighted Sum
                  ┌──────────────┐      ↓
                  │  Minimize    │  ┌────────────┐
                  │  Features    │  │  FITNESS   │
                  └──────────────┘  │    F(x)    │
                         ↓          └────────────┘
                  Weight: 0.01
                         ↓
        Component 2: 0.01 × (selected/total)

FORMULA:
┌────────────────────────────────────────────────────────┐
│ F(x) = 0.99×(1 - Acc) + 0.01×(N_selected / N_total)  │
│                                                        │
│ Where:                                                 │
│   Acc = Classification accuracy                       │
│   N_selected = Number of selected features            │
│   N_total = Total number of features (30)            │
│                                                        │
│ Goal: MINIMIZE F(x)                                   │
└────────────────────────────────────────────────────────┘
```

### Visio Tips
- Use **Equation editor** for formulas
- **Flowchart** showing components merging
- **Weights** clearly labeled with percentages
- **Example calculation** in a separate box

---

## 📝 How to Create in Visio

### Step-by-Step Process

1. **Open Microsoft Visio**
   - File → New → Flowchart (or Basic Diagram)

2. **For Each Diagram**:
   - Select appropriate template (Flowchart, BPMN, Block Diagram)
   - Drag and drop shapes from the left panel
   - Connect shapes with arrows/connectors
   - Add text labels (double-click shapes)
   - Format colors, fonts, and styles

3. **Formatting Tips**:
   - **Consistent colors**: Use a color scheme throughout
   - **Font**: Arial or Calibri, size 10-12 for text, 14-16 for titles
   - **Alignment**: Use Visio's alignment tools (Ctrl+Shift+arrow keys)
   - **Grid**: Enable snap-to-grid for clean layouts
   - **Export**: File → Save As → PNG or PDF (high resolution)

4. **Professional Touch**:
   - Add a **legend** for symbols and colors
   - Include **figure numbers** and captions
   - Use **page borders** for presentation
   - **Version control**: Save each diagram separately

---

## 📐 Recommended Visio Templates

| Diagram Type | Visio Template | Shapes to Use |
|--------------|----------------|---------------|
| 1. DIO Flowchart | Basic Flowchart | Process, Decision, Terminator |
| 2. Nested Optimization | Cross-Functional Flowchart | Swimlanes, Containers |
| 3. Movement Strategies | Block Diagram | 3D shapes, Arrows |
| 4. Experimental Design | BPMN Diagram | Process, Data stores |
| 5. Model Comparison | Block Diagram with Tree | Hierarchical blocks |
| 6. Trade-off | Charts and Graphs | Scatter plot template |
| 7. Fitness Function | Equation shapes | Mathematical notation |

---

## 🎨 Color Scheme Recommendation

```
Primary: #2c3e50 (Dark Blue) - Main flow
Secondary: #3498db (Blue) - Sub-processes
Success: #2ecc71 (Green) - DIO components
Warning: #f39c12 (Orange) - Decision points
Danger: #e74c3c (Red) - Error/boundary conditions
Info: #9b59b6 (Purple) - Evaluation steps
```

---

## ✅ Checklist Before Submission

- [ ] All diagrams have clear titles
- [ ] Figure numbers assigned (Figure 1, Figure 2, etc.)
- [ ] Captions written for each diagram
- [ ] Consistent color scheme across all diagrams
- [ ] Mathematical notation is correct
- [ ] Text is readable (minimum 10pt font)
- [ ] Arrows show clear flow direction
- [ ] Legend provided where necessary
- [ ] High-resolution exports (300 DPI minimum)
- [ ] File names follow convention (Fig1_DIO_Flowchart.png)

---

**Alternative Tools** (if Visio not available):
- **draw.io** (free, web-based)
- **Lucidchart** (online, similar to Visio)
- **yEd** (free, desktop application)
- **Microsoft PowerPoint** (basic diagrams)

---

**Files to Reference While Creating Diagrams**:
- `dio.py` - Algorithm implementation details
- `main.py` - Nested optimization structure
- `STATISTICAL_RESULTS.md` - Results for comparison diagrams
- `statistical_comparison_summary.csv` - Performance data

---

**Last Updated**: October 25, 2025  
**Purpose**: Research paper schema creation guide  
**Professor Requirement**: Use Microsoft Visio for all schemas
