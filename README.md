# Dholes-Inspired Optimization (DIO) for Feature Selection and Hyperparameter Tuning

This project implements the **Dholes-Inspired Optimization (DIO)** algorithm for simultaneous feature selection and hyperparameter optimization of Random Forest classifiers, tested on the Breast Cancer Wisconsin (Diagnostic) dataset.

## 🎯 Project Overview

The DIO algorithm is a nature-inspired metaheuristic optimization algorithm based on the cooperative hunting behavior of dholes (Asiatic wild dogs). This implementation uses DIO for:

1. **Feature Selection**: Identifying the most informative features from the dataset
2. **Hyperparameter Optimization**: Finding optimal Random Forest hyperparameters
3. **Nested Optimization**: Combining both tasks where hyperparameter optimization is the outer loop and feature selection is the inner loop

## 🏆 Key Results

### Performance on Breast Cancer Dataset

- **Test Accuracy**: 100% (Perfect classification!)
- **Features Used**: 8 out of 30 (73% feature reduction)
- **Optimized Hyperparameters**:
  - n_estimators: 193
  - max_depth: 13
  - min_samples_split: 4
  - min_samples_leaf: 1

### Comparison with Baseline Models

| Rank | Model | Accuracy | Features |
|------|-------|----------|----------|
| 🥇 1st | **DIO-Optimized RF** | **100.00%** | **8** |
| 🥈 2nd | XGBoost (Selected) | 99.42% | 8 |
| 🥉 3rd | RF Default (Selected) | 98.83% | 8 |
| 4th | Logistic Regression | 97.66% | 30 |
| 5th | RF Default (All) | 97.08% | 30 |
| 6th | XGBoost (All) | 96.49% | 30 |
| 7th | Gradient Boosting | 95.91% | 30 |
| 8th | KNN | 95.91% | 30 |
| 9th | Naive Bayes | 94.15% | 30 |
| 10th | SVM | 93.57% | 30 |

## 📁 Project Structure

```
Dio_expose/
├── dio.py                              # DIO algorithm implementation
├── main.py                             # Main script with optimization and comparison
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
├── model_comparison_results.csv        # Results table (generated)
├── optimization_results.json           # Optimization details (generated)
├── model_comparison_visualization.png  # Comparison charts (generated)
└── roc_curve_comparison.png           # ROC curves (generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/dio-optimization.git
cd dio-optimization
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

Run the main optimization script:

```bash
python main.py
```

This will:
1. Load the Breast Cancer dataset from scikit-learn
2. Run nested DIO optimization (hyperparameter → feature selection)
3. Compare DIO-optimized Random Forest with baseline models
4. Generate visualizations and save results

### Output Files

After running `main.py`, the following files will be generated:

- **`model_comparison_results.csv`**: Detailed comparison metrics for all models
- **`optimization_results.json`**: Best features and hyperparameters found by DIO
- **`model_comparison_visualization.png`**: 6-panel visualization including:
  - Accuracy comparison
  - F1-Score comparison
  - Training time comparison
  - Top 3 models detailed metrics
  - Confusion matrix
  - Feature importance
- **`roc_curve_comparison.png`**: ROC curves for key models

## 🧠 Algorithm Details

### DIO Algorithm

The Dholes-Inspired Optimization algorithm simulates the hunting strategies of dhole packs:

1. **Chasing (Exploitation)**: Dholes move toward the best solution (alpha dhole)
2. **Scouting (Exploration)**: Dholes explore new areas by following random pack members
3. **Pack Cooperation**: Dholes adjust positions based on the pack center

### Nested Optimization Structure

```
Outer Loop: Hyperparameter Optimization
├── For each hyperparameter set:
│   └── Inner Loop: Feature Selection
│       ├── Test different feature combinations
│       └── Return best feature subset
└── Select hyperparameters with best feature selection fitness
```

### Fitness Functions

**Feature Selection Fitness**:
```
fitness = 0.99 * (1 - accuracy) + 0.01 * (n_selected / n_total)
```
- Balances accuracy maximization with feature minimization
- 99% weight on accuracy, 1% weight on feature count

**Hyperparameter Fitness**:
- The fitness of a hyperparameter set is determined by the best feature selection fitness achieved with those parameters

## 🔧 Customization

### Adjusting DIO Parameters

In `main.py`, you can modify:

```python
# Hyperparameter optimization (outer loop)
hp_dio = DIO(
    objective_function=hyperparameter_objective_function,
    search_space=hp_search_space,
    n_dholes=5,          # Number of candidate solutions
    max_iterations=10    # Number of optimization iterations
)

# Feature selection (inner loop)
fs_dio = DIO(
    objective_function=feature_selection_objective_function,
    search_space=fs_search_space,
    n_dholes=10,         # Number of candidate solutions
    max_iterations=20    # Number of optimization iterations
)
```

### Hyperparameter Search Space

Modify the search ranges in `main.py`:

```python
hp_search_space = [
    [10, 200],    # n_estimators
    [1, 20],      # max_depth (1 = None)
    [2, 10],      # min_samples_split
    [1, 10]       # min_samples_leaf
]
```

## 📊 Visualizations

The project generates comprehensive visualizations:

1. **Accuracy Bar Chart**: Compare all models
2. **F1-Score Comparison**: Performance metrics
3. **Training Time**: Computational efficiency
4. **Detailed Metrics**: Top 3 models comparison
5. **Confusion Matrix**: DIO-optimized model predictions
6. **Feature Importance**: Most important selected features
7. **ROC Curves**: Model discrimination capability

## 📚 Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

See `requirements.txt` for specific versions.

## 🔬 Research Reference

This implementation is based on the DIO algorithm. For the original research paper, please refer to:

**Ali El Romeh, Václav Snášel, Seyedali Mirjalili** - "Dholes-Inspired Optimization (DIO): A Nature-Inspired Algorithm for Engineering Optimization Problems", *Cluster Computing*, 2025.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- Original DIO algorithm by Ali El Romeh, Václav Snášel, and Seyedali Mirjalili
- Breast Cancer Wisconsin (Diagnostic) dataset from UCI Machine Learning Repository
- scikit-learn community for excellent machine learning tools

---

**Note**: This is an educational implementation. For production use, consider additional validation and testing.
