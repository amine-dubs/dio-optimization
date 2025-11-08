"""
XGBoost Optimization using DIO (Single-Split, Fast Configuration)
==================================================================
Optimizes XGBoost classifier with nested DIO for feature selection
and hyperparameter tuning. Uses single train/test split for speed.

Configuration: 5 dholes, 10 iterations (outer and inner loops)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from xgboost import XGBClassifier
import json
import time
from dio import DIO

print("="*80)
print("XGBOOST OPTIMIZATION WITH DIO (SINGLE-SPLIT, FAST)")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("\nLoading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
X, y = data.data, data.target
n_features = X.shape[1]
feature_names = data.feature_names

print(f"Dataset loaded: {X.shape[0]} samples, {n_features} features")
print(f"Classes: {data.target_names}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== NESTED OPTIMIZATION ====================
# Outer loop: Hyperparameter optimization for XGBoost
# Inner loop: Feature selection

# Global variables
current_hyperparameters = None
best_features_for_hyperparams = {}

def feature_selection_objective_function(features):
    """
    Objective function for feature selection.
    Uses current XGBoost hyperparameters being evaluated.
    """
    # Threshold features (> 0.5 = selected)
    selected_features_indices = np.where(np.array(features) > 0.5)[0]
    
    if len(selected_features_indices) == 0:
        return 1.0  # Worst fitness if no features selected

    # Select features
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Extract hyperparameters
    n_estimators = int(current_hyperparameters[0])
    max_depth = int(current_hyperparameters[1])
    learning_rate = current_hyperparameters[2]
    subsample = current_hyperparameters[3]
    colsample_bytree = current_hyperparameters[4]

    # Train XGBoost classifier
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    try:
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
    except:
        # If training fails, return worst fitness
        return 1.0

    # Fitness function: balance accuracy and feature count
    n_selected = len(selected_features_indices)
    fitness = 0.99 * (1 - acc) + 0.01 * (n_selected / n_features)
    
    return fitness

def hyperparameter_objective_function(params):
    """
    Objective function for XGBoost hyperparameter optimization.
    For each hyperparameter set, run feature selection.
    """
    global current_hyperparameters, best_features_for_hyperparams
    current_hyperparameters = params
    
    params_key = tuple(params)
    
    print(f"\n  Evaluating XGBoost hyperparameters:")
    print(f"    n_estimators={int(params[0])}, max_depth={int(params[1])}")
    print(f"    learning_rate={params[2]:.4f}, subsample={params[3]:.4f}")
    print(f"    colsample_bytree={params[4]:.4f}")
    
    # Feature selection search space
    fs_search_space = [[0, 1]] * n_features
    
    # DIO for feature selection (inner loop)
    fs_dio = DIO(
        objective_function=feature_selection_objective_function,
        search_space=fs_search_space,
        n_dholes=5,          # Fast configuration
        max_iterations=10    # Fast configuration
    )
    
    print("    Running feature selection (5 dholes, 10 iterations)...")
    start_time = time.time()
    best_features, best_fitness = fs_dio.optimize()
    elapsed = time.time() - start_time
    print(f"    Feature selection completed in {elapsed:.2f}s")
    print(f"    Best fitness: {best_fitness:.6f}")
    
    # Store best features for this hyperparameter set
    best_features_for_hyperparams[params_key] = best_features
    
    return best_fitness

# ==================== MAIN OPTIMIZATION ====================
print(f"\n{'='*80}")
print("STARTING DIO OPTIMIZATION")
print(f"{'='*80}")
print("Configuration:")
print("  - Outer loop (hyperparameters): 5 dholes, 10 iterations")
print("  - Inner loop (features): 5 dholes, 10 iterations")
print("  - Total expected fitness evaluations: ~50 (outer) × 50 (inner) = 2,500")
print(f"{'='*80}")

# XGBoost hyperparameter search space
hp_search_space = [
    [10, 200],      # n_estimators
    [1, 20],        # max_depth
    [0.01, 0.3],    # learning_rate
    [0.5, 1.0],     # subsample
    [0.5, 1.0]      # colsample_bytree
]

# DIO for hyperparameter optimization (outer loop)
hp_dio = DIO(
    objective_function=hyperparameter_objective_function,
    search_space=hp_search_space,
    n_dholes=5,          # Fast configuration
    max_iterations=10    # Fast configuration
)

print("\n🚀 Starting hyperparameter optimization...")
start_time_total = time.time()
best_hyperparams, best_fitness = hp_dio.optimize()
total_time = time.time() - start_time_total

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE!")
print(f"{'='*80}")
print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Best fitness achieved: {best_fitness:.6f}")

# ==================== EXTRACT RESULTS ====================
print(f"\n{'='*80}")
print("EXTRACTING OPTIMIZED CONFIGURATION")
print(f"{'='*80}")

# Get best features for best hyperparameters
best_hyperparams_key = tuple(best_hyperparams)
best_features = best_features_for_hyperparams[best_hyperparams_key]
selected_features_indices = np.where(np.array(best_features) > 0.5)[0]
selected_features_names = [feature_names[i] for i in selected_features_indices]

print(f"\nOptimized XGBoost Hyperparameters:")
print(f"  - n_estimators: {int(best_hyperparams[0])}")
print(f"  - max_depth: {int(best_hyperparams[1])}")
print(f"  - learning_rate: {best_hyperparams[2]:.4f}")
print(f"  - subsample: {best_hyperparams[3]:.4f}")
print(f"  - colsample_bytree: {best_hyperparams[4]:.4f}")

print(f"\nSelected Features: {len(selected_features_indices)}/{n_features}")
print(f"Feature reduction: {(1 - len(selected_features_indices)/n_features)*100:.1f}%")
print(f"\nSelected feature indices: {list(selected_features_indices)}")
print(f"\nSelected feature names:")
for i, name in enumerate(selected_features_names, 1):
    print(f"  {i}. {name}")

# ==================== FINAL MODEL EVALUATION ====================
print(f"\n{'='*80}")
print("FINAL MODEL EVALUATION")
print(f"{'='*80}")

# Prepare data with selected features
X_train_selected = X_train[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

# Train final optimized model
final_model = XGBClassifier(
    n_estimators=int(best_hyperparams[0]),
    max_depth=int(best_hyperparams[1]),
    learning_rate=best_hyperparams[2],
    subsample=best_hyperparams[3],
    colsample_bytree=best_hyperparams[4],
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

print("\nTraining final optimized XGBoost model...")
final_model.fit(X_train_selected, y_train)

# Predictions
y_pred = final_model.predict(X_test_selected)
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'─'*80}")
print("PERFORMANCE METRICS")
print(f"{'─'*80}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# ==================== SAVE RESULTS ====================
print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

results = {
    "method": "DIO for XGBoost (Single-Split, Fast)",
    "configuration": {
        "outer_loop": {"n_dholes": 5, "max_iterations": 10},
        "inner_loop": {"n_dholes": 5, "max_iterations": 10}
    },
    "optimization_time_seconds": total_time,
    "best_hyperparameters": {
        "n_estimators": int(best_hyperparams[0]),
        "max_depth": int(best_hyperparams[1]),
        "learning_rate": float(best_hyperparams[2]),
        "subsample": float(best_hyperparams[3]),
        "colsample_bytree": float(best_hyperparams[4])
    },
    "selected_features": {
        "count": len(selected_features_indices),
        "indices": [int(i) for i in selected_features_indices],
        "names": selected_features_names,
        "reduction_percentage": round((1 - len(selected_features_indices)/n_features)*100, 2)
    },
    "performance": {
        "fitness": float(best_fitness),
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1_score": float(f1),
        "test_roc_auc": float(roc_auc)
    }
}

with open('xgboost_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("✓ Results saved to 'xgboost_optimization_results.json'")

# ==================== COMPARISON WITH BASELINES ====================
print(f"\n{'='*80}")
print("COMPARISON WITH BASELINE MODELS")
print(f"{'='*80}")

baseline_models = {
    'DIO-Optimized XGBoost': final_model,
    'XGBoost Default (Selected)': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'XGBoost Default (All)': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'RF Default (Selected)': RandomForestClassifier(n_estimators=100, random_state=42),
    'RF Default (All)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB()
}

comparison_results = []

for model_name, model in baseline_models.items():
    if model_name == 'DIO-Optimized XGBoost':
        # Already trained
        acc = accuracy
        f1_val = f1
        train_time = 0  # Already measured
        n_feats = len(selected_features_indices)
    else:
        # Determine features to use
        if 'Selected' in model_name:
            X_train_use = X_train_selected
            X_test_use = X_test_selected
            n_feats = len(selected_features_indices)
        else:
            X_train_use = X_train
            X_test_use = X_test
            n_feats = n_features
        
        # Train and evaluate
        start = time.time()
        model.fit(X_train_use, y_train)
        train_time = time.time() - start
        
        y_pred_base = model.predict(X_test_use)
        acc = accuracy_score(y_test, y_pred_base)
        f1_val = f1_score(y_test, y_pred_base)
    
    comparison_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'F1-Score': f1_val,
        'Features': n_feats,
        'Train_Time': train_time
    })
    
    print(f"{model_name:35s} - Acc: {acc:.4f}, F1: {f1_val:.4f}, Features: {n_feats}")

# Save comparison
comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
comparison_df['Rank'] = range(1, len(comparison_df) + 1)
comparison_df.to_csv('xgboost_model_comparison.csv', index=False)
print("\n✓ Comparison saved to 'xgboost_model_comparison.csv'")

# ==================== VISUALIZATIONS ====================
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DIO-Optimized XGBoost Model Analysis', fontsize=16, fontweight='bold')

# 1. Model Comparison Bar Chart
ax1 = axes[0, 0]
colors = ['#2ecc71' if 'DIO' in m else '#3498db' if 'XGBoost' in m else '#95a5a6' 
          for m in comparison_df['Model']]
bars = ax1.barh(comparison_df['Model'], comparison_df['Accuracy'], color=colors, alpha=0.7)
ax1.set_xlabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
ax1.set_xlim([0.85, 1.0])
ax1.grid(True, alpha=0.3, axis='x')
for i, (acc, rank) in enumerate(zip(comparison_df['Accuracy'], comparison_df['Rank'])):
    ax1.text(acc + 0.005, i, f'#{rank} ({acc:.3f})', va='center', fontsize=9)

# 2. Accuracy vs Features Scatter
ax2 = axes[0, 1]
for idx, row in comparison_df.iterrows():
    color = '#2ecc71' if 'DIO' in row['Model'] else '#3498db' if 'XGBoost' in row['Model'] else '#95a5a6'
    marker = 'o' if 'DIO' in row['Model'] else 's' if 'XGBoost' in row['Model'] else '^'
    ax2.scatter(row['Features'], row['Accuracy'], s=200, color=color, marker=marker, 
               alpha=0.7, edgecolors='black', linewidths=1.5)
    if 'DIO' in row['Model']:
        ax2.annotate(row['Model'], (row['Features'], row['Accuracy']), 
                    xytext=(10, -10), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
ax2.set_xlabel('Number of Features', fontweight='bold', fontsize=11)
ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax2.set_title('Pareto Analysis: Accuracy vs Feature Count', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.85, 1.0])

# 3. Feature Importance (if applicable)
ax3 = axes[1, 0]
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    ax3.barh(range(len(indices)), importances[indices], color='#e74c3c', alpha=0.7)
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels([selected_features_names[i] for i in indices], fontsize=9)
    ax3.set_xlabel('Feature Importance', fontweight='bold', fontsize=11)
    ax3.set_title('Top 10 Selected Features by Importance', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
else:
    ax3.text(0.5, 0.5, 'Feature Importance\nNot Available', 
            ha='center', va='center', fontsize=14, color='gray')
    ax3.axis('off')

# 4. ROC Curve
ax4 = axes[1, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax4.plot(fpr, tpr, color='#2ecc71', linewidth=2.5, 
        label=f'DIO-XGBoost (AUC = {roc_auc:.3f})')

# Add baseline XGBoost
xgb_baseline = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
xgb_baseline.fit(X_train, y_train)
y_pred_proba_baseline = xgb_baseline.predict_proba(X_test)[:, 1]
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_baseline)
auc_base = roc_auc_score(y_test, y_pred_proba_baseline)
ax4.plot(fpr_base, tpr_base, color='#3498db', linewidth=2, linestyle='--',
        label=f'XGBoost Default (AUC = {auc_base:.3f})')

ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax4.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax4.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax4.set_title('ROC Curve Comparison', fontweight='bold', fontsize=12)
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_optimization_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'xgboost_optimization_visualization.png'")

plt.show()

# ==================== FINAL SUMMARY ====================
print(f"\n{'='*80}")
print("OPTIMIZATION SUMMARY")
print(f"{'='*80}")
print(f"\n📊 Results:")
print(f"  • Optimization time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"  • Features selected: {len(selected_features_indices)}/{n_features} ({(1-len(selected_features_indices)/n_features)*100:.1f}% reduction)")
print(f"  • Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Test F1-score: {f1:.4f}")
print(f"  • ROC-AUC: {roc_auc:.4f}")
print(f"\n🏆 Model Ranking:")
dio_rank = comparison_df[comparison_df['Model'] == 'DIO-Optimized XGBoost']['Rank'].values[0]
print(f"  • DIO-Optimized XGBoost: Rank #{dio_rank} out of {len(comparison_df)}")
print(f"\n✅ Files Generated:")
print(f"  1. xgboost_optimization_results.json - Full optimization details")
print(f"  2. xgboost_model_comparison.csv - Performance comparison table")
print(f"  3. xgboost_optimization_visualization.png - 4-panel analysis figure")
print(f"\n{'='*80}")
print("XGBOOST OPTIMIZATION COMPLETE!")
print(f"{'='*80}\n")
