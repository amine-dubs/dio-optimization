"""
XGBoost DIO Optimization for CIFAR-10 Features (Fast Configuration)
====================================================================
Nested optimization: Hyperparameters + Feature Selection
Uses small subset for speed (complete in 1-2 hours)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from dio import DIO

print("="*80)
print("XGBOOST DIO OPTIMIZATION - CIFAR-10 IMAGE FEATURES")
print("="*80)

# ==================== LOAD FEATURES ====================
print("\n[1/6] Loading features...")

# Check for NPZ file (from Colab)
npz_locations = [
    './data/cifar10_resnet50_features.npz',
    'cifar10_resnet50_features.npz',
    './cifar10_resnet50_features.npz'
]

data_loaded = False
for npz_file in npz_locations:
    if os.path.exists(npz_file):
        print(f"Loading from: {npz_file}")
        data = np.load(npz_file)
        X_train = data['train_features']
        y_train = data['train_labels']
        X_test = data['test_features']
        y_test = data['test_labels']
        data_loaded = True
        break

if not data_loaded:
    print("ERROR: No feature file found!")
    print("Place cifar10_resnet50_features.npz in ./data/ folder")
    exit(1)

print(f"✓ Full dataset loaded - Train: {X_train.shape}, Test: {X_test.shape}")

# ==================== USE SMALL SUBSET ====================
print("\n[2/6] Creating small subset for fast optimization...")

# Use only 2,000 train + 500 test samples (stratified)
y_train = y_train.flatten()
y_test = y_test.flatten()

X_train, _, y_train, _ = train_test_split(
    X_train, y_train, train_size=2000, stratify=y_train, random_state=42
)
X_test, _, y_test, _ = train_test_split(
    X_test, y_test, train_size=500, stratify=y_test, random_state=42
)

print(f"✓ Using subset - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Train classes: {np.bincount(y_train)}")
print(f"  Test classes: {np.bincount(y_test)}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]
print(f"  Feature dimension: {n_features}")

# ==================== NESTED DIO SETUP ====================
print("\n[3/6] Setting up nested DIO optimization...")
print("Configuration:")
print("  Outer (Hyperparameters): 3 dholes, 8 iterations")
print("  Inner (Features): 3 dholes, 8 iterations")
print("  Expected time: 1-2 hours")

current_hyperparameters = None
best_features_for_hyperparams = {}

def feature_selection_objective(features):
    """Inner loop: Feature selection"""
    # Threshold features
    selected_idx = np.where(np.array(features) > 0.5)[0]
    
    if len(selected_idx) == 0:
        return 1.0
    
    # Select features
    X_tr_sel = X_train[:, selected_idx]
    X_te_sel = X_test[:, selected_idx]
    
    # Extract hyperparameters
    n_est = int(current_hyperparameters[0])
    max_d = int(current_hyperparameters[1])
    lr = current_hyperparameters[2]
    
    # Train XGBoost
    clf = XGBClassifier(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=lr,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0,
        n_jobs=-1
    )
    
    try:
        clf.fit(X_tr_sel, y_train)
        y_pred = clf.predict(X_te_sel)
        acc = accuracy_score(y_test, y_pred)
    except:
        return 1.0
    
    # Fitness: 95% accuracy + 5% feature reduction
    n_sel = len(selected_idx)
    fitness = 0.95 * (1 - acc) + 0.05 * (n_sel / n_features)
    
    return fitness

def hyperparameter_objective(params):
    """Outer loop: Hyperparameter optimization"""
    global current_hyperparameters, best_features_for_hyperparams
    current_hyperparameters = params
    
    params_key = tuple(params)
    
    print(f"\n  Testing XGBoost [n_est={int(params[0])}, depth={int(params[1])}, lr={params[2]:.3f}]")
    
    # Feature selection search space
    fs_search_space = [[0, 1]] * n_features
    
    # DIO for feature selection
    fs_dio = DIO(
        objective_function=feature_selection_objective,
        search_space=fs_search_space,
        n_dholes=3,
        max_iterations=8
    )
    
    start = time.time()
    best_features, best_fitness = fs_dio.optimize()
    elapsed = time.time() - start
    
    n_selected = len(np.where(np.array(best_features) > 0.5)[0])
    print(f"    → Features: {n_selected}/{n_features}, Fitness: {best_fitness:.6f}, Time: {elapsed:.1f}s")
    
    # Store best features
    best_features_for_hyperparams[params_key] = best_features
    
    return best_fitness

# ==================== RUN OPTIMIZATION ====================
print("\n[4/6] Running DIO optimization...")
print("="*80)

# XGBoost hyperparameter search space
hp_search_space = [
    [30, 100],     # n_estimators (reduced range)
    [3, 10],       # max_depth (reduced range)
    [0.01, 0.3]    # learning_rate
]

# Outer DIO
hp_dio = DIO(
    objective_function=hyperparameter_objective,
    search_space=hp_search_space,
    n_dholes=3,
    max_iterations=8
)

start_total = time.time()
best_hp, best_fitness = hp_dio.optimize()
total_time = time.time() - start_total

print("\n" + "="*80)
print("✓ OPTIMIZATION COMPLETE!")
print("="*80)
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Best fitness: {best_fitness:.6f}")

# ==================== EXTRACT RESULTS ====================
print("\n[5/6] Extracting optimized configuration...")

best_hp_key = tuple(best_hp)
best_features = best_features_for_hyperparams[best_hp_key]
selected_idx = np.where(np.array(best_features) > 0.5)[0]

print(f"\nOptimized Hyperparameters:")
print(f"  n_estimators: {int(best_hp[0])}")
print(f"  max_depth: {int(best_hp[1])}")
print(f"  learning_rate: {best_hp[2]:.4f}")

print(f"\nSelected Features: {len(selected_idx)}/{n_features}")
print(f"Feature reduction: {(1 - len(selected_idx)/n_features)*100:.1f}%")

# ==================== FINAL EVALUATION ====================
print("\n[6/6] Evaluating final model...")

X_train_sel = X_train[:, selected_idx]
X_test_sel = X_test[:, selected_idx]

final_model = XGBClassifier(
    n_estimators=int(best_hp[0]),
    max_depth=int(best_hp[1]),
    learning_rate=best_hp[2],
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0,
    n_jobs=-1
)

final_model.fit(X_train_sel, y_train)

y_pred = final_model.predict(X_test_sel)
y_pred_proba = final_model.predict_proba(X_test_sel)[:, 1] if len(np.unique(y_train)) == 2 else None

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")

# ==================== COMPARISON WITH BASELINE ====================
print("\n" + "="*80)
print("COMPARISON WITH BASELINE")
print("="*80)

# Baseline XGBoost (default, all features)
baseline = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average='weighted')

print(f"\nDIO-Optimized XGBoost:")
print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Features: {len(selected_idx)}")
print(f"\nBaseline XGBoost (Default):")
print(f"  Accuracy: {acc_base:.4f}, F1: {f1_base:.4f}, Features: {n_features}")
print(f"\nImprovement:")
print(f"  Accuracy: {(acc - acc_base)*100:+.2f}%")
print(f"  Features: {(1 - len(selected_idx)/n_features)*100:.1f}% reduction")

# ==================== SAVE RESULTS ====================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    "dataset": "CIFAR-10 ResNet50 Features (Subset)",
    "subset_size": {"train": 2000, "test": 500},
    "optimization_time_minutes": round(total_time/60, 2),
    "configuration": {
        "outer_loop": {"n_dholes": 3, "max_iterations": 8},
        "inner_loop": {"n_dholes": 3, "max_iterations": 8}
    },
    "best_hyperparameters": {
        "n_estimators": int(best_hp[0]),
        "max_depth": int(best_hp[1]),
        "learning_rate": float(best_hp[2])
    },
    "selected_features": {
        "count": len(selected_idx),
        "total": n_features,
        "indices": [int(i) for i in selected_idx],
        "reduction_percentage": round((1 - len(selected_idx)/n_features)*100, 2)
    },
    "performance": {
        "dio_optimized": {"accuracy": float(acc), "f1_score": float(f1)},
        "baseline": {"accuracy": float(acc_base), "f1_score": float(f1_base)},
        "improvement": {
            "accuracy_gain": float((acc - acc_base)*100),
            "feature_reduction": float((1 - len(selected_idx)/n_features)*100)
        }
    }
}

with open('cifar10_xgboost_dio_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("✓ Results saved to 'cifar10_xgboost_dio_results.json'")

# ==================== VISUALIZATION ====================
print("\nGenerating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('DIO-Optimized XGBoost on CIFAR-10 Features', fontsize=14, fontweight='bold')

# 1. Accuracy comparison
ax1 = axes[0]
models = ['DIO-Optimized\nXGBoost', 'Baseline\nXGBoost']
accuracies = [acc, acc_base]
colors = ['#2ecc71', '#3498db']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
ax1.set_ylim([min(accuracies)-0.05, max(accuracies)+0.05])
ax1.grid(True, alpha=0.3, axis='y')
for bar, acc_val in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
            f'{acc_val:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. Feature count comparison
ax2 = axes[1]
features = [len(selected_idx), n_features]
bars2 = ax2.bar(models, features, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Features', fontweight='bold', fontsize=11)
ax2.set_title('Feature Count Comparison', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
for bar, feat_val in zip(bars2, features):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 20,
            f'{feat_val}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('cifar10_xgboost_dio_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to 'cifar10_xgboost_dio_visualization.png'")
plt.show()

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)
print(f"\n✅ Completed in {total_time/60:.1f} minutes")
print(f"✅ Selected {len(selected_idx)}/{n_features} features ({(1-len(selected_idx)/n_features)*100:.1f}% reduction)")
print(f"✅ Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ Improvement over baseline: {(acc-acc_base)*100:+.2f}%")
print(f"\n📁 Files generated:")
print(f"  1. cifar10_xgboost_dio_results.json")
print(f"  2. cifar10_xgboost_dio_visualization.png")
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80 + "\n")
