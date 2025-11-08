"""
DIO Optimization with Cross-Validation for Better Generalization
=================================================================
This version uses k-fold CV during optimization to find hyperparameters
that generalize across multiple data partitions, not just one split.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
import json
import time
import warnings
warnings.filterwarnings('ignore')

from dio import DIO

print("="*80)
print("DIO OPTIMIZATION WITH CROSS-VALIDATION")
print("="*80)
print("\n🎯 Goal: Find hyperparameters that generalize across multiple data splits")
print("📊 Method: k-fold CV during DIO fitness evaluation")
print("⚡ Configuration:")
print("   - Outer loop (Hyperparameters): 5 dholes, 10 iterations")
print("   - Inner loop (Features): 10 dholes, 20 iterations")
print("   - Cross-validation: 5 folds")
print("\n" + "="*80)

# ========== GLOBAL CONFIGURATION ==========
K_FOLDS = 5  # Number of folds for cross-validation
RANDOM_STATE = 42

# Load dataset
data = load_breast_cancer()
X_full = data.data
y_full = data.target
feature_names = data.feature_names

# Create train/test split (70/30) - this is our final holdout test set
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X_full, y_full, test_size=0.3, random_state=RANDOM_STATE, stratify=y_full
)

print(f"\n📊 Dataset: {data.target_names}")
print(f"   Total samples: {len(X_full)}")
print(f"   Training samples (for CV): {len(X_train_full)}")
print(f"   Test samples (final holdout): {len(X_test_final)}")
print(f"   Features: {X_full.shape[1]}")

# ========== GLOBAL VARIABLES FOR NESTED OPTIMIZATION ==========
current_hyperparameters = None
best_features_for_hyperparams = {}  # Cache best features for each hyperparameter set

def hyperparameter_key(hp):
    """Create a hashable key from hyperparameters"""
    return tuple(sorted(hp.items()))

# ========== FEATURE SELECTION (INNER LOOP) WITH CV ==========
def feature_selection_objective_function(feature_vector):
    """
    Fitness function for feature selection.
    Uses k-fold CV to evaluate feature subset with current hyperparameters.
    """
    # Convert continuous values to binary mask (threshold at 0.5)
    feature_mask = feature_vector > 0.5
    selected_features = np.where(feature_mask)[0]
    
    # Require at least 3 features
    if len(selected_features) < 3:
        return 1.0  # Maximum penalty
    
    # Extract hyperparameters
    hp = current_hyperparameters
    n_estimators = int(hp['n_estimators'])
    max_depth = int(hp['max_depth']) if hp['max_depth'] > 1 else None
    min_samples_split = int(hp['min_samples_split'])
    min_samples_leaf = int(hp['min_samples_leaf'])
    
    # Use k-fold cross-validation on training data
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
        X_train_fold = X_train_full[train_idx][:, selected_features]
        X_val_fold = X_train_full[val_idx][:, selected_features]
        y_train_fold = y_train_full[train_idx]
        y_val_fold = y_train_full[val_idx]
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(accuracy)
    
    # Average accuracy across folds
    mean_accuracy = np.mean(cv_scores)
    
    # Fitness = balance accuracy and feature count
    # 99% weight on accuracy, 1% weight on feature reduction
    n_selected = len(selected_features)
    n_total = len(feature_vector)
    
    fitness = 0.99 * (1 - mean_accuracy) + 0.01 * (n_selected / n_total)
    
    return fitness

# ========== HYPERPARAMETER OPTIMIZATION (OUTER LOOP) WITH CV ==========
def hyperparameter_objective_function(hp_vector):
    """
    Fitness function for hyperparameter optimization.
    For each hyperparameter set, runs feature selection with CV.
    """
    global current_hyperparameters, best_features_for_hyperparams
    
    # Extract hyperparameters from vector
    hyperparameters = {
        'n_estimators': hp_vector[0],
        'max_depth': hp_vector[1],
        'min_samples_split': hp_vector[2],
        'min_samples_leaf': hp_vector[3]
    }
    
    # Create key for caching
    hp_key = hyperparameter_key(hyperparameters)
    
    # Check if we already optimized features for these hyperparameters
    if hp_key in best_features_for_hyperparams:
        return best_features_for_hyperparams[hp_key]['fitness']
    
    # Set current hyperparameters for inner loop
    current_hyperparameters = hyperparameters
    
    # Run feature selection optimization (INNER LOOP)
    print(f"  Testing: n_est={int(hp_vector[0])}, depth={int(hp_vector[1])}, "
          f"split={int(hp_vector[2])}, leaf={int(hp_vector[3])}")
    
    fs_search_space = np.array([[0, 1]] * X_train_full.shape[1])  # Binary mask for each feature
    
    fs_dio = DIO(
        objective_function=feature_selection_objective_function,
        search_space=fs_search_space,
        n_dholes=10,  # Feature selection population
        max_iterations=20  # Feature selection iterations
    )
    
    best_feature_vector, best_fitness = fs_dio.optimize()
    
    # Store best features for this hyperparameter set
    feature_mask = best_feature_vector > 0.5
    selected_features = np.where(feature_mask)[0]
    
    best_features_for_hyperparams[hp_key] = {
        'fitness': best_fitness,
        'features': selected_features,
        'feature_vector': best_feature_vector
    }
    
    print(f"    → CV Fitness: {best_fitness:.6f}, Features: {len(selected_features)}/30")
    
    return best_fitness

# ========== RUN NESTED DIO OPTIMIZATION ==========
print("\n" + "="*80)
print("STARTING NESTED DIO OPTIMIZATION WITH CV")
print("="*80)
print("\nOuter Loop: Hyperparameter Optimization (with CV)")
print("  - This will take longer due to k-fold CV in fitness evaluation")
print("  - But hyperparameters should generalize better!\n")

start_time = time.time()

# Define hyperparameter search space
hp_search_space = np.array([
    [10, 200],   # n_estimators
    [1, 20],     # max_depth (1 means None)
    [2, 10],     # min_samples_split
    [1, 10]      # min_samples_leaf
])

# Run outer loop optimization
hp_dio = DIO(
    objective_function=hyperparameter_objective_function,
    search_space=hp_search_space,
    n_dholes=5,  # Hyperparameter population
    max_iterations=10  # Hyperparameter iterations
)

best_hp_vector, best_overall_fitness = hp_dio.optimize()

optimization_time = time.time() - start_time

# Extract best configuration
best_hp_key = hyperparameter_key({
    'n_estimators': best_hp_vector[0],
    'max_depth': best_hp_vector[1],
    'min_samples_split': best_hp_vector[2],
    'min_samples_leaf': best_hp_vector[3]
})

best_config = best_features_for_hyperparams[best_hp_key]
best_features = best_config['features']

best_hyperparameters = {
    'n_estimators': int(best_hp_vector[0]),
    'max_depth': int(best_hp_vector[1]) if best_hp_vector[1] > 1 else None,
    'min_samples_split': int(best_hp_vector[2]),
    'min_samples_leaf': int(best_hp_vector[3])
}

print("\n" + "="*80)
print("✓ OPTIMIZATION COMPLETE")
print("="*80)
print(f"⏱️  Total optimization time: {optimization_time:.2f} seconds")
print(f"\n🎯 Best Configuration (CV-Optimized):")
print(f"   CV Fitness: {best_overall_fitness:.6f}")
print(f"\n📊 Selected Features: {len(best_features)}/30 ({(1-len(best_features)/30)*100:.1f}% reduction)")
print(f"   Indices: {list(best_features)}")
print(f"\n⚙️  Optimized Hyperparameters:")
for key, value in best_hyperparameters.items():
    print(f"   {key}: {value}")

# ========== EVALUATE ON FINAL HOLDOUT TEST SET ==========
print("\n" + "="*80)
print("FINAL EVALUATION ON HOLDOUT TEST SET")
print("="*80)

# Train final model on full training set with selected features
X_train_selected = X_train_full[:, best_features]
X_test_selected = X_test_final[:, best_features]

final_model = RandomForestClassifier(
    n_estimators=best_hyperparameters['n_estimators'],
    max_depth=best_hyperparameters['max_depth'],
    min_samples_split=best_hyperparameters['min_samples_split'],
    min_samples_leaf=best_hyperparameters['min_samples_leaf'],
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_model.fit(X_train_selected, y_train_full)
y_pred_final = final_model.predict(X_test_selected)
y_pred_proba_final = final_model.predict_proba(X_test_selected)[:, 1]

final_accuracy = accuracy_score(y_test_final, y_pred_final)
final_f1 = f1_score(y_test_final, y_pred_final, average='weighted')
final_precision = precision_score(y_test_final, y_pred_final, average='weighted')
final_recall = recall_score(y_test_final, y_pred_final, average='weighted')

print(f"\n🏆 DIO-Optimized RF (CV-based):")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   F1-Score:  {final_f1:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")
print(f"   Features:  {len(best_features)}/30")

# ========== COMPARE WITH BASELINE MODELS ==========
print("\n" + "="*80)
print("COMPARING WITH BASELINE MODELS")
print("="*80)

baseline_models = {
    'RF Default (All)': {
        'model': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'features': 'all'
    },
    'RF Default (Selected)': {
        'model': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'features': 'selected'
    },
    'XGBoost (All)': {
        'model': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
        'features': 'all'
    },
    'XGBoost (Selected)': {
        'model': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
        'features': 'selected'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'features': 'all'
    },
    'SVM': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'features': 'all'
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=5),
        'features': 'all'
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'features': 'all'
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'features': 'all'
    }
}

results = []

# Add DIO-optimized model
results.append({
    'Model': 'DIO-Optimized RF (CV)',
    'Accuracy': final_accuracy,
    'F1-Score': final_f1,
    'Precision': final_precision,
    'Recall': final_recall,
    'Features': len(best_features),
    'y_pred': y_pred_final,
    'y_pred_proba': y_pred_proba_final
})

# Evaluate baseline models
print("\nEvaluating baseline models...")
for model_name, config in baseline_models.items():
    if config['features'] == 'selected':
        X_train_use = X_train_selected
        X_test_use = X_test_selected
    else:
        X_train_use = X_train_full
        X_test_use = X_test_final
    
    start = time.time()
    config['model'].fit(X_train_use, y_train_full)
    train_time = time.time() - start
    
    y_pred = config['model'].predict(X_test_use)
    
    # Get probability predictions
    if hasattr(config['model'], 'predict_proba'):
        y_pred_proba = config['model'].predict_proba(X_test_use)[:, 1]
    else:
        y_pred_proba = None
    
    n_features = len(best_features) if config['features'] == 'selected' else X_train_full.shape[1]
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test_final, y_pred),
        'F1-Score': f1_score(y_test_final, y_pred, average='weighted'),
        'Precision': precision_score(y_test_final, y_pred, average='weighted'),
        'Recall': recall_score(y_test_final, y_pred, average='weighted'),
        'Training Time (s)': train_time,
        'Features': n_features,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    
    print(f"  ✓ {model_name}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)
print(results_df[['Rank', 'Model', 'Accuracy', 'F1-Score', 'Features']].to_string(index=False))

# ========== SAVE RESULTS ==========
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save to CV-specific folder
import os
os.makedirs('cv_optimization', exist_ok=True)

# Save comparison results
results_df[['Rank', 'Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Features']].to_csv(
    'cv_optimization/model_comparison_cv.csv', index=False
)
print("✓ Results saved to 'cv_optimization/model_comparison_cv.csv'")

# Save optimization configuration
optimization_results = {
    'method': 'DIO with k-fold Cross-Validation',
    'k_folds': K_FOLDS,
    'optimization_time_seconds': optimization_time,
    'best_hyperparameters': {
        'n_estimators': best_hyperparameters['n_estimators'],
        'max_depth': best_hyperparameters['max_depth'],
        'min_samples_split': best_hyperparameters['min_samples_split'],
        'min_samples_leaf': best_hyperparameters['min_samples_leaf']
    },
    'selected_features': {
        'count': len(best_features),
        'indices': list(map(int, best_features)),
        'names': [feature_names[i] for i in best_features],
        'reduction_percentage': round((1 - len(best_features) / len(feature_names)) * 100, 2)
    },
    'performance': {
        'cv_fitness': float(best_overall_fitness),
        'test_accuracy': float(final_accuracy),
        'test_f1_score': float(final_f1),
        'test_precision': float(final_precision),
        'test_recall': float(final_recall)
    },
    'dio_configuration': {
        'outer_loop': {'population': 5, 'iterations': 10},
        'inner_loop': {'population': 10, 'iterations': 20}
    }
}

with open('cv_optimization/optimization_results_cv.json', 'w') as f:
    json.dump(optimization_results, f, indent=4)
print("✓ Configuration saved to 'cv_optimization/optimization_results_cv.json'")

# ========== VISUALIZATIONS ==========
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('DIO Optimization with Cross-Validation - Model Comparison', 
             fontsize=18, fontweight='bold', y=0.995)

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71' if 'DIO' in model else '#3498db' if 'Selected' in model else '#95a5a6' 
          for model in results_df['Model']]
bars = ax1.barh(range(len(results_df)), results_df['Accuracy'], color=colors, alpha=0.8)
ax1.set_yticks(range(len(results_df)))
ax1.set_yticklabels(results_df['Model'], fontsize=9)
ax1.set_xlabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Model Accuracy Comparison (CV-Optimized)', fontweight='bold', fontsize=12)
ax1.set_xlim([0.88, 1.0])
ax1.grid(True, alpha=0.3, axis='x')
for i, (acc, rank) in enumerate(zip(results_df['Accuracy'], results_df['Rank'])):
    ax1.text(acc + 0.003, i, f'{acc:.4f} (#{rank})', va='center', fontsize=8)

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 3, 2)
ax2.barh(range(len(results_df)), results_df['F1-Score'], color=colors, alpha=0.8)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['Model'], fontsize=9)
ax2.set_xlabel('F1-Score', fontweight='bold', fontsize=11)
ax2.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
ax2.set_xlim([0.88, 1.0])
ax2.grid(True, alpha=0.3, axis='x')
for i, f1 in enumerate(results_df['F1-Score']):
    ax2.text(f1 + 0.003, i, f'{f1:.4f}', va='center', fontsize=8)

# 3. Features Used
ax3 = plt.subplot(2, 3, 3)
feature_colors = ['#2ecc71' if f <= 10 else '#f39c12' if f <= 20 else '#e74c3c' 
                  for f in results_df['Features']]
ax3.barh(range(len(results_df)), results_df['Features'], color=feature_colors, alpha=0.8)
ax3.set_yticks(range(len(results_df)))
ax3.set_yticklabels(results_df['Model'], fontsize=9)
ax3.set_xlabel('Number of Features', fontweight='bold', fontsize=11)
ax3.set_title('Feature Count (Lower = Better)', fontweight='bold', fontsize=12)
ax3.axvline(x=30, color='red', linestyle='--', linewidth=2, label='All features', alpha=0.7)
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend(fontsize=9)
for i, feat in enumerate(results_df['Features']):
    ax3.text(feat + 0.5, i, f'{feat}/30', va='center', fontsize=8)

# 4. Top 3 Models Detail
ax4 = plt.subplot(2, 3, 4)
top3 = results_df.head(3)
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.25

for i, (_, row) in enumerate(top3.iterrows()):
    values = [row['Accuracy'], row['F1-Score'], row['Precision'], row['Recall']]
    ax4.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)

ax4.set_xlabel('Metrics', fontweight='bold', fontsize=11)
ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
ax4.set_title('Top 3 Models - Detailed Metrics', fontweight='bold', fontsize=12)
ax4.set_xticks(x + width)
ax4.set_xticklabels(metrics, fontsize=9)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0.88, 1.0])

# 5. Confusion Matrix for DIO-Optimized
ax5 = plt.subplot(2, 3, 5)
cm = confusion_matrix(y_test_final, results_df[results_df['Model'] == 'DIO-Optimized RF (CV)']['y_pred'].values[0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=True,
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
ax5.set_xlabel('Predicted', fontweight='bold', fontsize=11)
ax5.set_ylabel('Actual', fontweight='bold', fontsize=11)
ax5.set_title('Confusion Matrix (DIO CV-Optimized)', fontweight='bold', fontsize=12)

# 6. Feature Importance
ax6 = plt.subplot(2, 3, 6)
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]
selected_feature_names = [feature_names[i] for i in best_features]
top_n = min(10, len(best_features))
ax6.barh(range(top_n), importances[indices[:top_n]], color='#2ecc71', alpha=0.8)
ax6.set_yticks(range(top_n))
ax6.set_yticklabels([selected_feature_names[indices[i]] for i in range(top_n)], fontsize=9)
ax6.set_xlabel('Importance', fontweight='bold', fontsize=11)
ax6.set_title(f'Top {top_n} Feature Importances (CV-Optimized)', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3, axis='x')
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('cv_optimization/model_comparison_visualization_cv.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'cv_optimization/model_comparison_visualization_cv.png'")

# ROC Curves
fig2, ax = plt.subplots(figsize=(10, 8))
for _, row in results_df.iterrows():
    if row['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test_final, row['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        linestyle = '-' if 'DIO' in row['Model'] else '--' if 'Selected' in row['Model'] else ':'
        linewidth = 3 if 'DIO' in row['Model'] else 2
        ax.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
                label=f"{row['Model']} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curves - CV-Optimized vs Baselines', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cv_optimization/roc_curves_cv.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved to 'cv_optimization/roc_curves_cv.png'")

plt.show()

# ========== FINAL SUMMARY ==========
print("\n" + "="*80)
print("CV-BASED OPTIMIZATION COMPLETE")
print("="*80)
print(f"\n🎯 Key Results:")
print(f"   ✓ Optimization method: k-fold Cross-Validation (k={K_FOLDS})")
print(f"   ✓ Optimization time: {optimization_time:.2f} seconds")
print(f"   ✓ Test accuracy: {final_accuracy:.4f}")
print(f"   ✓ Features selected: {len(best_features)}/30 ({(1-len(best_features)/30)*100:.1f}% reduction)")
print(f"   ✓ Rank: #{results_df[results_df['Model']=='DIO-Optimized RF (CV)']['Rank'].values[0]}")
print(f"\n📁 Files saved to 'cv_optimization/' folder:")
print(f"   1. optimization_results_cv.json")
print(f"   2. model_comparison_cv.csv")
print(f"   3. model_comparison_visualization_cv.png")
print(f"   4. roc_curves_cv.png")
print(f"\n💡 These hyperparameters should generalize better than single-split optimization!")
print("="*80)
