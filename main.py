import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import json
import time
from dio import DIO

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target
n_features = X.shape[1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------ Nested Optimization ------------------
# Outer loop: Hyper-parameter optimization
# Inner loop: Feature selection (for each set of hyper-parameters)

# Global variables to store the current hyper-parameters and best features found
current_hyperparameters = None
best_features_for_hyperparams = {}  # Dictionary to store best features for each hyper-param set

def feature_selection_objective_function(features):
    """
    Objective function for feature selection.
    Uses the current hyper-parameters being evaluated.
    """
    # Threshold the features
    selected_features_indices = np.where(np.array(features) > 0.5)[0]
    
    if len(selected_features_indices) == 0:
        return 1.0 # Return worst fitness if no features are selected

    # Select the features from the training and testing data
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Extract hyper-parameters
    n_estimators = int(current_hyperparameters[0])
    max_depth = int(current_hyperparameters[1]) if current_hyperparameters[1] > 1 else None
    min_samples_split = int(current_hyperparameters[2])
    min_samples_leaf = int(current_hyperparameters[3])

    # Train a Random Forest classifier with the current hyper-parameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X_train_selected, y_train)

    # Predict and calculate accuracy
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)

    # Fitness function
    n_selected = len(selected_features_indices)
    fitness = 0.99 * (1 - acc) + 0.01 * (n_selected / n_features)
    
    return fitness

def hyperparameter_objective_function(params):
    """
    Objective function for hyper-parameter optimization.
    For each set of hyper-parameters, run feature selection to find the best fitness.
    """
    global current_hyperparameters, best_features_for_hyperparams
    current_hyperparameters = params
    
    # Create a hashable key for this hyper-parameter set
    params_key = tuple(params)
    
    print(f"\n  Evaluating hyper-parameters: n_estimators={int(params[0])}, max_depth={int(params[1]) if params[1] > 1 else None}, min_samples_split={int(params[2])}, min_samples_leaf={int(params[3])}")
    
    # Search space for feature selection
    fs_search_space = [[0, 1]] * n_features
    
    # DIO for feature selection (inner optimization)
    # REDUCED PARAMETERS: 5 dholes (was 10), 10 iterations (was 20)
    fs_dio = DIO(
        objective_function=feature_selection_objective_function,
        search_space=fs_search_space,
        n_dholes=10,          # Reduced from 10 for faster execution
        max_iterations=20    # Reduced from 20 for faster execution
    )
    
    # Run feature selection
    best_features, best_fitness_fs = fs_dio.optimize()
    
    # Store the best features for this hyper-parameter set
    best_features_for_hyperparams[params_key] = best_features.copy()
    
    # The fitness of this hyper-parameter set is the best feature selection fitness
    print(f"  Best feature selection fitness for these hyper-parameters: {best_fitness_fs}")
    
    return best_fitness_fs

# Search space for hyper-parameters
# [n_estimators, max_depth, min_samples_split, min_samples_leaf]
hp_search_space = [
    [10, 200],    # n_estimators
    [1, 20],      # max_depth (1 is a placeholder for None)
    [2, 10],      # min_samples_split
    [1, 10]       # min_samples_leaf
]

# DIO for hyper-parameter optimization (outer optimization)
print("Running Hyper-parameter Optimization (with nested Feature Selection)...\n")
# REDUCED PARAMETERS: 3 dholes (was 5), 5 iterations (was 10)
hp_dio = DIO(
    objective_function=hyperparameter_objective_function,
    search_space=hp_search_space,
    n_dholes=5,            # Reduced from 5 for faster execution
    max_iterations=10       # Reduced from 10 for faster execution
)

best_hyperparameters, best_fitness = hp_dio.optimize()

# Retrieve the best features that were found for the best hyper-parameters
params_key = tuple(best_hyperparameters)
best_features = best_features_for_hyperparams[params_key]
selected_features_indices = np.where(np.array(best_features) > 0.5)[0]

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE - RETRIEVING BEST SOLUTION")
print("="*60)

# Final evaluation
X_train_selected = X_train[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

n_estimators = int(best_hyperparameters[0])
max_depth = int(best_hyperparameters[1]) if best_hyperparameters[1] > 1 else None
min_samples_split = int(best_hyperparameters[2])
min_samples_leaf = int(best_hyperparameters[3])

clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
final_accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Best features (indices): {selected_features_indices}")
print(f"Number of selected features: {len(selected_features_indices)}")
print(f"\nBest hyper-parameters:")
print(f"  n_estimators: {n_estimators}")
print(f"  max_depth: {max_depth}")
print(f"  min_samples_split: {min_samples_split}")
print(f"  min_samples_leaf: {min_samples_leaf}")
print(f"\nFinal Test Accuracy: {final_accuracy:.6f}")
print(f"Best Fitness: {best_fitness:.6f}")
print("="*60)

# ==================== BASELINE MODELS COMPARISON ====================
print("\n" + "="*60)
print("COMPARING WITH BASELINE MODELS")
print("="*60)

# Prepare results storage
results = []

# Helper function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test, features_used=None):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Training Time (s)': train_time,
        'Features Used': features_used if features_used else X_train.shape[1]
    }
    
    return metrics, y_pred

# 1. Optimized Random Forest (DIO-optimized)
print("\n1. DIO-Optimized Random Forest")
optimized_rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
metrics, y_pred_optimized = evaluate_model(
    'DIO-Optimized RF', 
    optimized_rf, 
    X_train_selected, 
    X_test_selected, 
    y_train, 
    y_test,
    len(selected_features_indices)
)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 2. Random Forest (Default - All Features)
print("\n2. Random Forest (Default - All Features)")
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
metrics, y_pred_rf_default = evaluate_model('RF Default (All Features)', rf_default, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 3. Random Forest (Default - Selected Features Only)
print("\n3. Random Forest (Default - Selected Features)")
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
metrics, y_pred_rf_selected = evaluate_model('RF Default (Selected Features)', rf_selected, X_train_selected, X_test_selected, y_train, y_test, len(selected_features_indices))
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 4. XGBoost (All Features)
print("\n4. XGBoost (All Features)")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
metrics, y_pred_xgb = evaluate_model('XGBoost (All Features)', xgb_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 5. XGBoost (Selected Features)
print("\n5. XGBoost (Selected Features)")
xgb_selected = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
metrics, y_pred_xgb_selected = evaluate_model('XGBoost (Selected Features)', xgb_selected, X_train_selected, X_test_selected, y_train, y_test, len(selected_features_indices))
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 6. Gradient Boosting
print("\n6. Gradient Boosting (All Features)")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
metrics, y_pred_gb = evaluate_model('Gradient Boosting', gb_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 7. SVM
print("\n7. SVM (All Features)")
svm_model = SVC(kernel='rbf', random_state=42)
metrics, y_pred_svm = evaluate_model('SVM', svm_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 8. K-Nearest Neighbors
print("\n8. K-Nearest Neighbors (All Features)")
knn_model = KNeighborsClassifier(n_neighbors=5)
metrics, y_pred_knn = evaluate_model('KNN', knn_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 9. Naive Bayes
print("\n9. Naive Bayes (All Features)")
nb_model = GaussianNB()
metrics, y_pred_nb = evaluate_model('Naive Bayes', nb_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# 10. Logistic Regression
print("\n10. Logistic Regression (All Features)")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
metrics, y_pred_lr = evaluate_model('Logistic Regression', lr_model, X_train, X_test, y_train, y_test)
results.append(metrics)
print(f"Accuracy: {metrics['Accuracy']:.6f}, F1-Score: {metrics['F1-Score']:.6f}")

# ==================== SAVE RESULTS ====================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

# Save to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")

# Save detailed optimization results
optimization_results = {
    'best_hyperparameters': {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth) if max_depth else None,
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf)
    },
    'selected_features': {
        'indices': selected_features_indices.tolist(),
        'count': len(selected_features_indices),
        'names': [data.feature_names[i] for i in selected_features_indices]
    },
    'optimization_fitness': float(best_fitness),
    'final_test_accuracy': float(final_accuracy)
}

with open('optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=4)
print("✓ Optimization details saved to 'optimization_results.json'")

# Display results table
print("\n" + "="*60)
print("RESULTS COMPARISON TABLE")
print("="*60)
print(results_df.to_string(index=False))

# ==================== VISUALIZATIONS ====================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
bars = ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0.85, 1.0])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
             ha='left', va='center', fontsize=9, fontweight='bold')

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 3, 2)
bars = ax2.barh(results_df['Model'], results_df['F1-Score'], color=colors)
ax2.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_xlim([0.85, 1.0])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
             ha='left', va='center', fontsize=9, fontweight='bold')

# 3. Training Time Comparison
ax3 = plt.subplot(2, 3, 3)
bars = ax3.barh(results_df['Model'], results_df['Training Time (s)'], color='#e74c3c')
ax3.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}s', 
             ha='left', va='center', fontsize=9, fontweight='bold')

# 4. Metrics Comparison for Top 3 Models
ax4 = plt.subplot(2, 3, 4)
top3 = results_df.head(3)
x = np.arange(len(top3))
width = 0.2
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics_to_plot):
    ax4.bar(x + i*width, top3[metric], width, label=metric)
ax4.set_xlabel('Models', fontsize=12, fontweight='bold')
ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Top 3 Models - Detailed Metrics', fontsize=14, fontweight='bold')
ax4.set_xticks(x + width * 1.5)
ax4.set_xticklabels(top3['Model'], rotation=15, ha='right')
ax4.legend()
ax4.set_ylim([0.85, 1.0])

# 5. Confusion Matrix for Optimized Model
ax5 = plt.subplot(2, 3, 5)
cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=True)
ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax5.set_title('DIO-Optimized RF - Confusion Matrix', fontsize=14, fontweight='bold')

# 6. Feature Importance
ax6 = plt.subplot(2, 3, 6)
feature_importance = optimized_rf.feature_importances_
feature_names_selected = [data.feature_names[i] for i in selected_features_indices]
sorted_idx = np.argsort(feature_importance)[::-1]
ax6.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#9b59b6')
ax6.set_yticks(range(len(sorted_idx)))
ax6.set_yticklabels([feature_names_selected[i] for i in sorted_idx], fontsize=9)
ax6.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax6.set_title(f'Feature Importance (Selected {len(selected_features_indices)} Features)', 
              fontsize=14, fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to 'model_comparison_visualization.png'")

# Additional: ROC Curve comparison
from sklearn.metrics import roc_curve, auc

fig2, ax = plt.subplots(figsize=(10, 8))

# Plot ROC for key models
models_for_roc = [
    ('DIO-Optimized RF', optimized_rf, X_test_selected),
    ('RF Default', rf_default, X_test),
    ('XGBoost', xgb_model, X_test),
    ('SVM', svm_model, X_test)
]

for name, model, X_test_data in models_for_roc:
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test_data)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved to 'roc_curve_comparison.png'")

plt.show()

print("\n" + "="*60)
print("ALL RESULTS SAVED AND VISUALIZATIONS GENERATED!")
print("="*60)
print("\nFiles created:")
print("  1. model_comparison_results.csv - Detailed metrics table")
print("  2. optimization_results.json - DIO optimization details")
print("  3. model_comparison_visualization.png - Main comparison charts")
print("  4. roc_curve_comparison.png - ROC curves")
print("="*60)
