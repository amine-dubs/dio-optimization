"""
Statistical Model Comparison with XGBoost-Optimized Configuration
=================================================================
Compares DIO-XGBoost-optimized model with baseline models using 30 independent runs
for statistical significance. Uses the single-split optimized hyperparameters from
the fast DIO optimization (5 dholes, 10 iterations).
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import json
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STATISTICAL MODEL COMPARISON - XGBOOST-OPTIMIZED CONFIGURATION")
print("="*80)

# ========== LOAD XGBOOST-OPTIMIZED CONFIGURATION ==========
print("\nLoading DIO XGBoost-optimized configuration...")
with open('xgboost_optimization_results.json', 'r') as f:
    opt_config = json.load(f)

best_hyperparams = opt_config['best_hyperparameters']
selected_features = opt_config['selected_features']['indices']

print(f"\n✓ Loaded XGBoost-optimized configuration:")
print(f"  - Optimization method: {opt_config['method']}")
print(f"  - Selected features: {len(selected_features)}/30 ({opt_config['selected_features']['reduction_percentage']:.1f}% reduction)")
print(f"  - Feature indices: {selected_features}")
print(f"  - n_estimators: {best_hyperparams['n_estimators']}")
print(f"  - max_depth: {best_hyperparams['max_depth']}")
print(f"  - learning_rate: {best_hyperparams['learning_rate']:.4f}")
print(f"  - subsample: {best_hyperparams['subsample']:.4f}")
print(f"  - colsample_bytree: {best_hyperparams['colsample_bytree']:.4f}")

# ========== CONFIGURATION ==========
NUM_RUNS = 30  # Statistical significance
RANDOM_STATE_BASE = 42

print(f"\n{'='*80}")
print(f"RUNNING {NUM_RUNS} INDEPENDENT TRIALS")
print(f"{'='*80}")
print(f"⏱️  Estimated time: ~2-5 minutes")
print(f"📊  Statistical significance: Wilcoxon signed-rank test")

# ========== MODELS DEFINITION ==========
def get_models():
    """Define all models to compare"""
    models = {
        'DIO-XGBoost-Optimized': {
            'model': XGBClassifier(
                n_estimators=best_hyperparams['n_estimators'],
                max_depth=best_hyperparams['max_depth'],
                learning_rate=best_hyperparams['learning_rate'],
                subsample=best_hyperparams['subsample'],
                colsample_bytree=best_hyperparams['colsample_bytree'],
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ),
            'features': 'selected',
            'color': '#e67e22',
            'marker': 'o'
        },
        'XGBoost Default (XGB-Selected)': {
            'model': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'features': 'selected',
            'color': '#e74c3c',
            'marker': 's'
        },
        'XGBoost (All)': {
            'model': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'features': 'all',
            'color': '#f39c12',
            'marker': 'd'
        },
        'RF Default (XGB-Selected)': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'features': 'selected',
            'color': '#3498db',
            'marker': '^'
        },
        'RF Default (All)': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'features': 'all',
            'color': '#9b59b6',
            'marker': 'v'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'features': 'all',
            'color': '#1abc9c',
            'marker': 'p'
        },
        'SVM': {
            'model': SVC(kernel='rbf', random_state=42),
            'features': 'all',
            'color': '#34495e',
            'marker': 'h'
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'features': 'all',
            'color': '#16a085',
            'marker': '*'
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'features': 'all',
            'color': '#95a5a6',
            'marker': 'D'
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'features': 'all',
            'color': '#7f8c8d',
            'marker': 'X'
        }
    }
    return models

# ========== LOAD DATA ==========
data = load_breast_cancer()
X = data.data
y = data.target

# ========== RUN MULTIPLE TRIALS ==========
all_results = {model_name: {
    'accuracy': [],
    'f1_score': [],
    'precision': [],
    'recall': [],
    'train_time': []
} for model_name in get_models().keys()}

start_time_all = time.time()

for run in range(NUM_RUNS):
    print(f"\n{'─'*80}")
    print(f"Run {run + 1}/{NUM_RUNS}")
    print(f"{'─'*80}")
    
    # Use different random state for each run to get different train/test splits
    random_state = RANDOM_STATE_BASE + run
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Get selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Test each model
    models = get_models()
    for model_name, model_config in models.items():
        # Select features
        if model_config['features'] == 'selected':
            X_train_use = X_train_selected
            X_test_use = X_test_selected
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # Train and evaluate
        start_time = time.time()
        model_config['model'].fit(X_train_use, y_train)
        train_time = time.time() - start_time
        
        y_pred = model_config['model'].predict(X_test_use)
        
        # Store metrics
        all_results[model_name]['accuracy'].append(accuracy_score(y_test, y_pred))
        all_results[model_name]['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
        all_results[model_name]['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        all_results[model_name]['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        all_results[model_name]['train_time'].append(train_time)
        
        print(f"  {model_name:35s} - Acc: {all_results[model_name]['accuracy'][-1]:.4f}")

total_time = time.time() - start_time_all
print(f"\n{'='*80}")
print(f"✓ Completed {NUM_RUNS} runs in {total_time:.2f} seconds")
print(f"{'='*80}")

# ========== COMPUTE STATISTICS ==========
print(f"\n{'='*80}")
print("STATISTICAL ANALYSIS")
print(f"{'='*80}")

stats_summary = []
for model_name, metrics in all_results.items():
    summary = {
        'Model': model_name,
        'Mean_Accuracy': np.mean(metrics['accuracy']),
        'Std_Accuracy': np.std(metrics['accuracy']),
        'Min_Accuracy': np.min(metrics['accuracy']),
        'Max_Accuracy': np.max(metrics['accuracy']),
        'Mean_F1': np.mean(metrics['f1_score']),
        'Std_F1': np.std(metrics['f1_score']),
        'Mean_Precision': np.mean(metrics['precision']),
        'Mean_Recall': np.mean(metrics['recall']),
        'Mean_Time': np.mean(metrics['train_time']),
        'Features': len(selected_features) if 'Selected' in model_name or 'XGBoost' in model_name and 'All' not in model_name else 30
    }
    stats_summary.append(summary)

stats_df = pd.DataFrame(stats_summary)
stats_df = stats_df.sort_values('Mean_Accuracy', ascending=False).reset_index(drop=True)
stats_df['Rank'] = range(1, len(stats_df) + 1)

# ========== WILCOXON SIGNED-RANK TEST ==========
print("\nWilcoxon Signed-Rank Test (vs DIO-XGBoost-Optimized):")
print("─" * 80)

dio_accuracy = all_results['DIO-XGBoost-Optimized']['accuracy']
statistical_tests = []

for model_name in all_results.keys():
    if model_name != 'DIO-XGBoost-Optimized':
        model_accuracy = all_results[model_name]['accuracy']
        statistic, p_value = stats.wilcoxon(dio_accuracy, model_accuracy)
        
        # Determine significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        statistical_tests.append({
            'Model': model_name,
            'Statistic': statistic,
            'p-value': p_value,
            'Significance': sig,
            'Mean_Diff': np.mean(dio_accuracy) - np.mean(model_accuracy)
        })
        
        print(f"{model_name:35s} - p={p_value:.6f} {sig:3s} (diff={np.mean(dio_accuracy) - np.mean(model_accuracy):+.4f})")

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")

stats_tests_df = pd.DataFrame(statistical_tests)
stats_tests_df = stats_tests_df.sort_values('p-value')

# ========== SAVE RESULTS ==========
print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

# Save statistical summary
stats_df.to_csv('xgboost_statistical_comparison_summary.csv', index=False)
print("✓ Statistical summary saved to 'xgboost_statistical_comparison_summary.csv'")

# Save statistical tests
stats_tests_df.to_csv('xgboost_statistical_significance_tests.csv', index=False)
print("✓ Statistical tests saved to 'xgboost_statistical_significance_tests.csv'")

# Save all raw results
raw_results_df = pd.DataFrame()
for model_name, metrics in all_results.items():
    temp_df = pd.DataFrame(metrics)
    temp_df['Model'] = model_name
    temp_df['Run'] = range(1, NUM_RUNS + 1)
    raw_results_df = pd.concat([raw_results_df, temp_df], ignore_index=True)

raw_results_df.to_csv('xgboost_all_runs_detailed_results.csv', index=False)
print("✓ Detailed results (all runs) saved to 'xgboost_all_runs_detailed_results.csv'")

# Display summary table
print(f"\n{'='*80}")
print("SUMMARY STATISTICS (30 RUNS - XGBOOST-OPTIMIZED)")
print(f"{'='*80}")
display_cols = ['Rank', 'Model', 'Mean_Accuracy', 'Std_Accuracy', 'Mean_F1', 'Features']
print(stats_df[display_cols].to_string(index=False))

# ========== VISUALIZATIONS ==========
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Statistical Model Comparison - XGBoost-Optimized (30 Independent Runs)', 
             fontsize=18, fontweight='bold', y=0.995)

models_config = get_models()

# 1. Box Plot - Accuracy Distribution
ax1 = plt.subplot(2, 3, 1)
accuracy_data = [all_results[model]['accuracy'] for model in stats_df['Model']]
bp = ax1.boxplot(accuracy_data, labels=range(1, len(stats_df)+1), patch_artist=True)
for patch, model_name in zip(bp['boxes'], stats_df['Model']):
    patch.set_facecolor(models_config[model_name]['color'])
ax1.set_xlabel('Model Rank', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy Distribution Across 30 Runs', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.85, 1.01])

# 2. Mean Accuracy with Error Bars
ax2 = plt.subplot(2, 3, 2)
x_pos = np.arange(len(stats_df))
colors_ordered = [models_config[model]['color'] for model in stats_df['Model']]
ax2.barh(x_pos, stats_df['Mean_Accuracy'], xerr=stats_df['Std_Accuracy'],
         color=colors_ordered, alpha=0.7, capsize=5)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(stats_df['Model'], fontsize=8)
ax2.set_xlabel('Mean Accuracy ± Std', fontweight='bold', fontsize=11)
ax2.set_title('Mean Accuracy with Standard Deviation', fontweight='bold', fontsize=12)
ax2.set_xlim([0.85, 1.01])
ax2.grid(True, alpha=0.3, axis='x')
for i, (mean, std) in enumerate(zip(stats_df['Mean_Accuracy'], stats_df['Std_Accuracy'])):
    ax2.text(mean + std + 0.005, i, f'{mean:.3f}±{std:.3f}', 
             va='center', fontsize=7)

# 3. F1-Score Comparison
ax3 = plt.subplot(2, 3, 3)
f1_data = [all_results[model]['f1_score'] for model in stats_df['Model']]
bp = ax3.boxplot(f1_data, labels=range(1, len(stats_df)+1), patch_artist=True)
for patch, model_name in zip(bp['boxes'], stats_df['Model']):
    patch.set_facecolor(models_config[model_name]['color'])
ax3.set_xlabel('Model Rank', fontweight='bold', fontsize=11)
ax3.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
ax3.set_title('F1-Score Distribution Across 30 Runs', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.85, 1.01])

# 4. Statistical Significance Heatmap
ax4 = plt.subplot(2, 3, 4)
model_names_short = []
for model in stats_df['Model'][:6]:  # Top 6 models
    model_short = model.replace('DIO-XGBoost-Optimized', 'DIO-XGB').replace('XGBoost Default', 'XGB-Def').replace(' (XGB-Selected)', '(S)').replace(' (All)', '(A)')[:20]
    model_names_short.append(model_short)
    
# Create p-value matrix
p_matrix = []
for model1 in stats_df['Model'][:6]:
    row = []
    for model2 in stats_df['Model'][:6]:
        if model1 == model2:
            row.append(1.0)
        else:
            acc1 = all_results[model1]['accuracy']
            acc2 = all_results[model2]['accuracy']
            _, p_val = stats.wilcoxon(acc1, acc2)
            row.append(p_val)
    p_matrix.append(row)

im = ax4.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.05)
ax4.set_xticks(range(6))
ax4.set_yticks(range(6))
ax4.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=7)
ax4.set_yticklabels(model_names_short, fontsize=7)
ax4.set_title('Pairwise Statistical Significance (p-values)', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax4, label='p-value')
for i in range(6):
    for j in range(6):
        text = ax4.text(j, i, f'{p_matrix[i][j]:.3f}',
                       ha="center", va="center", color="black", fontsize=6)

# 5. Accuracy vs Training Time
ax5 = plt.subplot(2, 3, 5)
for model_name in stats_df['Model']:
    acc = all_results[model_name]['accuracy']
    time_vals = all_results[model_name]['train_time']
    ax5.scatter(time_vals, acc, 
               color=models_config[model_name]['color'],
               marker=models_config[model_name]['marker'],
               s=50, alpha=0.6, label=model_name if model_name in stats_df['Model'][:3] else "")
ax5.set_xlabel('Training Time (seconds)', fontweight='bold', fontsize=11)
ax5.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax5.set_title('Accuracy vs Training Time (All Runs)', fontweight='bold', fontsize=12)
ax5.legend(loc='lower right', fontsize=7)
ax5.grid(True, alpha=0.3)

# 6. Convergence Plot - Mean Accuracy by Rank
ax6 = plt.subplot(2, 3, 6)
ranks = stats_df['Rank']
mean_accs = stats_df['Mean_Accuracy']
std_accs = stats_df['Std_Accuracy']
ax6.errorbar(ranks, mean_accs, yerr=std_accs, fmt='o-', 
            color='#e67e22', linewidth=2, markersize=8,
            capsize=5, capthick=2)
ax6.fill_between(ranks, mean_accs - std_accs, mean_accs + std_accs, 
                 alpha=0.2, color='#e67e22')
ax6.set_xlabel('Model Rank', fontweight='bold', fontsize=11)
ax6.set_ylabel('Mean Accuracy', fontweight='bold', fontsize=11)
ax6.set_title('Mean Accuracy by Rank (with Std Dev)', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(ranks)
ax6.set_ylim([0.85, 1.01])

plt.tight_layout()
plt.savefig('xgboost_statistical_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'xgboost_statistical_comparison_visualization.png'")

# ========== ADDITIONAL PUBLICATION-QUALITY PLOTS ==========

# Plot 2: Individual Model Performance Trends
fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle('Individual Model Performance Across 30 Runs (XGBoost-Optimized)', fontsize=16, fontweight='bold')

for idx, (model_name, ax) in enumerate(zip(stats_df['Model'], axes.flatten())):
    runs = range(1, NUM_RUNS + 1)
    acc = all_results[model_name]['accuracy']
    f1 = all_results[model_name]['f1_score']
    
    ax.plot(runs, acc, 'o-', color=models_config[model_name]['color'], 
           linewidth=2, markersize=4, label='Accuracy', alpha=0.7)
    ax.plot(runs, f1, 's--', color=models_config[model_name]['color'], 
           linewidth=1.5, markersize=3, label='F1-Score', alpha=0.5)
    
    ax.axhline(np.mean(acc), color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_title(model_name, fontsize=8, fontweight='bold')
    ax.set_xlabel('Run', fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_ylim([0.85, 1.01])
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('xgboost_individual_model_trends.png', dpi=300, bbox_inches='tight')
print("✓ Individual trends saved to 'xgboost_individual_model_trends.png'")

plt.show()

# ========== FINAL SUMMARY ==========
print(f"\n{'='*80}")
print("STATISTICAL COMPARISON COMPLETE - XGBOOST-OPTIMIZED")
print(f"{'='*80}")
print(f"\n📊 Summary:")
print(f"  • Optimization method: {opt_config['method']}")
print(f"  • Runs completed: {NUM_RUNS}")
print(f"  • Models compared: {len(stats_df)}")
print(f"  • Total evaluations: {NUM_RUNS * len(stats_df)}")
print(f"  • Execution time: {total_time:.2f} seconds")
print(f"\n🏆 Top 3 Models (by mean accuracy):")
for i in range(min(3, len(stats_df))):
    print(f"  {i+1}. {stats_df.iloc[i]['Model']:35s} - {stats_df.iloc[i]['Mean_Accuracy']:.4f} ± {stats_df.iloc[i]['Std_Accuracy']:.4f}")
print(f"\n✨ XGBoost-Optimized Model Performance:")
dio_xgb_stats = stats_df[stats_df['Model'] == 'DIO-XGBoost-Optimized'].iloc[0]
print(f"  • Rank: #{dio_xgb_stats['Rank']}")
print(f"  • Mean Accuracy: {dio_xgb_stats['Mean_Accuracy']:.4f} ± {dio_xgb_stats['Std_Accuracy']:.4f}")
print(f"  • Features used: {dio_xgb_stats['Features']}/30 ({opt_config['selected_features']['reduction_percentage']:.1f}% reduction)")
print(f"  • Mean training time: {dio_xgb_stats['Mean_Time']:.4f} seconds")
print(f"\n📁 Files generated:")
print(f"  1. xgboost_statistical_comparison_summary.csv")
print(f"  2. xgboost_statistical_significance_tests.csv")
print(f"  3. xgboost_all_runs_detailed_results.csv")
print(f"  4. xgboost_statistical_comparison_visualization.png")
print(f"  5. xgboost_individual_model_trends.png")
print(f"\n{'='*80}")
