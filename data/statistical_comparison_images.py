"""
Statistical Model Comparison for CIFAR-10 Image Features
===========================================================
Compares the DIO-XGBoost-optimized model with baseline models using 30 independent runs
for statistical significance, based on the configuration from 'optimize_xgboost_images.py'.
"""
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("="*80)
print("STATISTICAL COMPARISON - DIO-XGBOOST ON CIFAR-10 FEATURES")
print("="*80)

# ========== 1. LOAD OPTIMIZED CONFIGURATION ==========
config_path = 'data/cifar10_xgboost_dio_results.json'
print(f"\n[1/7] Loading DIO-optimized configuration from '{config_path}'...")
with open(config_path, 'r') as f:
    opt_config = json.load(f)

best_hyperparams = opt_config['best_hyperparameters']
selected_feature_indices = opt_config['selected_features']['indices']
n_selected_features = len(selected_feature_indices)
n_total_features = opt_config['selected_features']['total']

print("✓ Configuration loaded:")
print(f"  - Selected Features: {n_selected_features}/{n_total_features} ({opt_config['selected_features']['reduction_percentage']:.1f}% reduction)")
print(f"  - Hyperparameters (XGBoost):")
for key, val in best_hyperparams.items():
    print(f"    - {key}: {val}")

# ========== 2. LOAD DATASET ==========
print("\n[2/7] Loading CIFAR-10 ResNet50 features...")
try:
    data = np.load('data/cifar10_resnet50_features.npz')
    X_train_full = data['train_features']
    y_train_full = data['train_labels'].flatten()
    X_test_full = data['test_features']
    y_test_full = data['test_labels'].flatten()
    print(f"✓ Full dataset loaded.")
except FileNotFoundError:
    print("❌ Error: 'data/cifar10_resnet50_features.npz' not found. Please run the feature extraction script first.")
    exit()

# Use the same small subset as in the optimization script for consistency
print("\n[2/7] Creating consistent small subset for statistical analysis...")
X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train_full, y_train_full, train_size=2000, stratify=y_train_full, random_state=42
)
X_test_subset, _, y_test_subset, _ = train_test_split(
    X_test_full, y_test_full, train_size=500, stratify=y_test_full, random_state=42
)
X_full = np.vstack([X_train_subset, X_test_subset])
y_full = np.hstack([y_train_subset, y_test_subset])

print(f"✓ Using consistent subset of {X_full.shape[0]} samples for all 30 trials.")

# ========== 3. DEFINE MODELS ==========
print("\n[3/7] Defining models for comparison...")

def get_models():
    """Define models optimized for image features (tree-based models only)"""
    models = {
        'DIO-XGBoost-Optimized': {
            'model': XGBClassifier(
                **best_hyperparams,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            ),
            'features': 'selected'
        },
        'XGBoost Default (DIO-Selected)': {
            'model': XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            ),
            'features': 'selected'
        },
        'XGBoost Default (All)': {
            'model': XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            ),
            'features': 'all'
        },
        'Random Forest (DIO-Selected)': {
            'model': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'features': 'selected'
        },
        'Random Forest (All)': {
            'model': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'features': 'all'
        }
    }
    return models

print("✓ Models defined: DIO-XGBoost-Optimized, XGBoost variants, Random Forest variants")

# ========== 4. RUN 30 INDEPENDENT TRIALS ==========
NUM_RUNS = 30
RANDOM_STATE_BASE = 42

print(f"\n[4/7] Starting {NUM_RUNS} independent trials...")
print(f"⏱️  Estimated time: ~3-5 minutes")
print(f"📊  Statistical significance: Wilcoxon signed-rank test")

# Store all results
all_results = {model_name: {
    'accuracy': [],
    'f1_score': [],
    'precision': [],
    'recall': []
} for model_name in get_models().keys()}

import time
start_time_all = time.time()

for run in range(NUM_RUNS):
    random_state = RANDOM_STATE_BASE + run
    
    # Split data for this run
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=random_state, stratify=y_full
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get selected features
    X_train_selected = X_train_scaled[:, selected_feature_indices]
    X_test_selected = X_test_scaled[:, selected_feature_indices]

    # Test each model
    models = get_models()
    for model_name, model_config in models.items():
        # Select features
        if model_config['features'] == 'selected':
            X_train_use = X_train_selected
            X_test_use = X_test_selected
        else:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        
        # Train and predict
        model_config['model'].fit(X_train_use, y_train)
        y_pred = model_config['model'].predict(X_test_use)

        # Store metrics
        all_results[model_name]['accuracy'].append(accuracy_score(y_test, y_pred))
        all_results[model_name]['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
        all_results[model_name]['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        all_results[model_name]['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    
    if (run + 1) % 5 == 0:
        print(f"  - Completed run {run + 1}/{NUM_RUNS}")

total_time = time.time() - start_time_all
print(f"✓ Completed {NUM_RUNS} runs in {total_time:.2f} seconds")

# ========== 5. COMPUTE STATISTICS ==========
print(f"\n[5/7] Computing statistical summary...")

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
        'Features': len(selected_feature_indices) if 'Selected' in model_name or ('DIO' in model_name and 'All' not in model_name) else n_total_features
    }
    stats_summary.append(summary)

stats_df = pd.DataFrame(stats_summary)
stats_df = stats_df.sort_values('Mean_Accuracy', ascending=False).reset_index(drop=True)
stats_df['Rank'] = range(1, len(stats_df) + 1)

# ========== 6. WILCOXON SIGNED-RANK TEST ==========
print("\n[6/7] Performing Wilcoxon signed-rank test...")
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

# ========== 7. SAVE RESULTS ==========
print(f"\n[7/7] Saving results and visualizations...")

# Save statistical summary
stats_df.to_csv('data/cifar10_statistical_comparison_summary.csv', index=False)
print("✓ Statistical summary saved to 'data/cifar10_statistical_comparison_summary.csv'")

# Save statistical tests
stats_tests_df.to_csv('data/cifar10_statistical_significance_tests.csv', index=False)
print("✓ Statistical tests saved to 'data/cifar10_statistical_significance_tests.csv'")

# Save all raw results
raw_results_df = pd.DataFrame()
for model_name, metrics in all_results.items():
    temp_df = pd.DataFrame(metrics)
    temp_df['Model'] = model_name
    temp_df['Run'] = range(1, NUM_RUNS + 1)
    raw_results_df = pd.concat([raw_results_df, temp_df], ignore_index=True)

raw_results_df.to_csv('data/cifar10_all_runs_detailed_results.csv', index=False)
print("✓ Detailed results saved to 'data/cifar10_all_runs_detailed_results.csv'")

# Display summary table
print(f"\n{'='*80}")
print("SUMMARY STATISTICS (30 RUNS - CIFAR-10 FEATURES)")
print(f"{'='*80}")
display_cols = ['Rank', 'Model', 'Mean_Accuracy', 'Std_Accuracy', 'Mean_F1', 'Features']
print(stats_df[display_cols].to_string(index=False))

# ========== VISUALIZATIONS ==========
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

# Create boxplot visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Statistical Model Comparison - CIFAR-10 Image Features (30 Runs)', 
             fontsize=16, fontweight='bold')

# 1. Box Plot - Accuracy Distribution
accuracy_data = [all_results[model]['accuracy'] for model in stats_df['Model']]
bp = ax1.boxplot(accuracy_data, labels=range(1, len(stats_df)+1), patch_artist=True)
colors = ['#e67e22', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_xlabel('Model Rank', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy Distribution Across 30 Runs', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. Mean Accuracy with Error Bars
x_pos = np.arange(len(stats_df))
ax2.barh(x_pos, stats_df['Mean_Accuracy'], xerr=stats_df['Std_Accuracy'],
         color=colors, alpha=0.7, capsize=5)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(stats_df['Model'], fontsize=9)
ax2.set_xlabel('Mean Accuracy ± Std', fontweight='bold', fontsize=11)
ax2.set_title('Mean Accuracy with Standard Deviation', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
for i, (mean, std) in enumerate(zip(stats_df['Mean_Accuracy'], stats_df['Std_Accuracy'])):
    ax2.text(mean + std + 0.01, i, f'{mean:.3f}±{std:.3f}', 
             va='center', fontsize=8)

plt.tight_layout()
plt.savefig('data/cifar10_statistical_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'data/cifar10_statistical_comparison_visualization.png'")
plt.close()

# ========== FINAL SUMMARY ==========
print(f"\n{'='*80}")
print("STATISTICAL COMPARISON COMPLETE - CIFAR-10 IMAGE FEATURES")
print(f"{'='*80}")
print(f"\n📊 Summary:")
print(f"  • Dataset: CIFAR-10 ResNet50 Features (Subset: 2500 samples)")
print(f"  • Runs completed: {NUM_RUNS}")
print(f"  • Models compared: {len(stats_df)}")
print(f"  • Total evaluations: {NUM_RUNS * len(stats_df)}")
print(f"  • Execution time: {total_time:.2f} seconds")
print(f"\n🏆 Top 3 Models (by mean accuracy):")
for i in range(min(3, len(stats_df))):
    print(f"  {i+1}. {stats_df.iloc[i]['Model']:35s} - {stats_df.iloc[i]['Mean_Accuracy']:.4f} ± {stats_df.iloc[i]['Std_Accuracy']:.4f}")
print(f"\n✨ DIO-XGBoost-Optimized Performance:")
dio_stats = stats_df[stats_df['Model'] == 'DIO-XGBoost-Optimized'].iloc[0]
print(f"  • Rank: #{dio_stats['Rank']}")
print(f"  • Mean Accuracy: {dio_stats['Mean_Accuracy']:.4f} ± {dio_stats['Std_Accuracy']:.4f}")
print(f"  • Features used: {dio_stats['Features']}/{n_total_features} ({(1-dio_stats['Features']/n_total_features)*100:.1f}% reduction)")
print(f"\n📁 Files generated:")
print(f"  1. data/cifar10_statistical_comparison_summary.csv")
print(f"  2. data/cifar10_statistical_significance_tests.csv")
print(f"  3. data/cifar10_all_runs_detailed_results.csv")
print(f"  4. data/cifar10_statistical_comparison_visualization.png")
print(f"\n{'='*80}")
