"""
Benchmark Testing Script for DIO Algorithm
===========================================
Tests DIO on standard benchmark functions from the original paper
with REDUCED parameters for faster execution on local PC.

Original Paper Settings:
- Population: 30
- Iterations: 500
- Runs: 30

Reduced Settings (for faster testing):
- Population: 10
- Iterations: 100
- Runs: 5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dio import DIO
from benchmark_functions import BenchmarkFunctions
import time
import json

# Set random seed
np.random.seed(42)

# ========== CONFIGURATION ==========
# FULL PAPER PARAMETERS FOR ACCURATE COMPARISON (~1 HOUR EXECUTION)
CONFIG = {
    'n_dholes': 30,           # Paper setting: 30 population
    'max_iterations': 500,    # Paper setting: 500 iterations
    'num_runs': 30,           # Paper setting: 30 independent runs
    'test_functions': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 
                       'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14'],  # All functions
}

# To run QUICK test (2-3 minutes), use this instead:
# CONFIG = {
#     'n_dholes': 10,
#     'max_iterations': 100,
#     'num_runs': 5,
#     'test_functions': ['F1', 'F5', 'F9', 'F10'],
# }

print("="*70)
print("DIO BENCHMARK TESTING")
print("="*70)
print(f"Configuration:")
print(f"  Population Size: {CONFIG['n_dholes']} (Paper: 30)")
print(f"  Max Iterations: {CONFIG['max_iterations']} (Paper: 500)")
print(f"  Independent Runs: {CONFIG['num_runs']} (Paper: 30)")
print(f"  Test Functions: {len(CONFIG['test_functions'])}")
print()
print(f"⏱️  ESTIMATED EXECUTION TIME: ~60-90 minutes")
print(f"📊  PAPER-LEVEL SETTINGS FOR ACCURATE COMPARISON")
print("="*70)

# Initialize benchmark functions
benchmarks = BenchmarkFunctions()

# Storage for results
results = []

# ========== RUN BENCHMARKS ==========
total_functions = len(CONFIG['test_functions'])
start_time_all = time.time()

for func_idx, func_id in enumerate(CONFIG['test_functions'], 1):
    print(f"\n{'='*70}")
    print(f"Testing {func_id}: {benchmarks.functions_info[func_id]['name']}")
    print(f"Progress: {func_idx}/{total_functions} functions ({func_idx/total_functions*100:.1f}%)")
    
    # Time estimation
    elapsed = time.time() - start_time_all
    if func_idx > 1:
        avg_time_per_func = elapsed / (func_idx - 1)
        remaining_funcs = total_functions - func_idx + 1
        est_remaining = avg_time_per_func * remaining_funcs
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | Est. Remaining: {est_remaining/60:.1f} min")
    print(f"{'='*70}")
    
    # Get function details
    func = benchmarks.get_function(func_id)
    bounds = benchmarks.get_bounds(func_id)
    dim = benchmarks.get_dimension(func_id)
    fmin = benchmarks.get_optimum(func_id)
    
    print(f"  Dimension: {dim}")
    print(f"  Range: {bounds[0]}")
    print(f"  Global Minimum: {fmin}")
    print(f"  Type: {benchmarks.functions_info[func_id]['type']}")
    
    # Run multiple independent trials
    best_fitness_runs = []
    convergence_curves = []
    exec_times = []
    
    for run in range(CONFIG['num_runs']):
        print(f"\n  Run {run + 1}/{CONFIG['num_runs']}...", end=" ")
        
        start_time = time.time()
        
        # Create DIO optimizer
        dio = DIO(
            objective_function=func,
            search_space=bounds,
            n_dholes=CONFIG['n_dholes'],
            max_iterations=CONFIG['max_iterations']
        )
        
        # Run optimization
        best_solution, best_fitness = dio.optimize()
        
        exec_time = time.time() - start_time
        
        best_fitness_runs.append(best_fitness)
        exec_times.append(exec_time)
        
        print(f"Best Fitness: {best_fitness:.6e} (Time: {exec_time:.2f}s)")
    
    # Calculate statistics
    mean_fitness = np.mean(best_fitness_runs)
    std_fitness = np.std(best_fitness_runs)
    min_fitness = np.min(best_fitness_runs)
    max_fitness = np.max(best_fitness_runs)
    mean_time = np.mean(exec_times)
    
    # Store results
    results.append({
        'Function': func_id,
        'Name': benchmarks.functions_info[func_id]['name'],
        'Type': benchmarks.functions_info[func_id]['type'],
        'Dimension': dim,
        'Global_Min': fmin,
        'Best': min_fitness,
        'Worst': max_fitness,
        'Mean': mean_fitness,
        'Std': std_fitness,
        'Avg_Time': mean_time
    })
    
    print(f"\n  Summary:")
    print(f"    Best: {min_fitness:.6e}")
    print(f"    Worst: {max_fitness:.6e}")
    print(f"    Mean: {mean_fitness:.6e}")
    print(f"    Std: {std_fitness:.6e}")
    print(f"    Avg Time: {mean_time:.2f}s")

# ========== SAVE RESULTS ==========
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('benchmark_results.csv', index=False)
print("✓ Results saved to 'benchmark_results.csv'")

# Save configuration
config_save = {
    'config': CONFIG,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'note': 'FULL PAPER SETTINGS - Results comparable to original DIO paper (Tables 6-9).'
}

with open('benchmark_config.json', 'w') as f:
    json.dump(config_save, f, indent=4)
print("✓ Configuration saved to 'benchmark_config.json'")

# ========== DISPLAY RESULTS TABLE ==========
print(f"\n{'='*70}")
print("BENCHMARK RESULTS")
print(f"{'='*70}")
print(results_df.to_string(index=False))

# ========== VISUALIZATION ==========
print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DIO Benchmark Results', fontsize=16, fontweight='bold')

# 1. Mean Fitness Comparison
ax1 = axes[0, 0]
bars = ax1.barh(results_df['Function'], results_df['Mean'], color='#3498db')
ax1.set_xlabel('Mean Fitness (log scale)', fontweight='bold')
ax1.set_title('Mean Fitness by Function', fontweight='bold')
ax1.set_xscale('log')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.2e}', ha='left', va='center', fontsize=8)

# 2. Best vs Worst
ax2 = axes[0, 1]
x = np.arange(len(results_df))
width = 0.35
ax2.bar(x - width/2, results_df['Best'], width, label='Best', color='#2ecc71')
ax2.bar(x + width/2, results_df['Worst'], width, label='Worst', color='#e74c3c')
ax2.set_xlabel('Function', fontweight='bold')
ax2.set_ylabel('Fitness (log scale)', fontweight='bold')
ax2.set_title('Best vs Worst Fitness', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Function'])
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Standard Deviation
ax3 = axes[1, 0]
bars = ax3.barh(results_df['Function'], results_df['Std'], color='#9b59b6')
ax3.set_xlabel('Standard Deviation (log scale)', fontweight='bold')
ax3.set_title('Std Deviation by Function', fontweight='bold')
ax3.set_xscale('log')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.2e}', ha='left', va='center', fontsize=8)

# 4. Execution Time
ax4 = axes[1, 1]
bars = ax4.barh(results_df['Function'], results_df['Avg_Time'], color='#f39c12')
ax4.set_xlabel('Average Time (seconds)', fontweight='bold')
ax4.set_title('Average Execution Time', fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}s', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('benchmark_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'benchmark_visualization.png'")

# ========== COMPARISON WITH PAPER (TEMPLATE) ==========
print(f"\n{'='*70}")
print("PAPER COMPARISON TEMPLATE")
print(f"{'='*70}")
print("""
✅ RUNNING WITH FULL PAPER CONFIGURATION

Current Settings Match Paper Exactly:
- Population: 30 dholes
- Iterations: 500 
- Independent Runs: 30
- Functions: All 14 benchmark functions (F1-F14)

Expected Results (from DIO Paper Tables 6-9):
- F1 (Sphere): Mean ~10^-94 to 10^-100 (near-zero)
- F2 (Schwefel 2.22): Mean ~10^-50 to 10^-60
- F3 (Schwefel 1.2): Mean ~10^-75 to 10^-85
- F9 (Rastrigin): Mean ~0 (very close to global optimum)
- F10 (Ackley): Mean ~10^-15 (near-zero)

Statistical Significance:
- Wilcoxon rank-sum test will show DIO vs other algorithms
- Results should be comparable to PSO, GA, DE, GWO, etc.
- Check paper Table 10 for statistical comparisons

⏱️  Total Estimated Time: 60-90 minutes
💾  Results will be saved to benchmark_results.csv
""")


plt.show()

print(f"\n{'='*70}")
print("BENCHMARK TESTING COMPLETE!")
print(f"{'='*70}")

# Calculate total time
total_time = time.time() - start_time_all
print(f"\n⏱️  Total Execution Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print(f"📊  Functions Tested: {len(CONFIG['test_functions'])}")
print(f"🔢  Total Runs: {CONFIG['num_runs'] * len(CONFIG['test_functions'])}")
print(f"📈  Total Evaluations: {CONFIG['n_dholes'] * CONFIG['max_iterations'] * CONFIG['num_runs'] * len(CONFIG['test_functions']):,}")

print(f"\nFiles generated:")
print(f"  1. benchmark_results.csv - Numerical results")
print(f"  2. benchmark_config.json - Configuration used")
print(f"  3. benchmark_visualization.png - Visual comparison")
print(f"{'='*70}")
