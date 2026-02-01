import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ahpy import Compare



# Load APEATO results
with open("/hoame/kiyotoka/Desktop/os/YAFS/apeato_simulation_results.json") as f:
    data = json.load(f)

stats = data["statistics"]

apeato_energy = stats["energy"]["average_per_task"]
apeato_latency = stats["latency"]["average_per_task"]
apeato_deadline = stats["deadline_performance"]["success_rate_percent"] / 100

# Baselines
df = pd.DataFrame({
    "Strategy": ["Local", "Edge", "Cloud", "APEATO"],
    "Energy":  [2.5, 0.15, 0.45, apeato_energy],
    "Latency": [26.5, 52.5, 310.0, apeato_latency],
    "DeadlineSuccess": [0.70, 0.70, 0.85, apeato_deadline]
})


# ===========================================================
# 1️⃣ Weighted Sum Method (WSM)
# ===========================================================
weights = {"Energy": 0.4, "Latency": 0.3, "DeadlineSuccess": 0.3}

# normalize (minimize Energy, min Latency, maximize Deadline)
df_norm = df.copy()
df_norm["Energy"] = df["Energy"].min() / df["Energy"]
df_norm["Latency"] = df["Latency"].min() / df["Latency"]
df_norm["DeadlineSuccess"] = df["DeadlineSuccess"] / df["DeadlineSuccess"].max()

df["WSM_Score"] = (
    df_norm["Energy"] * weights["Energy"] +
    df_norm["Latency"] * weights["Latency"] +
    df_norm["DeadlineSuccess"] * weights["DeadlineSuccess"]
)


# ===========================================================
# 2️⃣ TOPSIS (Manual Implementation)
# ===========================================================
matrix = df[["Energy", "Latency", "DeadlineSuccess"]].to_numpy()
norm = matrix / np.sqrt((matrix ** 2).sum(axis=0))

# Weights
w = np.array([0.4, 0.3, 0.3])
norm_w = norm * w

ideal_best = np.array([
    norm_w[:,0].min(),   # Energy MIN
    norm_w[:,1].min(),   # Latency MIN
    norm_w[:,2].max()    # Deadline MAX
])

ideal_worst = np.array([
    norm_w[:,0].max(),
    norm_w[:,1].max(),
    norm_w[:,2].min()
])

d_best = np.sqrt(((norm_w - ideal_best) ** 2).sum(axis=1))
d_worst = np.sqrt(((norm_w - ideal_worst) ** 2).sum(axis=1))

df["TOPSIS_Score"] = d_worst / (d_best + d_worst)


# ===========================================================
# 3️⃣ Analytic Hierarchy Process (AHP)
# ===========================================================
# ==========================================
# AHP (using ahpy.Compare)
# ==========================================

# ==========================================
# AHP (Compatible with AHPy)
# ==========================================
# ==========================================
# AHP (Compatible with AHPy)
# ==========================================

# Define the comparison matrix in the correct format
criteria_comparisons = {
    ('Energy', 'Latency'): 3,           # Energy is moderately more important than Latency
    ('Energy', 'DeadlineSuccess'): 5,   # Energy is strongly more important than DeadlineSuccess
    ('Latency', 'DeadlineSuccess'): 4   # Latency is moderately to strongly more important than DeadlineSuccess
}

# Create the comparison object
criteria_compare = Compare('Criteria', criteria_comparisons, precision=4)

# Get the weights
criteria_weights = criteria_compare.target_weights

# Normalize the weights to sum to 1 (just to be safe)
total = sum(criteria_weights.values())
criteria_weights = {k: v/total for k, v in criteria_weights.items()}

# Calculate AHP score
df["AHP_Score"] = (
    (1 / df["Energy"]) * criteria_weights['Energy'] +
    (1 / df["Latency"]) * criteria_weights['Latency'] +
    df["DeadlineSuccess"] * criteria_weights['DeadlineSuccess']
)



# ===========================================================
# FINAL RANKING
# ===========================================================
df["Final_Rank"] = df[["WSM_Score", "TOPSIS_Score", "AHP_Score"]].mean(axis=1).rank(ascending=False)


# ===========================================================
# GRAPHS
# ===========================================================
# Update the save_plot function to use the images directory
def save_plot(title, y, filename):
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    plt.figure()
    plt.bar(df["Strategy"], y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{filename}")
    plt.close()

save_plot("WSM Score Comparison", df["WSM_Score"], "wsm_scores.png")
save_plot("TOPSIS Score Comparison", df["TOPSIS_Score"], "topsis_scores.png")
save_plot("AHP Score Comparison", df["AHP_Score"], "ahp_scores.png")

# Combined
plt.figure()
index = np.arange(len(df))
bar_width = 0.25

plt.bar(index, df["WSM_Score"], width=bar_width, label="WSM")
plt.bar(index + bar_width, df["TOPSIS_Score"], width=bar_width, label="TOPSIS")
plt.bar(index + 2 * bar_width, df["AHP_Score"], width=bar_width, label="AHP")

plt.xticks(index + bar_width, df["Strategy"], rotation=45)
plt.legend()
plt.title("Comparison of All MCDM Methods")
plt.tight_layout()
plt.savefig("images/combined_scores.png")
plt.close()


# PRINT RESULTS
print("\n===== FINAL RESULTS =====\n")
print(df)
print("\nGraphs saved in /mnt/data/")


# ===========================================================
# ACCURACY COMPARISON
# ===========================================================
print("\n===== ACCURACY COMPARISON (vs APEATO) =====")

# Get APEATO's values as baseline
apeato_values = df[df['Strategy'] == 'APEATO'][['Energy', 'Latency', 'DeadlineSuccess']].iloc[0]

# Calculate accuracy for each metric
metrics = ['Energy', 'Latency', 'DeadlineSuccess']
for metric in metrics:
    print(f"\n--- {metric} ---")
    for idx, row in df.iterrows():
        if row['Strategy'] != 'APEATO':
            if metric == 'DeadlineSuccess':
                # Higher is better
                accuracy = (row[metric] / apeato_values[metric]) * 100
            else:
                # Lower is better (for Energy and Latency)
                accuracy = (apeato_values[metric] / row[metric]) * 100
            print(f"{row['Strategy']} vs APEATO: {accuracy:.2f}%")

# Visualize accuracy comparison
plt.figure(figsize=(12, 6))
metrics = ['Energy', 'Latency', 'DeadlineSuccess']
x = np.arange(len(metrics))
width = 0.25

for i, strategy in enumerate(['Local', 'Edge', 'Cloud']):
    accuracies = []
    for metric in metrics:
        if metric == 'DeadlineSuccess':
            acc = (df[df['Strategy'] == strategy][metric].values[0] / 
                   df[df['Strategy'] == 'APEATO'][metric].values[0]) * 100
        else:
            acc = (df[df['Strategy'] == 'APEATO'][metric].values[0] / 
                   df[df['Strategy'] == strategy][metric].values[0]) * 100
        accuracies.append(acc)
    
    plt.bar(x + i*width, accuracies, width, label=strategy)

plt.xlabel('Metrics')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison with APEATO (Higher is better)')
plt.xticks(x + width, metrics)
plt.legend()
plt.tight_layout()
plt.savefig("images/accuracy_comparison.png")
plt.close()

print("\nAccuracy comparison plot saved to images/accuracy_comparison.png")



# ===========================================================
# COMPREHENSIVE COMPARISON MATRIX
# ===========================================================
print("\n===== COMPREHENSIVE COMPARISON MATRIX =====")

# Function to compare two strategies
def compare_strategies(strat1, strat2, metric):
    val1 = df[df['Strategy'] == strat1][metric].values[0]
    val2 = df[df['Strategy'] == strat2][metric].values[0]
    
    if metric == 'DeadlineSuccess':
        # Higher is better
        return (val1 / val2) * 100 if val2 != 0 else 0
    else:
        # Lower is better
        return (val2 / val1) * 100 if val1 != 0 else 0

# Create comparison matrix for each metric
metrics = ['Energy', 'Latency', 'DeadlineSuccess']
strategies = df['Strategy'].tolist()

for metric in metrics:
    print(f"\n--- {metric} Comparison Matrix (%) ---")
    print(f"{'vs':<10}", end="")
    for s in strategies:
        print(f"{s:<10}", end="")
    print()
    
    for s1 in strategies:
        print(f"{s1:<10}", end="")
        for s2 in strategies:
            if s1 == s2:
                print("100.0    ", end="")
            else:
                comparison = compare_strategies(s1, s2, metric)
                print(f"{comparison:>6.1f}   ", end="")
        print()

# Create a combined visualization
plt.figure(figsize=(15, 10))
n_metrics = len(metrics)
n_strategies = len(strategies)
bar_width = 0.15
index = np.arange(n_metrics)

for i, strategy in enumerate(strategies):
    values = []
    for metric in metrics:
        if metric == 'DeadlineSuccess':
            # Higher is better
            values.append(df[df['Strategy'] == strategy][metric].values[0] * 100)
        else:
            # Lower is better - convert to percentage of best
            best = df[metric].min() if metric != 'DeadlineSuccess' else df[metric].max()
            current = df[df['Strategy'] == strategy][metric].values[0]
            if metric != 'DeadlineSuccess':
                values.append((best / current) * 100)
            else:
                values.append((current / best) * 100)
    
    plt.bar(index + i*bar_width, values, bar_width, label=strategy)

plt.xlabel('Metrics')
plt.ylabel('Performance (% of best)')
plt.title('Comprehensive Performance Comparison (Higher is better)')
plt.xticks(index + bar_width*1.5, metrics)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("images/comprehensive_comparison.png", bbox_inches='tight')
plt.close()

print("\nComprehensive comparison plot saved to images/comprehensive_comparison.png")


# ===========================================================
# MCDM METHODS COMPARISON VISUALIZATION
# ===========================================================
print("\n===== MCDM METHODS COMPARISON =====")

# Prepare data for visualization
methods = ['WSM', 'TOPSIS', 'AHP']
scores = {
    'Local': [df[df['Strategy'] == 'Local']['WSM_Score'].values[0],
              df[df['Strategy'] == 'Local']['TOPSIS_Score'].values[0],
              df[df['Strategy'] == 'Local']['AHP_Score'].values[0]],
    'Edge': [df[df['Strategy'] == 'Edge']['WSM_Score'].values[0],
             df[df['Strategy'] == 'Edge']['TOPSIS_Score'].values[0],
             df[df['Strategy'] == 'Edge']['AHP_Score'].values[0]],
    'Cloud': [df[df['Strategy'] == 'Cloud']['WSM_Score'].values[0],
              df[df['Strategy'] == 'Cloud']['TOPSIS_Score'].values[0],
              df[df['Strategy'] == 'Cloud']['AHP_Score'].values[0]],
    'APEATO': [df[df['Strategy'] == 'APEATO']['WSM_Score'].values[0],
               df[df['Strategy'] == 'APEATO']['TOPSIS_Score'].values[0],
               df[df['Strategy'] == 'APEATO']['AHP_Score'].values[0]]
}

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.suptitle('MCDM Methods Comparison', fontsize=16)

# Plot each method
for i, method in enumerate(methods):
    method_scores = [scores[strategy][i] for strategy in ['Local', 'Edge', 'Cloud', 'APEATO']]
    axs[i].bar(['Local', 'Edge', 'Cloud', 'APEATO'], method_scores, 
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axs[i].set_title(f'{method} Scores')
    axs[i].set_ylim(0, 1.1)
    axs[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for j, v in enumerate(method_scores):
        axs[i].text(j, v + 0.02, f'{v:.3f}', ha='center')

# Add method descriptions
descriptions = [
    "• Normalizes scores\n• Weights: Energy 40%, \n  Latency 30%, \n  Deadline 30%",
    "• Minimizes distance to ideal\n• Maximizes distance from worst\n• Considers all objectives",
    "• Uses pairwise comparisons\n• Calculates consistency\n• Derives weights objectively"
]

for i, desc in enumerate(descriptions):
    axs[i].text(0.5, -0.3, desc, transform=axs[i].transAxes, 
                ha='center', va='top', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig("images/mcdm_comparison.png", bbox_inches='tight', dpi=300)
plt.close()

print("MCDM methods comparison plot saved to images/mcdm_comparison.png")

