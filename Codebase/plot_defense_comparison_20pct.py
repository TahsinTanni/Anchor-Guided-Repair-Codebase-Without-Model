"""
Visualize defense effectiveness for 20% noise attack
Compares clean, attacked (20% noise), and defended models across FP16, INT8, and 4-BIT quantization
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open('qwen2.5_0.5b_perplexity_comparison_wikitext2.json', 'r') as f:
    baseline_data = json.load(f)

with open('noisy_quantized_perplexity_wikitext2.json', 'r') as f:
    noisy_data = json.load(f)

with open('defended_20pct_v2_results.json', 'r') as f:
    defended_data = json.load(f)

# Extract baseline (clean) perplexities
clean_fp16 = baseline_data['fp16']['perplexity']
clean_int8 = baseline_data['int8']['perplexity']
clean_4bit = baseline_data['4bit']['perplexity']

# Extract attacked (20% noise) perplexities
attacked_fp16 = noisy_data['noise_levels']['20%']['results']['fp16']['perplexity']
attacked_int8 = noisy_data['noise_levels']['20%']['results']['int8']['perplexity']
attacked_4bit = noisy_data['noise_levels']['20%']['results']['4bit']['perplexity']

# Extract defended perplexities
defended_fp16 = defended_data['perplexity']['fp16']
defended_int8 = defended_data['perplexity']['int8']
defended_4bit = defended_data['perplexity']['4bit']

# Prepare data for plotting
precision_types = ['FP16', 'INT8', '4-BIT']
clean_perplexities = [clean_fp16, clean_int8, clean_4bit]
attacked_perplexities = [attacked_fp16, attacked_int8, attacked_4bit]
defended_perplexities = [defended_fp16, defended_int8, defended_4bit]

# Calculate recovery percentages
recovery_percentages = []
for clean, attacked, defended in zip(clean_perplexities, attacked_perplexities, defended_perplexities):
    degradation = attacked - clean
    recovery = attacked - defended
    recovery_pct = (recovery / degradation) * 100
    recovery_percentages.append(recovery_pct)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Perplexity comparison
x = np.arange(len(precision_types))
width = 0.25

bars1 = ax1.bar(x - width, clean_perplexities, width, label='Clean (Baseline)', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, attacked_perplexities, width, label='Attacked (20% Noise)', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x + width, defended_perplexities, width, label='Defended (Anchored)', color='#3498db', alpha=0.8)

ax1.set_xlabel('Precision Type', fontsize=12, fontweight='bold')
ax1.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_title('Defense Against 20% Weight Noise Attack', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(precision_types)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: Recovery percentage
bars4 = ax2.bar(precision_types, recovery_percentages, color=['#9b59b6', '#e67e22', '#1abc9c'], alpha=0.8)

ax2.set_xlabel('Precision Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recovery Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Attack Degradation Recovery', fontsize=14, fontweight='bold')
ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Recovery')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend()

# Add value labels on bars
for bar, pct in zip(bars4, recovery_percentages):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('defense_comparison_20pct.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'defense_comparison_20pct.png'")

# Print summary statistics
print("\n" + "="*70)
print("DEFENSE EFFECTIVENESS SUMMARY - 20% NOISE ATTACK")
print("="*70)
print(f"\nHyperparameters: λ=1e-2, steps=300, lr=1e-5")
print("\nPerplexity Results:")
print(f"{'Precision':<10} {'Clean':<10} {'Attacked':<10} {'Defended':<10} {'Recovery':<12}")
print("-" * 70)
for i, prec in enumerate(precision_types):
    print(f"{prec:<10} {clean_perplexities[i]:<10.2f} {attacked_perplexities[i]:<10.2f} "
          f"{defended_perplexities[i]:<10.2f} {recovery_percentages[i]:<12.1f}%")

print("\nDegradation Analysis:")
for i, prec in enumerate(precision_types):
    degradation = ((attacked_perplexities[i] - clean_perplexities[i]) / clean_perplexities[i]) * 100
    defense_improvement = ((attacked_perplexities[i] - defended_perplexities[i]) / attacked_perplexities[i]) * 100
    print(f"{prec}: Attack caused {degradation:.2f}% increase, Defense reduced by {defense_improvement:.2f}%")

print("="*70)
