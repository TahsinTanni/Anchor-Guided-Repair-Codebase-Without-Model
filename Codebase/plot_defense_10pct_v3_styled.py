"""
Styled visualization for 10% noise attack defense with new hyperparameters (v3)
Lambda=1e-4, steps=120, lr=2e-5
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open('qwen2.5_0.5b_perplexity_comparison_wikitext2.json', 'r') as f:
    baseline_data = json.load(f)

with open('noisy_quantized_perplexity_wikitext2.json', 'r') as f:
    noisy_data = json.load(f)

with open('defended_10pct_v3_results.json', 'r') as f:
    defended_data = json.load(f)

# Extract data
clean_fp16 = baseline_data['fp16']['perplexity']
clean_int8 = baseline_data['int8']['perplexity']
clean_4bit = baseline_data['4bit']['perplexity']

attacked_fp16 = noisy_data['noise_levels']['10%']['results']['fp16']['perplexity']
attacked_int8 = noisy_data['noise_levels']['10%']['results']['int8']['perplexity']
attacked_4bit = noisy_data['noise_levels']['10%']['results']['4bit']['perplexity']

defended_fp16 = defended_data['perplexity']['fp16']
defended_int8 = defended_data['perplexity']['int8']
defended_4bit = defended_data['perplexity']['4bit']

# Prepare data for plotting
precision_types = ['FP16', 'INT8', '4-BIT']
clean_perplexities = [clean_fp16, clean_int8, clean_4bit]
attacked_perplexities = [attacked_fp16, attacked_int8, attacked_4bit]
defended_perplexities = [defended_fp16, defended_int8, defended_4bit]

# Calculate metrics
def calc_recovery(clean, attacked, defended):
    degradation = attacked - clean
    recovery = attacked - defended
    return (recovery / degradation) * 100

recovery_fp16 = calc_recovery(clean_fp16, attacked_fp16, defended_fp16)
recovery_int8 = calc_recovery(clean_int8, attacked_int8, defended_int8)
recovery_4bit = calc_recovery(clean_4bit, attacked_4bit, defended_4bit)

# Attack degradation percentages
attack_deg_fp16 = ((attacked_fp16 - clean_fp16) / clean_fp16) * 100
attack_deg_int8 = ((attacked_int8 - clean_int8) / clean_int8) * 100
attack_deg_4bit = ((attacked_4bit - clean_4bit) / clean_4bit) * 100

# Defended degradation percentages (compared to clean)
defended_deg_fp16 = ((defended_fp16 - clean_fp16) / clean_fp16) * 100
defended_deg_int8 = ((defended_int8 - clean_int8) / clean_int8) * 100
defended_deg_4bit = ((defended_4bit - clean_4bit) / clean_4bit) * 100

# Create the plot
plt.figure(figsize=(14, 8))

# Plot lines
x_pos = [0, 1, 2]
plt.plot(x_pos, clean_perplexities, 'o-', linewidth=2.5, markersize=10, 
         color='#2196F3', label='Clean Model (Baseline)', zorder=3)
plt.plot(x_pos, attacked_perplexities, 'X--', linewidth=2.5, markersize=12, 
         color='#F44336', label='Attacked Model (10% Noise)', zorder=2)
plt.plot(x_pos, defended_perplexities, 'D-.', linewidth=2.5, markersize=10, 
         color='#4CAF50', label='Defended Model (Weight Anchoring)', zorder=3)

# Add value labels
for i, (clean, attacked, defended) in enumerate(zip(clean_perplexities, attacked_perplexities, defended_perplexities)):
    plt.text(i, clean, f'{clean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2196F3')
    plt.text(i, attacked, f'{attacked:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#F44336')
    plt.text(i, defended, f'{defended:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#4CAF50')

# Styling
plt.xlabel('Precision Type', fontsize=13, fontweight='bold')
plt.ylabel('Perplexity', fontsize=13, fontweight='bold')
plt.title('Defense Effectiveness: Clean vs Attacked vs Defended Models\nQwen2.5-0.5B with 10% Noise Attack - WikiText-2 Test Set', 
          fontsize=15, fontweight='bold', pad=20)
plt.xticks(x_pos, precision_types, fontsize=12)
plt.yticks(fontsize=11)
plt.legend(fontsize=11, loc='upper left', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Add text boxes with metrics - positioned to avoid covering the lines
# Defense Recovery box (top left, below legend)
recovery_text = f"Defense Recovery:\nFP16: {recovery_fp16:.1f}%\nINT8: {recovery_int8:.1f}%\n4-BIT: {recovery_4bit:.1f}%"
plt.text(0.02, 0.65, recovery_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=2))

# Attack Degradation box (top right)
attack_text = f"Attack Degradation:\nFP16: +{attack_deg_fp16:.1f}%\nINT8: +{attack_deg_int8:.1f}%\n4-BIT: +{attack_deg_4bit:.1f}%"
plt.text(0.98, 0.98, attack_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2))

# Defended Degradation box (middle right)
defended_text = f"Defended Degradation:\nFP16: {defended_deg_fp16:+.1f}%\nINT8: {defended_deg_int8:+.1f}%\n4-BIT: {defended_deg_4bit:+.1f}%"
plt.text(0.98, 0.65, defended_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))

plt.tight_layout()
plt.savefig('defense_effectiveness_10pct_v3.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'defense_effectiveness_10pct_v3.png'")
print(f"\nHyperparameters: λ=1e-4, steps=120, lr=2e-5, grad_accum=1")
print(f"Defense Recovery: FP16={recovery_fp16:.1f}%, INT8={recovery_int8:.1f}%, 4-BIT={recovery_4bit:.1f}%")
