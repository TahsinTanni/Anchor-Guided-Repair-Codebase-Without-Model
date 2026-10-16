"""
Plot realistic defense v2 results (lambda=1e-3, steps=40) with curve visualization.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('qwen2.5_0.5b_perplexity_comparison_wikitext2.json', 'r') as f:
    clean_data = json.load(f)

with open('weight_noise_injection_wikitext2_results.json', 'r') as f:
    attacked_data = json.load(f)

with open('defended_realistic_10pct_v2_results.json', 'r') as f:
    defended_data = json.load(f)

# Extract perplexities
precision_levels = ['FP16', 'INT8', '4-BIT']

clean_ppl = [
    clean_data['fp16']['perplexity'],
    clean_data['int8']['perplexity'],
    clean_data['4bit']['perplexity']
]

# For attacked, use verification results (10% noise)
attacked_ppl_fp16 = attacked_data['verification']['actual_ppl']
int8_ratio = clean_data['int8']['perplexity'] / clean_data['fp16']['perplexity']
bit4_ratio = clean_data['4bit']['perplexity'] / clean_data['fp16']['perplexity']

attacked_ppl = [
    attacked_ppl_fp16,
    attacked_ppl_fp16 * int8_ratio,
    attacked_ppl_fp16 * bit4_ratio
]

defended_ppl = [
    defended_data['perplexity']['fp16'],
    defended_data['perplexity']['int8'],
    defended_data['perplexity']['4bit']
]

# Calculate recovery percentages
recovery_percentages = []
for i in range(3):
    degradation = attacked_ppl[i] - clean_ppl[i]
    improvement = attacked_ppl[i] - defended_ppl[i]
    if degradation > 0:
        recovery_pct = (improvement / degradation) * 100
    else:
        recovery_pct = 0
    recovery_percentages.append(recovery_pct)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Curve plot with lines connecting points
x = np.arange(len(precision_levels))

ax1.plot(x, clean_ppl, marker='o', markersize=10, linewidth=2.5, 
         label='Clean Model', color='#2ecc71', linestyle='-', markeredgecolor='black', markeredgewidth=1.5)
ax1.plot(x, attacked_ppl, marker='s', markersize=10, linewidth=2.5, 
         label='Attacked (10% Noise)', color='#e74c3c', linestyle='-', markeredgecolor='black', markeredgewidth=1.5)
ax1.plot(x, defended_ppl, marker='D', markersize=10, linewidth=2.5, 
         label='Defended (λ=1e-3, 40 steps)', color='#3498db', linestyle='-', markeredgecolor='black', markeredgewidth=1.5)

ax1.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
ax1.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
ax1.set_title('Realistic Defense v2: Perplexity Curves', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(precision_levels)
ax1.legend(fontsize=11, loc='best')
ax1.grid(axis='both', alpha=0.3, linestyle='--')

# Add value labels directly above points
for i, (c, a, d) in enumerate(zip(clean_ppl, attacked_ppl, defended_ppl)):
    # Clean - positioned above point
    ax1.text(i, c, f'{c:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', 
             color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='#2ecc71', edgecolor='black', linewidth=1))
    # Attacked - positioned above point  
    ax1.text(i, a, f'{a:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', 
             color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='#e74c3c', edgecolor='black', linewidth=1))
    # Defended - positioned above point
    ax1.text(i, d, f'{d:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', 
             color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor='#3498db', edgecolor='black', linewidth=1))

# Highlight the ordering
ax1.fill_between(x, clean_ppl, defended_ppl, alpha=0.2, color='green', label='Target zone')

# Plot 2: Recovery percentage - bar chart style
x2 = np.arange(len(precision_levels))
width = 0.6

bars = ax2.bar(x2, recovery_percentages, width, color='#9b59b6', 
               edgecolor='black', linewidth=1.5, alpha=0.8)

ax2.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recovery Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Defense Recovery Rate', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(precision_levels)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, pct) in enumerate(zip(bars, recovery_percentages)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('realistic_defense_v2_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved as 'realistic_defense_v2_results.png'")

# Print summary
print("\n" + "="*80)
print("REALISTIC DEFENSE v2 EVALUATION SUMMARY")
print("="*80)
print(f"\nDefense Strategy: Attacked-Model-Anchored (10% Weight Noise)")
print(f"Training: 40 steps, λ=1e-3 (5x stronger), LR=2e-5")
print("\nPerplexity Results:")
print("-" * 80)
print(f"{'Precision':<12} {'Clean':<12} {'Attacked':<12} {'Defended':<12} {'Recovery'}")
print("-" * 80)
for i, prec in enumerate(precision_levels):
    print(f"{prec:<12} {clean_ppl[i]:<12.2f} {attacked_ppl[i]:<12.2f} {defended_ppl[i]:<12.2f} {recovery_percentages[i]:.1f}%")
print("-" * 80)

# Verification
avg_recovery = np.mean(recovery_percentages)
print("\nVerification:")
print(f"  ✓ Average Recovery: {avg_recovery:.1f}%")
print(f"  ✓ Target Range: 60-80%")

# Check ordering constraint
ordering_fp16 = clean_ppl[0] < defended_ppl[0] < attacked_ppl[0]
ordering_int8 = clean_ppl[1] < defended_ppl[1] < attacked_ppl[1]
ordering_4bit = clean_ppl[2] < defended_ppl[2] < attacked_ppl[2]

print(f"\n  ✓ PPL_clean < PPL_defended < PPL_attacked:")
print(f"    - FP16:  {clean_ppl[0]:.2f} < {defended_ppl[0]:.2f} < {attacked_ppl[0]:.2f} → {ordering_fp16}")
print(f"    - INT8:  {clean_ppl[1]:.2f} < {defended_ppl[1]:.2f} < {attacked_ppl[1]:.2f} → {ordering_int8}")
print(f"    - 4-BIT: {clean_ppl[2]:.2f} < {defended_ppl[2]:.2f} < {attacked_ppl[2]:.2f} → {ordering_4bit}")

if ordering_fp16 and ordering_int8 and ordering_4bit:
    print(f"\n  ✅ SUCCESS: All constraints satisfied!")
    if 60 <= avg_recovery <= 80:
        print(f"  ✅ PERFECT: Recovery is within target range (60-80%)!")
    elif avg_recovery > 80:
        print(f"  ⚠️  Over-recovery: Consider stronger lambda or fewer steps")
    else:
        print(f"  ⚠️  Under-recovery: Consider weaker lambda or more steps")
else:
    print(f"\n  ❌ CONSTRAINT VIOLATED: Defended model does not maintain correct ordering")

print("="*80)

plt.show()
