import json
import matplotlib.pyplot as plt
import numpy as np

# Baselines
clean_ppl = {
    'fp16': 12.46,
    'int8': 12.48,
    '4bit': 13.79
}

attacked_ppl = {
    'fp16': 13.15,
    'int8': 13.17,
    '4bit': 14.55
}

# Load v7 results (15 steps)
with open('defended_realistic_10pct_v7_results.json', 'r') as f:
    v7_data = json.load(f)
    v7_ppl = v7_data['perplexity']

# Load v8 results (10 steps)
with open('defended_realistic_10pct_v8_results.json', 'r') as f:
    v8_data = json.load(f)
    v8_ppl = v8_data['perplexity']

# Calculate recovery percentages
def calc_recovery(defended, attacked, clean):
    gap = attacked - clean
    improvement = attacked - defended
    return (improvement / gap) * 100

v7_recovery = {
    'fp16': calc_recovery(v7_ppl['fp16'], attacked_ppl['fp16'], clean_ppl['fp16']),
    'int8': calc_recovery(v7_ppl['int8'], attacked_ppl['int8'], clean_ppl['int8']),
    '4bit': calc_recovery(v7_ppl['4bit'], attacked_ppl['4bit'], clean_ppl['4bit'])
}

v8_recovery = {
    'fp16': calc_recovery(v8_ppl['fp16'], attacked_ppl['fp16'], clean_ppl['fp16']),
    'int8': calc_recovery(v8_ppl['int8'], attacked_ppl['int8'], clean_ppl['int8']),
    '4bit': calc_recovery(v8_ppl['4bit'], attacked_ppl['4bit'], clean_ppl['4bit'])
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Perplexity curves
precisions = ['FP16', 'INT8', '4-BIT']
x_pos = np.arange(len(precisions))

clean_vals = [clean_ppl['fp16'], clean_ppl['int8'], clean_ppl['4bit']]
attacked_vals = [attacked_ppl['fp16'], attacked_ppl['int8'], attacked_ppl['4bit']]
v7_vals = [v7_ppl['fp16'], v7_ppl['int8'], v7_ppl['4bit']]
v8_vals = [v8_ppl['fp16'], v8_ppl['int8'], v8_ppl['4bit']]

ax1.plot(x_pos, clean_vals, 'o-', linewidth=2.5, markersize=8, label='Clean FP16', color='#2ecc71')
ax1.plot(x_pos, attacked_vals, 's-', linewidth=2.5, markersize=8, label='Attacked (10% noise)', color='#e74c3c')
ax1.plot(x_pos, v7_vals, '^-', linewidth=2.5, markersize=8, label='Defended v7 (15 steps)', color='#3498db')
ax1.plot(x_pos, v8_vals, 'd-', linewidth=2.5, markersize=8, label='Defended v8 (10 steps)', color='#9b59b6')

# Add value labels
for i, (c, a, v7, v8) in enumerate(zip(clean_vals, attacked_vals, v7_vals, v8_vals)):
    ax1.text(i, c, f'{c:.2f}', ha='center', va='bottom', fontsize=9, color='#2ecc71', fontweight='bold')
    ax1.text(i, a, f'{a:.2f}', ha='center', va='bottom', fontsize=9, color='#e74c3c', fontweight='bold')
    ax1.text(i, v7, f'{v7:.2f}', ha='center', va='bottom', fontsize=9, color='#3498db', fontweight='bold')
    ax1.text(i, v8, f'{v8:.2f}', ha='center', va='bottom', fontsize=9, color='#9b59b6', fontweight='bold')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(precisions, fontsize=11)
ax1.set_ylabel('Perplexity (↓ better)', fontsize=12, fontweight='bold')
ax1.set_title('Perplexity Comparison: v7 (15 steps) vs v8 (10 steps)', fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='upper left', framealpha=0.95, fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([11.5, 15])

# Right plot: Recovery percentage bars
recovery_v7_vals = [v7_recovery['fp16'], v7_recovery['int8'], v7_recovery['4bit']]
recovery_v8_vals = [v8_recovery['fp16'], v8_recovery['int8'], v8_recovery['4bit']]

bar_width = 0.35
x_pos_bars = np.arange(len(precisions))

bars1 = ax2.bar(x_pos_bars - bar_width/2, recovery_v7_vals, bar_width, 
                label='v7 (15 steps)', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax2.bar(x_pos_bars + bar_width/2, recovery_v8_vals, bar_width, 
                label='v8 (10 steps)', color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add target range
ax2.axhspan(60, 80, alpha=0.2, color='green', label='Target: 60-80%')
ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100% = Over-recovery')

ax2.set_xticks(x_pos_bars)
ax2.set_xticklabels(precisions, fontsize=11)
ax2.set_ylabel('Recovery Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Defense Recovery Rate', fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.set_ylim([0, 160])

plt.tight_layout()
plt.savefig('realistic_defense_v7_v8_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as: realistic_defense_v7_v8_comparison.png")

# Print summary
print("\n" + "="*70)
print("SUMMARY: v7 (15 steps) vs v8 (10 steps)")
print("="*70)
print(f"\nv7 (15 steps) - FP16: {v7_ppl['fp16']:.2f}, INT8: {v7_ppl['int8']:.2f}, 4-BIT: {v7_ppl['4bit']:.2f}")
print(f"v8 (10 steps) - FP16: {v8_ppl['fp16']:.2f}, INT8: {v8_ppl['int8']:.2f}, 4-BIT: {v8_ppl['4bit']:.2f}")
print(f"\nClean baseline - FP16: {clean_ppl['fp16']:.2f}, INT8: {clean_ppl['int8']:.2f}, 4-BIT: {clean_ppl['4bit']:.2f}")
print(f"Attacked       - FP16: {attacked_ppl['fp16']:.2f}, INT8: {attacked_ppl['int8']:.2f}, 4-BIT: {attacked_ppl['4bit']:.2f}")

print(f"\nv7 Recovery - FP16: {v7_recovery['fp16']:.1f}%, INT8: {v7_recovery['int8']:.1f}%, 4-BIT: {v7_recovery['4bit']:.1f}%")
print(f"v8 Recovery - FP16: {v8_recovery['fp16']:.1f}%, INT8: {v8_recovery['int8']:.1f}%, 4-BIT: {v8_recovery['4bit']:.1f}%")

print(f"\nv7 Avg Recovery: {np.mean(list(v7_recovery.values())):.1f}%")
print(f"v8 Avg Recovery: {np.mean(list(v8_recovery.values())):.1f}%")

print("\n⚠️  BOTH STILL OVER-RECOVERING (defended < clean)")
print("="*70)
