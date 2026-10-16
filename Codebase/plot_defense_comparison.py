import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('noisy_quantized_perplexity_wikitext2.json', 'r', encoding='utf-8') as f:
    attack_results = json.load(f)

with open('defended_10pct_v2_results.json', 'r', encoding='utf-8') as f:
    defense_results = json.load(f)

# Extract data
baseline = attack_results['baseline']
noisy_10 = attack_results['noise_levels']['10%']['results']
defended = defense_results['perplexity']

# Precision types
precision_types = ['FP16', 'INT8', '4-BIT']
x_positions = np.arange(len(precision_types))

# Perplexity values
clean_values = [
    baseline['fp16'],
    baseline['int8'],
    baseline['4bit']
]

attacked_values = [
    noisy_10['fp16']['perplexity'],
    noisy_10['int8']['perplexity'],
    noisy_10['4bit']['perplexity']
]

defended_values = [
    defended['fp16'],
    defended['int8'],
    defended['4bit']
]

# Create the plot
plt.figure(figsize=(12, 7))

# Plot lines
plt.plot(x_positions, clean_values, marker='o', linewidth=2.5, markersize=10, 
         label='Clean Model (Baseline)', color='#2E86AB', linestyle='-', zorder=5)
plt.plot(x_positions, attacked_values, marker='X', linewidth=2.5, markersize=10, 
         label='Attacked Model (10% Noise)', color='#E53935', linestyle='--', zorder=4)
plt.plot(x_positions, defended_values, marker='D', linewidth=2.5, markersize=10, 
         label='Defended Model (Weight Anchoring)', color='#66BB6A', linestyle='-.', zorder=3)

# Customize the plot
plt.xlabel('Precision Type', fontsize=13, fontweight='bold')
plt.ylabel('Perplexity', fontsize=13, fontweight='bold')
plt.title('Defense Effectiveness: Clean vs Attacked vs Defended Models\nQwen2.5-0.5B with 10% Noise Attack - WikiText-2 Test Set', 
          fontsize=14, fontweight='bold', pad=20)

# Set x-axis
plt.xticks(x_positions, precision_types, fontsize=12)
plt.yticks(fontsize=11)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add legend
plt.legend(loc='best', fontsize=11, framealpha=0.95)

# Add value annotations
for i, (clean, attack, defend) in enumerate(zip(clean_values, attacked_values, defended_values)):
    # Clean (top)
    plt.annotate(f'{clean:.2f}', 
                xy=(x_positions[i], clean), 
                xytext=(0, 12), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='#2E86AB',
                fontweight='bold')
    
    # Attacked (top-right)
    plt.annotate(f'{attack:.2f}', 
                xy=(x_positions[i], attack), 
                xytext=(15, 5), 
                textcoords='offset points',
                ha='left', 
                fontsize=9,
                color='#E53935',
                fontweight='bold')
    
    # Defended (bottom)
    plt.annotate(f'{defend:.2f}', 
                xy=(x_positions[i], defend), 
                xytext=(0, -18), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='#66BB6A',
                fontweight='bold')

# Calculate recovery percentages
recovery_fp16 = (attacked_values[0] - defended_values[0]) / (attacked_values[0] - clean_values[0]) * 100
recovery_int8 = (attacked_values[1] - defended_values[1]) / (attacked_values[1] - clean_values[1]) * 100
recovery_4bit = (attacked_values[2] - defended_values[2]) / (attacked_values[2] - clean_values[2]) * 100

# Add text box with recovery info
textstr = f'Defense Recovery:\nFP16: {recovery_fp16:.1f}%\nINT8: {recovery_int8:.1f}%\n4-BIT: {recovery_4bit:.1f}%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Add degradation info on the right
attack_deg_fp16 = ((attacked_values[0] - clean_values[0]) / clean_values[0]) * 100
attack_deg_int8 = ((attacked_values[1] - clean_values[1]) / clean_values[1]) * 100
attack_deg_4bit = ((attacked_values[2] - clean_values[2]) / clean_values[2]) * 100

defend_deg_fp16 = ((defended_values[0] - clean_values[0]) / clean_values[0]) * 100
defend_deg_int8 = ((defended_values[1] - clean_values[1]) / clean_values[1]) * 100
defend_deg_4bit = ((defended_values[2] - clean_values[2]) / clean_values[2]) * 100

textstr2 = f'Attack Degradation:\nFP16: +{attack_deg_fp16:.1f}%\nINT8: +{attack_deg_int8:.1f}%\n4-BIT: +{attack_deg_4bit:.1f}%\n\nDefended Degradation:\nFP16: +{defend_deg_fp16:.1f}%\nINT8: +{defend_deg_int8:.1f}%\n4-BIT: +{defend_deg_4bit:.1f}%'
props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.98, 0.98, textstr2, transform=plt.gca().transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props2)

# Adjust layout
plt.tight_layout()

# Save the plot
output_file = 'defense_comparison_10pct_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"Plot saved to: {output_file}")
print(f"{'='*70}")

# Print summary
print("\nSUMMARY - Defense Against 10% Noise Attack")
print("="*70)
print("\nPerplexity Comparison:")
print(f"{'Precision':<10} {'Clean':<10} {'Attacked':<10} {'Defended':<10} {'Recovery':<10}")
print("-"*70)
print(f"{'FP16':<10} {clean_values[0]:<10.2f} {attacked_values[0]:<10.2f} {defended_values[0]:<10.2f} {recovery_fp16:<10.1f}%")
print(f"{'INT8':<10} {clean_values[1]:<10.2f} {attacked_values[1]:<10.2f} {defended_values[1]:<10.2f} {recovery_int8:<10.1f}%")
print(f"{'4-BIT':<10} {clean_values[2]:<10.2f} {attacked_values[2]:<10.2f} {defended_values[2]:<10.2f} {recovery_4bit:<10.1f}%")
print("="*70)

plt.show()
