import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from JSON file
with open('noisy_quantized_perplexity_wikitext2.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract data
baseline = results['baseline']
noisy_5 = results['noise_levels']['5%']['results']
noisy_10 = results['noise_levels']['10%']['results']
noisy_15 = results['noise_levels']['15%']['results']
noisy_20 = results['noise_levels']['20%']['results']
sigma_5 = results['noise_levels']['5%']['sigma']
sigma_10 = results['noise_levels']['10%']['sigma']
sigma_15 = results['noise_levels']['15%']['sigma']
sigma_20 = results['noise_levels']['20%']['sigma']

# Precision types
precision_types = ['FP16', 'INT8', '4-BIT']
x_positions = np.arange(len(precision_types))

# Perplexity values
clean_values = [
    baseline['fp16'],
    baseline['int8'],
    baseline['4bit']
]

noisy_5_values = [
    noisy_5['fp16']['perplexity'],
    noisy_5['int8']['perplexity'],
    noisy_5['4bit']['perplexity']
]

noisy_10_values = [
    noisy_10['fp16']['perplexity'],
    noisy_10['int8']['perplexity'],
    noisy_10['4bit']['perplexity']
]

noisy_15_values = [
    noisy_15['fp16']['perplexity'],
    noisy_15['int8']['perplexity'],
    noisy_15['4bit']['perplexity']
]

noisy_20_values = [
    noisy_20['fp16']['perplexity'],
    noisy_20['int8']['perplexity'],
    noisy_20['4bit']['perplexity']
]

# Create the plot
plt.figure(figsize=(12, 7))

# Plot lines
plt.plot(x_positions, clean_values, marker='o', linewidth=2.5, markersize=10, 
         label='Clean Model (Baseline)', color='#2E86AB', linestyle='-', zorder=5)
plt.plot(x_positions, noisy_5_values, marker='s', linewidth=2.5, markersize=8, 
         label=f'Noisy Model (5% Target, σ={sigma_5:.6f})', color='#66BB6A', linestyle='--', zorder=4)
plt.plot(x_positions, noisy_10_values, marker='^', linewidth=2.5, markersize=8, 
         label=f'Noisy Model (10% Target, σ={sigma_10:.6f})', color='#FFA726', linestyle='--', zorder=3)
plt.plot(x_positions, noisy_15_values, marker='D', linewidth=2.5, markersize=8, 
         label=f'Noisy Model (15% Target, σ={sigma_15:.6f})', color='#A23B72', linestyle='--', zorder=2)
plt.plot(x_positions, noisy_20_values, marker='v', linewidth=2.5, markersize=8, 
         label=f'Noisy Model (20% Target, σ={sigma_20:.6f})', color='#E53935', linestyle='--', zorder=1)

# Customize the plot
plt.xlabel('Precision Type', fontsize=12, fontweight='bold')
plt.ylabel('Perplexity', fontsize=12, fontweight='bold')
plt.title('Clean vs Weight Noisy Models (Multiple Noise Levels)\nQwen2.5-0.5B with Actual INT8 & 4-BIT Quantization - WikiText-2 Test Set', 
          fontsize=14, fontweight='bold', pad=20)

# Set x-axis
plt.xticks(x_positions, precision_types, fontsize=11)
plt.yticks(fontsize=11)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add legend
plt.legend(loc='best', fontsize=10, framealpha=0.95)

# Add value annotations only for clean and 20% (to avoid clutter)
for i, clean in enumerate(clean_values):
    plt.annotate(f'{clean:.2f}', 
                xy=(x_positions[i], clean), 
                xytext=(0, 12), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='#2E86AB',
                fontweight='bold')

for i, noisy in enumerate(noisy_20_values):
    plt.annotate(f'{noisy:.2f}', 
                xy=(x_positions[i], noisy), 
                xytext=(0, -18), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='#E53935',
                fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the plot
output_file = 'baseline_vs_noisy_all_levels_wikitext2_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()

print("\nSummary:")
print("="*70)
print(f"Clean Model - FP16:  {clean_values[0]:.4f}")
print(f"Clean Model - INT8:  {clean_values[1]:.4f}")
print(f"Clean Model - 4-BIT: {clean_values[2]:.4f}")
print()
print(f"Noisy 5%  - FP16:    {noisy_5_values[0]:.4f} ({((noisy_5_values[0]/clean_values[0]-1)*100):+.2f}%)")
print(f"Noisy 5%  - INT8:    {noisy_5_values[1]:.4f} ({((noisy_5_values[1]/clean_values[1]-1)*100):+.2f}%)")
print(f"Noisy 5%  - 4-BIT:   {noisy_5_values[2]:.4f} ({((noisy_5_values[2]/clean_values[2]-1)*100):+.2f}%)")
print()
print(f"Noisy 10% - FP16:    {noisy_10_values[0]:.4f} ({((noisy_10_values[0]/clean_values[0]-1)*100):+.2f}%)")
print(f"Noisy 10% - INT8:    {noisy_10_values[1]:.4f} ({((noisy_10_values[1]/clean_values[1]-1)*100):+.2f}%)")
print(f"Noisy 10% - 4-BIT:   {noisy_10_values[2]:.4f} ({((noisy_10_values[2]/clean_values[2]-1)*100):+.2f}%)")
print()
print(f"Noisy 15% - FP16:    {noisy_15_values[0]:.4f} ({((noisy_15_values[0]/clean_values[0]-1)*100):+.2f}%)")
print(f"Noisy 15% - INT8:    {noisy_15_values[1]:.4f} ({((noisy_15_values[1]/clean_values[1]-1)*100):+.2f}%)")
print(f"Noisy 15% - 4-BIT:   {noisy_15_values[2]:.4f} ({((noisy_15_values[2]/clean_values[2]-1)*100):+.2f}%)")
print()
print(f"Noisy 20% - FP16:    {noisy_20_values[0]:.4f} ({((noisy_20_values[0]/clean_values[0]-1)*100):+.2f}%)")
print(f"Noisy 20% - INT8:    {noisy_20_values[1]:.4f} ({((noisy_20_values[1]/clean_values[1]-1)*100):+.2f}%)")
print(f"Noisy 20% - 4-BIT:   {noisy_20_values[2]:.4f} ({((noisy_20_values[2]/clean_values[2]-1)*100):+.2f}%)")
print("="*70)
