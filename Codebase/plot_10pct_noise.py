import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from JSON file
with open('noisy_quantized_perplexity_wikitext2.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract data
baseline = results['baseline']
noisy_10 = results['noise_levels']['10%']['results']
sigma_10 = results['noise_levels']['10%']['sigma']

# Precision types
precision_types = ['FP16', 'INT8', '4-BIT']
x_positions = np.arange(len(precision_types))

# Perplexity values
clean_values = [
    baseline['fp16'],
    baseline['int8'],
    baseline['4bit']
]

noisy_10_values = [
    noisy_10['fp16']['perplexity'],
    noisy_10['int8']['perplexity'],
    noisy_10['4bit']['perplexity']
]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(x_positions, clean_values, marker='o', linewidth=2.5, markersize=10, 
         label='Clean Model (Baseline)', color='#2E86AB', linestyle='-', zorder=5)
plt.plot(x_positions, noisy_10_values, marker='^', linewidth=2.5, markersize=10, 
         label=f'Noisy Model (10% Target, σ={sigma_10:.6f})', color='#FFA726', linestyle='--', zorder=4)

# Customize the plot
plt.xlabel('Precision Type', fontsize=13, fontweight='bold')
plt.ylabel('Perplexity', fontsize=13, fontweight='bold')
plt.title('Clean vs Noisy Models with 10% Noise Level\nQwen2.5-0.5B - WikiText-2 Test Set', 
          fontsize=14, fontweight='bold', pad=20)

# Set x-axis
plt.xticks(x_positions, precision_types, fontsize=12)
plt.yticks(fontsize=11)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add legend
plt.legend(loc='best', fontsize=11, framealpha=0.95)

# Add value annotations
for i, (clean, noisy) in enumerate(zip(clean_values, noisy_10_values)):
    plt.annotate(f'{clean:.2f}', 
                xy=(x_positions[i], clean), 
                xytext=(0, 12), 
                textcoords='offset points',
                ha='center', 
                fontsize=10,
                color='#2E86AB',
                fontweight='bold')
    
    plt.annotate(f'{noisy:.2f}', 
                xy=(x_positions[i], noisy), 
                xytext=(0, -18), 
                textcoords='offset points',
                ha='center', 
                fontsize=10,
                color='#FFA726',
                fontweight='bold')

# Calculate and display degradation percentages
degradation_fp16 = ((noisy_10_values[0] - clean_values[0]) / clean_values[0]) * 100
degradation_int8 = ((noisy_10_values[1] - clean_values[1]) / clean_values[1]) * 100
degradation_4bit = ((noisy_10_values[2] - clean_values[2]) / clean_values[2]) * 100

# Add text box with degradation info
textstr = f'Degradation vs Clean:\nFP16: {degradation_fp16:+.2f}%\nINT8: {degradation_int8:+.2f}%\n4-BIT: {degradation_4bit:+.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Adjust layout
plt.tight_layout()

# Save the plot
output_file = 'noisy_10pct_wikitext2_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()
