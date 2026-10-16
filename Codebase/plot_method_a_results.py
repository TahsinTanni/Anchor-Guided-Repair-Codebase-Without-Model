"""
Plot Method A Defense Results
Current parameters: lambda=5e-5, steps=120, lr=2e-5
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

NOISE_LEVELS = [5, 10, 15, 20]

for noise in NOISE_LEVELS:
    results_file = f'defended_adapted_{noise}pct_results.json'
    
    # Skip if results file doesn't exist
    if not os.path.exists(results_file):
        print(f"⚠ Skipping {noise}% - {results_file} not found")
        continue
    
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data
    precision_levels = ['FP16', 'INT8', '4-BIT']
    clean_ppls = [
        results['precision_levels']['fp16']['ppl_clean_adapted'],
        results['precision_levels']['int8']['ppl_clean_adapted'],
        results['precision_levels']['4bit']['ppl_clean_adapted']
    ]
    attacked_ppls = [
        results['precision_levels']['fp16']['ppl_attacked'],
        results['precision_levels']['int8']['ppl_attacked'],
        results['precision_levels']['4bit']['ppl_attacked']
    ]
    defended_ppls = [
        results['precision_levels']['fp16']['ppl_defended'],
        results['precision_levels']['int8']['ppl_defended'],
        results['precision_levels']['4bit']['ppl_defended']
    ]
    recovery_rates = [
        results['precision_levels']['fp16']['recovery_pct'],
        results['precision_levels']['int8']['recovery_pct'],
        results['precision_levels']['4bit']['recovery_pct']
    ]

    # Load defense stats (to show total loss formula values)
    defended_dir = f'saved_models/wikitext2_defended_adapted_fp16_{noise}pct'
    stats_path = os.path.join(defended_dir, 'defense_stats.json')
    defense_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as sf:
            defense_stats = json.load(sf)

    lambda_anchor = None
    ce_mean = None
    anchor_mean = None
    total_mean = None
    if defense_stats:
        lambda_anchor = defense_stats.get('lambda_anchor')
        comps = defense_stats.get('loss_components') or {}
        ce_mean = comps.get('ce_loss_mean')
        anchor_mean = comps.get('anchor_loss_mean')
        total_mean = comps.get('total_loss_mean')
        if total_mean is None:
            total_mean = defense_stats.get('final_train_loss')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Perplexity comparison
    x = np.arange(len(precision_levels))
    ax1.plot(x, clean_ppls, 'o-', color='#2ecc71', linewidth=2, markersize=10, label='Clean (Task-Adapted)')
    ax1.plot(x, attacked_ppls, 's-', color='#e74c3c', linewidth=2, markersize=10, label=f'Attacked ({noise}% Noise)')
    ax1.plot(x, defended_ppls, 'd-', color='#3498db', linewidth=2, markersize=10, label='Defended (Method A)')

    # Add value labels
    for i in range(len(precision_levels)):
        ax1.text(i, clean_ppls[i], f'{clean_ppls[i]:.2f}', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.7, edgecolor='none'),
                 fontsize=9, color='white', fontweight='bold')
        ax1.text(i, attacked_ppls[i], f'{attacked_ppls[i]:.2f}', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.7, edgecolor='none'),
                 fontsize=9, color='white', fontweight='bold')
        ax1.text(i, defended_ppls[i], f'{defended_ppls[i]:.2f}', ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.7, edgecolor='none'),
                 fontsize=9, color='white', fontweight='bold')

    ax1.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Method A: Task-Adapted Clean Baseline Defense\n{noise}% Noise Attack',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(precision_levels)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Recovery rates
    colors = ['#9b59b6', '#9b59b6', '#9b59b6']
    bars = ax2.bar(precision_levels, recovery_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, rate in zip(bars, recovery_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recovery Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Defense Recovery Rate', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim([0, 100])
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Recovery (Upper Bound)')
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Target: 50-80%')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Show total-loss formula values (if available)
    # total_loss = ce_loss + lambda_anchor * anchor_loss
    loss_lines = [
        'Total loss components:',
        'total = ce + λ * anchor'
    ]
    if lambda_anchor is not None:
        loss_lines.append(f"λ_anchor = {lambda_anchor}")
    if ce_mean is not None:
        loss_lines.append(f"ce_loss(mean) = {ce_mean:.6f}")
    if anchor_mean is not None:
        loss_lines.append(f"anchor_loss(mean) = {anchor_mean:.6f}")
    if total_mean is not None:
        loss_lines.append(f"total_loss(mean) = {total_mean:.6f}")
    if not defense_stats:
        loss_lines.append('(defense_stats.json not found)')

    fig.text(0.98, 0.02, "\n".join(loss_lines), ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    out_file = f'method_a_defense_results_{noise}pct.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {out_file}")
    plt.show()
