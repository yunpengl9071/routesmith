#!/usr/bin/env python3
"""
Generate comparison figures between baseline and quality-fixed experiments.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load metrics
baseline_file = Path(__file__).parent / 'metrics.json'
quality_file = Path(__file__).parent / 'quality_fixed_metrics.json'

with open(baseline_file) as f:
    baseline = json.load(f)

with open(quality_file) as f:
    quality = json.load(f)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quality-Fixed Reward Function: Comparison with Baseline', fontsize=14, fontweight='bold')

# 1. Cost Comparison
ax1 = axes[0, 0]
categories = ['Total Cost ($)']
baseline_costs = [baseline.get('cumulative_cost_routed', baseline.get('cumulative_cost_baseline', 224))]
quality_costs = [quality['cumulative_cost_routed']]
x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_costs, width, label='Baseline', color='#ff6b6b')
bars2 = ax1.bar(x + width/2, quality_costs, width, label='Quality-Fixed', color='#4ecdc4')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Total Cost Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.bar_label(bars1, fmt='$%.1f')
ax1.bar_label(bars2, fmt='$%.1f')

# Calculate savings
savings = ((baseline_costs[0] - quality_costs[0]) / baseline_costs[0]) * 100
ax1.annotate(f'-{savings:.1f}%', xy=(0, quality_costs[0]), fontsize=12, color='green', fontweight='bold')

# 2. Quality Rate Comparison
ax2 = axes[0, 1]
baseline_quality = baseline.get('avg_quality', 0.56) * 100
quality_rate = quality['acceptable_quality_rate']
categories = ['Quality Rate (%)']
bars1 = ax2.bar(x - width/2, [baseline_quality], width, label='Baseline', color='#ff6b6b')
bars2 = ax2.bar(x + width/2, [quality_rate], width, label='Quality-Fixed', color='#4ecdc4')
ax2.set_ylabel('Quality Rate (%)')
ax2.set_title('Acceptable Quality Rate (≥5/10)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.bar_label(bars1, fmt='%.1f%%')
ax2.bar_label(bars2, fmt='%.1f%%')

improvement = ((quality_rate - baseline_quality) / baseline_quality) * 100
ax2.annotate(f'+{improvement:.1f}%', xy=(0, quality_rate), fontsize=12, color='green', fontweight='bold')

# 3. Routing Distribution
ax3 = axes[1, 0]
tiers = ['Premium', 'Standard', 'Economy']
baseline_dist = [40, 40, 20]  # From baseline metrics
quality_dist = [
    quality['routing_distribution']['premium'],
    quality['routing_distribution']['standard'],
    quality['routing_distribution']['economy']
]

x = np.arange(len(tiers))
bars1 = ax3.bar(x - width/2, baseline_dist, width, label='Baseline', color='#ff6b6b')
bars2 = ax3.bar(x + width/2, quality_dist, width, label='Quality-Fixed', color='#4ecdc4')
ax3.set_ylabel('Queries')
ax3.set_title('Routing Distribution')
ax3.set_xticks(x)
ax3.set_xticklabels(tiers)
ax3.legend()

# 4. New Metrics
ax4 = axes[1, 1]
new_metrics = ['Premium Retry\nRate', 'Cost/Valid\nResponse ($)', 'Failures\nPrevented']
new_values = [
    quality['premium_retry_rate'],
    quality['cost_per_valid_response'],
    quality['failures_prevented']
]
colors = ['#45b7d1', '#96ceb4', '#ffeaa7']
bars = ax4.bar(new_metrics, new_values, color=colors)
ax4.set_ylabel('Value')
ax4.set_title('New Quality-Fixed Metrics')
ax4.bar_label(bars, fmt='%.1f')

# Add summary text
summary_text = f"""
Summary:
• Cost Reduction: {savings:.1f}%
• Quality Improvement: +{improvement:.1f}%
• Acceptable Quality: {quality_rate:.1f}%
• Premium Retries: {quality['failures_prevented']}
"""

plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save figure
output_path = Path(__file__).parent.parent / 'figures_real' / 'real_fig8_quality_fixed_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison figure to: {output_path}")

# Also create a summary stats file
summary = {
    'baseline': {
        'total_cost': baseline_costs[0],
        'quality_rate': baseline_quality,
        'routing': dict(zip(tiers, baseline_dist))
    },
    'quality_fixed': {
        'total_cost': quality_costs[0],
        'quality_rate': quality_rate,
        'routing': quality['routing_distribution'],
        'premium_retry_rate': quality['premium_retry_rate'],
        'cost_per_valid_response': quality['cost_per_valid_response'],
        'failures_prevented': quality['failures_prevented']
    },
    'improvements': {
        'cost_reduction_percent': savings,
        'quality_improvement_percent': improvement
    }
}

summary_file = Path(__file__).parent / 'quality_fixed_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to: {summary_file}")
