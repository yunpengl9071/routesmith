"""
Comparison figure: RouteSmith vs RouteLLM
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['RouteSmith\n(Thompson Sampling)', 'RouteLLM SW\n(Our Impl)', 'RouteLLM\nRandom', 'Static\n(All Strong)']
savings = [67.2, 54.6, 46.8, 0]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(methods, savings, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, savings):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Cost Savings (%)', fontsize=12)
ax.set_title('RouteSmith vs RouteLLM: Cost Reduction Comparison\n(Benchmark: 53 Diverse Queries)', fontsize=14)
ax.set_ylim(0, 80)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

# Add legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/yliulupo/projects/routesmith/report/figures/comparison_routellm.png', dpi=150, bbox_inches='tight')
print('Saved comparison figure')
