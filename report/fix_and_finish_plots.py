#!/usr/bin/env python3
"""
Fix and finish generating real experiment plots.
Handles the zero-cost economy tier case.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path.home() / "projects" / "routesmith" / "report" / "figures_real"

# Configure plot style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

print("=" * 60)
print("Finishing Real Experiment Plots")
print("=" * 60)

# Load real experiment data
results_path = Path.home() / "projects" / "routesmith" / "report" / "real_100_queries" / "results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# ============================================================
# FIGURE 5: Cost-Quality Tradeoff (Fixed)
# ============================================================
print("Generating Figure 5: Cost-Quality Tradeoff (Fixed)...")

fig5, ax5 = plt.subplots(figsize=(10, 6))

# Create scatter plot with tier coloring
colors = df['selected_tier'].map({'premium': '#3498db', 'economy': '#e74c3c'})
scatter = ax5.scatter(df['cost_usd'], df['quality_score'], 
                      c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add trend lines for each tier (handling zero variance)
for tier, color in [('premium', '#3498db'), ('economy', '#e74c3c')]:
    tier_data = df[df['selected_tier'] == tier]
    if len(tier_data) > 1:
        x = tier_data['cost_usd'].values
        y = tier_data['quality_score'].values
        
        # Check if we can fit a line (need variance in x)
        if np.var(x) > 1e-10:  # Has variance
            try:
                m, b = np.polyfit(x, y, 1)
                x_range = np.linspace(min(x), max(x), 100)
                ax5.plot(x_range, m * x_range + b, color=color, 
                        linewidth=2, linestyle='--', alpha=0.8,
                        label=f'{tier.capitalize()} trend')
            except:
                pass  # Skip if polyfit fails
        else:
            # No variance in x, just plot horizontal line at mean y
            mean_y = np.mean(y)
            ax5.axhline(y=mean_y, color=color, linewidth=2, 
                       linestyle='--', alpha=0.8,
                       label=f'{tier.capitalize()} mean')

# Calculate Pareto frontier (non-dominated points)
points = df[['cost_usd', 'quality_score']].values
pareto_mask = np.ones(len(points), dtype=bool)
for i, (cost_i, quality_i) in enumerate(points):
    for j, (cost_j, quality_j) in enumerate(points):
        if i != j and cost_j <= cost_i and quality_j >= quality_i:
            pareto_mask[i] = False
            break

pareto_points = df[pareto_mask]
ax5.scatter(pareto_points['cost_usd'], pareto_points['quality_score'], 
           color='gold', s=150, marker='*', edgecolors='black', linewidth=1,
           label='Pareto Frontier', zorder=5)

ax5.set_xlabel('Cost per Query (USD)', fontsize=12)
ax5.set_ylabel('Quality Score', fontsize=12)
ax5.set_title('Real Experiment: Cost-Quality Tradeoff\n(Scatter Plot with Pareto Frontier)', 
              fontsize=13, pad=15)

# Add legend for tiers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Premium Tier'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Economy Tier'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Pareto Frontier')
]
ax5.legend(handles=legend_elements, loc='upper right')

# Add efficiency annotation
avg_quality = df['quality_score'].mean()
avg_cost = df['cost_usd'].mean()
ax5.annotate(f'Avg: ${avg_cost:.4f}, Q={avg_quality:.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig5.savefig(OUTPUT_DIR / "real_fig5_cost_quality_tradeoff.png", dpi=300, bbox_inches='tight')
plt.close(fig5)
print(f"  Saved: {OUTPUT_DIR / 'real_fig5_cost_quality_tradeoff.png'}")

# ============================================================
# FIGURE 6: Token Usage vs Cost
# ============================================================
print("Generating Figure 6: Token Usage Analysis...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Token distribution by tier
tier_data = []
for tier in ['premium', 'economy']:
    tier_tokens = df[df['selected_tier'] == tier]['total_tokens'].values
    tier_data.append(tier_tokens)

bp1 = ax6a.boxplot(tier_data, labels=['Premium', 'Economy'], patch_artist=True,
                  boxprops=dict(alpha=0.8, linewidth=1.2),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(linewidth=1.2))

colors_box = ['#3498db', '#e74c3c']
for patch, color in zip(bp1['boxes'], colors_box):
    patch.set_facecolor(color)

ax6a.set_ylabel('Tokens per Query', fontsize=12)
ax6a.set_title('Token Usage by Tier', fontsize=13)

# Add mean annotation
for i, data in enumerate(tier_data):
    mean_val = np.mean(data)
    ax6a.text(i+1, mean_val, f'μ={mean_val:.1f}', 
              ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Cost vs Tokens scatter
colors_scatter = df['selected_tier'].map({'premium': '#3498db', 'economy': '#e74c3c'})
scatter2 = ax6b.scatter(df['total_tokens'], df['cost_usd'], 
                       c=colors_scatter, s=60, alpha=0.7, edgecolors='black')

ax6b.set_xlabel('Tokens per Query', fontsize=12)
ax6b.set_ylabel('Cost per Query (USD)', fontsize=12)
ax6b.set_title('Cost vs Token Usage', fontsize=13)

# Add correlation annotation (only for premium tier since economy has zero cost)
premium_df = df[df['selected_tier'] == 'premium']
if len(premium_df) > 1:
    correlation = premium_df['total_tokens'].corr(premium_df['cost_usd'])
    ax6b.annotate(f'Premium correlation: r = {correlation:.3f}', 
                  xy=(0.05, 0.95), xycoords='axes fraction',
                  fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    ax6b.annotate('Economy: Zero cost, variable tokens', 
                  xy=(0.05, 0.95), xycoords='axes fraction',
                  fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig6.savefig(OUTPUT_DIR / "real_fig6_token_analysis.png", dpi=300, bbox_inches='tight')
plt.close(fig6)
print(f"  Saved: {OUTPUT_DIR / 'real_fig6_token_analysis.png'}")

# ============================================================
# BONUS: Category-wise Cost Savings
# ============================================================
print("Generating Bonus Figure: Category-wise Analysis...")

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Average cost by category
category_costs = df.groupby('category')['cost_usd'].mean().sort_values()
bars = ax7a.bar(range(len(category_costs)), category_costs.values, 
                color=['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c'],
                alpha=0.8, edgecolor='black')

ax7a.set_xticks(range(len(category_costs)))
ax7a.set_xticklabels([c.capitalize() for c in category_costs.index], rotation=45, ha='right')
ax7a.set_ylabel('Average Cost per Query (USD)', fontsize=12)
ax7a.set_title('Cost by Query Category', fontsize=13)

# Add value labels
for bar, cost in zip(bars, category_costs.values):
    height = bar.get_height()
    ax7a.annotate(f'${cost:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

# Right: Tier selection by category
category_tier = pd.crosstab(df['category'], df['selected_tier'], normalize='index')
category_tier.plot(kind='bar', stacked=True, ax=ax7b,
                   color={'premium': '#3498db', 'economy': '#e74c3c'})

ax7b.set_xlabel('Query Category', fontsize=12)
ax7b.set_ylabel('Proportion of Queries', fontsize=12)
ax7b.set_title('Tier Selection by Category', fontsize=13)
ax7b.legend(title='Tier', loc='upper right')
ax7b.set_xticklabels([c.capitalize() for c in category_tier.index], rotation=45, ha='right')

plt.tight_layout()
fig7.savefig(OUTPUT_DIR / "real_fig7_category_analysis.png", dpi=300, bbox_inches='tight')
plt.close(fig7)
print(f"  Saved: {OUTPUT_DIR / 'real_fig7_category_analysis.png'}")

print("\n" + "=" * 60)
print("SUCCESS: All real experiment plots generated!")
print(f"Total plots: 7")
print("=" * 60)

# List all generated files
print("\nGenerated files:")
for i, f in enumerate(sorted(OUTPUT_DIR.glob("real_fig*.png")), 1):
    print(f"  {i}. {f.name}")