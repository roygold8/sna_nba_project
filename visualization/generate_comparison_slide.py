#!/usr/bin/env python3
"""
Generate 1 vs 2 vs 3 comparison slide visual.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'slides')

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('dark_background')

# ============================================
# SLIDE: 1 vs 2 vs 3 Comparison
# ============================================

fig, ax = plt.subplots(figsize=(14, 8))

# Data from the presentation findings
categories = ['THE STAR\n(1 Player)', 'THE DUO\n(2 Players)', 'THE TRIO\n(3 Players)']
correlations = [0.35, 0.42, 0.47]  # Approximate values from findings
examples = ['Luka, Harden', 'Jokić-Murray', 'Warriors Big 3']
colors = ['#FF6B6B', '#FECA57', '#4ECDC4']

# Create horizontal bars
y_pos = np.arange(len(categories))
bars = ax.barh(y_pos, correlations, color=colors, height=0.6, edgecolor='white', linewidth=2)

# Customize
ax.set_xlim(0, 0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=16, fontweight='bold')
ax.set_xlabel('Correlation with Win % (r)', fontsize=14)
ax.invert_yaxis()

# Add correlation values and examples on bars
for i, (bar, corr, example) in enumerate(zip(bars, correlations, examples)):
    # Correlation value
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
            f'r = +{corr:.2f}', va='center', fontsize=14, fontweight='bold', color='white')
    # Example
    ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
            example, va='center', ha='center', fontsize=11, color='black', fontweight='bold')

# Title
ax.set_title('Star vs Duo vs Trio: Which Predicts Success Best?', 
             fontsize=22, fontweight='bold', pad=20, color='white')

# Add takeaway at bottom
ax.text(0.3, -0.5, '📈 MORE connected stars = MORE wins', 
        fontsize=14, ha='center', color='#4ECDC4', fontweight='bold',
        transform=ax.get_yaxis_transform())

# Add "WINNER" indicator
ax.annotate('BEST\nPREDICTOR', xy=(0.47, 2), xytext=(0.55, 2),
            fontsize=10, fontweight='bold', color='#4ECDC4',
            arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=2),
            va='center')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'star_duo_trio_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()


# ============================================
# SLIDE: Build Up Visual (1 → 2 → 3)
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

configs = [
    ('THE STAR', 1, '#FF6B6B', 0.35, 'Max Weighted Degree'),
    ('THE DUO', 2, '#FECA57', 0.42, 'Top 2 Edge Weight'),
    ('THE TRIO', 3, '#4ECDC4', 0.47, 'Top 3 Avg Degree'),
]

for ax, (title, n_players, color, corr, metric) in zip(axes, configs):
    # Draw players as circles
    if n_players == 1:
        positions = [(0.5, 0.5)]
    elif n_players == 2:
        positions = [(0.35, 0.5), (0.65, 0.5)]
    else:
        positions = [(0.5, 0.7), (0.3, 0.35), (0.7, 0.35)]
    
    # Draw connections
    if n_players >= 2:
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=color, linewidth=4, alpha=0.7)
    
    # Draw player nodes
    for pos in positions:
        circle = plt.Circle(pos, 0.08, color=color, zorder=5)
        ax.add_patch(circle)
    
    # Labels
    ax.text(0.5, 0.95, title, fontsize=18, fontweight='bold', 
            ha='center', va='top', color=color)
    ax.text(0.5, 0.08, f'r = +{corr:.2f}', fontsize=16, fontweight='bold',
            ha='center', color='white')
    ax.text(0.5, 0.0, metric, fontsize=10, ha='center', color='#888888')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

# Add arrows between subplots
fig.text(0.355, 0.5, '→', fontsize=30, ha='center', va='center', color='white')
fig.text(0.645, 0.5, '→', fontsize=30, ha='center', va='center', color='white')

plt.suptitle('Building the Championship Core', fontsize=22, fontweight='bold', 
             color='white', y=1.02)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'star_duo_trio_buildup.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()

print("\n✅ Comparison visuals generated!")
