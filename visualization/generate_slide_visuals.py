#!/usr/bin/env python3
"""
Generate visualizations for presentation slides.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'slides')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set dark style for presentation
plt.style.use('dark_background')

# ============================================
# SLIDE 3: Regular Season vs Playoffs Comparison
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Regular Season factors
reg_factors = ['Star Duo\nConcentration', 'Top 3 Player\nDegree', 'Total Volume']
reg_correlations = [0.42, 0.47, 0.38]
reg_colors = ['#4ECDC4', '#4ECDC4', '#4ECDC4']

# Playoff factors
playoff_factors = ['Network\nDensity', 'Clustering', 'Equal\nDistribution']
playoff_correlations = [0.46, 0.38, 0.39]
playoff_colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B']

# Left plot: Regular Season
ax1 = axes[0]
bars1 = ax1.barh(reg_factors, reg_correlations, color=reg_colors, height=0.6)
ax1.set_xlim(0, 0.6)
ax1.set_xlabel('Correlation (r)', fontsize=12)
ax1.set_title('Regular Season Success', fontsize=16, fontweight='bold', color='#4ECDC4')
ax1.axvline(x=0.3, color='white', linestyle='--', alpha=0.3, label='Moderate threshold')

# Add correlation values on bars
for bar, corr in zip(bars1, reg_correlations):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'r = +{corr:.2f}', va='center', fontsize=11, fontweight='bold')

# Right plot: Playoff Success
ax2 = axes[1]
bars2 = ax2.barh(playoff_factors, playoff_correlations, color=playoff_colors, height=0.6)
ax2.set_xlim(0, 0.6)
ax2.set_xlabel('Correlation (r)', fontsize=12)
ax2.set_title('Playoff Success', fontsize=16, fontweight='bold', color='#FF6B6B')
ax2.axvline(x=0.3, color='white', linestyle='--', alpha=0.3)

# Add correlation values on bars
for bar, corr in zip(bars2, playoff_correlations):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'r = +{corr:.2f}', va='center', fontsize=11, fontweight='bold')

plt.suptitle('What Predicts NBA Team Success?', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'regular_vs_playoff.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', 
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()


# ============================================
# SLIDE: The Playoff Shift (Arrow Diagram)
# ============================================

fig, ax = plt.subplots(figsize=(12, 6))

# Two columns of text boxes
reg_x, playoff_x = 0.2, 0.8
center_x = 0.5

# Title
ax.text(0.5, 0.95, 'The Playoff Shift', fontsize=24, fontweight='bold', 
        ha='center', va='top', color='white')

# Regular Season column
ax.text(reg_x, 0.75, 'REGULAR SEASON', fontsize=14, fontweight='bold', 
        ha='center', color='#4ECDC4')
ax.text(reg_x, 0.60, '• Star Duo (+0.42)', fontsize=12, ha='center', color='white')
ax.text(reg_x, 0.50, '• Top 3 Degree (+0.47)', fontsize=12, ha='center', color='white')
ax.text(reg_x, 0.40, '• Volume (+0.38)', fontsize=12, ha='center', color='white')
ax.text(reg_x, 0.25, '"Let stars cook"', fontsize=11, ha='center', color='#888888', style='italic')

# Arrow
ax.annotate('', xy=(0.65, 0.5), xytext=(0.35, 0.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=3))
ax.text(0.5, 0.55, 'SHIFT', fontsize=10, ha='center', color='#888888')

# Playoff column
ax.text(playoff_x, 0.75, 'PLAYOFFS', fontsize=14, fontweight='bold', 
        ha='center', color='#FF6B6B')
ax.text(playoff_x, 0.60, '• Density (+0.46)', fontsize=12, ha='center', color='white')
ax.text(playoff_x, 0.50, '• Clustering (+0.38)', fontsize=12, ha='center', color='white')
ax.text(playoff_x, 0.40, '• Equal Dist. (+0.39)', fontsize=12, ha='center', color='white')
ax.text(playoff_x, 0.25, '"Depth wins rings"', fontsize=11, ha='center', color='#888888', style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

output_path = os.path.join(OUTPUT_DIR, 'playoff_shift.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()


# ============================================
# SLIDE: K-Core Concept Diagram
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))

# Draw concentric circles representing k-core levels
circles = [
    (0.5, 0.5, 0.45, '#333344', 'Periphery (Bench)'),
    (0.5, 0.5, 0.30, '#444466', 'Rotation (K=2)'),
    (0.5, 0.5, 0.15, '#FF6B6B', 'THE TRIO (K=3+)'),
]

for x, y, r, color, label in circles:
    circle = plt.Circle((x, y), r, color=color, alpha=0.7)
    ax.add_patch(circle)

# Add player nodes in the core
core_positions = [(0.5, 0.6), (0.4, 0.45), (0.6, 0.45)]
for pos in core_positions:
    circle = plt.Circle(pos, 0.04, color='#4ECDC4', zorder=5)
    ax.add_patch(circle)

# Draw triangle connecting core players
triangle = plt.Polygon(core_positions, fill=False, edgecolor='#4ECDC4', 
                        linewidth=2, linestyle='--', zorder=4)
ax.add_patch(triangle)

# Labels
ax.text(0.5, 0.95, 'The K-Core Structure', fontsize=20, fontweight='bold', 
        ha='center', color='white')

ax.text(0.5, 0.52, 'K-CORE\nTRIO', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white')

ax.text(0.5, 0.1, 'High degree core = Championship potential', 
        fontsize=12, ha='center', color='#888888', style='italic')

# Legend
ax.text(0.05, 0.85, '● Core Trio (K=3+)', fontsize=10, color='#FF6B6B')
ax.text(0.05, 0.80, '● Rotation Players', fontsize=10, color='#444466')
ax.text(0.05, 0.75, '● Bench/Periphery', fontsize=10, color='#333344')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

output_path = os.path.join(OUTPUT_DIR, 'kcore_diagram.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()


# ============================================
# SLIDE: Championship Formula
# ============================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.text(0.5, 0.90, 'The Championship Formula', fontsize=24, fontweight='bold', 
        ha='center', color='white')

# Formula box
formula_text = """
    REGULAR SEASON  =  Star Power  +  Volume
                       (Duo)         (Assists)

         ↓ PLAYOFF SHIFT ↓

    CHAMPIONSHIP  =  Density  +  Clustering  +  Depth
                     (Everyone)  (Triangles)    (Equal)
"""

ax.text(0.5, 0.55, formula_text, fontsize=14, ha='center', va='center', 
        color='#4ECDC4', family='monospace')

# Bridge
ax.text(0.5, 0.15, 'THE BRIDGE: K-Core Trio', fontsize=16, fontweight='bold',
        ha='center', color='#FF6B6B')
ax.text(0.5, 0.08, 'Strong Duo + Integrated 3rd Option', fontsize=12,
        ha='center', color='#888888')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

output_path = os.path.join(OUTPUT_DIR, 'championship_formula.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f"✓ Saved: {output_path}")
plt.close()

print(f"\\n✅ All slide visuals generated in {OUTPUT_DIR}")
