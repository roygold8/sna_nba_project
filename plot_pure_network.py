"""
Generate Pure Network Rankings Visualization
No Win% - Only Network Metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# Load data
df = pd.read_csv('output_2025_26_prediction/championship_prediction_2025_26_final.csv')

# Calculate pure network score
df['Pure_Network_Score'] = 0.25*df['Hierarchy'] + 0.25*df['Star_Power'] + 0.25*df['Core_Concentration'] + 0.25*df['Order']

# Get top 10 with actual profiles (score > 50)
df_top = df[df['Pure_Network_Score'] > 50].sort_values('Pure_Network_Score', ascending=False).head(10)

# Create figure
fig = plt.figure(figsize=(20, 14))

# ===== 1. Bar Chart - Top 10 by Network Score =====
ax1 = fig.add_subplot(2, 2, 1)
top_10 = df_top.sort_values('Pure_Network_Score', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.95, len(top_10)))[::-1]
bars = ax1.barh(top_10['Team'], top_10['Pure_Network_Score'], color=colors, edgecolor='black', linewidth=1.5)

for bar, (_, row) in zip(bars, top_10.iterrows()):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{row["Pure_Network_Score"]:.1f}', va='center', fontsize=14, fontweight='bold')

ax1.set_xlabel('Network Structure Score', fontsize=16, fontweight='bold')
ax1.set_title('TOP 10 TEAMS BY NETWORK METRICS\n(Pure SNA - No Win%)', fontsize=18, fontweight='bold')
ax1.set_xlim(0, 105)
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3, axis='x')

# ===== 2. Grouped Bar - Metric Breakdown =====
ax2 = fig.add_subplot(2, 2, 2)
top_5 = df_top.head(5)
x = np.arange(len(top_5))
width = 0.2

metrics = ['Hierarchy', 'Star_Power', 'Core_Concentration', 'Order']
colors_m = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
labels_m = ['Hierarchy\n(Std Degree)', 'Star Power\n(Max Degree)', 'Core Conc.\n(Top 3-4)', 'Order\n(Low Entropy)']

for i, (metric, color, label) in enumerate(zip(metrics, colors_m, labels_m)):
    values = top_5[metric].values
    ax2.bar(x + i*width, values, width, label=label, color=color, edgecolor='black', alpha=0.85)

ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(top_5['Team'], fontsize=14, fontweight='bold')
ax2.set_ylabel('Metric Score (0-100)', fontsize=14, fontweight='bold')
ax2.set_title('TOP 5: Network Metric Breakdown', fontsize=18, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.set_ylim(0, 110)
ax2.tick_params(axis='y', labelsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# ===== 3. Scatter - Hierarchy vs Star Power =====
ax3 = fig.add_subplot(2, 2, 3)
scatter = ax3.scatter(df_top['Hierarchy'], df_top['Star_Power'], 
                      s=df_top['Pure_Network_Score']*5, c=df_top['Core_Concentration'],
                      cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidths=2)

for _, row in df_top.iterrows():
    ax3.annotate(row['Team'], (row['Hierarchy'], row['Star_Power']),
                fontsize=14, fontweight='bold', xytext=(5, 5), textcoords='offset points')

# Draw elite zone
ax3.axhline(y=85, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Elite Star Power')
ax3.axvline(x=85, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Elite Hierarchy')

ax3.set_xlabel('Hierarchy (Std of Weighted Degree)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Star Power (Max Weighted Degree)', fontsize=14, fontweight='bold')
ax3.set_title('HELIOCENTRIC MODEL\nHierarchy vs Star Power', fontsize=18, fontweight='bold')
ax3.legend(loc='lower right', fontsize=11)
ax3.tick_params(labelsize=12)

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Core Concentration', fontsize=12)

# ===== 4. Radar Chart - Top 5 =====
ax4 = fig.add_subplot(2, 2, 4, polar=True)
top_5 = df_top.head(5)
metrics = ['Hierarchy', 'Star_Power', 'Core_Concentration', 'Order']
labels = ['Hierarchy\n(Std Degree)', 'Star Power\n(Max Degree)', 'Core Focus\n(Top 3-4)', 'Order\n(Low Entropy)']

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

colors_r = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

for i, (_, row) in enumerate(top_5.iterrows()):
    values = [row[m] for m in metrics]
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=3, markersize=10, color=colors_r[i], label=row['Team'])
    ax4.fill(angles, values, alpha=0.15, color=colors_r[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(labels, fontsize=14, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.tick_params(axis='y', labelsize=12)
ax4.set_title('TOP 5 NETWORK PROFILES', fontsize=18, fontweight='bold', pad=25)
ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=14, title='Teams', title_fontsize=16)

plt.suptitle('2025-26 NBA CHAMPIONSHIP PREDICTION\nPure Network Structure Analysis (No Win%)', 
             fontsize=22, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output_2025_26_prediction/03_pure_network_rankings.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] Saved: output_2025_26_prediction/03_pure_network_rankings.png')
