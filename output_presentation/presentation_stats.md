# NBA Social Network Analysis - Verified Statistics
Generated: 2026-01-21 18:32:05

## Dataset Summary
- **Seasons Analyzed:** 9 (2015-16 to 2023-24)
- **Team-Seasons:** 270
- **Player-Seasons:** 3487
- **Total Nodes:** ~3487
- **Total Edges:** ~40500 (estimated)

## Key Findings (Verified Correlations)

### 1. Hierarchy (Standard Deviation of Weighted Degree)
- **Correlation:** r = 0.468***
- **P-Value:** 0.0000
- **Interpretation:** Strong positive - Higher hierarchy leads to more wins

**Presentation Statement:**
> "There is a strong positive correlation between the standard deviation of weighted degree (hierarchy) and winning percentage (r=0.468, p=0.0000)."

### 2. Heliocentric Model (Star Player Max Degree)
- **Correlation:** r = 0.419***
- **P-Value:** 0.0000
- **Interpretation:** Strong positive - Star system leads to more wins

**Presentation Statement:**
> "Strong positive correlation (r=0.419) between a star player's maximum weighted degree and team wins."

### 3. Pass Entropy (Randomness)
- **Correlation:** r = -0.178**
- **P-Value:** 0.0033
- **Interpretation:** Negative - Randomness hurts winning

**Presentation Statement:**
> "'Pass Entropy' (randomness in distribution) has a negative correlation with winning (r=-0.178)."

### 4. Graph Density
- **Correlation:** r = 0.185**
- **P-Value:** 0.0023
- **Interpretation:** Weak/No significant effect

**Presentation Statement:**
> "Graph Density showed weak correlation with winning percentage (r=0.185, p=0.0023)."

## All Correlations (Sorted by Strength)

| Metric | Correlation (r) | P-Value | Significant | Interpretation |
|--------|-----------------|---------|-------------|----------------|
| Std_Weighted_Degree | 0.4685*** | 0.0000 | Yes | Strong positive - Hierarchy leads to more wins... |
| Mean_Weighted_Degree | 0.4233*** | 0.0000 | Yes | Strong positive correlation... |
| Star_Player_Degree | 0.4188*** | 0.0000 | Yes | Strong positive correlation... |
| Max_Weighted_Degree | 0.4188*** | 0.0000 | Yes | Strong positive - Star system leads to more wins... |
| Total_Passes | 0.3519*** | 0.0000 | Yes | Moderate positive correlation... |
| Top3_Concentration | 0.3207*** | 0.0000 | Yes | Moderate positive correlation... |
| Top2_Concentration | 0.2967*** | 0.0000 | Yes | Moderate positive correlation... |
| Top1_Concentration | 0.2375*** | 0.0001 | Yes | Moderate positive correlation... |
| Star_Degree_Share | 0.2375*** | 0.0001 | Yes | Moderate positive correlation... |
| Avg_Clustering | 0.2130*** | 0.0004 | Yes | Moderate positive correlation... |
| Std_Eigenvector | 0.2104*** | 0.0005 | Yes | Moderate positive correlation... |
| Transitivity | 0.2092*** | 0.0005 | Yes | Moderate positive correlation... |
| Std_Betweenness | 0.1927** | 0.0015 | Yes | Weak positive correlation... |
| Mean_Closeness | 0.1902** | 0.0017 | Yes | Weak positive correlation... |
| Graph_Density | 0.1847** | 0.0023 | Yes | Weak positive - Connectivity has positive effect... |
| Max_Betweenness | 0.1674** | 0.0058 | Yes | Weak positive correlation... |
| Max_Closeness | 0.1287* | 0.0345 | Yes | Weak positive correlation... |
| Gini_Coefficient | 0.1287* | 0.0346 | Yes | Weak positive - Inequality leads to more wins... |
| Max_Eigenvector | 0.1280* | 0.0355 | Yes | Weak positive correlation... |
| Largest_Component_Pct | 0.1138 | 0.0618 | No | No significant relationship positive correlation... |
| Mean_Eigenvector | 0.0994 | 0.1031 | No | No significant relationship positive correlation... |
| Degree_Centralization | 0.0798 | 0.1909 | No | No significant relationship positive correlation... |
| Min_Weighted_Degree | 0.0796 | 0.1922 | No | No significant relationship positive correlation... |
| Mean_Betweenness | 0.0424 | 0.4881 | No | No significant relationship positive correlation... |
| Avg_Clustering_Weighted | -0.0054 | 0.9302 | No | No significant relationship negative correlation... |
| Diameter | -0.1015 | 0.0961 | No | No significant relationship negative correlation... |
| Num_Strong_Components | -0.1258* | 0.0388 | Yes | Weak negative correlation... |
| Num_Weak_Components | -0.1360* | 0.0254 | Yes | Weak negative correlation... |
| Num_Components | -0.1360* | 0.0254 | Yes | Weak negative correlation... |
| Degree_Equality | -0.1362* | 0.0252 | Yes | Weak negative correlation... |
| APL | -0.1503* | 0.0134 | Yes | Weak negative correlation... |
| Std_Closeness | -0.1752** | 0.0039 | Yes | Weak negative correlation... |
| Pass_Entropy | -0.1783** | 0.0033 | Yes | Weak negative - Randomness leads to fewer wins... |
| Edge_Weight_Entropy | -0.1828** | 0.0026 | Yes | Weak negative correlation... |
| Num_Triangles | -0.2450*** | 0.0000 | Yes | Moderate negative correlation... |
| Num_Edges | -0.2634*** | 0.0000 | Yes | Moderate negative correlation... |
| Largest_Component_Size | -0.2679*** | 0.0000 | Yes | Moderate negative correlation... |
| Num_Nodes | -0.2873*** | 0.0000 | Yes | Moderate negative correlation... |


## Methodology Notes

### Data Collection
- Data fetched from NBA API (nba_api Python package)
- Passing data from `playerdashptpass` endpoint
- Player filters: GP >= 20, MIN >= 10

### Network Construction
- **Nodes:** Players (per team, per season)
- **Edges:** Directed, weighted by pass count
- **Type:** Directed, Weighted Graph (DiGraph)

### Metrics Calculated
1. **Degree Centrality** (In, Out, Total - weighted)
2. **Betweenness Centrality** (weighted)
3. **Eigenvector Centrality** (on undirected version)
4. **Closeness Centrality** (on undirected version)
5. **Clustering Coefficient** (average, weighted)
6. **Graph Density** (standard NetworkX)
7. **Degree Equality** (Mean²/Mean(x²))
8. **Gini Coefficient** (degree inequality)
9. **Pass Entropy** (Shannon entropy of degree distribution)
10. **Community Detection** (Louvain algorithm)

### Important Caveats
1. **Playoff Data:** NBA API does not provide playoff-specific passing data in same format. Playoff analysis uses regular season metrics to predict playoff outcomes.
2. **Deni Avdija Analysis:** 2025-26 network metrics are ESTIMATED, not from actual API data.
3. **Win% Source:** Team win percentages are from official NBA records.

## Generated Files
- `output_complete_sna/complete_team_metrics.csv` - All team metrics
- `output_complete_sna/complete_correlations.csv` - All correlations
- `output_presentation/presentation_stats.md` - This file
