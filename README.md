# NBA Social Network Analysis Project

Analysis of NBA team passing/assist networks to understand how ball movement relates to team success.

## 🏀 The Network Explained

```
NBA ASSIST NETWORK (nba_assist_network.gexf)
├── NODES = Players (465 in 2023-24 season)
├── EDGES = Assist connections (Player A → Player B means A assisted B)
└── EDGE WEIGHT = Number of assists given during the season
```

### Example
If **LeBron James** assisted **Anthony Davis** 50 times during the season:

```
LeBron James ──(50)──► Anthony Davis
```

This creates a **directed edge** from LeBron to Anthony Davis with **weight = 50**.

---

## 📊 Network Metrics

| Metric | Hebrew | Description |
|--------|--------|-------------|
| **Degree** | דרגה | Total number of unique teammates connected with |
| **In_Degree** | - | Number of teammates who assisted to this player |
| **Out_Degree** | - | Number of teammates this player assisted to |
| **Weighted_Degree** | - | Total assists involved (given + received) |
| **Betweenness** | מרכזיות ביניים | How often a player lies on shortest paths between others |
| **Clustering** | מקדם אשכול | How interconnected a player's neighbors are (Connectedness) |
| **Team_Density** | צפיפות רשת | How connected teammates are with each other |

---

## 📁 Project Structure

```
nba_project/
├── data/                       # Raw NBA data & success metrics
├── output/                     # Generated networks & visualizations
├── data_collection/            # Scripts for fetching/loading data
│   ├── fetch_nba_data.py
│   ├── fetch_success_metrics.py
│   └── success_data_loader.py
├── network_construction/       # Scripts for building networks
│   └── build_network.py
├── analysis/                   # Analysis scripts
│   ├── analyze_networks_comprehensive.py
│   ├── analyze_team_success.py
│   ├── build_player_metrics_df.py
│   └── compare_top123.py
├── visualization/              # Visualization scripts
│   ├── visualize_network.py
│   ├── generate_improved_viz.py
│   └── generate_slide_visuals.py
├── notebooks/                  # Interactive analysis
│   └── nba_network_analysis.ipynb
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start

### 1. Generate Player Metrics DataFrame
```bash
```bash
python analysis/build_player_metrics_df.py
```
Creates `player_network_metrics_2023-24.csv` with:
- Network metrics (Degree, Betweenness, Clustering)
- Team success metrics (WinPCT, PlayoffScore)
- Player performance stats (PTS, AST, REB)

### 2. Generate Interactive Visualization
```bash
```bash
python visualization/generate_improved_viz.py
```
Creates `assist_network_2023-24_improved.html` - open in browser to explore.

---

## 📈 DataFrame Columns

### Network Metrics
| Column | Description |
|--------|-------------|
| `Player_Name` | שם שחקן |
| `Team` | קבוצה (קיצור) |
| `Season` | עונה |
| `Degree` | דרגה - מספר קשרים |
| `Betweenness` | מרכזיות ביניים |
| `Clustering` | Connectedness |
| `Team_Density` | צפיפות רשת הקבוצה |
| `Assists_Given` | אסיסטים שנתן |
| `Assists_Received` | אסיסטים שקיבל |

### Team Success Metrics
| Column | Description |
|--------|-------------|
| `Team_WinPCT` | אחוז ניצחונות |
| `Team_PlayoffRank` | דירוג פלייאוף |
| `Team_PlayoffWins` | ניצחונות בפלייאוף |
| `Team_PlayoffScore` | ציון הצלחה (0-6) |

### Player Performance (Top 100 scorers only)
| Column | Description |
|--------|-------------|
| `PTS` | נקודות למשחק |
| `AST` | אסיסטים למשחק |
| `REB` | ריבאונדים למשחק |
| `GP` | משחקים ששיחק |
| `EFF` | יעילות |

---

## ⚠️ Notes

- **Performance stats missing for some players**: The `scoring_leaders.csv` contains only TOP 100 scorers. Players outside top 100 will have NaN for PTS, AST, REB, etc.
- **Edge weights stored as strings in GEXF**: When reading the network, convert weights to float using `float(data.get('weight', 0))`.

---

## 🎯 Research Goal

Analyze whether teams with stronger assist networks (higher density, more balanced distribution) achieve more success (higher win percentage, deeper playoff runs).

---

## 📚 Data Sources

- **NBA Stats API** - Player passing and performance data
- **Seasons**: 2014-15 to 2024-25

