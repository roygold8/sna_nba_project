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
sna_nba_project/
├── data/
│   └── 2023-24/
│       ├── passing_*.json      # Raw NBA passing data
│       ├── playoff_scores.csv  # Team playoff performance
│       ├── scoring_leaders.csv # Top 100 scorers stats
│       └── team_standings.csv  # Team win percentages
├── output/
│   ├── nba_assist_network.gexf # Assist network graph
│   └── nba_pass_network.gexf   # Pass network graph
├── build_network.py            # Build network from raw data
├── build_player_metrics_df.py  # Create player metrics DataFrame
├── fetch_nba_data.py           # Fetch data from NBA API
├── fetch_success_metrics.py    # Fetch team success metrics
├── generate_improved_viz.py    # Generate HTML visualization
└── nba_network_analysis.ipynb  # Main analysis notebook
```

---

## 🚀 Quick Start

### 1. Generate Player Metrics DataFrame
```bash
python build_player_metrics_df.py
```
Creates `player_network_metrics_2023-24.csv` with:
- Network metrics (Degree, Betweenness, Clustering)
- Team success metrics (WinPCT, PlayoffScore)
- Player performance stats (PTS, AST, REB)

### 2. Generate Interactive Visualization
```bash
python generate_improved_viz.py
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

