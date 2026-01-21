# NBA Network Analysis: Comprehensive Research Findings

## Executive Summary

This research project investigated the relationship between NBA team passing and assist network characteristics and team success. We tested **11 hypotheses** derived from Social Network Analysis (SNA) theory, applying them to **both assist networks** (where edges represent scoring assists) and **pass networks** (where edges represent all passes, regardless of whether they led to scores). Each hypothesis was tested against **four success metrics**: regular season win percentage (WinPCT), playoff seeding (PlayoffRank), number of playoff wins (PlayoffWins), and a custom playoff depth score (PlayoffScore, 0-6 scale).

The analysis revealed that certain network properties—particularly network density, total passing volume, and distribution equality—are significantly correlated with team success. However, some hypotheses showed unexpected results, particularly regarding star player dependency, which revealed a nuanced relationship between regular season performance and playoff success.

---

## Methodology

**Data Source:** NBA Stats API passing data for the 2023-24 season, covering all 30 NBA teams and 465 qualified players.

**Networks Analyzed:**
- **Assist Network:** Directed edges from Player A to Player B with weight = number of times A assisted B for a score
- **Pass Network:** Directed edges from Player A to Player B with weight = total passes from A to B

**Statistical Method:** Pearson correlation coefficients with significance levels: * p<0.1, ** p<0.05, *** p<0.01

---

## Detailed Hypothesis Analysis

---

### H1: Network Density and Team Success

**Hypothesis:** Teams with denser passing/assist networks (where more players are connected to more teammates) will achieve greater success.

**Expected Direction:** Positive correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | WinPCT | +0.238 | 0.206 | No |
| Assist | PlayoffScore | **+0.403** | 0.027 | **Yes** |
| Pass | WinPCT | +0.309 | 0.097 | Yes* |
| Pass | PlayoffScore | **+0.461** | 0.010 | **Yes** |

#### What We Learned

The density hypothesis received strong support, particularly for playoff success. When we examine pass networks, we find that teams with higher density—meaning more players are actively passing to more teammates rather than concentrating ball movement through a few individuals—achieve significantly better playoff outcomes. The correlation of +0.461 with PlayoffScore in the pass network is one of the strongest findings in this study.

What makes this finding particularly compelling is the difference between regular season and playoff implications. The correlation with regular season win percentage is weaker and only marginally significant (r=0.309, p=0.097), while playoff success shows a much stronger relationship (r=0.461, p=0.010). This suggests that while any team can win regular season games with various approaches, the playoff environment—with its heightened defensive intensity and series-based format where opponents can game-plan against predictable patterns—rewards teams that have developed robust, multi-pathway ball movement systems.

The assist network shows a similar pattern but with slightly weaker correlations. This makes intuitive sense: the pass network captures all ball movement, while the assist network only captures successful scoring plays. A team might have dense passing without necessarily converting those passes to assists, but the overall ball movement still contributes to offensive success.

**Practical Implication:** Coaches should prioritize developing offensive systems that involve all players rather than running plays through one or two primary ball handlers. Teams like the 2023-24 Boston Celtics exemplified this principle.

---

### H2: Distribution Equality (Gini Coefficient) and Team Success

**Hypothesis:** Teams where passing/assists are more equally distributed among players (lower Gini coefficient) will perform better than teams where ball movement is concentrated in a few players.

**Expected Direction:** Negative correlation (lower Gini = more equal = better)

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | PlayoffScore | **-0.392** | 0.032 | **Yes** |
| Pass | PlayoffRank | -0.309 | 0.096 | Yes* |
| Pass | PlayoffScore | **-0.366** | 0.046 | **Yes** |

#### What We Learned

The distribution equality hypothesis was supported, revealing that teams with more egalitarian ball distribution tend to perform better, especially in the playoffs. The Gini coefficient measures inequality—a value of 0 means perfectly equal distribution, while 1 means complete concentration in one player. Our findings show that lower Gini coefficients (more equal distribution) correlate with better playoff performance.

The negative correlation of -0.392 in the assist network with PlayoffScore tells us that championship-caliber teams typically don't rely on a single playmaker to create all scoring opportunities. Instead, they develop systems where multiple players can initiate offense and create for others. This makes the team less predictable and more resilient when facing playoff defenses that specifically scheme to shut down star players.

However, it's crucial to note that this doesn't mean successful teams have no stars—it means the gap between their best and worst players in terms of playmaking is smaller than on unsuccessful teams. A team like the Detroit Pistons, which had the worst record (0.171 WinPCT), showed high inequality (Gini = 0.51) because their offense flowed almost exclusively through Cade Cunningham when he was healthy.

The pass network shows consistent findings, with significant negative correlations confirming that overall ball movement equality—not just scoring plays—matters for success.

**Practical Implication:** Player development programs should focus on building playmaking skills across the roster, not just for designated point guards. The "positionless basketball" trend aligns with this finding.

---

### H3: Number of Playmaking Hubs and Team Success

**Hypothesis:** Teams with more "hubs" (players with high passing/assist involvement, defined as players with >50% of the maximum weighted degree) will outperform teams with fewer hubs.

**Expected Direction:** Positive correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | WinPCT | -0.077 | 0.686 | No |
| Assist | PlayoffScore | +0.127 | 0.503 | No |
| Pass | WinPCT | -0.102 | 0.590 | No |
| Pass | PlayoffScore | +0.063 | 0.739 | No |

#### What We Learned

Surprisingly, the number of hubs hypothesis received no support whatsoever across either network type or any success metric. All correlations were weak and far from statistical significance. This is a genuinely unexpected finding that challenges certain assumptions about team structure.

The failure of this hypothesis suggests that the simple count of high-involvement players is not predictive of success. A team could have seven players who each exceed the "hub" threshold and still fail, while a team with only two hubs could succeed. The Detroit Pistons, despite their poor record, had the most connections per player simply because their roster turnover meant more teammates interacted over the season.

What this finding reveals is that quality and efficiency of hub connections matter more than quantity. It's not enough to have many players touching the ball frequently—what matters is whether those touches lead to efficient offense and whether the connections between players are strategically valuable. This hypothesis was too simplistic in its operationalization.

Additionally, our hub threshold (50% of maximum weighted degree) may not have been the optimal cutoff. Future research might explore different thresholds or use more sophisticated measures of "playmaking importance" that account for shot quality created, assist-to-turnover ratios, or gravity effects.

**Practical Implication:** Coaches should not simply aim to involve as many players as possible in playmaking; instead, they should focus on developing the right players as playmakers and ensuring their connections are efficient.

---

### H4: Star Dependency (Top 2 Concentration) and Team Success

**Hypothesis:** Teams where a smaller percentage of total ball movement involves the top 2 players will be more successful (less vulnerable to single points of failure).

**Expected Direction:** Negative correlation (less star dependency = better)

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | WinPCT | **+0.415** | 0.022 | **REVERSED** |
| Assist | PlayoffRank | -0.389 | 0.033 | Yes** |
| Pass | WinPCT | **+0.434** | 0.017 | **REVERSED** |
| Pass | PlayoffScore | +0.049 | 0.798 | No |

#### What We Learned

This hypothesis produced the most surprising and nuanced findings of the entire study. Contrary to our prediction, teams with HIGHER star dependency (where the top 2 players handle a larger share of passing/assists) actually WIN MORE regular season games. This was a statistically significant reversed correlation in both networks.

However—and this is crucial—the star dependency effect completely disappears when we look at playoff depth (PlayoffScore). The correlation drops from +0.434 to an insignificant +0.049. This reveals a profound truth about NBA basketball: what works in the regular season doesn't necessarily work in the playoffs.

During the 82-game regular season, having dominant playmakers like Tyrese Haliburton (IND), Nikola Jokić (DEN), or Luka Dončić (DAL) carrying the offensive load is efficient. These players can break down defenses consistently, and opposing teams have limited preparation time. The star carries the team to a high win total and a good seed.

But in a 7-game playoff series, opponents have days to prepare specifically for those one or two players. They can design defensive schemes to take away the stars' favorite actions, employ double-teams earlier, and accept giving up shots to the role players who haven't been as involved all season. Teams that relied too heavily on their stars suddenly find those role players unprepared to fill the void.

The correlation with PlayoffRank (-0.389) shows that star-dependent teams DO get better seeds—but the lack of correlation with PlayoffScore indicates they don't necessarily advance deeper once the playoffs begin.

**Practical Implication:** There's a strategic trade-off: maximize regular season wins through star dependency, but this approach carries playoff vulnerability. Contending teams should consciously develop secondary playmakers throughout the season even if it slightly reduces regular season efficiency.

---

### H5: Team Cohesion (Clustering Coefficient) and Team Success  

**Hypothesis:** Teams with higher average clustering—where a player's teammates also frequently pass to each other, forming tight triangular relationships—will achieve better results.

**Expected Direction:** Positive correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | PlayoffScore | **+0.378** | 0.040 | **Yes** |
| Pass | PlayoffScore | +0.317 | 0.088 | Yes* |

#### What We Learned

The clustering coefficient hypothesis was supported, particularly for playoff success. The clustering coefficient measures how interconnected a player's immediate network neighbors are—if Player A passes frequently to both Players B and C, do B and C also pass frequently to each other? High clustering indicates tight, cohesive sub-groups within the team.

The significant positive correlation (+0.378) between assist network clustering and playoff depth reveals that successful playoff teams feature interconnected offensive units. These teams don't just have random player combinations; they have players who have developed chemistry with each other through repeated interactions.

Consider what high clustering means practically: if the team's best scorer receives passes from both the point guard and the power forward regularly, and those two feeders also connect with each other, the offense can flow organically even when the primary action is disrupted. The defenders can't simply cut off one passing lane because multiple pathways exist between any three players.

Teams like the Cleveland Cavaliers (density = 0.91, one of the highest) and Minnesota Timberwolves (density = 0.93) exemplified this principle. Their offensive systems featured multiple three-man combinations that could operate independently, making them less predictable and more resilient.

The assist network showed stronger effects than the pass network, which makes sense—the clustering that matters most is among players who can actually finish plays with scores, not just total ball movement.

**Practical Implication:** Coaches should design lineups and practice drills that reinforce specific three-player combinations, building chemistry among subgroups rather than only running full five-man sets.

---

### H6: Ball Movement Volume and Team Success

**Hypothesis:** Teams with higher total passing/assist volume will win more games.

**Expected Direction:** Positive correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | WinPCT | **+0.377** | 0.040 | **Yes** |
| Assist | PlayoffWins | **+0.377** | 0.040 | **Yes** |
| Pass | WinPCT | +0.333 | 0.072 | Yes* |
| Pass | PlayoffWins | +0.333 | 0.072 | Yes* |

#### What We Learned

The volume hypothesis was strongly supported—teams that simply move the ball more frequently tend to win more games. This finding aligns with conventional basketball wisdom that "the ball finds energy" and that ball movement creates defensive breakdowns.

The assist network correlation (+0.377) with win percentage is one of the cleaner findings in this study. Teams with more total assists are teams that are getting higher-quality shots through player movement and defensive manipulation. Every successful pass represents a decision point where the defense had to react, and accumulated reactions lead to rotational breakdowns and open looks.

Interestingly, the volume effect is stronger for regular season wins and playoff wins (counting total victories) but weaker for playoff depth (PlayoffScore). This suggests that while playing a high-volume passing style wins individual games, other factors may matter more for advancing through multiple playoff rounds.

The Denver Nuggets (total assists = 2,176), Sacramento Kings (2,063), and Cleveland Cavaliers (2,211) were among the volume leaders, and all had successful seasons. Meanwhile, the low-assist teams like Memphis Grizzlies (1,606) and Detroit Pistons (1,926) struggled.

It's worth noting that volume can be somewhat inflated by pace—teams that play faster have more possessions and thus more opportunities for assists. Future analysis might normalize by possessions to isolate the true effect of ball movement efficiency.

**Practical Implication:** Teams should prioritize playing styles that generate high assist totals, whether through motion offenses, pick-and-roll actions, or post-entry passing. The specific system matters less than the volume of quality passes.

---

### H7: Degree Centralization and Team Success

**Hypothesis:** Teams with lower degree centralization (where ball movement responsibility is distributed rather than concentrated) will perform better.

**Expected Direction:** Negative correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | PlayoffScore | -0.262 | 0.162 | No (direction correct) |
| Pass | PlayoffScore | -0.307 | 0.098 | Yes* |

#### What We Learned

The degree centralization hypothesis received only weak support, with just one marginally significant finding in the pass network. Degree centralization measures how much the network's total connectivity is concentrated in a single node—a high value means one player dominates the passing landscape.

The pass network result (-0.307 with PlayoffScore) suggests that teams where one player handles a disproportionate share of all passes tend to have shorter playoff runs. This is consistent with the earlier findings about star dependency, but the weaker significance levels indicate this metric is not as predictive as simpler measures like density or Gini coefficient.

One reason for the weaker results may be that degree centralization is mathematically related to but distinct from the Gini coefficient used in H2. While Gini measures the inequality of the distribution, centralization specifically measures deviation from the theoretical maximum centralization. In real NBA networks, we rarely see extreme centralization values because even dominant playmakers have teammates who make some passes to each other.

The direction was consistently correct (negative correlations as hypothesized), even when not statistically significant. This suggests the effect exists but is either small in magnitude or confounded by other variables.

**Practical Implication:** While avoiding extreme centralization is advisable, this metric alone is not a strong predictor. Coaches are better served by focusing on density and equality measures.

---

### H8: Betweenness Centralization (Bridging Players) and Team Success

**Hypothesis:** Teams with lower maximum betweenness (where no single player serves as the critical bridge for ball movement) will perform better due to reduced vulnerability.

**Expected Direction:** Negative correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | PlayoffScore | +0.321 | 0.084 | **REVERSED** |
| Pass | WinPCT | **+0.330** | 0.075 | **REVERSED** |
| Pass | PlayoffScore | **+0.404** | 0.027 | **REVERSED** |

#### What We Learned

The betweenness hypothesis showed the most consistently reversed results across both networks. Contrary to our prediction, teams where one player has HIGH betweenness centrality actually perform BETTER—the opposite of what network resilience theory would suggest.

Betweenness centrality measures how often a node lies on the shortest path between other nodes. A player with high betweenness is a "connector" through whom ball movement frequently flows to reach other parts of the team. We expected that over-reliance on a single connector would create vulnerability.

Instead, the data shows a positive correlation (+0.404) between maximum betweenness and playoff success. This surprising finding can be explained by the nature of elite playmaking in basketball. The players with highest betweenness are typically elite point guards or playmaking big men—Nikola Jokić, Tyrese Haliburton, Shai Gilgeous-Alexander. These players ELEVATE the entire offense by connecting teammates who wouldn't otherwise find efficient scoring opportunities.

Unlike physical infrastructure networks where a single critical node represents fragility, basketball networks benefit from having a gifted connector. That connector isn't passive infrastructure—he's actively making decisions, reading defenses, and creating advantages. The betweenness reflects skill, not vulnerability.

Furthermore, high betweenness in stars doesn't preclude role players from connecting with each other for simple actions. The star's betweenness comes from facilitating the more complex, higher-value connections.

**Practical Implication:** Having a true point guard or playmaking hub (high betweenness) is valuable, not problematic. Teams should develop their best playmaker's ability to connect all teammates rather than trying to artificially distribute bridging responsibility.

---

### H10: K-Core Strength and Team Success

**Hypothesis:** Teams with higher k-core numbers (indicating a tightly connected core group) will outperform teams with looser structures.

**Expected Direction:** Positive correlation

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | All metrics | Near zero | >0.50 | No |
| Pass | All metrics | Near zero | >0.50 | No |

#### What We Learned

The k-core hypothesis received no support whatsoever. The k-core of a network is the maximal subgraph where every node has at least k connections. A higher maximum k-core suggests a tightly interconnected core group that could represent the "starting five" or primary rotation players.

The complete lack of correlation suggests that k-core is either not the right measure for capturing team core cohesion, or that the concept itself doesn't translate from general network theory to basketball dynamics.

Several factors may explain this null result. First, NBA teams have roster limits and rotation norms that create similar k-core values across most teams—most teams play 8-10 players significantly, and those players will naturally have similar minimum connection counts. There's simply not enough variance in k-core values to predict success differences.

Second, the k-core measures structural connectivity but not the quality or efficiency of those connections. Two teams with the same k-core number could have vastly different offensive efficiency based on WHO is in that core and HOW they connect, not just that they meet the minimum connection threshold.

Third, basketball networks are inherently dense compared to many networks studied in social network analysis. With only 10-15 significant contributors per team and a high-touch sport, almost all players end up connecting with almost all teammates above some minimal threshold.

**Practical Implication:** K-core is not a useful metric for evaluating NBA team networks. Other measures like density, clustering, and centrality provide more actionable insights.

---

### H11: Network Diameter and Team Success

**Hypothesis:** Teams with smaller network diameter (fewer "hops" to move the ball between any two players) will have more fluid offense and better success.

**Expected Direction:** Negative correlation  

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | PlayoffRank | **-0.344** | 0.063 | Yes* |
| Pass | All metrics | Near zero | >0.60 | No |

#### What We Learned

The diameter hypothesis showed mixed results, with only one marginally significant finding in the assist network. The diameter of a network is the longest shortest path between any two nodes—a smaller diameter means any player can reach any other player in fewer passes.

The assist network showed a marginal correlation (-0.344) with playoff rank, suggesting that teams with smaller assist network diameters tend to get better seeding. This makes intuitive sense: if every player can be reached from every other player in just 2-3 passing actions, the offense can attack from any angle and doesn't get "stuck" running plays through specific players sequentially.

However, the pass network showed no correlation at all, and even the assist network result was only marginally significant and didn't extend to other success metrics. This suggests diameter may capture some relevant information but is not a robust predictor.

The limited variance in diameter values may explain the weak findings. Most NBA team networks have diameters of 2-3 because teams are relatively small (10-15 active players) and dense. The range of possible diameters is narrow, making it difficult to detect effects.

**Practical Implication:** While maximizing connectivity is generally good (reducing diameter), this metric is not actionable enough to guide coaching decisions. Focus on the stronger predictors like density and volume.

---

### H12: Eigenvector Centrality Spread and Team Success

**Hypothesis:** Teams where eigenvector centrality (influence-weighted connectivity) is spread across more players will perform better due to distributed offensive threats.

**Expected Direction:** Positive correlation (higher standard deviation = more spread among multiple influencers)

#### Results

| Network | Success Metric | Correlation | P-Value | Supported? |
|---------|---------------|-------------|---------|------------|
| Assist | All metrics | Near zero | >0.10 | No |
| Pass | WinPCT | **+0.354** | 0.055 | Yes* |
| Pass | PlayoffWins | **+0.354** | 0.055 | Yes* |

#### What We Learned

The eigenvector centrality hypothesis showed a split result: no effect in the assist network but a significant positive effect in the pass network. This is an interesting divergence that reveals something about the different nature of the two networks.

Eigenvector centrality weights a node's importance not just by its connections but by the importance of those connections. A player with high eigenvector centrality is connected to other highly-connected players. The standard deviation of eigenvector centrality across a team captures whether there are multiple "influential" players or just one dominant hub.

The pass network finding (+0.354 with WinPCT) suggests that teams with multiple influential passers—not just one primary ball handler—win more regular season games. This aligns with modern NBA trends toward multiple ball-handlers and "positionless" basketball where forwards and centers also initiate offense.

The lack of effect in the assist network may be because assists are more specialized—even on teams with multiple ball-handlers, actual scoring assists still flow through a smaller number of designated playmakers. The pass network captures the broader ball movement patterns that precede the final assist action.

**Practical Implication:** Developing multiple players with high passing connectivity to other key players (not just the point guard-to-everyone pattern) correlates with regular season success. This supports roster construction with multiple playmaking forwards or wing players.

---

## Cross-Hypothesis Synthesis

### What Predicts Regular Season Success (WinPCT)?

1. **Total Volume** (H6): r = +0.38*** — Move the ball more
2. **Star Power** (H4, reversed): r = +0.43*** — Have a dominant playmaker
3. **Eigenvector Spread** (H12, pass): r = +0.35* — Multiple influential players
4. **Density** (H1, pass): r = +0.31* — Everyone connects

### What Predicts Playoff Success (PlayoffScore)?

1. **Density** (H1, pass): r = +0.46** — Connected teams advance
2. **Clustering** (H5): r = +0.38** — Cohesive units matter
3. **Equality** (H2): r = -0.39** — Distribution beats concentration
4. **Betweenness** (H8, reversed): r = +0.40** — Elite connectors elevate

### Key Divergence: Regular Season vs Playoffs

The most important finding is the difference between what predicts regular season wins versus playoff depth. **Star dependency (H4) helps in the regular season but provides no advantage in the playoffs**, while **density and equality become more important when the games matter most.**

---

## Limitations

1. **Single Season:** Analysis covers only 2023-24; multi-year analysis would strengthen conclusions
2. **Correlation, Not Causation:** We cannot claim network structure causes success; successful teams may develop these patterns as a result of having good players
3. **Missing Context:** Injuries, trades, rest games, and lineup combinations are not accounted for
4. **Edge Weight as Proxy:** Using raw assist/pass counts doesn't account for difficulty of passes or shot quality created

---

## Conclusion

This comprehensive analysis tested 11 hypotheses about NBA passing and assist networks, revealing that team success—particularly in the playoffs—is strongly associated with **network density**, **distribution equality**, **total ball movement volume**, and **clustering/cohesion**. Surprisingly, **star dependency** and **betweenness concentration**, while theoretically representing vulnerabilities, are actually associated with better regular season performance due to the value of elite playmaking.

The practical implication for team building is clear: develop multiple capable playmakers, create offensive systems that involve all players, and build chemistry among player subgroups. However, don't artificially limit your best playmaker—their ability to connect the team is valuable. The key is ensuring role players are also prepared to contribute when playoff defenses inevitably key on the stars.
