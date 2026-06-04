# Counter-Strike: Global Offensive — Win Probability ML Pipeline

This repository contains a Python-based ML pipeline for analyzing CS:GO match data and predicting round win probability. The project extends prior state-of-the-art models by introducing **player position embeddings** as a novel input feature — something prior work (Xenopoulos et al. 2022) explicitly omitted despite positioning being one of the most strategically significant factors in the game.

Most win probability models for CS:GO consider *what* players have (health, armor, weapons, number alive) but ignore *where* they are. This pipeline adds spatial context by encoding real-time player coordinates as tokenized strings and training a Word2Vec model on those sequences — treating map positions like a language with semantic structure. Those embeddings are fed as additional features into an XGBoost classifier alongside standard game-state attributes.

**Result: accuracy improves from 64.8% → 67.5% on 200 parsed Mirage matches.**

## Pipeline Overview
CS:GO demo files (.parquet)
↓
ETL layer (Pandas) — 33 parquet files/match, 20+ relational tables
↓
Feature engineering:
• Standard features: players alive (T/CT), equipment value
• Position encoding: (x, y) → divided by 256 → tokenized string per tick
↓
Word2Vec training — sentences = full rounds, words = per-tick position snapshots
(3-second skip between words to capture strategic movement, not just static snapshots)
↓
XGBoost classifier — standard features + 8 position embedding vectors
↓
Confusion matrix / F1 / feature importance evaluation

## Requirements

Make sure you have the following Python libraries installed:

- pandas
- glob
- xgboost
- scikit-learn
- os
- gensim

```bash
pip install pandas glob2 xgboost scikit-learn gensim
```

## Dataset Collection

Before running the code, make sure to set the correct path for your CS:GO demo files. Update the `dir_name` variable with the location of your stored demo files (sample demos are included in this repository).

```python
dir_name = "C:\\path\\to\\your\\Demos"
```

A sample of 50 demos is included in this repository. For more demos and the full dataset schema, see the links below.

## Key Findings

- Position embeddings shifted feature importance toward `num_t_alive` over `num_ct_alive` — consistent with the real-game intuition that a single T-side player's positioning can swing a round more than a lone CT holding passively.
- CT equipment value retained higher relative importance than T equipment value even with embeddings added — correct, since CT utility (smokes, molotovs) controls bombsite access.
- Each of the 8 position embedding vectors carried roughly equal importance (~0.063 each), as expected when querying a single representative tick per round.
- F1 score was slightly higher *without* embeddings (0.70 vs 0.65), suggesting the current encoding granularity (÷256) may over-generalize positions — a known tradeoff worth tuning.

## Potential Improvements

- Finer coordinate quantization (÷64 instead of ÷256) for more precise spatial resolution
- Add per-player health, gun, and utility state at the queried tick
- Test on less CT-biased maps (Mirage historically skews CT-sided)
- Swap XGBoost for a neural architecture to better leverage the embedding geometry
- Build a live demo viewer showing tick-by-tick win probability in real time

## Important Links

- [PureSkill.gg Data Docs](https://docs.pureskill.gg/datascience/adx/csgo/csds/spec)
- [Column Reference (Google Sheets)](https://docs.google.com/spreadsheets/d/11tDzUNBq9zIX6_9Rel__fdAUezAQzSnh5AVYzCP060c/htmlview)
- [PureSkill CS:GO DSDK](https://github.com/pureskillgg/csgo-dsdk)
- [PureSkill DSDK](https://github.com/pureskillgg/dsdk)
