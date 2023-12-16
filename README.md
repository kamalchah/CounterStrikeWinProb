# Counter-Strike: Global Offensive (CS:GO) Data Analysis

This repository contains Python code for analyzing and modeling CS:GO match data. The code focuses on extracting meaningful features from CS:GO demo files, training a model with and without word embeddings, and evaluating its performance in predicting win probability.

## Requirements

Make sure you have the following Python libraries installed:

- pandas
- glob
- xgboost
- scikit-learn
- os
- gensim

You can install them using the following command:

```bash
pip install pandas glob2 xgboost scikit-learn gensim
```

Dataset Collection
Before running the code, make sure to set the correct path for your CS:GO demo files. Update the dir_name variable with the location of your stored CS:GO demo files (download the sample ones in this repository).
A sample of 50 demos is included in this repository. For more demos visit https://docs.pureskill.gg/datascience/adx/csgo/csds/spec/#round_end---single_event

Replace this directory with the location of which your demos are stored as files containing parquet files.
dir_name = "C:\\Users\\kAMAL\\Desktop\\pureskill\\Demos"


Important links if needed by reader:
https://docs.pureskill.gg/datascience/adx/csgo/csds/spec
https://docs.google.com/spreadsheets/d/11tDzUNBq9zIX6_9Rel__fdAUezAQzSnh5AVYzCP060c/htmlview
https://github.com/pureskillgg/csgo-dsdk
https://github.com/pureskillgg/csgo-dsdk
https://github.com/pureskillgg/dsdk
