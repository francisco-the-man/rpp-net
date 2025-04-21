# -*- coding: utf-8 -*-
"""
Post-processing glue: concatenate per-chunk feature CSVs and join ground truth.

1. Glob data/features/results_chunk_*.csv -> concat
2. Read data/rpp_targets.csv
3. Left-join on 'doi' -> data/master_features.csv

Run after the SLURM array completes:

    python src/merge_results.py

The resulting master file is the single source-of-truth table for modelling.
It contains one row per original RPP paper and ~20 engineered features; missing
rows (failed fetches) will have NaNs for feature columns.
"""

import glob, pandas as pd, pathlib

chunks = glob.glob("data/features/results_chunk_*.csv")
dfs = []
for f in chunks:
    try:
        dfs.append(pd.read_csv(f))
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {f}")
if not dfs:
    raise RuntimeError("No non-empty chunk files found!")
df = pd.concat(dfs, ignore_index=True)
targets = pd.read_csv("data/rpp_targets.csv")
master = targets.merge(df, on="doi", how="left")
pathlib.Path("data").mkdir(exist_ok=True)
master.to_csv("data/master_features.csv", index=False)
print("Wrote data/master_features.csv  🔥")