# -*- coding: utf-8 -*-
"""
Chunker for target-paper list.

Reads data/rpp_targets.csv, shuffles deterministically, and partitions it into
--n_chunks approximately equal CSV files under data/chunks/.

This is run once before submitting the job array:

    python src/split_targets.py --n_chunks 100

Columns in the original CSV are preserved so that per-chunk files still contain
pub_year and repl_year for later cutoff logic.
"""

import pandas as pd, argparse, pathlib, math, random
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_chunks", type=int, required=True)
    args = ap.parse_args()

    # Check if input file exists
    input_path = pathlib.Path("data/rpCB_targets.csv")
    if not input_path.exists():
        log.error(f"Input file {input_path} not found")
        return 1

    log.info(f"Reading target papers from {input_path}")
    df = pd.read_csv(input_path)
    original_count = len(df)
    
    # Check for and remove duplicates
    df = df.drop_duplicates(subset=['doi'])
    if len(df) < original_count:
        log.warning(f"Removed {original_count - len(df)} duplicate DOIs")
    
    log.info(f"Shuffling {len(df)} papers")
    df = df.sample(frac=1, random_state=42)          # shuffle
    
    n = math.ceil(len(df) / args.n_chunks)
    chunks_dir = pathlib.Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Splitting into {args.n_chunks} chunks with ~{n} papers each")
    for i in range(args.n_chunks):
        chunk = df.iloc[i*n:(i+1)*n]
        output_path = chunks_dir / f"chunk_{i:02}.csv"
        chunk.to_csv(output_path, index=False)
        log.info(f"Wrote {len(chunk)} papers to {output_path}")
    
    log.info("Splitting complete")
    return 0

if __name__ == "__main__":
    exit(main())