# -*- coding: utf-8 -*-
"""
SLURM task wrapper: fetch -> featureify -> write CSV (one chunk per job).

Each task:

1. Reads data/chunks/chunk_<ID>.csv, a list of DOIs plus meta columns
2. For every DOI:
     a. Determine cutoff_year = replication_year (if present) else pub_year
     b. Call fetch_network_sync(...)
     c. Persist raw JSON to data/networks_raw/{doi}.json
     d. Call features_from_network(...) -> dict
3. Appends all rows to data/features/results_chunk_<ID>.csv

CLI flags (parsed with argparse)
--------------------------------
--chunk_id      Two-digit chunk number (00-99)
--max_depth     BFS depth (default 2)
--max_nodes     Node budget cap (default 1000)
--n_concurrent  Concurrent HTTP requests per task (default 32)

Logging: INFO level to stdout; each processed DOI logs success/fail status.

Exit status is 0 even if some DOIs fail; errors are logged and those rows omitted.
The merge step will outer-join on DOI, so missing rows appear as NaNs downstream.
"""

# run_worker.py
import argparse, logging, pathlib, pandas as pd, orjson as json
from fetch_network import fetch_network_sync
from compute_features import features_from_network

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_id", required=True)
    p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--max_nodes", type=int, default=1000)
    p.add_argument("--n_concurrent", type=int, default=32)
    return p.parse_args()

def main():
    args = parse()
    cid = f"{int(args.chunk_id):02}"
    chunk_path = pathlib.Path(f"data/chunks/chunk_{cid}.csv")
    out_path   = pathlib.Path(f"data/features/results_chunk_{cid}.csv")
    out_rows   = []

    df = pd.read_csv(chunk_path)
    log.info(f"Processing {len(df)} DOIs in chunk {cid}")
    for i, doi in enumerate(df.doi):
        log.info(f"[{i+1}/{len(df)}] Processing {doi}")
        try:
            # publication year used as cut‑off unless repl_year available
            meta_row = df[df.doi==doi].iloc[0]
            cutoff = meta_row.get("repl_year", meta_row.pub_year)
            net = fetch_network_sync(doi, cutoff, args.max_depth,
                                     args.max_nodes, args.n_concurrent)
            pathlib.Path("data/networks_raw").mkdir(parents=True, exist_ok=True)
            with open(f"data/networks_raw/{doi.replace('/','_')}.json","wb") as f:
                f.write(json.dumps(net))
            feats = features_from_network(net, doi)
            out_rows.append(feats)
            log.info("✅ %s done", doi)
        except Exception as e:
            log.exception("❌ %s failed: %s", doi, e)

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    log.info(f"Completed chunk {cid}: {len(out_rows)} successful, {len(df) - len(out_rows)} failed")

if __name__ == "__main__":
    main()