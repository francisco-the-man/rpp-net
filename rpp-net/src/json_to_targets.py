"""
Convert papers_info.json to rpp_targets.csv format.

This script takes the papers_info.json file created by clean_data.py
and converts it to the rpp_targets.csv format expected by the RPP-Net pipeline.

Usage:
    python src/json_to_targets.py
"""

import json
import pandas as pd
import pathlib
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    # Input and output paths
    input_path = pathlib.Path("rpp-net/data/papers_info.json")
    output_path = pathlib.Path("rpp-net/data/rpp_targets.csv")
    
    if not input_path.exists():
        log.error(f"Input file {input_path} not found")
        return 1
    
    log.info(f"Reading paper information from {input_path}")
    
    # Load the JSON data
    with open(input_path, 'r') as f:
        papers_info = json.load(f)
    
    log.info(f"Loaded information for {len(papers_info)} papers")
    
    # Convert to DataFrame
    rows = []
    for doi, paper_data in papers_info.items():
        # Extract publication year from date
        pub_year = None
        if paper_data.get('date'):
            try:
                pub_year = int(paper_data['date'].split('-')[0])
            except (ValueError, IndexError):
                pass
        
        # Create row with required fields
        row = {
            'doi': doi,
            'title': paper_data.get('paper_name', ''),
            'authors': paper_data.get('paper_authors', ''),
            'pub_year': pub_year,
            'journal': paper_data.get('journal', ''),
            'effect_size_original': paper_data.get('effect_size_original'),
            'effect_size_replication': paper_data.get('effect_size_replication'),
            'study_num': paper_data.get('study_num')
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    
    # Ensure all required columns are present
    required_columns = ['doi', 'pub_year']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        log.error(f"Missing required columns: {missing_columns}")
        return 1
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    log.info(f"Saved {len(df)} papers to {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 