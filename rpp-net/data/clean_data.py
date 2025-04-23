'''
Cleans PsychRP data from RP_Psych_Updated.csv
Grabs paper number, title, authors, year, and effect size original and reproduced
Need the following to find date: Journal (O),Volume (O),Issue (O),Pages (O),
            paper_num = str(row['Study Num'])
            paper_name = str(row['Study Title (O)'])
            paper_authors = str(row['Authors (O)'])
            journal = str(row['Journal (O)'])
            volume = str(row['Volume (O)'])
            issue = str(row['Issue (O)'])
            pages = str(row['Pages (O)'])
            effect_size_original = row['Effect size (O)']
            effect_size_replication = row['Effect Size (R)']
            replication_p_value = row['P-value (R)']
            replication_direction = row['Direction (R)']
            replication_significance => calculated from replication_p_value and replication_direction
For each extracted item, looks up the doi using OpenAlex and finds the date 
of publication. We are doing some disambiguation here to make sure we find the right paper
(using journal, volume, issue, pages, and first author's last name). This is because the
RPP doesn't directly have a DOI.
Creates a json file with the following structure:
{
    "[DOI]": {
        "study_num": [num],
        "paper_name": [title],
        "paper_authors": [authors],
        "journal": [journal],
        "date": [date],
        "effect_size_original": [original_effect_size],
        "effect_size_replication": [replication_effect_size],
        "replication_p_value": [replication_p_value],
        "replication_direction": [replication_direction],
        "replication_significance": [replication_significance]
    }
}

Also creates a CSV file with the same information for easier analysis.
'''

import json
import pandas as pd
import requests
import time
from datetime import datetime
from pyalex import Works
import re
import os

def parse_p_value(p_val):
    """Parse p-values in various formats to float."""
    if pd.isna(p_val):
        return None
    
    # If already a number, return it
    try:
        return float(p_val)
    except ValueError:
        pass
    
    # Convert string to lowercase
    p_str = str(p_val).strip()
    
    # Handle scientific notation like "2.2 x 10-16"
    sci_match = re.search(r'(\d+\.?\d*)\s*[x×]\s*10[-−](\d+)', p_str)
    if sci_match:
        try:
            base = float(sci_match.group(1))
            exponent = int(sci_match.group(2))
            return base * (10 ** -exponent)
        except (ValueError, IndexError):
            pass
    
    # Handle "less than" notation
    if p_str.startswith('<'):
        # Extract the number after <
        match = re.search(r'<\s*(.+)', p_str)
        if match:
            try:
                return float(match.group(1)) * 0.5  # Conservative estimate
            except ValueError:
                pass
    
    # Handle "prep > .99" or similar
    if 'prep' in p_str.lower() and '>' in p_str:
        return 0.01  # Assuming this means very significant
    
    # Try direct conversion, replacing commas with periods
    try:
        return float(p_str.replace(',', '.'))
    except ValueError:
        print(f"Could not parse p-value: {p_val}")
        return None


def find_doi(title, authors, journal=None, volume=None, issue=None, pages=None):
    """
    Find DOI using OpenAlex API based on paper title and authors
    """
    try:
        clean_title = title.strip().replace('"', '').replace("'", "")
        print(f"Searching for: {clean_title}")
        
        # Extract first author's last name
        first_author_last_name = None
        if authors and authors != 'nan':
            # Split authors string and get first author
            author_parts = authors.split(',')[0].strip().split()
            if author_parts:
                # Last name is typically the last part of the name
                first_author_last_name = author_parts[-1].lower()
                print(f"First author last name: {first_author_last_name}")
        
        # Search for works with the title
        works = Works().search(clean_title)
        results = works.get()
        
        if not results:
            print(f"No results found for: {clean_title}")
            return None
        
        # Filter by author if available
        if first_author_last_name:
            filtered_results = []
            for work in results:
                if 'authorships' in work and work['authorships']:
                    first_work_author = work['authorships'][0]
                    if 'author' in first_work_author and 'display_name' in first_work_author['author']:
                        author_name = first_work_author['author']['display_name'].lower()
                        if first_author_last_name in author_name:
                            filtered_results.append(work)
            
            # Use filtered results if we found any
            if filtered_results:
                print(f"Found {len(filtered_results)} results matching author {first_author_last_name}")
                results = filtered_results
        
        # If we still have multiple results, score them based on additional criteria
        if len(results) > 1:
            scored_results = []
            for work in results:
                score = 0
                
                # Score based on title similarity
                if 'title' in work and work['title']:
                    work_title = work['title'].lower()
                    if clean_title.lower() == work_title:
                        score += 10
                    elif clean_title.lower() in work_title or work_title in clean_title.lower():
                        score += 5
                
                # Score based on journal match
                if journal and journal != 'nan' and 'primary_location' in work and 'source' in work['primary_location']:
                    source = work['primary_location']['source']
                    if 'display_name' in source and journal.lower() in source['display_name'].lower():
                        score += 3
                
                # Score based on volume match
                if volume and volume != 'nan' and 'biblio' in work and 'volume' in work['biblio']:
                    if str(work['biblio']['volume']) == str(volume):
                        score += 2
                
                # Score based on issue match
                if issue and issue != 'nan' and 'biblio' in work and 'issue' in work['biblio']:
                    if str(work['biblio']['issue']) == str(issue):
                        score += 2
                
                # Score based on page match
                if pages and pages != 'nan' and 'biblio' in work and 'first_page' in work['biblio']:
                    first_page = pages.split('-')[0].strip() if '-' in pages else pages.strip()
                    if str(work['biblio']['first_page']) == str(first_page):
                        score += 2
                
                # Add to scored results
                scored_results.append((work, score))
            
            # Sort by score (highest first)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            print(f"Scored results: {[(work['doi'], score) for work, score in scored_results]}")
            
            # Use the highest scoring result
            if scored_results:
                results = [scored_results[0][0]]
        
        # Return the DOI of the first result
        if results and 'doi' in results[0]:
            doi = results[0]['doi']
            if doi:
                return doi.replace('https://doi.org/', '')
        return None
    
    except Exception as e:
        print(f"Error finding DOI for {title}: {e}")
        return None

def get_publication_date(doi):
    """
    Get publication date from DOI using OpenAlex API
    """
    try:
        if not doi:
            return None
        
        work = Works().filter(doi=doi).get()[0]
        
        # look for the publication date
        if 'publication_date' in work:
            return work['publication_date']
        return None
    
    except Exception as e:
        print(f"Error getting publication date for DOI {doi}: {e}")
        return None

def main():
    # load data
    try:
        df = pd.read_csv('data/RPP_full_cleaned.csv', encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    csv_data = []

    for index, row in df.iterrows():
        try:
            study_num = str(row['Study Num'])
            paper_name = str(row['Study Title (O)'])
            paper_authors = str(row['Authors (O)'])
            
            try:
                effect_size_original = row['T_r..O.']
                print(f"Effect size original for study {study_num}: {effect_size_original}")
            except (ValueError, TypeError):
                print(f"Error parsing effect size orig for study {study_num}: {row['Effect size (O)']} -> {effect_size_original}")
                continue
                
            try:
                effect_size_replication = row['T_r..R.']
                print(f"Effect size replication for study {study_num}: {effect_size_replication}")
            except (ValueError, TypeError):
                print(f"Error parsing effect size for study {study_num}: {row['Effect Size (R)']} -> {effect_size_replication}")
                continue

            try:
                replication_p_value = parse_p_value(row['P-value (R)'])
                print(f"Replication p-value for study {study_num}: {replication_p_value}")
            except (ValueError, TypeError):
                print(f"Error parsing p-value for study {study_num}: {row['P-value (R)']} -> {replication_p_value}")
                continue

            try:
                replication_direction = str(row['Direction (R)'])
            except (ValueError, TypeError):
                print(f"Error parsing direction for study {study_num}: {row['Direction (R)']} -> {replication_direction}")
                continue
            
            # note: using 0.05 as threshold p-value for significance
            replication_significance = (replication_p_value <= 0.05 and replication_direction.lower() == 'same')
            print(f"Replication significance for study {study_num}: {replication_significance}")

            journal = str(row['Journal (O)'])
            volume = str(row['Volume (O)'])
            issue = str(row['Issue (O)'])
            pages = str(row['Pages (O)'])
            doi = find_doi(paper_name, paper_authors, journal, volume, issue, pages)
            
            if not doi:
                print(f"Skipping study {study_num}: No DOI found")
                continue

            pub_date = get_publication_date(doi)
            
            
            # Add to CSV data
            csv_data.append({
                "doi": doi,
                "study_num": study_num,
                "paper_name": paper_name,
                "paper_authors": paper_authors,
                "journal": journal,
                "volume": volume,
                "issue": issue,
                "pages": pages,
                "date": pub_date,
                "effect_size_original": effect_size_original,
                "effect_size_replication": effect_size_replication,
                "replication_p_value": replication_p_value,
                "replication_direction": replication_direction,
                "replication_significance": replication_significance
            })
            
            # small delay to avoid rate limit
            time.sleep(0.5)
            
            print(f"Processed study {study_num}: {doi}")
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    
    # Save CSV output
    try:
        csv_df = pd.DataFrame(csv_data)
        csv_path = 'data/rpp_targets_cleaned.csv'
        csv_df.to_csv(csv_path, index=False)
        print(f"Successfully wrote data for {len(csv_data)} papers to {csv_path}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    main()