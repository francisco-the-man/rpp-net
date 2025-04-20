'''
Cleans PsychRP data from RP_Psych_Updated.csv
Grabs paper number, title, authors, year, and effect size original and reproduced
Need the following to find date: Journal (O),Volume (O),Issue (O),Pages (O),
            paper_num = str(row['Study Num'])
            paper_name = str(row['Study Title (O)'])
            paper_authors = str(row['Authors (O)'])
            effect_size_original = row['T_r..O.']
            effect_size_replication = row['T_r..R.']
            journal = str(row['Journal (O)'])
            volume = str(row['Volume (O)'])
            issue = str(row['Issue (O)'])
            pages = str(row['Pages (O)'])
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
        "effect_size_original": [original_effect_size],
        "effect_size_replication": [replication_effect_size],
        "journal": [journal],
        "date": [date]
    }
'''

import json
import pandas as pd
import requests
import time
from datetime import datetime
from pyalex import Works


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
        df = pd.read_csv('rpp-net/data/RP_Psych_Updated.csv', encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    output_data = {}

    for index, row in df.iterrows():
        try:
            study_num = str(row['Study Num'])
            paper_name = str(row['Study Title (O)'])
            paper_authors = str(row['Authors (O)'])
            
            try:
                effect_size_original = float(row['T_r..O.'])
            except (ValueError, TypeError):
                effect_size_original = float('nan')
                # skip the row if no original effect size
                continue
                
            try:
                effect_size_replication = float(row['T_r..R.'])
            except (ValueError, TypeError):
                effect_size_replication = float('nan')
                # skip the row if no replication effect size
                continue
                
            journal = str(row['Journal (O)'])
            volume = str(row['Volume (O)'])
            issue = str(row['Issue (O)'])
            pages = str(row['Pages (O)'])
            doi = find_doi(paper_name, paper_authors, journal, volume, issue, pages)
            
            if not doi:
                print(f"Skipping study {study_num}: No DOI found")
                continue

            pub_date = get_publication_date(doi)
            
            data_entry = {
                "study_num": study_num,
                "paper_name": paper_name,
                "paper_authors": paper_authors,
                "effect_size_original": effect_size_original,
                "effect_size_replication": effect_size_replication,
                "journal": journal,
                "volume": volume,
                "issue": issue,
                "pages": pages,
                "date": pub_date
            }
            
            output_data[doi] = data_entry
            
            # small delay to avoid rate limit
            time.sleep(0.5)
            
            print(f"Processed study {study_num}: {doi}")
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    try:
        with open('rpp-net/data/papers_info.json', 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
        print(f"Successfully wrote data for {len(output_data)} papers to papers_info.json")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")

if __name__ == "__main__":
    main()