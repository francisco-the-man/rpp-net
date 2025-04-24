# -*- coding: utf-8 -*-
"""
Graph‑feature extractor (citation structure + authorship homophily).

Given one raw ego‑network (the dict produced by `fetch_network.py`), this
computes a single feature vector suitable for tabular ML:

Structural metrics
------------------
1. n_nodes           – ego network size
2. in/out degree     – citations into / out of root
3. clustering coef   – local transitivity at root
4. betweenness       – root's betweenness on undirected ego
5. Louvain modularity
6. σ small‑worldness – Humphries‑Gurney formulation

Authorship‑homophily metrics
----------------------------
7. Gini coefficient  – inequality of citation counts
8. assort_inst       – institutional assortativity
9. assort_citation   – citation count assortativity
10. pbi_mean         – PBI mean

Implementation notes
--------------------
* Builds a directed citation graph (G) and an author–paper bipartite graph (B);
  authors are projected to Aproj for assortativity.
* Handles missing attributes by assigning "Unknown" so NetworkX can still compute.
* Returns `None` where a metric is undefined (e.g. single‑category assortativity).

Usage
-----
feats = features_from_network(raw_net_dict, root_doi="10.1037/...")
# → {'doi': '10.1037/…', 'n_nodes': 713, 'in_deg': 15, …}
"""

import networkx as nx, numpy as np
from typing import Dict, Any
import community as community_louvain  # Import python-louvain as community
import requests
import os

BASE = os.getenv("OPENALEX_ENDPOINT", "https://api.openalex.org")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

def time_slice_citation_count(author, year):
    '''
    Calculate the h-index for an author at a specific year.
    '''
    # get author citation data from pyalex using author id:
    print("DEBUG: author =", author, "year =", year)
    author_id = author.get("id")
    if author_id and author_id.startswith("https://openalex.org/"):
        author_id = author_id.split("/")[-1]
    url = f"{BASE}/authors/{author_id}?select=counts_by_year"
    if OPENALEX_API_KEY:
        url += f"&api_key={OPENALEX_API_KEY}"
    print(f"Fetching {url}")
    author_info = requests.get(url).json()
    author_info = author_info.get("counts_by_year", [])

    for entry in author_info:
        year_count = entry.get("year")
        if year <= year_count:
            sliced_count = entry.get("cited_by_count")
            print(f'Picked {sliced_count} citations for {author_id} in {year}')
            return sliced_count
    return 0

def build_author_graph(net):
    # Map author_id -> {'dois': set, 'inst_ids': set}
    author_info = {}
    doi_to_authors = {}

    # Build author info and doi->authors mapping
    for doi, meta in net["nodes"].items():
        authors = []
        for auth in meta.get("authorships", []):
            author = auth.get("author", {})
            author_id = author.get("id")
            
            if not author_id:
                continue
            # Get all institution ids from affiliations
            inst_ids = [a.get("id") for a in auth.get("institutions", [])]
            if author_id not in author_info:
                year = meta.get("publication_year")
                author_info[author_id] = {'dois': set(), 'inst_ids': set(), 'citation_count': time_slice_citation_count(author, year)}
            author_info[author_id]['dois'].add(doi)
            author_info[author_id]['inst_ids'].update(inst_ids)
            authors.append(author_id)
        doi_to_authors[doi] = authors

    G = nx.DiGraph()
    # Add nodes
    for author_id, info in author_info.items():
        G.add_node(author_id, dois=list(info['dois']), inst_ids=list(info['inst_ids']))

    # Add edges: for each citation, connect authors of citing to authors of cited
    for citing_doi, cited_doi in net["edges"]:
        citing_authors = doi_to_authors.get(citing_doi, [])
        cited_authors = doi_to_authors.get(cited_doi, [])
        for a1 in citing_authors:
            for a2 in cited_authors:
                    G.add_edge(a1, a2)
    return G

def gini_coefficient(author_graph):
    # Get weighted out-degrees, using max(0, degree) to ensure non-negative values
    weighted_out_degrees = [max(0, d) for n, d in author_graph.out_degree(weight='weight')]
    if sum(weighted_out_degrees) == 0:
        return 0

    sorted_degrees = sorted(weighted_out_degrees)
    cumulative_degrees = np.cumsum(sorted_degrees)

    # Calculate percentages
    total_degree = cumulative_degrees[-1]
    y = cumulative_degrees / total_degree
    x = np.arange(1, len(y) + 1) / len(y)

    # Calculate Gini coefficient
    B = np.trapezoid(y, x)
    A = 0.5 - B  # Area between line of equality and Lorenz curve
    gini = A / (A + B)

    return gini

def assortativity_by_institution(G):
    """Compute institution assortativity coefficient for the graph."""
    # Create a new graph with first institution ID only (as string)
    H = nx.DiGraph()
    
    for node, data in G.nodes(data=True):
        # Get the first institution ID if available
        inst_id = None
        if data.get('inst_ids') and len(data['inst_ids']) > 0:
            inst_id = str(data['inst_ids'][0])  # Convert to string to ensure hashability
            
        if inst_id:  # Only add nodes with institution data
            H.add_node(node, inst_id=inst_id)
    
    # Add edges between nodes that exist in H
    for u, v in G.edges():
        if H.has_node(u) and H.has_node(v):
            H.add_edge(u, v)
    
    if H.number_of_nodes() < 2:
        print("Not enough nodes with institution data")
        return None
        
    try:
        # Compute assortativity coefficient using single institution ID
        assortativity = nx.attribute_assortativity_coefficient(H, 'inst_id')
        return assortativity
    except Exception as e:
        print(f"Error computing assortativity: {e}")
        return None

def assortativity_by_citation_count(G):
    # Create a new graph with citation count only (as an int)
    H = nx.DiGraph()
    for node, data in G.nodes(data=True):
        citation_count = data.get('citation_count', 0)
        if citation_count > 0:
            H.add_node(node, citation_count=citation_count)
    
    for u, v in G.edges():
        if H.has_node(u) and H.has_node(v):
            H.add_edge(u, v)
    
    if H.number_of_nodes() < 2:
        print("Not enough nodes with citation count data")
        return None

    assortativity = nx.attribute_assortativity_coefficient(H, 'citation_count')
    return assortativity

def pbi_mean(G, prestige_attr="citation_count"):
    """
    PBI_mean  =  (mean prestige of CITED authors  –  mean prestige of ALL authors)
                 /  SD(prestige of ALL authors)

    • Baseline = every author node in the ego graph
    • Cited    = tail nodes of any edge (receive ≥1 intra-ego citation)
    """
    # --- baseline distribution ---
    base_scores = [d[prestige_attr] for n, d in G.nodes(data=True)
                   if d.get(prestige_attr) is not None]
    if len(base_scores) < 2 or np.std(base_scores) == 0:
        return None

    # --- cited set ---
    cited_nodes  = {v for _, v in G.edges()}     # heads of edges
    cited_scores = [G.nodes[v][prestige_attr] for v in cited_nodes
                    if G.nodes[v].get(prestige_attr) is not None]
    if not cited_scores:
        return 0.0

    diff = np.mean(cited_scores) - np.mean(base_scores)
    return diff / np.std(base_scores)

def features_from_network(net: Dict[str, Any], root_doi: str) -> dict:
    # ---------------- Build graphs -----------------
    G = nx.DiGraph(net["edges"])
    for doi, meta in net["nodes"].items():
        G.add_node(doi, **meta)

    Ud = G.to_undirected()
    ego = nx.ego_graph(G, root_doi)
    Ud_ego = ego.to_undirected()

    feats = {
        "doi": root_doi,
        "n_nodes": Ud_ego.number_of_nodes(),
        "in_deg":  G.in_degree(root_doi),
        "out_deg": G.out_degree(root_doi),
        "clust_coef": nx.clustering(Ud_ego, root_doi),
        "betweenness": nx.betweenness_centrality(Ud_ego)[root_doi],
    }

    # -------------- Modularity ---------------------
    partition = community_louvain.best_partition(Ud_ego)
    feats["modularity"] = community_louvain.modularity(partition, Ud_ego)


    # -------------- Small‑world σ ------------------
    try:
        C = nx.average_clustering(Ud_ego)
        L = nx.average_shortest_path_length(Ud_ego)
        R = nx.random_reference(Ud_ego, niter=5, connectivity=False)
        Cr = nx.average_clustering(R)
        Lr = nx.average_shortest_path_length(R)
        feats["sigma_sw"] = (C/Cr) / (L/Lr)
    except Exception:
        feats["sigma_sw"] = None

    # -------------- Author Graph ---------------------
    author_graph = build_author_graph(net)
    feats["gini"] = gini_coefficient(author_graph)
    feats["assort_inst"] = assortativity_by_institution(author_graph)
    feats["assort_citation"] = assortativity_by_citation_count(author_graph)
    feats["pbi_mean"] = pbi_mean(author_graph)

    return feats