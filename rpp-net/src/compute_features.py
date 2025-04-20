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
6. Gini coefficient  – inequality of citation counts
7. σ small‑worldness – Humphries‑Gurney formulation

Authorship‑homophily metrics
----------------------------
8. assort_inst       – institutional assortativity
9. assort_country    – country assortativity
10. assort_topic     – concept/topic assortativity
11. assort_gender    – gender assortativity
12. root_same_inst_frac – fraction of co‑authors sharing root‑author institution

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

def _gini(values):
    if not values: return None
    cum = np.cumsum(sorted(values)) / sum(values)
    return 1 - 2 * np.trapz(cum, dx = 1/len(values))

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

    # -------------- Gini of citations --------------
    cites = [G.out_degree(n) for n in Ud_ego.nodes]
    feats["gini"] = _gini(cites)

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

    # -------------- Homophily ----------------------
    # Build bipartite author–paper graph  (mini‑scope inside ego)
    B = nx.Graph()
    for author_id, doi in net["author_edges"]:
        B.add_node(author_id, type="author")
        B.add_node(doi, type="paper")
        B.add_edge(author_id, doi)

    # And use the author metadata directly:
    attr_inst, attr_country, attr_topic, attr_gender = {}, {}, {}, {}

    for author_id, author_data in net["authors"].items():
        # Extract institution and country
        if 'last_known_institution' in author_data:
            inst = author_data['last_known_institution']
            if inst:
                inst_id = inst['id'].split('/')[-1]
                attr_inst[author_id] = inst_id
                if 'country_code' in inst:
                    attr_country[author_id] = inst['country_code']
        
        # Extract topic from works
        if 'x_concepts' in author_data and author_data['x_concepts']:
            attr_topic[author_id] = author_data['x_concepts'][0]['id']
        
        # Extract gender if available
        if 'gender' in author_data:
            attr_gender[author_id] = author_data['gender']

    authors = {n for n,d in B.nodes(data=True) if d["type"]=="author"}
    Aproj   = nx.algorithms.bipartite.weighted_projected_graph(B, authors)

    def assort(attr_dict, label):
        nx.set_node_attributes(Aproj, attr_dict, "x")
        try:
            return nx.attribute_assortativity_coefficient(Aproj, "x")
        except ZeroDivisionError:  # single‑category
            return None

    feats |= {
        "assort_inst":    assort(attr_inst,    "inst"),
        "assort_country": assort(attr_country, "country"),
        "assort_topic":   assort(attr_topic,   "topic"),
        "assort_gender":  assort(attr_gender,  "gender"),
    }

    # root‑centric fallback
    root_authors = []
    for author_edge in net["author_edges"]:
        if author_edge[1] == root_doi:  # If paper is root
            root_authors.append(author_edge[0])  # Add author ID

    tot=same=0
    for ra in root_authors:
        for nb in Aproj.neighbors(ra):
            tot += 1
            if attr_inst.get(ra) == attr_inst.get(nb): same += 1
    feats["root_same_inst_frac"] = same/tot if tot else None

    return feats