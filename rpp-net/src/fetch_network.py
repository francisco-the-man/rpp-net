'''
Async network fetcher for OpenAlex; safe for SLURM job arrays.

This module builds citation networks from the OpenAlex API by:
1. Starting with a root paper (identified by DOI)
2. Fetching its metadata and references
3. Recursively crawling referenced papers up to a specified depth
4. Building a network where nodes are papers and edges are citations
5. Pruning the network based on publication year and size constraints

The crawler uses asynchronous requests with concurrency control to efficiently
handle large networks while respecting API rate limits.

Usage:
    # Async usage
    network = await get_ego_network(
        doi="10.1037/0022-3514.90.5.751",
        cutoff_year=2010,
        max_depth=2
    )
    
    # Synchronous usage
    network = fetch_network_sync(
        doi="10.1037/0022-3514.90.5.751",
        cutoff_year=2010,
        max_depth=2
    )

Returns a dictionary with:
    - nodes: Paper metadata keyed by DOI
    - edges: List of (source_doi, target_doi) citation pairs
    - authors: Author metadata keyed by author ID
    - author_edges: List of (author_id, paper_doi) author-paper pairs
    - root_meta: Metadata for the root paper
'''
from __future__ import annotations
import asyncio, aiohttp, nest_asyncio, logging, os, re, time
import orjson as json

nest_asyncio.apply()
BASE = os.getenv("OPENALEX_ENDPOINT", "https://api.openalex.org")

log = logging.getLogger(__name__)

async def _get_json(session: aiohttp.ClientSession, url: str) -> dict:
    #retries for robustness
    retries = 3
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=60) as r:
                r.raise_for_status()
                return await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == retries - 1:
                raise
            log.warning(f"Request failed: {e}. Retrying ({attempt+1}/{retries})...")
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

async def _crawl(root_doi: str,
                 cutoff: int,
                 max_depth: int,
                 max_nodes: int,
                 n_conc: int) -> dict:
    sem   = asyncio.Semaphore(n_conc)
    nodes, edges = {}, []
    frontier = [(root_doi, 0)]
    visited  = set()
    authors = {}
    author_edges = []

    log.info(f"Starting network crawl from {root_doi}")

    async with aiohttp.ClientSession() as sess:
        while frontier and len(nodes) < max_nodes:
            doi, depth = frontier.pop(0)
            if doi in visited: continue
            visited.add(doi)

            encoded_doi = doi.replace("/", "%2F")
            url = f"{BASE}/works/doi:{encoded_doi}"
            async with sem: meta = await _get_json(sess, url)
            if meta.get("publication_year", 0) > cutoff:
                continue                                         # temporal slice

            nodes[doi] = meta
            if depth < max_depth:
                for ref in meta.get("referenced_works", []):
                    ref_doi = ref.split("/")[-1]
                    edges.append((doi, ref_doi))
                    frontier.append((ref_doi, depth + 1))

            if 'authorships' in meta:
                for authorship in meta['authorships']:
                    if 'author' in authorship and 'id' in authorship['author']:
                        author_id = authorship['author']['id']
                        if author_id not in authors:
                            authors[author_id] = authorship['author']
                        author_edges.append((author_id, doi))

    log.info(f"Completed network with {len(nodes)} nodes and {len(edges)} edges")
    if root_doi not in nodes:
        log.warning(f"Root DOI {root_doi} not found in final network")
        return {"nodes": nodes, "edges": edges, "authors": authors, "author_edges": author_edges, "root_meta": {}}
    return {"nodes": nodes, "edges": edges, "authors": authors, "author_edges": author_edges, "root_meta": nodes[root_doi]}

# Public façade -----------------------------------------------------------
async def get_ego_network(doi: str,
                          cutoff_year: int,
                          max_depth: int = 2,
                          max_nodes: int = 1_000,
                          n_concurrent: int = 32) -> dict:
    """Return a pruned ego‑network around DOI.
    Parameters mirror CLI flags in run_worker.py."""
    return await _crawl(doi, cutoff_year, max_depth, max_nodes, n_concurrent)

# Convenience sync wrapper so run_worker can call it blocking
def fetch_network_sync(*args, **kw):
    return asyncio.run(get_ego_network(*args, **kw))