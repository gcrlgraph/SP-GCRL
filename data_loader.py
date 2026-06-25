import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from scipy.io import mmread
import pandas as pd
import os
from pathlib import Path

def get_data_paths():
    """Automatically scan the data folder to get all dataset paths"""
    base_dir = Path(__file__).resolve().parent / 'data'
    
    # Map dataset filenames to identifiers
    dataset_patterns = {
        'fb-pages-public-figure.edges': 'fb_public',
        'fb-pages-tvshow.edges': 'fb_tvshow',
        'soc-epinions.mtx': 'soc_epinions',
        'soc-advogato.edges': 'soc_advogato',
        'out.petster-hamster-household': 'petster_hamster',
        'fb-pages-politician.edges': 'fb_politician',
    }
    
    data_paths = {}
    # Recursively search all files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in dataset_patterns:
                dataset_name = dataset_patterns[file]
                data_paths[dataset_name] = str(Path(root) / file)
    
    return data_paths

def load_data(dataset_name):
    """Load graph datasets in various formats, returning PyG Data and NetworkX graph objects"""
    # Get dataset paths
    data_paths = get_data_paths()
    
    file_path = data_paths.get(dataset_name)
    if file_path is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Unified node ID mapping
    def map_nodes(edges):
        # Ensure edges have shape (N, 2)
        if edges.ndim != 2 or edges.shape[1] != 2:
             raise ValueError(f"Invalid edge data shape, expected (N, 2), got {edges.shape}")
        nodes = np.unique(edges[:, :2])
        mapping = {n: i for i, n in enumerate(nodes)}
        # Use vectorize to apply the mapping
        mapped_edges = np.vectorize(mapping.get)(edges[:, :2])
        return mapped_edges, nodes.size

    edges = None
    # Unified loading logic
    try:
        if dataset_name == 'soc_epinions':
            # Read mtx file using loadtxt, skip comments and header
            # Matrix Market headers start with %%, comments with %
            # We need to skip these lines and the matrix dimension line
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data_lines = []
            header_skipped = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                # Skip the first non-comment line (matrix dimensions)
                if not header_skipped:
                    header_skipped = True
                    continue
                data_lines.append(line)

            # Load data from processed lines
            if not data_lines:
                 raise ValueError("Failed to extract any data lines from soc_epinions.mtx")
            
            # mtx files are typically space-separated: row col [value]
            edges_data = np.loadtxt(data_lines, delimiter=' ', usecols=(0, 1), dtype=np.float64)
            
            # Matrix Market indices are 1-based, convert to 0-based
            edges = (edges_data - 1).astype(np.int64)

        elif dataset_name in ['fb_public', 'fb_tvshow']:
            # Facebook datasets use comma-separated values
            edges = np.loadtxt(file_path, delimiter=',',
                             usecols=(0,1), dtype=np.int64)

        elif dataset_name == 'soc_advogato':
            # soc-advogato.edges uses space-separated values, skip comments, ignore weight column
            edges = np.loadtxt(file_path, delimiter=' ',
                             comments='%', usecols=(0,1), dtype=np.int64)
                             
        elif dataset_name == 'petster_hamster':
            # out.petster-hamster-household uses tab-separated values, skip comments
            edges = np.loadtxt(file_path, delimiter='\t',
                             comments='%', usecols=(0,1), dtype=np.int64)
                             
        elif dataset_name == 'fb_politician':
            # fb-pages-politician.edges uses comma-separated values
            edges = np.loadtxt(file_path, delimiter=',',
                             usecols=(0,1), dtype=np.int64)
        
        else:
            raise ValueError(f"Unhandled dataset loading logic: {dataset_name}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading file {file_path}: {e}")

    if edges is None or edges.size == 0:
         raise ValueError(f"Failed to load any edge data from {file_path}")

    # Remap node IDs
    mapped_edges, num_nodes = map_nodes(edges)

    # Convert to PyG data format (unweighted)
    edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)

    # Create unweighted directed NetworkX graph
    G = nx.DiGraph()
    # Ensure edges are integer type, no weights added
    G.add_edges_from([(int(u), int(v)) for u, v in mapped_edges])

    # Verify graph properties
    if not nx.is_directed(G):
        raise ValueError(f"Graph {dataset_name} is not directed")
    
    # Ensure graph is unweighted
    for u, v in G.edges():
        if G[u][v]:  # if edge has any attributes
            G[u][v].clear()  # clear all edge attributes

    # Verify unweighted
    if any(G[u][v] for u, v in G.edges()):
        raise ValueError(f"Failed to convert graph {dataset_name} to unweighted")

    return Data(edge_index=edge_index, num_nodes=num_nodes), G

