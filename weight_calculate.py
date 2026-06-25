import torch
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def degree_calculate(data):
    """
    Compute edge weights based on target node in-degree.

    For each edge (u -> v), the weight is: w(u,v) = 1 / in_degree(v).
    A small epsilon (1e-8) is added to prevent division by zero.

    Args:
        data (torch_geometric.data.Data): Graph data with edge_index and num_nodes.

    Returns:
        torch_geometric.data.Data: New Data object with computed edge_attr (edge weights).
    """
    target_nodes = data.edge_index[1]
    in_degree = torch.bincount(target_nodes, minlength=data.num_nodes).float()

    # Edge weight = 1 / in_degree(target), with epsilon for numerical stability
    edge_weight = 1.0 / (in_degree[target_nodes] + 1e-8)

    return Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_weight,
        num_nodes=data.num_nodes
    ).to(device)

