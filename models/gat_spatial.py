"""
GAT-based spatial encoder for traffic graph.

Each node (sensor) attends to its neighbors via multi-head graph attention,
learning which nearby roads most influence it at each time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops


class GATSpatialEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network for spatial feature extraction.

    Input:  (B, N, T, F)  — batch, nodes, time steps, features
    Output: (B, N, T, hidden_dim)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Stacked GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim // num_heads  # Each head outputs out_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,    # Concatenate head outputs → hidden_dim
                    add_self_loops=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x:            (B, N, T, F)
            edge_index:   (2, E) — graph connectivity
            return_attention: whether to return attention weights for viz

        Returns:
            out:          (B, N, T, hidden_dim)
            attn_weights: list of (E, heads) tensors if return_attention else None
        """
        B, N, T, F = x.shape
        attn_weights_all = []

        # Project input
        x = self.input_proj(x)  # (B, N, T, hidden_dim)

        # Reshape for PyG: treat each (batch, time) as independent graph
        # PyG GATConv expects (B*T*N, F)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, N, hidden_dim)
        x = x.view(B * T, N, self.hidden_dim)    # (B*T, N, hidden_dim)
        x_flat = x.view(B * T * N, self.hidden_dim)  # node batch

        # Replicate edge_index for all batches
        # edge_index: (2, E) → (2, B*T*E) with offset
        E = edge_index.shape[1]
        edge_index_batched = self._batch_edge_index(edge_index, B * T, N)

        # Apply GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x_flat

            if return_attention:
                x_flat, (edge_idx, alpha) = gat(
                    x_flat, edge_index_batched, return_attention_weights=True
                )
                # Store only the first batch's attention (for visualization)
                attn_weights_all.append(alpha[:E])  # (E, heads)
            else:
                x_flat = gat(x_flat, edge_index_batched)

            x_flat = self.activation(x_flat)
            x_flat = self.dropout(x_flat)

            # Residual connection
            if self.residual:
                x_flat = x_flat + residual

            # Layer norm (reshape for norm, reshape back)
            x_flat = norm(x_flat)

        # Reshape back: (B*T*N, hidden_dim) → (B, N, T, hidden_dim)
        out = x_flat.view(B, T, N, self.hidden_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, N, T, hidden_dim)

        if return_attention:
            return out, attn_weights_all
        return out

    def _batch_edge_index(
        self, edge_index: torch.Tensor, num_graphs: int, num_nodes: int
    ) -> torch.Tensor:
        """Tile edge_index for a batch of graphs with node offset."""
        edge_list = []
        for i in range(num_graphs):
            edge_list.append(edge_index + i * num_nodes)
        return torch.cat(edge_list, dim=1)


class AdaptiveAdjacency(nn.Module):
    """
    Learnable adaptive adjacency matrix (Graph WaveNet-style).
    Generates a soft adjacency from learned node embeddings.
    """

    def __init__(self, num_nodes: int, embedding_dim: int = 10):
        super().__init__()
        self.emb1 = nn.Embedding(num_nodes, embedding_dim)
        self.emb2 = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.emb1.weight)
        nn.init.xavier_uniform_(self.emb2.weight)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns dense soft adjacency (N, N) for visualization or
        dense GCN operations.
        """
        e1 = self.emb1(node_ids)  # (N, D)
        e2 = self.emb2(node_ids)  # (N, D)
        adj = F.relu(torch.mm(e1, e2.t()))
        # Row-normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        return adj
