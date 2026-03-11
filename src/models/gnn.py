"""
Graph Neural Network for Maritime Route Cost Prediction.
Predicts dynamic edge costs (travel time, fuel, risk) for A* pathfinding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
log.info(f"Using device: {DEVICE}")


# ── 1. Model ──────────────────────────────────────────────────────────────────

class MaritimeGNN(nn.Module):
    """
    Graph Attention Network that predicts edge traversal cost.

    Architecture:
        - 3 x GAT layers (node embeddings)
        - Edge MLP (combines src + dst embeddings + edge features)
        - Output: scalar cost per edge
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout

        # GAT layers
        self.conv1 = GATConv(node_features, hidden_channels,
                             heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels,
                             heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels,
                             heads=1, concat=False, dropout=dropout)

        # Edge cost MLP
        # Input = src_embed + dst_embed + edge_features
        edge_input_dim = hidden_channels * 2 + edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # cost must be positive
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x          : Node features  [N, node_features]
            edge_index : Graph edges    [2, E]
            edge_attr  : Edge features  [E, edge_features]
        Returns:
            edge_costs : Predicted cost [E, 1]
        """
        # Node embeddings via GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)  # [N, hidden_channels]

        # Edge cost prediction
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_costs = self.edge_mlp(edge_input)

        return edge_costs


# ── 2. Graph Data Builder ─────────────────────────────────────────────────────

def build_pyg_graph(
    nodes_path: str = "data/processed/graph_nodes.parquet",
    edges_path: str = "data/processed/graph_edges.parquet",
) -> Data:
    """
    Convert port graph DataFrames into a PyTorch Geometric Data object.
    """
    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)

    log.info(f"Building PyG graph: {len(nodes_df)} nodes, {len(edges_df)} edges")

    # ── Node features ──
    # Normalize lat/lon to [-1, 1]
    nodes_df["lat_norm"] = nodes_df["lat"] / 90.0
    nodes_df["lon_norm"] = nodes_df["lon"] / 180.0

    # Port degree (connectivity)
    out_degree = edges_df.groupby("departure_port_id").size()
    in_degree  = edges_df.groupby("arrival_port_id").size()
    nodes_df["out_degree"] = nodes_df["port_id"].map(out_degree).fillna(0)
    nodes_df["in_degree"]  = nodes_df["port_id"].map(in_degree).fillna(0)

    node_feat_cols = ["lat_norm", "lon_norm", "out_degree", "in_degree"]
    x = torch.tensor(
        nodes_df[node_feat_cols].values,
        dtype=torch.float
    )

    # ── Edge index ──
    # Remap port_ids to contiguous indices
    port_id_to_idx = {pid: idx for idx, pid in enumerate(nodes_df["port_id"])}

    src = edges_df["departure_port_id"].map(port_id_to_idx).values
    dst = edges_df["arrival_port_id"].map(port_id_to_idx).values
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0),
        dtype=torch.long
    )

    # ── Edge features ──
    # Normalize
    edges_df["dist_norm"]  = (
        edges_df["avg_distance_km"] / edges_df["avg_distance_km"].max()
    )
    edges_df["speed_norm"] = (
        edges_df["avg_speed"] / 30.0  # max 30 knots
    )
    edges_df["traffic_norm"] = (
        edges_df["n_vessels"] / edges_df["n_vessels"].max()
    )

    edge_feat_cols = ["dist_norm", "speed_norm", "traffic_norm"]
    edge_attr = torch.tensor(
        edges_df[edge_feat_cols].fillna(0).values,
        dtype=torch.float
    )

    # ── Ground truth cost ──
    # Proxy: distance / avg_speed (= travel time in hours)
    edges_df["cost"] = (
        edges_df["avg_distance_km"] / (edges_df["avg_speed"] * 1.852 + 1e-6)
    )
    edges_df["cost_norm"] = edges_df["cost"] / edges_df["cost"].max()
    y = torch.tensor(edges_df["cost_norm"].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.port_id_to_idx = port_id_to_idx
    data.nodes_df = nodes_df
    data.edges_df = edges_df

    log.info(f"  Node features : {data.x.shape}")
    log.info(f"  Edge index    : {data.edge_index.shape}")
    log.info(f"  Edge features : {data.edge_attr.shape}")
    return data


# ── 3. Training ───────────────────────────────────────────────────────────────

def train_gnn(
    data: Data,
    hidden_channels: int = 64,
    num_heads: int = 4,
    dropout: float = 0.2,
    lr: float = 0.001,
    epochs: int = 200,
) -> MaritimeGNN:
    """Train the GNN on the port graph."""

    data = data.to(DEVICE)

    model = MaritimeGNN(
        node_features=data.x.shape[1],
        edge_features=data.edge_attr.shape[1],
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    log.info(f"\nTraining GNN on {DEVICE}...")
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(data.x, data.edge_index, data.edge_attr).squeeze()
        loss = F.mse_loss(pred, data.y)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Restore best model
    model.load_state_dict(best_state)
    log.info(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    return model


def save_model(model: MaritimeGNN, path: str = "data/processed/gnn_model.pt"):
    """Save model weights."""
    torch.save(model.state_dict(), path)
    log.info(f"Model saved to {path}")


def load_model(
    path: str,
    node_features: int,
    edge_features: int,
    hidden_channels: int = 64,
) -> MaritimeGNN:
    """Load model weights."""
    model = MaritimeGNN(node_features, edge_features, hidden_channels)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ── 4. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build graph
    data = build_pyg_graph()

    # Train
    model = train_gnn(data, epochs=200)

    # Save
    save_model(model)

    # Quick inference test
    model.eval()
    with torch.no_grad():
        costs = model(
            data.x.to(DEVICE),
            data.edge_index.to(DEVICE),
            data.edge_attr.to(DEVICE)
        ).squeeze().cpu().numpy()

    log.info(f"\nPredicted edge costs:")
    log.info(f"  Min  : {costs.min():.4f}")
    log.info(f"  Max  : {costs.max():.4f}")
    log.info(f"  Mean : {costs.mean():.4f}")