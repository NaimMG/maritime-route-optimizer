"""
Unit tests for Maritime Route Optimizer core functions.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock


# ── 1. Haversine distance ─────────────────────────────────────────────────────

from src.features.engineer import haversine_km

class TestHaversine:

    def test_same_point_is_zero(self):
        """Distance from a point to itself must be 0."""
        d = haversine_km(48.8566, 2.3522, 48.8566, 2.3522)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_paris_london(self):
        """Paris → London is ~340km."""
        d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
        assert 330 < d < 350

    def test_known_distance_la_long_beach(self):
        """LA → Long Beach is ~6km."""
        d = haversine_km(33.75, -118.25, 33.7667, -118.1833)
        assert 4 < d < 10

    def test_symmetry(self):
        """Distance A→B must equal B→A."""
        d1 = haversine_km(29.75, -95.28, 27.92, -82.45)
        d2 = haversine_km(27.92, -82.45, 29.75, -95.28)
        assert d1 == pytest.approx(d2, rel=1e-5)

    def test_vectorized_input(self):
        """Must work with numpy arrays."""
        lats = np.array([33.75, 29.75])
        lons = np.array([-118.25, -95.28])
        d = haversine_km(33.75, -118.25, lats, lons)
        assert len(d) == 2
        assert d[0] == pytest.approx(0.0, abs=1e-3)


# ── 2. Speed filtering ────────────────────────────────────────────────────────

from src.data.pipeline import filter_speed

class TestFilterSpeed:

    def _make_df(self, sog_values):
        return pd.DataFrame({'sog': sog_values, 'mmsi': [1]*len(sog_values)})

    def test_removes_stopped_vessels(self):
        """Vessels with SOG < 0.5 must be removed."""
        df = self._make_df([0.0, 0.3, 0.5, 1.0, 5.0])
        result = filter_speed(df)
        assert all(result['sog'] >= 0.5)

    def test_removes_outliers(self):
        """Vessels with SOG > 30 must be removed."""
        df = self._make_df([1.0, 15.0, 30.0, 31.0, 97.0])
        result = filter_speed(df)
        assert all(result['sog'] <= 30.0)

    def test_keeps_valid_speeds(self):
        """Valid speeds must be kept."""
        df = self._make_df([0.5, 5.0, 10.0, 20.0, 30.0])
        result = filter_speed(df)
        assert len(result) == 5

    def test_handles_nan(self):
        """NaN SOG values must be removed."""
        df = self._make_df([1.0, float('nan'), 5.0])
        result = filter_speed(df)
        assert len(result) == 2
        assert result['sog'].isna().sum() == 0

    def test_empty_dataframe(self):
        """Empty DataFrame must return empty DataFrame."""
        df = self._make_df([])
        result = filter_speed(df)
        assert len(result) == 0


# ── 3. Graph construction ─────────────────────────────────────────────────────

class TestGraphConstruction:

    def test_pyg_graph_shapes(self):
        """PyG graph must have correct tensor shapes."""
        from src.models.gnn import build_pyg_graph
        data = build_pyg_graph(
            nodes_path='data/processed/graph_nodes.parquet',
            edges_path='data/processed/graph_edges.parquet',
        )
        assert data.x.shape[1] == 4        # 4 node features
        assert data.edge_attr.shape[1] == 3 # 3 edge features
        assert data.edge_index.shape[0] == 2 # src, dst

    def test_pyg_graph_no_nan(self):
        """PyG graph tensors must not contain NaN."""
        from src.models.gnn import build_pyg_graph
        data = build_pyg_graph(
            nodes_path='data/processed/graph_nodes.parquet',
            edges_path='data/processed/graph_edges.parquet',
        )
        assert not torch.isnan(data.x).any()
        assert not torch.isnan(data.edge_attr).any()
        assert not torch.isnan(data.y).any()

    def test_edge_index_valid(self):
        """Edge indices must be within node range."""
        from src.models.gnn import build_pyg_graph
        data = build_pyg_graph(
            nodes_path='data/processed/graph_nodes.parquet',
            edges_path='data/processed/graph_edges.parquet',
        )
        n_nodes = data.x.shape[0]
        assert data.edge_index.max() < n_nodes
        assert data.edge_index.min() >= 0


# ── 4. Route optimizer ────────────────────────────────────────────────────────

class TestRouteOptimizer:

    @pytest.fixture(scope='class')
    def optimizer(self):
        from src.models.optimizer import MaritimeRouteOptimizer
        return MaritimeRouteOptimizer()

    def test_known_route_found(self, optimizer):
        """LA → Long Beach must return a valid route."""
        result = optimizer.optimize('Los Angeles', 'Long Beach')
        assert result.found is True
        assert result.total_distance_km > 0
        assert result.n_hops >= 1
        assert 'Los Angeles' in result.path_ports[0]

    def test_unknown_port_returns_not_found(self, optimizer):
        """Unknown port must return found=False."""
        result = optimizer.optimize('Atlantis', 'Nowhere')
        assert result.found is False

    def test_same_port_does_not_crash(self, optimizer):
        """Same origin and destination must not crash and return 0 distance."""
        result = optimizer.optimize('Houston', 'Houston')
        assert result.total_distance_km == pytest.approx(0.0, abs=1e-3)
        assert result.n_hops == 0

    def test_route_cost_is_positive(self, optimizer):
        """Route cost must always be positive."""
        result = optimizer.optimize('New Orleans', 'Gretna')
        if result.found:
            assert result.total_cost > 0

    def test_list_ports_not_empty(self, optimizer):
        """Port list must not be empty."""
        ports = optimizer.list_ports()
        assert len(ports) > 0
        assert 'name' in ports.columns
        assert 'lat' in ports.columns
        assert 'lon' in ports.columns