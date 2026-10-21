from typing import Dict, Any, Optional

from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch

from .mpn_modules import EdgeModule, NodeModule, GlobalModule


noop = lambda *args, **kwargs: None  # no-op function


class MPNStep(nn.Module):
    """
    Defines a single MessagePassingLayer that takes an observation graph and updates its node and edge
    features using different modules (Edge, Node, Global).
    It first updates the edge-features. The node-features are updated next using the new edge-features. Finally,
    it updates the global features using the new edge- & node-features. The updates are done through MLPs.
    """

    def __init__(self,
                 stack_config: Dict[str, Any],
                 latent_dimension: int,
                 scatter_reducers,
                 use_global_features: bool = False):
        """
        Args:
            stack_config:
                Configuration of the stack of GNN steps. Should contain keys
                "num_steps" (int),
                "residual_connections" (str: "none", "inner", "outer"),
                "step_mlp" (Dict[str, Any]). "step_mlp" is a dictionary for the general configuration of the MLP.
                    which should contain keys
                    "num_layers" (int),
                    "add_output_layer" (bool),
                    "activation_function" (str: "relu", "leakyrelu", "tanh", "silu"), and
                    "regularization" (Dict[str, Any]),
                        which should contain keys
                        "spectral_norm" (bool),
                        "dropout" (float),
                        "latent_normalization" (str: "batch_norm", "layer_norm" or None)
            latent_dimension:
                Dimension of the latent space.
            scatter_reducers:
                reduce operators from torch_scatter. Can be e.g. [scatter_mean]
            use_global_features:
                Whether to use global features or not.
        """
        super().__init__()
        self._latent_dimension = latent_dimension

        residual_connections: Optional[str] = stack_config.get("residual_connections")
        residual_connections = residual_connections.lower() if residual_connections is not None else None
        layer_norm: Optional[str] = stack_config.get("layer_norm")
        layer_norm = layer_norm.lower() if layer_norm is not None else None
        self.use_layer_norm = layer_norm in ["outer", "inner"]

        self.edge_module: Optional[nn.Module] = None
        self.node_module: Optional[nn.Module] = None
        self.global_module: Optional[nn.Module] = None

        self._old_graph: Dict[str, Any] = {}

        self._initialize_maybes()

        if residual_connections == "outer":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_outer_residual = self._add_graph_residuals
        elif residual_connections == "inner":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_inner_node_residual = self._add_node_residual
            self.maybe_inner_edge_residual = self._add_edge_residual

        if layer_norm == "outer":
            self.maybe_outer_layer_norm = self._graph_layer_norm
        elif layer_norm == "inner":
            self.maybe_inner_node_layer_norm = self._node_layer_norm
            self.maybe_inner_edge_layer_norm = self._edge_layer_norm

        self._global_layer_norms = None
        if use_global_features:
            self.maybe_store_global = self._store_global
            if residual_connections in ["outer", "inner"]:
                self.maybe_global_residual = self._add_global_residual
            if layer_norm in ["outer", "inner"]:
                self.maybe_global_layer_norm = self._global_layer_norm
                self._global_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)

        n_scatter_ops = len(scatter_reducers)

        edge_in_features = 3 * latent_dimension  # edge features, and the two participating nodes
        node_in_features = latent_dimension * (1 + n_scatter_ops)  # node and aggregated incoming edge features

        if use_global_features:
            edge_in_features += latent_dimension
            node_in_features += latent_dimension
            self.global_module = GlobalModule(
                in_features=latent_dimension * (2 * n_scatter_ops + 1),
                # global features, reduced edge features and reduced node features
                latent_dimension=latent_dimension,
                stack_config=stack_config,
                scatter_reducers=scatter_reducers
            )
        else:
            self.global_module = noop

        # edge module
        self.edge_module = EdgeModule(in_features=edge_in_features,
                                      latent_dimension=latent_dimension,
                                      stack_config=stack_config,
                                      scatter_reducers=scatter_reducers,
                                      use_global_features=use_global_features)

        # node module
        self.node_module = NodeModule(in_features=node_in_features,
                                      latent_dimension=latent_dimension,
                                      stack_config=stack_config,
                                      scatter_reducers=scatter_reducers,
                                      use_global_features=use_global_features)

        self.reset_parameters()

        if self.use_layer_norm:
            self._node_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
            self._edge_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
        else:
            self._node_layer_norms = None
            self._edge_layer_norms = None

    def _initialize_maybes(self):
        self.maybe_store_old_graph = noop

        self.maybe_outer_residual = noop
        self.maybe_inner_node_residual = noop
        self.maybe_inner_edge_residual = noop

        self.maybe_outer_layer_norm = noop
        self.maybe_inner_node_layer_norm = noop
        self.maybe_inner_edge_layer_norm = noop

        self.maybe_store_global = noop
        self.maybe_global_residual = noop
        self.maybe_global_layer_norm = noop

    def _store_old_graph(self, graph: Batch):
        self._store_nodes(graph)
        self._store_edges(graph)
        self.maybe_store_global(graph)

    def _add_graph_residuals(self, graph: Batch):
        self._add_node_residual(graph)
        self._add_edge_residual(graph)
        self.maybe_global_residual(graph)

    def _graph_layer_norm(self, graph: Batch) -> None:
        self._node_layer_norm(graph)
        self._edge_layer_norm(graph)
        self.maybe_global_layer_norm(graph)

    def _store_nodes(self, graph: Batch):
        self._old_graph["x"] = graph.x

    def _store_edges(self, graph: Batch):
        self._old_graph["edge_attr"] = graph.edge_attr

    def _store_global(self, graph: Batch):
        self._old_graph["u"] = graph.u

    def _add_node_residual(self, graph: Batch):
        graph.__setattr__("x", graph.x + self._old_graph["x"])

    def _add_edge_residual(self, graph: Batch):
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])

    def _add_global_residual(self, graph):
        graph.__setattr__("u", graph.u + self._old_graph["u"])

    def _node_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("x", self._node_layer_norms(graph.x))

    def _edge_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("edge_attr", self._edge_layer_norms(graph.edge_attr))

    def _global_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("u", self._global_layer_norms(graph.u))

    def reset_parameters(self):
        """
        Resets all the parameters for all modules
        """
        for item in [self.node_module, self.edge_module, self.global_module]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, graph: Data | Batch):
        """
        Computes the forward pass for this step inplace
        """
        self.maybe_store_old_graph(graph=graph)

        self.edge_module(graph)
        self.maybe_inner_edge_residual(graph=graph)
        self.maybe_inner_edge_layer_norm(graph=graph)

        self.node_module(graph)
        self.maybe_inner_node_residual(graph=graph)
        self.maybe_inner_node_layer_norm(graph=graph)

        self.global_module(graph)
        self.maybe_global_residual(graph=graph)
        self.maybe_global_layer_norm(graph=graph)

        self.maybe_outer_residual(graph=graph)
        self.maybe_outer_layer_norm(graph=graph)
