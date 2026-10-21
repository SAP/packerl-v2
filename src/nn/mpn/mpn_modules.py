from typing import Dict, Any, List, Callable

import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch

from src.nn import MLP


class BaseModule(nn.Module):
    """
    Base class for the node/edge/global feature update modules used in the MPN.
    """
    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: Dict[str, Any],
                 scatter_reducers: List[Callable],
                 use_global_features: bool = False):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            stack_config: Dictionary specifying the way that the gnn base should look like
            scatter_reducers: How to aggregate over the nodes/edges/globals. Can for example be [torch.scatter_mean]
            use_global_features: whether global features are used
        """
        super().__init__()

        self._scatter_reducers = scatter_reducers
        self._n_scatter_reducers = len(self._scatter_reducers)

        if self._n_scatter_reducers == 1:
            # saving on operations if only one reducer is used.
            self.multiscatter = self._no_multiscatter
        else:
            self.multiscatter = self._multiscatter

        mlp_config = stack_config.get("step_mlp")
        self._mlp = MLP(in_features=in_features,
                        latent_dimension=latent_dimension,
                        config=mlp_config)

        self.in_features = in_features
        self.latent_dimension = latent_dimension
        self.use_global_features = use_global_features

        if use_global_features:
            self._maybe_concat_global = self._concat_global
        else:
            self._maybe_concat_global = lambda x, y: x

    def _no_multiscatter(self, features: torch.Tensor, indices: torch.Tensor, dim: int, dim_size: int):
        return self._scatter_reducers[0](features, indices, dim=dim, dim_size=dim_size)

    def _multiscatter(self, features: torch.Tensor, indices: torch.Tensor, dim: int, dim_size: int):
        """
        This function is used to aggregate over the nodes/edges/globals. Uses an internal list of scatter recude
        operations such as min/mean/sum/max and outputs a concatenation of the reduced results.
        Only invoked if more than one reducer is used.
        """
        latent_dimension = features.shape[1]
        reduced = torch.zeros((dim_size, latent_dimension * self._n_scatter_reducers), device=features.device)
        for position, reducer in enumerate(self._scatter_reducers):
            reduced[:, features.shape[1] * position:features.shape[1] * (position + 1)] = \
                reducer(features, indices, dim=dim, dim_size=dim_size)

        # we have [[1,1.1,1.2,1.3,1.4],[2,2.1,2.2,2.3,2.4]]
        # and [[1,1.5,1.6,1.7,1.8],[2,2.5,2.6,2.7,2.8]]
        # and we want [[1,1.1,1.2,1.3,1.4,1,1.5,1.6,1.7,1.8],[2,2.1,2.2,2.3,2.4,2,2.5,2.6,2.7,2.8]]
        # this seems to be the correct thing to do:
        return reduced

    def _concat_global(self, features, graph):
        raise NotImplementedError(f"Module {type(self)} needs to implement _concat_global(self,features,graph)")


class EdgeModule(BaseModule):
    """
    Edge inputs are concatenated: Its own edge features, the features of the two participating nodes and optionally,
    global features are also concatenated to the input.
    """

    def forward(self, graph: Batch):
        """
        Computes edge updates in-place.
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)
        aggregated_features = self._maybe_concat_global(aggregated_features, graph)

        graph.__setattr__("edge_attr", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        source_indices, _ = graph.edge_index
        global_features = graph.u[graph.batch[source_indices]]
        return torch.cat([aggregated_features, global_features], 1)


class NodeModule(BaseModule):
    """
    Node inputs are concatenated: Its own node features, the reduced features of all incoming edges and optionally,
    global features are also concatenated to the input.
    """
    def forward(self, graph: Batch):
        """
        Compute updates for each node feature vector in-place.
        """
        aggregated_edge_features = self.multiscatter(features=graph.edge_attr, indices=graph.edge_index[1],
                                                     dim=0, dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)
        aggregated_features = self._maybe_concat_global(aggregated_features, graph)

        graph.__setattr__("x", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        global_features = graph.u[graph.batch]
        return torch.cat([aggregated_features, global_features], dim=1)


class GlobalModule(BaseModule):
    """
    Global feature network inputs are concatenated: Its own global features, the reduced features of all edges,
    and the reduced features of all nodes.
    """

    def forward(self, graph: Batch):
        """
        Compute updates for the global feature vector
            graph: Batch object of pytorch_geometric.data, represents a batch of homogeneous graphs
        Returns: None. in-place operation.
        """
        reduced_node_features = self.multiscatter(features=graph.x,
                                                  indices=graph.batch,
                                                  dim=0,
                                                  dim_size=graph.u.shape[0])
        source_indices, _ = graph.edge_index
        reduced_edge_features = self.multiscatter(features=graph.edge_attr,
                                                  indices=graph.batch[source_indices],
                                                  dim=0,
                                                  dim_size=graph.u.shape[0])
        aggregated_features = torch.cat([reduced_edge_features, reduced_node_features, graph.u], dim=1)
        graph.__setattr__("u", self._mlp(aggregated_features))
