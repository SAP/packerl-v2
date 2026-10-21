from typing import Dict, Optional, Any

from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch

from .mpn_embedding import MPNEmbedding
from .mpn_step import MPNStep


def get_scatter_reduce(name: str):
    """
    Translates a string to the appropriate function from torch_scatter.
    Args:
        name: the name of the scatter operation: "std", "mean", "max", "min", "sum"

    Returns: the appropriate function from torch_scatter with signature (src, index, dim)

    """
    if name == "mean":
        from torch_scatter import scatter_mean
        scatter_reduce = scatter_mean
    elif name == "min":
        from torch_scatter import scatter_min
        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif name == "max":
        from torch_scatter import scatter_max
        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif name == "sum":
        from torch_scatter import scatter_add
        scatter_reduce = scatter_add
    elif name == "std":
        from torch_scatter import scatter_std
        scatter_reduce = scatter_std
    else:
        raise ValueError(f"Unknown scatter reduce '{name}'")
    return scatter_reduce


class MPN(nn.Module):
    """
    The Message Passing Network (MPN) uses a stack of GNN Steps. Each step defines a single GNN pass.
    """
    def __init__(self,
                 in_node_features: int,
                 in_edge_features: int,
                 in_global_features: Optional[int],
                 latent_dimension: int,
                 mpn_config: Dict[str, Any]
                 ):
        super().__init__()

        use_global_features = in_global_features is not None

        # Embeddings
        embedding_config = mpn_config.get("embedding")
        self.node_input_embedding = MPNEmbedding(in_features=in_node_features,
                                                 latent_dimension=latent_dimension,
                                                 embedding_config=embedding_config)
        self.edge_input_embedding = MPNEmbedding(in_features=in_edge_features,
                                                 latent_dimension=latent_dimension,
                                                 embedding_config=embedding_config)
        if use_global_features:
            self.global_input_embedding = MPNEmbedding(in_features=in_global_features,
                                                       latent_dimension=latent_dimension,
                                                       embedding_config=embedding_config)
        else:
            self.global_input_embedding = None

        # Message Passing Stack
        scatter_reduce_strs = mpn_config.get("scatter_reduce")
        if isinstance(scatter_reduce_strs, str):
            scatter_reduce_strs = [scatter_reduce_strs]
        scatter_reducers = [get_scatter_reduce(name) for name in scatter_reduce_strs]

        stack_config = mpn_config.get("stack")
        self._num_steps: int = stack_config.get("num_steps")
        self._num_step_repeats: int = stack_config.get("num_step_repeats", 1)

        self._message_passing_steps = nn.ModuleList([MPNStep(stack_config=stack_config,
                                                             latent_dimension=latent_dimension,
                                                             scatter_reducers=scatter_reducers,
                                                             use_global_features=use_global_features)
                                                     for _ in range(self._num_steps)])

    def _input_embeddings(self, graph: Data | Batch):
        """
        Embeds the input features of the graph in place.
        """
        graph.__setattr__("x", self.node_input_embedding(graph.x))
        graph.__setattr__("edge_attr", self.edge_input_embedding(graph.edge_attr))
        if self.global_input_embedding is not None and graph.u is not None:
            graph.u = self.global_input_embedding(graph.u)

    def _message_passing_stack(self, graph: Data | Batch):
        """
        Runs the message passing stack (multiple message passing steps/repeats) in place.
        """
        for message_passing_step in self._message_passing_steps:
            for repeat in range(self._num_step_repeats):
                message_passing_step(graph=graph)

    def _forward_batch(self, graph: Batch) -> Batch:
        self._input_embeddings(graph)
        self._message_passing_stack(graph)
        return graph

    def forward(self, graph: Data | Batch) -> Data | Batch:
        """
        Performs a forward pass through the MPN for the given input batch of graphs, embedding the input features and
        performing multiple rounds of message passing. Note that this forward pass may not be deterministic
        wrt. floating point precision, because the used scatter_reduce functions are not.
        """
        if isinstance(graph, Batch):
            return self._forward_batch(graph)
        else:
            return self._forward_batch(Batch.from_data_list([graph])).to_data_list()[0]
