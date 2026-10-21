import torch
from torch import nn as nn, Tensor
from torch_geometric.data import Batch, Data

from src.algorithm.trainable._nexthop_actor import NextHopActor
from src.nn.mpn import MPN
from src.utils.observations import ObservationGraph, ObservationBatch


class FieldlinesTMLR24Actor(NextHopActor):
    """
    This next-hop actor architecture uses an MPN to provide per-edge values from the latent embedding.
    It is based on the architecture proposed in our TMLR 2024 paper.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, device: str):
        super().__init__(nn_config, device)
        self.latent_dimension = nn_config["latent_dimension"]
        self._embedding = MPN(in_node_features=feature_counts["node"],
                              in_edge_features=feature_counts["edge"],
                              in_global_features=feature_counts["global"],
                              latent_dimension=self.latent_dimension,
                              mpn_config=nn_config["mpn"]
                              ).to(self.device)
        self._per_dest_embedding = MPN(in_node_features=self.latent_dimension * 2 + 1,
                                       in_edge_features=self.latent_dimension,
                                       in_global_features=self.latent_dimension,
                                       latent_dimension=self.latent_dimension,
                                       mpn_config=nn_config["mpn"]
                                       ).to(self.device)
        self._score = nn.Linear(in_features=self.latent_dimension, out_features=1, device=device).to(self.device)

    def _prepare_graphs(self, input: ObservationGraph | ObservationBatch):
        """
        Creates batch_size * N graphs from the input graph, where the node features of
        the corresponding destination node are concatenated to the entire graph's node features.
        """
        # for each graph in the batch create N graphs with per-destination features
        input_graph_list, spdist_list, Ns, Es = self._unpack_input(input)
        NEs = []  # [num_nodes*num_edges for graph in batch]
        aug_graphs = []
        for g, spdist, N, E in zip(input_graph_list, spdist_list, Ns, Es):
            NEs.append(N * E)
            # create new per-node input features for N graphs, where for each segment i
            # the features of destination node i are concatenated to all nodes' features.
            aug_xs_features = [g.x.repeat(N, 1), g.x.repeat_interleave(N, dim=0)]

            # add shortest path distance as auxiliary feature if available
            if spdist is not None:
                aug_xs_features.append(spdist.unsqueeze(-1))
            else:
                aug_xs_features.append(torch.zeros((N * N, 1), device=self.device))

            # chunk creates features into N segments and create N graphs with per-destination features.
            aug_xs = torch.cat(aug_xs_features, dim=-1).chunk(N, dim=0)
            b_aug_graphs = [Data(x=x, edge_index=g.edge_index, edge_attr=g.edge_attr, u=g.u)
                            for x in aug_xs]
            aug_graphs.extend(b_aug_graphs)
        return aug_graphs, Ns, NEs

    def forward(self, input: Batch) -> Tensor:
        embedded_input = self._embedding(input)
        aug_graphs, graph_N, graph_NE = self._prepare_graphs(embedded_input)
        per_dest_embeddings = self._per_dest_embedding(Batch.from_data_list(aug_graphs))
        scores = self._score(per_dest_embeddings.edge_attr)  # shape: [sum([num_nodes*num_edges for graph in batch]), 1]
        scores_flattened_per_graph = [cur_graph_scores.reshape(N, -1).t().flatten()
                                      for cur_graph_scores, N in
                                      zip(scores.split(graph_NE, dim=0), graph_N)]  # per graph g: [g.num_edges * g.num_nodes]
        return torch.cat(scores_flattened_per_graph, dim=0)
