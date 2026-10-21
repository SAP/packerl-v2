import torch
from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from src.algorithm.trainable._nexthop_actor import NextHopActor
from src.nn import MPN
from src.utils.observations import ObservationGraph, ObservationBatch


class PlacerActor(NextHopActor):
    def __init__(self, nn_config: dict, feature_counts: dict, embedding_dim: int, device: str):
        super().__init__(nn_config, device)
        self._spdist_embed = MPN(in_node_features=feature_counts["node"],
                                 in_edge_features=feature_counts["edge"] + 1,
                                 in_global_features=feature_counts["global"],
                                 latent_dimension=nn_config["latent_dimension"],
                                 mpn_config=nn_config["mpn"]
                                 ).to(self.device)
        self._general_embed = MPN(in_node_features=feature_counts["node"] + nn_config["latent_dimension"],
                                  in_edge_features=feature_counts["edge"],
                                  in_global_features=feature_counts["global"],
                                  latent_dimension=nn_config["latent_dimension"],
                                  mpn_config=nn_config["mpn"]
                                  ).to(self.device)
        self._node_embed = nn.Linear(nn_config["latent_dimension"], embedding_dim).to(self.device)

    def _get_spdist_embeddings(self,
                               input_graph_list: list[ObservationGraph], spdist_list: list[Tensor],
                               Ns: list[int], Es: list[int]) -> Tensor:
        spdist_graph_list = []
        for g, spdist, N, E in zip(input_graph_list, spdist_list, Ns, Es):
            # create fully connected g `spdist_graph` that holds:
            #   - `g`'s node features as node features
            #   - an edge feature vector per edge i,j that holds the shortest path distance from i to j and,
            #     if i,j is also an edge in `g`, the edge features of that edge (otherwise zero-padded)
            #   - the global features of `g` as global features
            full_edge_index = torch.stack([
                torch.arange(N, device=self.device).repeat_interleave(N),
                torch.arange(N, device=self.device).repeat(N)
            ], dim=0)  # shape: [2, N*N]

            full_edge_features = torch.zeros((N * N, g.edge_attr.size(1)), device=self.device)
            # map original edges to their position in the fully connected g
            row, col = g.edge_index
            full_edge_positions = row * N + col  # shape: [E]
            full_edge_features[full_edge_positions] = g.edge_attr  # fill in original edge features
            # append shortest path distances as additional edge feature
            full_edge_features = torch.cat([full_edge_features, spdist.view(-1, 1)], dim=-1)
            full_graph = Data(x=g.x, edge_index=full_edge_index, edge_attr=full_edge_features, u=g.u)
            spdist_graph_list.append(full_graph)

        embed_spdist_input = Batch.from_data_list(spdist_graph_list).to(self.device)
        return self._spdist_embed(embed_spdist_input).x  # [total_num_nodes, latent_dimension]

    def forward(self, input: ObservationGraph | ObservationBatch) -> Tensor:
        input_graph_list, spdist_list, Ns, Es = self._unpack_input(input)

        # if we have shortest path distances, create additional input graphs encoding them
        if spdist_list[0] is not None:
            spdist_embeddings = self._get_spdist_embeddings(input_graph_list, spdist_list, Ns, Es)
        else:  # otherwise, just use zero embeddings
            spdist_embeddings = torch.zeros((input.x.shape[0], self.nn_config["latent_dimension"]), device=self.device)

        # get latent geometric node embeddings
        input.__setattr__("x", torch.cat([input.x, spdist_embeddings], dim=-1))
        return self._node_embed(self._general_embed(input).x)  # [total_num_nodes, embedding_dim]
