import torch
from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from src.algorithm.trainable._nexthop_actor import NextHopActor
from src.nn.mpn import MPN
from src.utils.observations import ObservationGraph, ObservationBatch


class FieldlinesDestActor(NextHopActor):
    """
    This next-hop policy network architecture uses an MPN to provide per-edge values from the latent embedding.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, device: str):
        super().__init__(nn_config, device)
        self.positional_embed = MPN(in_node_features=feature_counts["node"] * 2 + 1,
                                    in_edge_features=feature_counts["edge"],
                                    in_global_features=feature_counts["global"],
                                    latent_dimension=nn_config["latent_dimension"],
                                    mpn_config=nn_config["mpn"]
                                    ).to(self.device)
        self.per_dest_embedding_to_score = nn.Linear(in_features=2 * nn_config["latent_dimension"],
                                                     out_features=1,
                                                     device=device)
        self.lat_dim = nn_config["latent_dimension"]
        self.activation = nn.LeakyReLU()

    def forward(self, input: ObservationGraph | ObservationBatch) -> Tensor:

        # create E*N edge scores for each observation graph in the batch
        input_graph_list, spdist_list, Ns, Es = self._unpack_input(input)
        global_data_list = []
        for g, spdist, N, E in zip(input_graph_list, spdist_list, Ns, Es):

            if spdist is not None:
                spdist_g = spdist.view(N, N)
            else:
                spdist_g = torch.zeros((N, N), device=self.device)

            # for each destination node n:
            for n in range(N):
                stacked_x_i = torch.cat([
                    g.x,  # node features
                    spdist_g[:, n].unsqueeze(1),  # distances to destination node
                    g.x[n].unsqueeze(0).repeat(N, 1)  # node features of destination node
                ], dim=-1)  # shape: [N, lat_dim + 1 + lat_dim]
                graph_i = Data(x=stacked_x_i, edge_index=g.edge_index, edge_attr=g.edge_attr, u=g.u)
                global_data_list.append(graph_i)

        # One single GNN pass over ALL sub‑graphs
        embeddings = self.positional_embed(Batch.from_data_list(global_data_list))

        # Compute scores per edge and destination node one-by-one per graph in batch
        edge_scores = []
        ptr_edge = 0
        ptr_node = 0
        for N, E in zip(Ns, Es):

            # obtain edge and destination node embeddings
            edge_emb = embeddings.edge_attr[ptr_edge:ptr_edge + N * E].view(N, E, self.lat_dim)  # [N, E, D]
            ptr_edge += N * E
            node_emb = embeddings.x[ptr_node:ptr_node + N * N].view(N, N, self.lat_dim)
            ptr_node += N * N
            dest_emb = node_emb[torch.arange(N, device=self.device), torch.arange(N, device=self.device)]  # [N, D]
            dest_emb = dest_emb.unsqueeze(1).expand(-1, E, -1)  # [N, E, D]

            # obtain scores for each edge to each destination node
            cat = torch.cat([edge_emb, dest_emb], dim=-1)  # [N, E, 2D]
            scores = self.per_dest_embedding_to_score(cat).squeeze(-1)  # [N, E]
            scores = self.activation(scores)

            # transpose back to [edge, dest] to match the original edge index
            edge_scores.append(scores.transpose(0, 1).reshape(-1))  # [E*N]

        return torch.cat(edge_scores, dim=0)  # shape: [sum([g.E*g.N for g in batch]),]
