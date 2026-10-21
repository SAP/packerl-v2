import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, Batch

from src.algorithm.trainable._nexthop_actor import NextHopActor
from src.nn.mpn import MPN
from src.utils.observations import ObservationGraph, ObservationBatch


NODE_EMB_TAU = 2.0  # Initial temperature for node-embedding-based actor
NODE_EMB_TAU_BOUNDS = (0.5, 4.0)  # Bounds for temperature tau


class FieldlinesNodeEmbActor(NextHopActor):
    """
    This next-hop policy network architecture uses an MPN to provide node embeddings and derive edge scores from there.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, device: str):
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

        self.log_tau = nn.Parameter(torch.tensor(math.log(NODE_EMB_TAU), device=self.device))

    def tau(self) -> Tensor:
        # Positive and bounded τ; prevents extreme logit gaps.
        t = self.log_tau.exp()
        return t.clamp(NODE_EMB_TAU_BOUNDS[0], NODE_EMB_TAU_BOUNDS[1])

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

    @staticmethod
    def _get_pairwise_dists(node_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, input: ObservationGraph | ObservationBatch) -> Tensor:

        input_graph_list, spdist_list, Ns, Es = self._unpack_input(input)

        # if we have shortest path distances, create additional input graphs encoding them
        if spdist_list[0] is not None:
            spdist_embeddings = self._get_spdist_embeddings(input_graph_list, spdist_list, Ns, Es)
        else:  # otherwise, just use zero embeddings
            spdist_embeddings = torch.zeros((input.x.shape[0], self.nn_config["latent_dimension"]), device=self.device)

        # get latent geometric node embeddings
        input.__setattr__("x", torch.cat([input.x, spdist_embeddings], dim=-1))
        node_embeddings = self._general_embed(input).x  # [total_num_nodes, latent_dimension]

        # Convert latent node embeddings to edge scores grouped by traffic destination, each graph individually
        node_emb_per_graph = node_embeddings.split(Ns, dim=0)
        if isinstance(input, ObservationBatch):
            edge_dests_per_graph = input.edge_index_without_offset[1].split(Es)
        else:
            edge_dests_per_graph = [input.edge_index[1]]
        tau = self.tau()
        edge_scores = []
        for n, e, node_emb_g, edge_dests_g in zip(Ns, Es, node_emb_per_graph, edge_dests_per_graph):
            # For each possible traffic destination d, the score of an edge (u, v) is the negative distance between
            # the embedding of v and the embedding of d => Index dists accordingly, grouped by traffic dest. first
            pairwise_dists = self._get_pairwise_dists(node_emb_g)
            pairwise_logits = -pairwise_dists / tau  # Safe scaled logits (negative distance = higher score)

            edge_destinations = edge_dests_g.repeat_interleave(n, dim=0)
            traffic_destinations = torch.arange(n, device=self.device).repeat(e)
            edge_scores.append(pairwise_logits[edge_destinations, traffic_destinations])  # 1-dim tensor, numel=e*n

        edge_scores = torch.cat(edge_scores, dim=0)  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        return edge_scores


class FieldLinesNodeEmbHardBoundActor(FieldlinesNodeEmbActor):
    """
    This variant does not bound the embedding radii, but instead simply clamps the final squared distances.
    """
    @staticmethod
    def _get_pairwise_dists(node_emb: Tensor) -> Tensor:
        squared_norms = torch.sum(node_emb ** 2, dim=1, keepdim=True)
        pairwise_squared_dists = squared_norms + squared_norms.t() - 2 * torch.matmul(node_emb, node_emb.t())
        return pairwise_squared_dists.clamp_(min=0.0, max=4.0)


class FieldLinesNodeEmbCosineSimilarityActor(FieldlinesNodeEmbActor):
    """
    This variant L2-normalizes node embeddings to the unit sphere and uses cosine similarity to derive distances.
    """
    @staticmethod
    def _get_pairwise_dists(node_emb: Tensor, eps: float = 1e-8) -> Tensor:
        norms = node_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        direction = node_emb / norms
        cosine_similarity = direction @ direction.t()  # values in [-1, 1]
        # Squared Euclid on unit sphere in [0, 4] via cosine similarity
        pairwise_dist = 2 * (1 - cosine_similarity)
        return pairwise_dist.clamp_(min=0.0, max=4.0)


class FieldLinesNodeEmbSoftBoundActor(FieldlinesNodeEmbActor):
    """
    This variant uses a tanh to "softly" bound the radii of the embeddings prior to compute pairwise squared dists.
    It presents a middle ground between the two other variants.
    """
    @staticmethod
    def _get_pairwise_dists(node_emb: Tensor, R: float = 1.0, s: float = 1.0, eps : float = 1e-8) -> Tensor:
        norms = node_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        direction = node_emb / norms  # directions
        radius = R * torch.tanh(norms / s)  # smooth, bounded radii, values in [0, R)
        cosine_similarity = direction @ direction.t()  # values in [-1, 1]
        sq_radius = radius.pow(2)  # values in [0, R^2)
        pairwise_squared_dists = sq_radius + sq_radius.t() - 2.0 * (radius @ radius.t()) * cosine_similarity
        return pairwise_squared_dists.clamp_(min=0.0, max=(2 * R) ** 2)
