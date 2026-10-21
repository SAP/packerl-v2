import torch
from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from src.nn.mpn import MPN
from src.nn import MLP


class PPOValueNetwork(nn.Module):
    """
    Value network architecture for PPO
    """
    def __init__(self,
                 nn_config: dict,
                 feature_counts: dict,
                 global_pool: str,
                 device: str):
        self.device = device
        super(PPOValueNetwork, self).__init__()
        self._embedding = MPN(in_node_features=feature_counts["node"],
                              in_edge_features=feature_counts["edge"],
                              in_global_features=feature_counts["global"],
                              latent_dimension=nn_config["latent_dimension"],
                              mpn_config=nn_config["mpn"]
                              ).to(self.device)
        self._node_mlp = MLP(in_features=nn_config["latent_dimension"],
                             latent_dimension=nn_config["latent_dimension"],
                             config=nn_config["mlp"]
                             ).to(self.device)
        self._node_value = nn.Linear(in_features=nn_config["latent_dimension"], out_features=1,
                                     device=self.device)
        if global_pool == "mean":
            from torch_scatter import scatter_mean
            self._global_pool = scatter_mean
        elif global_pool == "sum":
            from torch_scatter import scatter_sum
            self._global_pool = scatter_sum
        else:
            raise ValueError(f"Unknown/Invalid global pooling method: {global_pool}")

    def forward(self, input: Data | Batch) -> Tensor:
        embeddings = self._embedding(input)
        node_hidden = self._node_mlp(embeddings.x)
        node_values = self._node_value(node_hidden)
        if isinstance(input, Batch):
            pool_idx = input.batch
        else:  # Single graph
            pool_idx = torch.zeros(embeddings.x.size(0), dtype=torch.long, device=self.device)
        return self._global_pool(node_values, pool_idx, dim=0)
