from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from src.nn.mpn import MPN


class MSlimActor(nn.Module):
    """
    Link-weight policy network architecture that leverages the MPN model for link weight prediction.
    """
    def __init__(self, nn_config: dict, feature_counts: dict, out_features: int, device: str,
                 receives_line_graph: bool = True):
        super(MSlimActor, self).__init__()
        self.nn_config = nn_config
        self.device = device
        self.receives_line_graph = receives_line_graph
        self._embedding = MPN(in_node_features=feature_counts["node"],
                              in_edge_features=feature_counts["edge"],
                              in_global_features=feature_counts["global"],
                              latent_dimension=nn_config["latent_dimension"],
                              mpn_config=nn_config["mpn"]
                              ).to(self.device)
        self._readout = nn.Linear(nn_config["latent_dimension"], out_features, device=device)

    def forward(self, input: Data | Batch) -> Tensor:
        hidden = self._embedding(input)
        if self.receives_line_graph:
            res = self._readout(hidden.x)  # link weights stem from node embeddings in line graph
        else:
            res = self._readout(hidden.edge_attr)  # link weights stem from edge embeddings in original graph
        return res
