from torch import Tensor
from torch_geometric.data import Data

from src.utils.canonical_bytes import canonical_bytes


class PackerlEnvAction(Data):
    """
    Wraps a PyG Data object representing assembled actions for the PackeRL environment.

    """
    def __init__(self,
                 edge_index: Tensor = None,
                 edge_attr: Tensor = None,
                 agent_ids: Tensor = None
                 ):
        """
        EnvActions require edge_index, edge_attr and agent_ids.
        """
        if edge_index is None:
            raise ValueError("edge_index must be provided.")
        if edge_attr is None:
            raise ValueError("edge_attr must be provided.")
        if agent_ids is None:
            raise ValueError("agent_ids must be provided.")

        super().__init__(edge_index=edge_index, edge_attr=edge_attr, agent_ids=agent_ids)

    def to_canonical_bytes(self) -> bytes:
        """
        Canonically encodes the action into a bytes object.
        """
        fields = (
            ("edge_index", self.edge_index),
            ("edge_attr", self.edge_attr),
            ("agent_ids", self.agent_ids),
        )
        return canonical_bytes(("pyg.Data", tuple((k, v) for k, v in fields)))