from typing import Self

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


from src.utils.torch import common_values


class ObservationGraph(Data):
    """
    An observation graph is a view of the network state from a single observer. The observer is specified by
    an observer ID which matches the respective agent ID, or is constants.GLOBAL_OBSERVER_ID for a global observer
    (i.e. "theoretical" global aggregation of agents' views without communication delay).

    The ObservationGraph contains:
        - the observer id, (agent ID of the observer),
        - node, edge and global features (x, edge_attr, and u; as seen by the observer),
        - an agent_ids tensor, with all agent IDs in the observation (as seen by the observer),
        - network connectivity between agents (edge_index, as seen by the observer).

    NOTE: The edge_index tensor uses contiguous 'inner' node IDs [0 ... num_nodes-1] to represent the graph structure.
     for any operation that requires the original/"outer" agent IDs, the agent_ids tensor must be used as a map.
    """
    def __init__(self,
                 x: Tensor = None,
                 edge_index: Tensor = None,
                 edge_attr: Tensor = None,
                 u: Tensor = None,
                 agent_ids: Tensor = None,
                 observer_ids: Tensor = None
                 ):
        """
        Constructs an ObservationGraph instance, checking for mandatory attributes and their shapes.
        """

        # If all mandatory attributes are None, assume this is an empty initialization for batching.
        if x is None and edge_index is None and edge_attr is None and u is None:
            super().__init__()
            return
        
        # Ensure the mandatory attributes are provided.
        if x is None:
            raise ValueError("x (node features) must be provided.")
        if edge_index is None:
            raise ValueError("edge_index must be provided.")
        if edge_attr is None:
            raise ValueError("edge_attr must be provided.")
        if u is None:
            raise ValueError("Global features 'u' must be provided.")
        if agent_ids is None:
            raise ValueError("agent_ids must be provided.")
        if observer_ids is None:
            raise ValueError("Observer ID must be provided.")

        # Check that 'u' has shape [1, num_global_features]
        if u.dim() != 2 or u.size(0) != 1:
            raise ValueError("u must be a 2D tensor with shape [1, num_global_features].")

        # Check that 'agent_ids' is a 1D tensor with the correct number of nodes.
        if agent_ids.dim() != 1 or agent_ids.size(0) != x.size(0):
            raise ValueError("agent_ids must be a 1D tensor with length equal to the number of nodes in x.")

        # Check that 'observer_ids' has shape [1]
        if observer_ids.dim() != 1 or observer_ids.size(0) != 1:
            raise ValueError("observer_ids must be a 1D tensor with shape [1].")

        # Call the parent constructor with all provided attributes.
        super().__init__(x=x,
                         edge_index=edge_index,
                         edge_attr=edge_attr,
                         u=u,
                         agent_ids=agent_ids,
                         observer_ids=observer_ids
                         )

    @property
    def edge_agent_index(self):
        """
        Returns: The edge_index mapped back to the original agent IDs (uses agent_ids for mapping).
        """
        return self.agent_ids[self.edge_index]

    @property
    def edge_agent_encoding(self):
        """
        Returns: A 1D tensor encoding the edges in the graph as a single integer value for each edge,
        where each edge is represented as `src * max_agent_id + dst`, allowing for efficient intersection checks.
        """
        max_agent_id = self.agent_ids.max().item() + 1
        edge_src, edge_dst = self.edge_agent_index
        return edge_src * max_agent_id + edge_dst

    def each_node_has_outgoing_edges(self) -> bool:
        """
        Returns: True if each node in the graph has at least one outgoing edge, False otherwise.
        """
        out_degree: Tensor = torch.bincount(self.edge_index[0], minlength=self.num_nodes)
        return (out_degree > 0).all().item()

    @classmethod
    def stack_feat(cls, base_graph: Self, top_graph: Self) -> Self:
        """
        Returns a new ObservationGraph object with the node, edge, and global features of the two input graphs stacked.
        The agent IDs of the two graphs as well as the set of edges must have a non-empty intersection.

        Args:
            base_graph: The base graph (bottom) to stack onto. Observer ID is taken from this graph.
            top_graph: The top graph to stack onto the base graph.

        Returns: A new ObservationGraph object with the stacked features.
        """
        # get common agents (nodes)
        common_agent_ids = common_values(base_graph.agent_ids, top_graph.agent_ids)
        if common_agent_ids.size(0) == 0:
            raise ValueError("No common agent IDs found between the two observation graphs.")

        # stack node features of common agents
        common_agent_idx_base = torch.where(torch.isin(base_graph.agent_ids, common_agent_ids))[0]
        common_agent_idx_top = torch.where(torch.isin(top_graph.agent_ids, common_agent_ids))[0]
        stacked_x = torch.cat((base_graph.x[common_agent_idx_base],
                               top_graph.x[common_agent_idx_top]), dim=1)

        # get common edges (requires encoding of 'outer', agent-based edge index to find intersection efficiently)
        max_agent_id = torch.cat((base_graph.agent_ids, top_graph.agent_ids)).max().item() + 1
        base_agent_src, base_agent_dst = base_graph.edge_agent_index
        top_agent_src, top_agent_dst = top_graph.edge_agent_index
        encoded_base_edges = base_agent_src * max_agent_id + base_agent_dst
        encoded_top_edges = top_agent_src * max_agent_id + top_agent_dst
        common_encoded_edges = common_values(encoded_base_edges, encoded_top_edges)
        if common_encoded_edges.size(0) == 0:
            raise ValueError("No common edges found between the two observation graphs.")
        common_agent_index = torch.stack((common_encoded_edges // max_agent_id, common_encoded_edges % max_agent_id))
        common_edge_index = torch.searchsorted(common_agent_ids, common_agent_index)

        # stack edge features of common edges
        common_edge_idx_base = torch.where(torch.isin(encoded_base_edges, common_encoded_edges))[0]
        common_edge_idx_top = torch.where(torch.isin(encoded_top_edges, common_encoded_edges))[0]
        stacked_edge_attr = torch.cat((base_graph.edge_attr[common_edge_idx_base],
                                       top_graph.edge_attr[common_edge_idx_top]), dim=1)

        # stack global features
        stacked_u = torch.cat([base_graph.u, top_graph.u], dim=1)

        stacked_graph = cls(x=stacked_x,
                            edge_index=common_edge_index,
                            edge_attr=stacked_edge_attr,
                            u=stacked_u,
                            agent_ids=common_agent_ids,
                            observer_ids=base_graph.observer_ids.clone()
                            )
        return stacked_graph

    def topology_equal(self, other: Self) -> bool:
        """
        Returns: True if the topology of this observation graph is equal to the other graph's topology
        (incl. agent IDs), False otherwise.
        """
        return self.edge_index.equal(other.edge_index) and self.agent_ids.equal(other.agent_ids)

    def to_line_graph(self, node_to_edge_feat: str) -> Data:
        """
        Converts the observation graph into its line graph representation by taking the original graph edges
        as new nodes, and drawing an edge between those new nodes that, as edges in the original graph,
        form a directed path of length two: E' = {(u, v), (w, x) | (u, v) in E; (w, x) in E; v = w}.
        The edge attributes of the original graph are used as node features in the line graph.
        Optionally, depending on the node_to_edge_feat parameter, the node features of the original graph become
        edge features of the new edges.

        Args:
            node_to_edge_feat (str): Mode for constructing node features of the new edges. Options:
                - "inner": use node features of the node that links the two edges,
                - "outer": use node features of the first edge's source node and the second edge's target node,
                - "all": use node features of all three nodes.

        Since we require the line graph only for inference in link-weight inference modules, we do not
        need information like agent_ids, observer_ids etc. in the resulting line graph.
        """
        edge_index, edge_attr = coalesce(self.edge_index, self.edge_attr, num_nodes=self.num_nodes)

        # Create optional edge features containing a combination of incident node features
        # - Broadcast to create a mask for matching row and col elements
        row, col = edge_index
        mask = row.unsqueeze(0) == col.unsqueeze(1)  # (num_edges, num_edges)
        new_edge_index = torch.nonzero(mask).T

        if self.x is None or self.x.shape[-1] == 0 or node_to_edge_feat == "none":
            new_edge_attr = torch.empty((new_edge_index.size(1), 0), dtype=edge_attr.dtype, device=edge_attr.device)
        else:
            node_features = self.x
            edge_src = row[new_edge_index[0]]  # Source of first edge
            edge_mid = col[new_edge_index[0]]  # Intermediate node
            edge_dst = col[new_edge_index[1]]  # Destination of second edge

            if node_to_edge_feat == "inner":
                new_edge_attr = node_features[edge_mid]
            elif node_to_edge_feat == "outer":
                new_edge_attr = torch.cat([node_features[edge_src], node_features[edge_dst]], dim=-1)
            elif node_to_edge_feat == "all":
                new_edge_attr = torch.cat([node_features[edge_src], node_features[edge_mid], node_features[edge_dst]],
                                          dim=-1)
            else:
                raise ValueError(f"Invalid node_to_edge_features mode: {self.node_to_edge_features}")

        # Create the line graph as a clone, using orig. edge attributes as node features and orig. global features
        return Data(x=edge_attr.clone(),
                    edge_index=new_edge_index,
                    edge_attr=new_edge_attr,
                    u=self.u.clone(),
                    )
