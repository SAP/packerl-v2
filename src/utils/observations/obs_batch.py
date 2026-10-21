from typing import List, Self

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from src.utils.observations.obs import Observation


class ObservationBatch(Batch):
    """
    A batch of Observations, each containing multiple ObservationGraphs. While this in theory would be a second-order
    batch, we circumvent the second layer by using the ``batch`` and ``ptr`` attribute to track the ObservationGraph
    objects directly, and creating an extra ``obs_ptr`` attribute that indicates the number of graphs per observation,
    allowing for a mapping between ObservationGraphs and their Observations.
    """
    @property
    def num_graphs(self) -> int:
        """
        Returns: The total amount of ObservationGraphs, i.e., sum of the number of graphs in each observation.
        """
        if hasattr(self, 'obs_ptr'):
            return self.obs_ptr[-1]
        else:
            raise ValueError("Can not infer the total number of graphs (ObservationBatch.num_graphs)")

    @property
    def num_nodes_per_graph(self):
        """
        Returns: The number of nodes in each ObservationGraph in the batch.
        """
        return self.ptr[1:] - self.ptr[:-1]

    @property
    def num_edges_per_graph(self):
        """
        Returns: The number of edges in each ObservationGraph in the batch.
        """
        graphs_per_edge_source = self.batch[self.edge_index[0]]
        return torch.bincount(graphs_per_edge_source, minlength=self.num_graphs)

    @property
    def num_graphs_per_observation(self) -> torch.Tensor:
        """
        Returns: The number of ObservationGraphs in each Observation.
        """
        if hasattr(self, 'obs_ptr'):
            return self.obs_ptr[1:] - self.obs_ptr[:-1]
        else:
            raise ValueError("Can not infer the number of graphs per observation")

    @property
    def num_observations(self) -> int:
        """
        Returns: The number of Observations in the batch.
        """
        if hasattr(self, 'obs_ptr'):
            return self.obs_ptr.numel() - 1
        else:
            raise ValueError("Can not infer the number of observations")

    @property
    def batch_size(self) -> int:
        return self.num_observations

    def __len__(self):
        return self.num_observations

    @property
    def edge_index_without_offset(self) -> Tensor:
        """
        Returns the edge index for all ObservationGraphs without the offset applied.
        This is useful for operations that require the original edge indices.
        """
        return self.edge_index - self.ptr[self.batch[self.edge_index[0]]]

    @property
    def edge_agent_index_without_offset(self) -> Tensor:
        """
        Returns: The edge agent index (not the inner edge index) WITHOUT the batching offset applied, such that
         the edge agent indices match the respective entries in self.agent_ids/self.observer_ids.
        """
        return self.agent_ids[self.edge_index]

    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: List[str] | None = None,
                       exclude_keys: List[str] | None = None):
        """
        De-implemented to force use of from_obs_list() method instead.
        """
        raise NotImplementedError("from_data_list() is not supported for ObservationBatch objects. "
                                  "Use from_obs_list() instead.")

    @classmethod
    def from_obs_list(cls,
                      obs_list: List[Observation],
                      follow_batch: List[str] | None = None,
                      exclude_keys: List[str] | None = None) -> Self:
        """
        Returns: An ObservationBatch object created from a list of Observation objects.
        """
        # Create lists for observation graphs, pointers, and central nodes
        obs_graphs = []
        obs_ptr = [0]
        obs_ptr_offset = 0
        obs_central_nodes = []
        obs_spdists = []
        obs_s_t_onehot = []
        for i, obs in enumerate(obs_list):
            if not isinstance(obs, Observation):
                raise ValueError(f"Element at index {i} is not an Observation object.")
            obs_ptr_offset += obs.num_graphs
            obs_ptr.append(obs_ptr_offset)
            obs_graphs.extend(obs.to_obs_graph_list(ignore_spdist=True))  # spdist handled separately below
            obs_central_nodes.append(obs.central_node)
            if hasattr(obs, 'spdist'):
                obs_spdists.append(obs.spdist)
            if hasattr(obs, 's_t_onehot'):
                obs_s_t_onehot.append(obs.s_t_onehot)

        # Create the batch object from the individual ObservationGraphs directly,
        # with special handling for the obs_ptr and central node tensor
        obs_batch = super().from_data_list(obs_graphs, follow_batch, exclude_keys)
        obs_batch.obs_ptr = torch.tensor(obs_ptr, dtype=torch.long)
        obs_batch.central_nodes = torch.stack(obs_central_nodes)
        if len(obs_spdists) > 0:
            obs_batch.spdists = obs_spdists
        if len(obs_s_t_onehot) > 0:
            obs_batch.s_t_onehot = torch.cat(obs_s_t_onehot, dim=0)  # shape: (num_observations, onehot_dim)

        return obs_batch

    @classmethod
    def to_data_list(self) -> List[BaseData]:
        """
        De-implemented to force use of to_obs_list() method instead.
        """
        raise NotImplementedError("to_data_list() is not supported for ObservationBatch objects. "
                                  "Use to_obs_list() instead.")

    def to_obs_list(self) -> List[Observation]:
        """
        Returns: A list of :class:`Observation` objects, each containing their respective ObservationGraphs.
        """
        # get ObservationGraphs directly as flat list
        obs_graph_list = [self.get_example(i) for i in range(self.num_graphs)]

        # slice list and re-create Observations
        obs_list = []
        start_idx = 0
        for i, (num_graphs, central_node) in enumerate(zip(self.num_graphs_per_observation, self.central_nodes)):
            cur_obs_graphs = obs_graph_list[start_idx:start_idx + num_graphs.item()]
            cur_obs = Observation.from_obs_graph_list(cur_obs_graphs, central_node=central_node)
            if hasattr(self, 'spdists'):
                cur_obs.spdist = self.spdists[i]
            if hasattr(self, 's_t_onehot'):
                cur_obs.s_t_onehot = self.s_t_onehot[i, :].unsqueeze(0)
            obs_list.append(cur_obs)
            start_idx += num_graphs

        return obs_list

    def to_line_graph(self, node_to_edge_feat: str) -> Self:
        """
        Converts the observations of the batch to line graphs.
        """
        obs_line_graph_list = [obs.to_line_graph(node_to_edge_feat) for obs in self.to_obs_list()]
        return ObservationBatch.from_obs_list(obs_line_graph_list)

    def get_actor_input(self,
                        actor_loc: str,
                        actor_input: str,
                        link_weights_as_input: bool = False) -> Self:
        """
        Args:
            actor_loc: The mode of operation for the agent, can be "single" or "multi".
            actor_input: The type of input graphs for the agent, can be "birdseye", "central", or "local".

        Returns: A view of this ObservationBatch object containing, for each observation,
         only the relevant observation graphs for the actor,
         depending on the agent deployment mode and graph input type.
        """
        obs_list = [obs.get_actor_input(actor_loc, actor_input, link_weights_as_input) for obs in self.to_obs_list()]
        return ObservationBatch.from_obs_list(obs_list)

    def get_value_input(self,
                        value_loc: str,
                        value_input: str,
                        link_weights_as_input: bool) -> Self:
        """
        Args:
            value_loc: The scope of the value input, can be "single", "multi", or "hybrid".
            value_input: The type of value input, can be "birdseye", "central", or "local".

        Returns: A view of this Observation object containing, for each observation,
         only the relevant observation graphs for the value input,
         depending on the value function scope and graph input type.

        """
        obs_list = [obs.get_value_input(value_loc, value_input, link_weights_as_input) for obs in self.to_obs_list()]
        return ObservationBatch.from_obs_list(obs_list)
