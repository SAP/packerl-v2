from typing import List, Self

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from src.utils.canonical_bytes import canonical_bytes, canonical_hash
from src.utils.constants import BIRDSEYE_OBSERVER_ID
from src.utils.observations.obs_graph import ObservationGraph


class Observation(Batch):
    """
    An 'observation' is the entirety of all observation graphs for a single network state.
    Each graph is from the perspective of a different observer, identified by their agent/observer ID.
    These IDs are non-negative,
    except for the value that indicates the 'birdseye' observer (constants.BIRDSEYE_OBSERVER_ID).

    We un-implement from_data_list() and to_data_list() to shift use to from_obs_graph_list() and to_obs_graph_list().
     These also call Batch.from_data_list() and Batch.to_data_list() with the appropriate arguments.
    """
    @property
    def num_nodes_per_graph(self) -> Tensor:
        """
        Returns: A tensor containing the number of nodes in each ObservationGraph.
        """
        return self.ptr[1:] - self.ptr[:-1]

    @property
    def num_edges_per_graph(self) -> Tensor:
        """
        Returns: A tensor containing the number of edges in each ObservationGraph.
        """
        graphs_per_edge_source = self.batch[self.edge_index[0]]
        return torch.bincount(graphs_per_edge_source, minlength=self.num_graphs)

    @property
    def edge_offsets(self) -> Tensor:
        """
        Returns: The offsets of the edge tensors for each graph in the batch.

        E.g. if the three graphs of the obs have [3, 5, 2] edges, this will return a tensor [0, 3, 8].
        """
        return torch.cat((
            torch.tensor([0], dtype=torch.long, device=self.ptr.device),
            torch.cumsum(self.num_edges_per_graph, dim=0)[:-1]
        ))

    @property
    def birdseye_observer_graph_idx(self):
        """
        Returns: A tensor with the index of the birdseye observer ID in self.observer_ids.
        """
        return (self.observer_ids == BIRDSEYE_OBSERVER_ID).nonzero(as_tuple=False).squeeze(-1)

    @property
    def central_observer_graph_idx(self):
        """
        Returns: A tensor with the index of the central observer ID in self.observer_ids.
        """
        return (self.observer_ids == self.central_node).nonzero(as_tuple=False).squeeze(-1)

    @property
    def local_observer_graph_idx(self) -> Tensor:
        """
        Returns: A tensor with the indices of all entries in self.observer_ids that are not the birdseye observer ID.
        """
        return (self.observer_ids != BIRDSEYE_OBSERVER_ID).nonzero(as_tuple=False).squeeze()

    @property
    def local_observer_ids(self) -> Tensor:
        """
        Returns: A tensor containing the observer IDs of all local observers (excluding the birdseye observer).
        """
        return self.observer_ids[self.local_observer_graph_idx]

    @property
    def edge_index_without_offset(self) -> Tensor:
        """
        Returns: The inner edge index (not the edge *agent* index) without the batching offset applied.
        """
        return self.edge_index - self.ptr[self.batch[self.edge_index[0]]]

    @property
    def edge_agent_index_without_offset(self) -> Tensor:
        """
        Returns: The edge agent index (not the inner edge index) WITHOUT the batching offset applied, such that
         the edge agent indices match the respective entries in self.agent_ids/self.observer_ids.
        """
        return self.agent_ids[self.edge_index]

    @property
    def agent_ids_with_offset(self):
        """
        Returns: The agent IDs with an offset applied to uniquely identify agent IDs across all graphs.
        """
        agent_id_bounds = self.agent_ids[self.ptr[:-1] + self.num_nodes_per_graph - 1] + 1
        agent_id_ptr = torch.cat((torch.tensor([0], dtype=torch.long, device=self.ptr.device),
                                 torch.cumsum(agent_id_bounds, dim=0))
                                )
        agent_id_offset = agent_id_ptr[:-1].repeat_interleave(self.num_nodes_per_graph)
        return self.agent_ids + agent_id_offset

    @property
    def edge_agent_index_with_offset(self) -> Tensor:
        """
        Returns: The edge agent index (not the inner edge index) WITH an offset applied to uniquely identify agent IDs
        across all graphs. This representation is compatible with self.agent_ids_with_offset.
        """
        return self.agent_ids_with_offset[self.edge_index]

    @classmethod
    def from_data_list(cls,
                       data_list: List[BaseData],
                       follow_batch: List[str] | None = None,
                       exclude_keys: List[str] | None = None):
        """
        De-implemented to force use of from_obs_graph_list() instead.
        """
        raise NotImplementedError("from_data_list() is not supported for Observation objects. "
                                  "Use from_obs_graph_list() instead.")

    @classmethod
    def from_obs_graph_list(cls,
                            obs_graph_list: List[ObservationGraph],
                            central_node: Tensor,
                            observer_ids: Tensor | None = None,
                            s_t_onehot: Tensor | None = None,
                            follow_batch: List[str] | None = None,
                            exclude_keys: List[str] | None = None):
        """
        Custom method to create an Observation object from a list of ObservationGraph objects.
        Includes handling for extra parameters like `central_node` and `observer_ids`.
        """
        obs = super(Observation, cls).from_data_list(obs_graph_list, follow_batch, exclude_keys)
        obs.central_node = central_node

        if observer_ids is not None:
            obs.observer_ids = observer_ids
        if s_t_onehot is not None:
            obs.s_t_onehot = s_t_onehot

        return obs

    def to_data_list(self) -> List[BaseData]:
        """
        De-implemented to force use of to_obs_graph_list() instead.
        """
        raise NotImplementedError("to_data_list() is not supported for Observation objects. "
                                  "Use to_obs_graph_list() instead.")

    def to_obs_graph_list(self, ignore_spdist: bool = False) -> List[ObservationGraph]:
        """
        This one simply is a rename for consistency with from_obs_graph_list().
        """
        obs_graph_list = super(Observation, self).to_data_list()
        if hasattr(self, "spdist") and not ignore_spdist:
            for obs_graph in obs_graph_list:
                obs_graph.spdist = self.spdist
        return obs_graph_list

    def to_line_graph(self, node_to_edge_feat: str) -> Self:
        """
        Converts all observation graphs into their line graph representations.
        """
        obs_line_graphs = [obs_graph.to_line_graph(node_to_edge_feat) for obs_graph in self.to_obs_graph_list()]
        return Observation.from_obs_graph_list(obs_line_graphs, central_node=self.central_node)

    def get_agent_count(self, actor_loc: str) -> int:
        """
        Returns: The number of agents in the observation based on the agent mode.
        """
        if actor_loc == "single":
            return 1
        elif actor_loc == "multi":
            return self.local_observer_ids.numel()
        else:
            raise ValueError(f"Unknown agent mode: {actor_loc}. Expected 'single' or 'multi'.")

    def get_obs_graph(self, observer_id: int) -> ObservationGraph:
        """
        Retrieve the observation graph for a specific observer ID.

        Args:
            observer_id (int): The ID of the observer.

        Returns:
            ObservationGraph: The observation graph for the specified observer.
        """
        idx = (self.observer_ids == observer_id).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            raise ValueError(f"Observer ID {observer_id} not found.")
        elif idx.numel() > 1:
            raise ValueError(f"Multiple graphs found for observer ID {observer_id}.")
        obs_graph = self.get_example(idx)
        return obs_graph

    def get_birdseye_obs_graph(self) -> ObservationGraph:
        """
        Returns: The birdseye graph from the observation.
        """
        return self.get_obs_graph(BIRDSEYE_OBSERVER_ID)

    def get_central_obs_graph(self) -> ObservationGraph:
        """
        Returns: The central graph from the observation.
        """
        return self.get_obs_graph(self.central_node)

    def _get_actor_input_list(self,
                              actor_loc: str,
                              actor_input: str,
                              link_weights_as_input: bool) -> (List[ObservationGraph], Tensor):
        """
        Returns: A selection of ObservationGraph objects from this Observation object,
            depending on the input and mode variants, destined for the actor function.

        Args:
            actor_loc (str): The mode of operation of the actor module, can be "single", "multi", or "hybrid".
            actor_input (str): The type of input graphs supplied to the actor module.
            link_weights_as_input (bool): Whether to include link weights as input for the actor module.

        NOTE: for module designs in the "multi" operation mode receiving "birdseye" or "central" input,
          while it would theoretically be faster/leaner to just do one inference on one graph and skip action assembly,
          we need full logprob tensors with values for every agent - otherwise, we can’t calculate per-agent ratios.
          Therefore, we return a list of identical graphs, even in "single" mode.
        """
        if actor_input == "birdseye":
            birdseye_obs_graph = self.get_birdseye_obs_graph()
            if actor_loc == "single":
                graph_list = [birdseye_obs_graph.clone()]
                observer_ids = birdseye_obs_graph.observer_ids
            else:  # actor_loc == "multi":
                birdseye_obs_graphs = [birdseye_obs_graph.clone() for _ in self.local_observer_graph_idx]
                # if we supply 'last_link_weights' as input for a multi-agent RL policy running on the birdseye view,
                # we need to manually override the last link weights of the respective birdseye graph with the
                # local ones, because 'last_link_weights' is provided/computed for/by the local observers of which
                # the birdseye observer never is a part.
                if link_weights_as_input:
                    # the birdseye graph has the 'ground truth' of edge_index, so we need to check which edges
                    # to take values from. Using the 1D tensor 'obs_graph.edge_agent_encoding', obtain the indices
                    # of the edges in the birdseye graph that correspond to the local observer's edges, and then
                    # override the last link weights in the birdseye graph (last column of edge_attr).
                    local_obs_graphs = self.index_select(self.local_observer_graph_idx)
                    for b_obs, l_obs in zip(birdseye_obs_graphs, local_obs_graphs):
                        max_agent_id = torch.cat((b_obs.agent_ids, l_obs.agent_ids)).max().item() + 1
                        bobs_edge_agent_src, bobs_edge_agent_dst = b_obs.edge_agent_index
                        lobs_edge_agent_src, lobs_edge_agent_dst = l_obs.edge_agent_index
                        bobs_encoding = bobs_edge_agent_src * max_agent_id + bobs_edge_agent_dst
                        lobs_encoding = lobs_edge_agent_src * max_agent_id + lobs_edge_agent_dst
                        # find indices in lobs_encoding that are also in bobs_encoding, in their order of appearance in bobs_encoding
                        source_idx = torch.where(bobs_encoding.unsqueeze(1) == lobs_encoding)[1]
                        # now find the indices in bobs_encoding that correspond to these matching indices
                        target_idx = torch.isin(bobs_encoding, lobs_encoding).nonzero(as_tuple=True)[0]
                        # finally, override the last link weights
                        b_obs.edge_attr[target_idx, -1] = l_obs.edge_attr[source_idx, -1]

                graph_list = birdseye_obs_graphs
                observer_ids = self.local_observer_ids

        elif actor_input == "central":
            central_obs_graph = self.get_central_obs_graph()
            if actor_loc == "single":
                graph_list = [central_obs_graph.clone()]
                observer_ids = central_obs_graph.observer_ids
            else:  # actor_loc == "multi":
                graph_list = [central_obs_graph.clone() for _ in self.local_observer_graph_idx]
                observer_ids = self.local_observer_ids

        else:  # actor_input == "local"
            if actor_loc == "single":
                raise ValueError(f"'{actor_loc}' is not supported for actor_input '{actor_input}'.")
            else: # actor_loc == "multi":
                graph_list = [lobs.clone() for lobs in self.index_select(self.local_observer_graph_idx)]
                observer_ids = self.local_observer_ids

        return graph_list, observer_ids

    def get_actor_input(self,
                        actor_loc: str,
                        actor_input: str,
                        link_weights_as_input: bool = False) -> Self:
        """
        Args:
            actor_loc: The mode of operation for the agent, can be "single" or "multi".
            actor_input: The type of input graphs for the agent, can be "birdseye", "central", or "local".

        Returns: A view of this Observation object containing only the relevant observation graphs for the actor,
          depending on the agent deployment mode and graph input type.
        """
        if actor_input not in ["birdseye", "central", "local"]:
            raise ValueError(f"Unknown agent input: {actor_input}. Expected 'birdseye', 'central', or 'local'.")
        if actor_loc not in ["single", "multi"]:
            raise ValueError(f"Unknown agent mode: {actor_loc}. Expected 'single' or 'multi'.")

        obs_graph_list, observer_ids = self._get_actor_input_list(actor_loc, actor_input, link_weights_as_input)
        actor_input = Observation.from_obs_graph_list(obs_graph_list,
                                                      central_node=self.central_node,
                                                      observer_ids=observer_ids)
        if hasattr(self, "spdist"):
            actor_input.spdist = self.spdist

        assert not torch.any(torch.isnan(actor_input.x)), "Actor input x contains NaN values"
        assert not torch.any(torch.isinf(actor_input.x)), "Actor input x contains Inf values"
        assert not torch.any(torch.isnan(actor_input.edge_attr)), "Actor input edge_attr contains NaN values"
        assert not torch.any(torch.isinf(actor_input.edge_attr)), "Actor input edge_attr contains Inf values"
        assert not torch.any(torch.isnan(actor_input.u)), "Actor input u contains NaN values"
        assert not torch.any(torch.isinf(actor_input.u)), "Actor input u contains Inf values"

        return actor_input

    def _get_value_input_list(self,
                              value_loc: str,
                              value_input: str,
                              link_weights_as_input: bool) -> (List[ObservationGraph], Tensor):
        """
        Returns: A selection of ObservationGraph objects from this Observation object,
            depending on the input and mode variants, destined for the value function.

        Args:
            value_loc (str): The mode of operation of the value module, can be "single", "multi", or "hybrid".
            value_input (str): The type of input graphs supplied to the value module.
            link_weights_as_input (bool): Whether to include link weights as input for the value module.

        NOTE: for module designs in the "multi" operation mode receiving "birdseye" or "central" input,
          while it would theoretically be faster/leaner to just do one inference on one graph and skip action assembly,
          we need full logprob tensors with values for every agent - otherwise, we can’t calculate per-agent ratios.
          Therefore, we return a list of identical graphs, even in "single" mode.
        """
        if value_input == "birdseye":
            birdseye_obs_graph = self.get_birdseye_obs_graph()
            if value_loc == "single":
                graph_list = [birdseye_obs_graph.clone()]
                observer_ids = birdseye_obs_graph.observer_ids
            else:  # mode_variant in ["multi", "hybrid"]:
                graph_list = [birdseye_obs_graph.clone() for _ in self.local_observer_graph_idx]
                observer_ids = self.local_observer_ids

        elif value_input == "central":
            central_obs_graph = self.get_central_obs_graph()
            if value_loc == "single":
                graph_list = [central_obs_graph.clone()]
                observer_ids = central_obs_graph.observer_ids
            else:  # mode_variant in ["single", "multi", "hybrid"]
                graph_list = [central_obs_graph.clone() for _ in self.local_observer_graph_idx]
                observer_ids = self.local_observer_ids

        else:  # input_variant == "local"
            if value_loc == "multi":
                graph_list = [lobs.clone() for lobs in self.index_select(self.local_observer_graph_idx)]
                observer_ids = self.local_observer_ids
            else:  # mode_variant in ["single", "hybrid"]
                raise ValueError(f"'{value_loc}' is not supported for 'local' input variant.")

        # remove last link weights from edge feature tensor if supplied
        if link_weights_as_input:
            for g in graph_list:
                g.edge_attr = g.edge_attr[:, :-1]

        # if we are in hybrid mode, we need to add the local observer graphs to the birdseye graph.
        # For these, also remove the last link weights if supplied.
        if value_loc == "hybrid":
            local_top_graphs = [lobs.clone() for lobs in self.index_select(self.local_observer_graph_idx)]
            if link_weights_as_input:
                for lobs in local_top_graphs:
                    lobs.edge_attr = lobs.edge_attr[:, :-1]
            graph_list = [ObservationGraph.stack_feat(base, top) for base, top in zip(graph_list, local_top_graphs)]

        return graph_list, observer_ids

    def get_value_input(self,
                        value_loc: str,
                        value_input: str,
                        link_weights_as_input: bool = False) -> Self:
        """
        Args:
            value_loc: The scope of the value input, can be "single", "multi", or "hybrid".
            value_input: The type of value input, can be "birdseye", "central", or "local".

        Returns: A view of this Observation object containing only the relevant observation graphs for the value input,
            depending on the value function scope and graph input type.
        """
        if value_input not in ["birdseye", "central", "local"]:
            raise ValueError(f"Unknown value input: {value_input}. Expected 'birdseye', 'central', or 'local'.")
        if value_loc not in ["single", "multi", "hybrid"]:
            raise ValueError(f"Unknown value scope: {value_loc}. Expected 'single', 'multi', or 'hybrid'.")

        obs_graph_list, observer_ids = self._get_value_input_list(value_loc, value_input, link_weights_as_input)
        value_input = Observation.from_obs_graph_list(obs_graph_list,
                                                      central_node=self.central_node,
                                                      observer_ids=observer_ids,
                                                      s_t_onehot=self.s_t_onehot)

        assert not torch.any(torch.isnan(value_input.x)), "Value input x contains NaN values"
        assert not torch.any(torch.isinf(value_input.x)), "Value input x contains Inf values"
        assert not torch.any(torch.isnan(value_input.edge_attr)), "Value input edge_attr contains NaN values"
        assert not torch.any(torch.isinf(value_input.edge_attr)), "Value input edge_attr contains Inf values"
        assert not torch.any(torch.isnan(value_input.u)), "Value input u contains NaN values"
        assert not torch.any(torch.isinf(value_input.u)), "Value input u contains Inf values"

        return value_input

    def to_canonical_bytes(self) -> bytes:
        """
        Converts the observation to a canonical bytes representation (used e.g. for debugging purposes)
        """
        fields = (
            ("x", self.x),
            ("edge_index", self.edge_index),
            ("edge_attr", self.edge_attr),
            ("u", self.u),
            ("agent_ids", self.agent_ids),
            ("observer_ids", self.observer_ids),
            ("central_node", self.central_node),
            ("s_t_onehot", self.s_t_onehot),
            ("batch", self.batch),
            ("ptr", self.ptr),
        )
        return canonical_bytes(("Observation", tuple((k, v) for k, v in fields)))

    def to_pretty_hash(self) -> str:
        """
        Converts the observation to a pretty hash string (used e.g. for debugging purposes)
        """
        return canonical_hash([self.to_canonical_bytes()])
