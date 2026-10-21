from typing import List

from torch import nn, Tensor

from src.utils.observations import ObservationGraph, ObservationBatch


class NextHopActor(nn.Module):
    """
    Base class for all next-hop policy nets. They are used to predict the next hop for packets
    at each routing node and for all possible destinations.
    """
    def __init__(self, nn_config, device):
        super(NextHopActor, self).__init__()
        self.nn_config = nn_config
        self.device = device

    def _unpack_input(self, input: ObservationGraph | ObservationBatch) \
        -> (List[ObservationGraph], List[Tensor], List[int], List[int]):
        input_graph_list, spdist_list, Ns, Es = [], [], [], []
        if isinstance(input, ObservationBatch):
            Ns = input.num_nodes_per_graph.tolist()
            Es = input.num_edges_per_graph.tolist()
            for obs in input.to_obs_list():
                input_graph_list.extend(obs.to_obs_graph_list())
                if hasattr(obs, "spdist"):
                    spdist_list.extend([obs.spdist] * obs.num_graphs)  # TODO all graphs use the same "global" spdist
        elif isinstance(input, ObservationGraph):
            input_graph_list = [input]
            Ns = [input.num_nodes]
            Es = [input.num_edges]
            if hasattr(input, "spdist"):
                spdist_list.append(input.spdist)
        if len(spdist_list) == 0:
            spdist_list = [None] * len(input_graph_list)

        return input_graph_list, spdist_list, Ns, Es
