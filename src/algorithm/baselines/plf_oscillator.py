import torch

from src.algorithm.baselines import BaselineCentralizedAlgorithm
from src.utils.observations import ObservationGraph


def build_node_layers(num_nodes: int):
    layers = [[0]]
    layer_size = 2
    i = 2
    while i < num_nodes - 1:
        next_layer = []
        for _ in range(layer_size):
            next_layer.append(i)
            i += 1
        layer_size += 1
        layers.append(next_layer)
    layers.append([1])
    return layers


def layers_to_next_hop_edges(node_layers, is_even_step: bool):

    next_hop_edges = []

    for layer_i in range(len(node_layers) - 1):
        src_layer = node_layers[layer_i]
        dst_layer = node_layers[layer_i + 1]

        if len(dst_layer) == 1:
            # last layer, connect all to the sink
            dst = dst_layer[0]
            for src in src_layer:
                next_hop_edges.append((src, dst))
            # irrelevant in practice but required to maintain sedi shape: connect sink to some node
            next_hop_edges.append((dst, src_layer[0]))
        else:
            # connect each node to a node in the next layer, alternating between two options
            for i, src in enumerate(src_layer):
                if is_even_step:
                    dst = dst_layer[i]
                else:
                    dst = dst_layer[i + 1]
                next_hop_edges.append((src, dst))

    return next_hop_edges


def next_hop_edges_to_action(next_hop_edges, obs_graph: ObservationGraph):

    N, E = obs_graph.num_nodes, obs_graph.num_edges
    edge_lookup = torch.full((N, N), -1, dtype=torch.long)
    rows = torch.arange(E, dtype=torch.long)
    edge_src, edge_dst = obs_graph.edge_index
    edge_lookup[edge_src, edge_dst] = rows

    # create a sparse action tensor of shape [E, N] (edges x nodes) to store the shortest path actions.
    action_raw = torch.zeros((E, N))
    for src, dst in next_hop_edges:
        i = int(edge_lookup[src, dst].item())  # index of next-hop edge from src_node_id
        action_raw[i] = 1
    action_raw = action_raw.view(-1)  # flatten to 1D tensor of shape [E*N,]

    # obtain action sedi as a 1D tensor of shape [N*N,], containing the selected edge destination indices
    action_sedi = action_raw.nonzero(as_tuple=False).squeeze()

    return action_raw, action_sedi


class PLFOscillatorAlgorithm(BaselineCentralizedAlgorithm):
    """
    TODO desc
    """
    action_raw_representation = "next_hop"

    def __init__(self, env):
        super().__init__(env)
        self.cached_even_action = None
        self.cached_odd_action = None

    def _get_action(self, obs_graph: ObservationGraph):
        """
        Overrides the baseline method to return a shortest-path routing from randomly generated link weights.
        """
        is_even_step = self._env.t % 2 == 0

        if is_even_step and self.cached_even_action is not None:
            return self.cached_even_action

        if not is_even_step and self.cached_odd_action is not None:
            return self.cached_odd_action

        # compute new actions
        node_layers = build_node_layers(obs_graph.num_nodes)
        next_hop_edges = layers_to_next_hop_edges(node_layers, is_even_step)
        action_raw, action_sedi = next_hop_edges_to_action(next_hop_edges, obs_graph)

        if is_even_step:
            self.cached_even_action = (action_raw, action_sedi)
        else:
            self.cached_odd_action = (action_raw, action_sedi)

        return action_raw, action_sedi
