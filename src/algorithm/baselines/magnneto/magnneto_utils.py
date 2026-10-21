from torch import Tensor

from src.utils.observations import Observation, ObservationGraph
from src.utils.types import TrafficMatrix


def traffic_matrix_to_str(tm: TrafficMatrix, agent_ids: Tensor, as_int=False) -> str:
    """
    Serialize a traffic matrix into a multi-line string, where each line contains a demand between routing agents.
    """
    N = tm.shape[0]
    demands = []

    for i in range(N):
        for j in range(N):
            if i != j:  # Skip diagonal elements
                demand_value = round(tm[i, j]) if as_int else tm[i, j]
                demands.append(f'demand_{len(demands)} {agent_ids[i].item()} {agent_ids[j].item()} {demand_value}')

    return f'DEMANDS {len(demands)}\nlabel src dest bw\n' + '\n'.join(demands)


def obs_to_str(obs: Observation | ObservationGraph,
               datarate_idx: int, delay_idx: int, capacity_scaling=1.0, as_int=False) -> str:
    """
    Serialize an ObservationGraph into a multi-line string, where each line contains an edge.
    """
    edge_lines = []

    # obs_graph.edge_index is a tensor of shape (2, num_edges)
    # obs_graph.edge_attr is a tensor of shape (num_edges, num_edge_features)
    datarates = capacity_scaling * obs.edge_attr[:, datarate_idx]  # shape: (num_edges,)
    delays = obs.edge_attr[:, delay_idx]  # shape: (num_edges,)
    if as_int:
        datarates = datarates.round().clamp(min=1).int()
        delays = delays.round().clamp(min=1).int()

    for i, (u, v) in enumerate(obs.edge_index.T):
        edge_lines.append(f"Link_{i} "
                          f"{obs.agent_ids[u].item()} "
                          f"{obs.agent_ids[v].item()} "
                          f"1 "
                          f"{datarates[i].item()} "
                          f"{delays[i].item()}")

    return f"EDGES {obs.num_edges}\nlabel src dest weight bw delay\n" + '\n'.join(edge_lines)
