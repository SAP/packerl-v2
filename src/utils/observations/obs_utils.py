import torch
from torch import Tensor

from src.utils.observations import ObservationGraph, ObservationBatch
from src.utils.logging import LOG_CONSOLE_FUNCTION


def get_nexthop_candidate_idx_single(obs: ObservationGraph) -> Tensor:
    """
    TODO desc
    """
    LOG_CONSOLE_FUNCTION()

    N, E = obs.num_nodes, obs.num_edges
    edge_src_expanded = obs.edge_index[0].unsqueeze(-1).expand(-1, N)  # shape: (E, N)
    dest_node_range = torch.arange(N).unsqueeze(0).expand(E, -1)  # shape: (E, N)
    nexthop_destination_idx = (edge_src_expanded * N + dest_node_range).view(-1)  # shape: (E * N)
    return nexthop_destination_idx


def get_nexthop_candidate_idx_batched(batch: ObservationBatch) -> Tensor:
    """
    Returns a flattened array of edge indices for an entire batch of graphs:

    The returned 1D tensor contains N_g * E_g indices for every observation graph g in the input batch,
    (concatenated along that one dimension). Each pair of edge and destination node gets an index (hence E*N values),
    and pairs of edge and destination node with common destination and edge _source_ nodes have the same index.

    For examples of how the result should look like, see the test in test/test_next_hop.py .

    NOTE: this is different from the tensors describing selected edge destination indices (sedi):
     Sedi tensors contain numbers from 0 to N*E - 1; if index i is in the tensor, it means that
     the edge with index i//N is selected as next-hop edge for destination node i%N.
     Comparison (nexthop_destination_idx as "ndi"):
      - numel: ndi has E or E*N (depending on policy design); sedi has N^2
      - min index: 0 for both
      - max index: N^2 - 1 for ndi; N*E - 1 for sedi

    NOTE 2: This function uses the inner node IDs of PyG, not PackeRL's "agent" IDs. That's no problem because we use
     the nexthop candidate indices only for scatter-sampling from a result tensor computed using inner node IDs.
    """
    LOG_CONSOLE_FUNCTION()

    # We create our own shifted edge source indices using offsets based on squared node counts per graph
    # (squared because the resulting nexthop destination indices, for each graph g, hold index values < N_g ** 2).
    N_per_graph = batch.num_nodes_per_graph
    E_per_graph = batch.num_edges_per_graph
    edge_idx_repeats = N_per_graph.repeat_interleave(E_per_graph)
    N_squared_ptr = torch.cat((torch.tensor([0], dtype=torch.long, device=N_per_graph.device),
                               batch.num_nodes_per_graph[:-1] ** 2))
    our_edge_index_offsets = torch.cumsum(N_squared_ptr, dim=0).repeat_interleave(E_per_graph)
    our_edge_source_idx = batch.edge_index_without_offset[0] * edge_idx_repeats + our_edge_index_offsets

    # We repeat each edge source index e of graph g for N_g times to prepare for the final step.
    expanded_edge_source_idx = our_edge_source_idx.repeat_interleave(edge_idx_repeats)

    # Obtain the destination node index offsets for each graph g.
    graph_range = torch.arange(batch.num_graphs, device=batch.batch.device)
    blocks_per_graph = N_per_graph * E_per_graph
    extended_graph_idx = graph_range.repeat_interleave(blocks_per_graph)
    global_block_idx = torch.arange(blocks_per_graph.sum(), device=batch.batch.device)
    block_ptr = blocks_per_graph.cumsum(dim=0)
    global_block_starts = torch.cat([torch.zeros(1, dtype=block_ptr.dtype, device=block_ptr.device), block_ptr[:-1]])
    unshifted_global_block_idx = global_block_idx - global_block_starts[extended_graph_idx]
    destination_node_range = unshifted_global_block_idx % N_per_graph[extended_graph_idx]

    # Finally, obtain the nexthop destination indices as described in the function docs
    # by adding graph-specific destination node index offsets to the expanded edge source indices.
    nexthop_destination_idx = expanded_edge_source_idx + destination_node_range
    return nexthop_destination_idx


def get_nexthop_candidate_idx(obs: ObservationGraph | ObservationBatch) -> Tensor:
    """
    Wrapper function that calls the single-graph or batched version of get_nexthop_candidate_idx_*
    depending on the type of the input observation:

    get unrolled edge-dest indices to properly index the edge_dest_actor_output tensor
    Caution: this works on PyG's inner node IDs (not PackeRL's agent IDs),
    and assumes that every node has at least one outgoing edge!
    """
    LOG_CONSOLE_FUNCTION()

    if isinstance(obs, ObservationBatch):
        return get_nexthop_candidate_idx_batched(obs)
    else:
        return get_nexthop_candidate_idx_single(obs)
