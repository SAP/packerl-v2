import sys
sys.path.append(".")
import pytest

import torch

from src.utils.observations import ObservationGraph, Observation, ObservationBatch
from src.utils.observations.obs_utils import get_nexthop_candidate_idx


# Helper function to create a graph easily
def create_obs_graph(num_nodes, edge_index) -> ObservationGraph:
    return ObservationGraph(
        x=torch.empty(num_nodes, 1),  # Mock node features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.empty(len(edge_index[0]), 1),  # Mock edge attributes,
        u=torch.empty(1, 1),  # Mock global features,
        agent_ids=torch.empty(num_nodes, dtype=torch.long),  # Mock agent IDs,
        observer_ids=torch.tensor([0], dtype=torch.long),  # Mock observer IDs
    )

# Batch 1
path4 = create_obs_graph(4, [[0, 1, 2], [1, 2, 3]])
expected_out_path4 = torch.tensor([0, 1, 2, 3,
                                   4, 5, 6, 7,
                                   8, 9, 10, 11])

path3 = create_obs_graph(3, [[0, 1], [1, 2]])
expected_out_path3 = torch.tensor([0, 1, 2,
                                   3, 4, 5])

obs1 = Observation.from_obs_graph_list([path4, path3],
                                       central_node=torch.tensor([0], dtype=torch.long))
batch1 = ObservationBatch.from_obs_list([obs1])
expected_out_batch1 = torch.tensor([0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    8, 9, 10, 11,

                                    16, 17, 18,
                                    19, 20, 21])  # for [e1g1, e1g2]

# Batch 2
path2 = create_obs_graph(2, [[0], [1]])  # 1 edge
expected_tensor_path2 = torch.tensor([0, 1])

path3_2 = create_obs_graph(3, [[0, 2], [2, 1]])  # 2 edges
expected_tensor_path3_2 = torch.tensor([0, 1, 2,
                                        6, 7, 8])

obs2 = Observation.from_obs_graph_list([path2, path3_2],
                                       central_node=torch.tensor([0], dtype=torch.long))
batch2 = ObservationBatch.from_obs_list([obs2])
expected_out_batch2 = torch.tensor([0, 1,

                                    4, 5, 6,
                                    10, 11, 12])  # for [e2g1, e2g2]

# Batch 3
ring3_plus_1 = create_obs_graph(4, [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])  # 6 edges
expected_out_ring3_plus_1 = torch.tensor([0, 1, 2, 3,
                                          4, 5, 6, 7,
                                          0, 1, 2, 3,
                                          8, 9, 10, 11,
                                          4, 5, 6, 7,
                                          8, 9, 10, 11])

path3_bi = create_obs_graph(3, [[0, 1, 1, 2], [1, 0, 2, 1]])  # 4 edges
expected_out_path3_bi = torch.tensor([0, 1, 2,
                                      3, 4, 5,
                                      3, 4, 5,
                                      6, 7, 8])

obs3 = Observation.from_obs_graph_list([ring3_plus_1, path3_bi],
                                       central_node=torch.tensor([0], dtype=torch.long))
batch3 = ObservationBatch.from_obs_list([obs3])
expected_out_batch3 = torch.tensor([0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    0, 1, 2, 3,
                                    8, 9, 10, 11,
                                    4, 5, 6, 7,
                                    8, 9, 10, 11,

                                    16, 17, 18,
                                    19, 20, 21,
                                    19, 20, 21,
                                    22, 23, 24])  # for [e3g1, e3g2]

# Batch 4
obs4 = Observation.from_obs_graph_list([ring3_plus_1, ring3_plus_1, path3_bi],
                                       central_node=torch.tensor([0], dtype=torch.long))
batch4 = ObservationBatch.from_obs_list([obs4])
expected_out_batch4 = torch.tensor([0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    0, 1, 2, 3,
                                    8, 9, 10, 11,
                                    4, 5, 6, 7,
                                    8, 9, 10, 11,

                                    16, 17, 18, 19,
                                    20, 21, 22, 23,
                                    16, 17, 18, 19,
                                    24, 25, 26, 27,
                                    20, 21, 22, 23,
                                    24, 25, 26, 27,

                                    32, 33, 34,
                                    35, 36, 37,
                                    35, 36, 37,
                                    38, 39, 40
                                    ]) # for [e3g1, e3g1, e3g2]

# Batch 5
batch5 = ObservationBatch.from_obs_list([obs3, obs2])
expected_out_batch5 = torch.tensor([0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    0, 1, 2, 3,
                                    8, 9, 10, 11,
                                    4, 5, 6, 7,
                                    8, 9, 10, 11,

                                    16, 17, 18,
                                    19, 20, 21,
                                    19, 20, 21,
                                    22, 23, 24,

                                    25, 26,

                                    29, 30, 31,
                                    35, 36, 37])  # for [e3g1, e3g2, e2g1, e2g2]

text_examples = [
    (path4, expected_out_path4),
    (path3, expected_out_path3),
    (batch1, expected_out_batch1),
    (path2, expected_tensor_path2),
    (path3_2, expected_tensor_path3_2),
    (batch2, expected_out_batch2),
    (ring3_plus_1, expected_out_ring3_plus_1),
    (path3_bi, expected_out_path3_bi),
    (batch3, expected_out_batch3),
    (batch4, expected_out_batch4),
    (batch5, expected_out_batch5),
]

@pytest.mark.parametrize("examples", text_examples)
def test_get_nexthop_candidate_idx(examples):
    obs_input, expected_tensor = examples
    nci = get_nexthop_candidate_idx(obs_input)
    assert torch.equal(nci, expected_tensor), f"Expected {expected_tensor}, but got {nci}"
