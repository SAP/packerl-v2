import torch

from src.utils.observations import Observation, ObservationGraph, ObservationBatch


# Helper function to create a graph easily
def create_obs_graph(num_nodes, edge_index, observer_id):
    return ObservationGraph(
        x=torch.empty(num_nodes, 1),  # Mock node features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.empty(len(edge_index[0]), 1),  # Mock edge attributes,
        u=torch.empty(1, 1),  # Mock global features,
        agent_ids=torch.arange(num_nodes, dtype=torch.long),  # Mock agent IDs,
        observer_ids=torch.tensor([observer_id], dtype=torch.long),  # Mock observer IDs
    )

# === create observations and batches =======================================================

ring2 = (2, [[0, 1], [1, 0]])  # 2 nodes, 2 edges
ring3 = (3, [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])  # 3 nodes, 6 edges

# Observation 1:
#  - One global observer (observer ID -1) observing a ring of 2 nodes.
#  - Two local observers (observer IDs 0 and 1) observing the same ring of 2 nodes.
#  - observer_ids: [-1, 0, 1]
#  - agent_ids: [0, 1, 0, 1, 0, 1]
# Total: 2+2+2=6 nodes, 2+2+2=6 edges
obs1 = Observation.from_obs_graph_list([
        create_obs_graph(*ring2, observer_id=-1),
        create_obs_graph(*ring2, observer_id=0),
        create_obs_graph(*ring2, observer_id=1),
    ], central_node=torch.tensor([0], dtype=torch.long)
)
obs1bN = 2  # number of nodes in the birdseye graph (observer -1) of obs 1
obs1bE = 2  # number of edges in the birdseye graph (observer -1) of obs 1
obs1n0N = 2  # number of nodes in observer 0's graph in obs 1
obs1n0E = 2  # number of edges in observer 0's graph in obs 1
obs1n1N = 2  # number of nodes in observer 1's graph in obs 1
obs1n1E = 2  # number of edges in observer 1's graph in obs 1
obs1cN = obs1n0N  # number of nodes in the central graph of obs 1
obs1cE = obs1n0E  # number of edges in the central graph of obs 1

# Observation 2:
#  - One global observer (observer ID -1) observing a ring of 2 nodes.
#  - One local observer (observer ID 0) observing a ring of 3 nodes.
#  - One local observer (observer ID 1) observing a ring of 2 nodes, like the global observer.
#  - observer_ids: [-1, 0, 1]
#  - agent_ids: [0, 1, 0, 1, 2, 0, 1]
# Total: 2+3+2=7 nodes, 2+6+2=10 edges
obs2 = Observation.from_obs_graph_list([
        create_obs_graph(*ring2, observer_id=-1),
        create_obs_graph(*ring3, observer_id=0),  # node 0 thinks there is a ring of 3 nodes
        create_obs_graph(*ring2, observer_id=1),
    ], central_node=torch.tensor([0], dtype=torch.long)
)
obs2bN = 2  # number of nodes in the birdseye graph (observer -1) of obs 2
obs2bE = 2  # number of edges in the birdseye graph (observer -1) of obs 2
obs2n0N = 3  # number of nodes in observer 0's graph in obs 2
obs2n0E = 6  # number of edges in observer 0's graph in obs 2
obs2n1N = 2  # number of nodes in observer 1's graph in obs 2
obs2n1E = 2  # number of edges in observer 1's graph in obs 2
obs2cN = obs2n0N  # number of nodes in the central graph of obs 2
obs2cE = obs2n0E  # number of edges in the central graph of obs 2

# Observation 3:
#  - One global observer (observer ID -1) observing a ring of 3 nodes.
#  - Three local observers (observer IDs 0, 1, and 2) observing the same ring of 3 nodes.
#  - observer_ids: [-1, 0, 1, 2]
#  - agent_ids: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
# Total: 3+3+3+3=12 nodes, 6+6+6+6=24 edges
obs3 = Observation.from_obs_graph_list([
        create_obs_graph(*ring3, observer_id=-1),
        create_obs_graph(*ring3, observer_id=0),
        create_obs_graph(*ring3, observer_id=1),
        create_obs_graph(*ring3, observer_id=2),
    ], central_node=torch.tensor([0], dtype=torch.long)
)
obs3bN = 3  # number of nodes in the birdseye graph (observer -1) of obs 3
obs3bE = 6  # number of edges in the birdseye graph (observer -1) of obs 3
obs3n0N = 3  # number of nodes in observer 0's graph in obs 3
obs3n0E = 6  # number of edges in observer 0's graph in obs 3
obs3n1N = 3  # number of nodes in observer 1's graph in obs 3
obs3n1E = 6  # number of edges in observer 1's graph in obs 3
obs3n2N = 3  # number of nodes in observer 2's graph in obs 3
obs3n2E = 6  # number of edges in observer 2's graph in obs 3
obs3cN = obs3n0N  # number of nodes in the central graph of obs 3
obs3cE = obs3n0E  # number of edges in the central graph of obs 3

# Batch 1: 6+7=13 nodes, 6+10=16 edges
obs_batch1 = ObservationBatch.from_obs_list([obs1, obs2])

# Batch 2: 7+6=13 nodes, 10+6=16 edges
obs_batch2 = ObservationBatch.from_obs_list([obs2, obs1])

# Batch 3: # 6+7+6=19 nodes, 6+10+6=22 edges
obs_batch3 = ObservationBatch.from_obs_list([obs1, obs2, obs1])

# Batch 4: 6+7+12=25 nodes, 6+10+24=40 edges
obs_batch4 = ObservationBatch.from_obs_list([obs1, obs2, obs3])
