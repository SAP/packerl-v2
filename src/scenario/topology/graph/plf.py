from networkx import Graph
import numpy as np

from src.scenario.config import BaseConfig, SingleOrChoice


class PLFConfig(BaseConfig):
    """
    Configuration for graphs of the Pyramid Lattice Funnel (PLF) family.
    """
    inner_depth: SingleOrChoice[int]
    bisection_bandwidth: SingleOrChoice[int]
    flat_delay_ms: SingleOrChoice[int]


def pascal_triangle_np(n: int):
    rows = [np.array([1], dtype=float)]
    for _ in range(n):
        rows.append(np.convolve(rows[-1], [1, 1]).astype(float))
    return rows


def generate_plf(inner_depth: int, bisection_bandwidth: int, flat_delay_ms: int, **kwargs):
    """
    Generate a Pyramid Lattice Funnel (PLF) graph as follows:
    - Source node 0.
    - Inner layers of depth `inner_depth`, with layer i having i+1 nodes,
       connected to the two closest nodes in layer i-1 except for nodes at the outer edges of the layer
       (like a Pascal's triangle/pyramid).
    - Sink node 1 connected to all nodes in the last inner layer.
    - All nodes have a 2-D Euclidean position. We create a linear spacing between -1 and 1 for the x-coordinates
      of each layer (meaning that we place the source at (-1, 0) and the sink at (1, 0)), and create y-axis linspaces
      for each layer such that the largest layer has y-coordinates between -1 and 1, and all other layers are scaled
      accordingly.
    - Uniform link delay `flat_delay_ms` for all edges.
    - Datarates for edges between layers are proportional to the values in Pascal's triangle (which correspond to the
      number of paths through each node from the start), and are further normalized and scaled such that
      the total datarate between any two layers is `bisection_bandwidth`.
    """
    if inner_depth < 1 or inner_depth > 16:
        raise ValueError("inner_depth must be in [1, 16]")
    g = Graph()
    g.add_node(0, pos=np.array([-1.0, 0.0]), active=True)  # Source node
    g.add_node(1, pos=np.array([1.0, 0.0]), active=True)   # Sink node

    # Add inner layers
    pascal_triangle = pascal_triangle_np(inner_depth)
    cur_node_idx = 2  # Start indexing inner nodes from 2
    prev_layer = [0]  # Start with the source node as the previous layer
    x_pos_linspace = np.linspace(-1, 1, inner_depth + 2)  # x-positions for each layer
    for layer in range(1, inner_depth + 1):   # inner layers 1 .. inner_depth, each with layer+1 nodes
        cur_layer = []
        y_range = layer / inner_depth  # Scale y-range based on layer depth
        y_pos_linspace = np.linspace(y_range, -y_range, layer + 1)  # y-positions for nodes in the current layer
        layer_dr_factors = np.repeat(pascal_triangle[layer-1], 2)
        layer_dr_factors /= np.sum(layer_dr_factors)
        layer_datarates = layer_dr_factors * bisection_bandwidth
        for i in range(layer + 1):
            # add node with calculated position as "passive" (no source/sink, just intermediate)
            g.add_node(cur_node_idx, pos=np.array([x_pos_linspace[layer], y_pos_linspace[i]]), active=False)
            if i == 0:
                # Connect leftmost node to the leftmost node of the previous layer
                g.add_edge(prev_layer[0], cur_node_idx, datarate=layer_datarates[0], delay=flat_delay_ms)
            elif i == layer:
                # Connect rightmost node to the rightmost node of the previous layer
                g.add_edge(prev_layer[-1], cur_node_idx, datarate=layer_datarates[-1], delay=flat_delay_ms)
            else:
                # Connect to the two closest nodes in the previous layer
                g.add_edge(prev_layer[i - 1], cur_node_idx, datarate=layer_datarates[2*i-1], delay=flat_delay_ms)
                g.add_edge(prev_layer[i], cur_node_idx, datarate=layer_datarates[2*i], delay=flat_delay_ms)
            cur_layer.append(cur_node_idx)
            cur_node_idx += 1
        prev_layer = cur_layer

    # add edges from all nodes of the last inner layer to the sink node. Datarates are proportional to the last row of pascal's triangle
    last_layer_datarates = (pascal_triangle[-1] / np.sum(pascal_triangle[-1])) * bisection_bandwidth
    for node, dr in zip(prev_layer, last_layer_datarates):
        g.add_edge(node, 1, datarate=dr, delay=flat_delay_ms)

    return g
