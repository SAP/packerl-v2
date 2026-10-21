import networkx as nx
from src.scenario.config import BaseConfig


DOT_TRAFFIC_SCALING = {
    "dot_b4": 12
}

class StaticDotConfig(BaseConfig):
    """
    Configuration for loading a static topology from a dot file.
    """
    dot_file: str  # Path to the dot file


def convert_pos_into_tuple(pos_str: str) -> (float, float):
    """
    Converts a position string from the dot file into a tuple of floats.
    Example inputs: "27,18!", '"27,18!"'
    Example output: (27.0, 18.0)
    """
    # Remove any surrounding quotes and exclamation marks
    pos_str = pos_str.strip('"').strip('!')
    # Split by comma and convert to float
    x_str, y_str = pos_str.split(',')
    return float(x_str), float(y_str)


def generate_static_dot(num_nodes, rng, rng_i, dot_file, **kwargs):
    """
    Loads a static topology from a dot file using networkx.read_dot.
    """
    # Create undirected nx.Graph from dot file
    G = nx.Graph(nx.nx_pydot.read_dot(dot_file))

    # Convert node labels to integers
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})

    # Convert global graph attributes from strings to integers
    custom_graph_attrs = G.graph.pop('graph', {})
    for k, v in custom_graph_attrs.items():
        G.graph[k] = float(v.strip('"'))

    # Convert nodes' position attributes from strings to tuples (if present)
    for n, data in G.nodes(data=True):
        if 'pos' in data:
            data['pos'] = convert_pos_into_tuple(data['pos'])

    # Convert edges' 'datarate' and 'delay' attributes from strings to integers (if present)
    for u, v, data in G.edges(data=True):
        if 'datarate' in data:
            data['datarate'] = int(data['datarate'])
        if 'delay' in data:
            data['delay'] = int(data['delay'])

    # Set graph name based on dot file name
    dot_file_stem = dot_file.split('/')[-1].split('.')[0]
    G.graph['name'] = f'dot_{dot_file_stem}'

    return G
