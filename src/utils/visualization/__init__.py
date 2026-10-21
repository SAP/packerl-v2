from typing import List

from PIL import Image

from .vis_action_stats import visualize_action_stats
from .vis_graph import visualize_obs_graph, visualize_action
from .vis_scenario import visualize_scenario_events
from .vis_step_stats import visualize_ep_step_stats
from .vis_topology import visualize_topology
from .vis_traffic import visualize_traffic_matrices
from .vis_utils import pad_images_to_largest, grid_image


def visualize_ep(ep_vis_material, ep_name) -> dict[str, Image.Image | List[Image.Image]]:
    """
    Logs visualizations of a single episode to disk, and W&B if desired
    """
    ep_vis = dict()

    # base topology: single PIL image with the network topology (no events, traffic, or actions)
    ep_vis[f'{ep_name}/scenario_topology'] = visualize_topology(ep_vis_material["scenario_topology"])

    # scenario events: single PIL image with one line-plot
    ep_vis[f'{ep_name}/scenario_events'] = visualize_scenario_events(ep_vis_material["scenario_events"])

    # observed traffic: list of PIL Images depicting colored matrices
    ep_vis[f'{ep_name}/traffic'] = visualize_traffic_matrices(ep_vis_material["traffic_matrices"])

    # action stats: single PIL image with a combined colored matrix+vector plot
    ep_vis[f'{ep_name}/action_stats'] = visualize_action_stats(ep_vis_material["oscillation_ratio"],
                                                               ep_vis_material["next_hop_spread"])

    # observed steps: list of PIL Images depicting:
    #  - on the left, the observed networks from a birdseye perspective,
    #  - on the right, the actions taken by the agent(s)
    # A) visualize the observation graphs
    graph_node_pos = ep_vis_material["graph_node_pos"]
    rendered_obs_graphs = [visualize_obs_graph(t, obs_graph, graph_node_pos)
                           for t, obs_graph in enumerate(ep_vis_material["obs_graphs"])]
    rendered_obs_graphs = pad_images_to_largest(rendered_obs_graphs)

    # B) visualize the actions taken by the agent(s)
    ac = ep_vis_material["actions"]
    ne = ep_vis_material["node_emb"] if "node_emb" in ep_vis_material else [graph_node_pos] * len(ac)
    rendered_action_graphs = [visualize_action(t, action, node_pos) for t, (action, node_pos) in enumerate(zip(ac, ne))]
    rendered_action_graphs = pad_images_to_largest(rendered_action_graphs)

    # C) combine the obs-action vis pairs into a single image per step
    rendered_pairs = zip(pad_images_to_largest(rendered_obs_graphs),
                         pad_images_to_largest(rendered_action_graphs))
    rendered_steps = [grid_image(pad_images_to_largest([obs, action])) for obs, action in rendered_pairs]
    ep_vis[f'{ep_name}/step'] = pad_images_to_largest(rendered_steps)

    # episode step statistics:  line-plots
    ep_vis |= visualize_ep_step_stats(ep_name, ep_vis_material['ep_step_stats'])

    return ep_vis
