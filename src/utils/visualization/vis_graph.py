from typing import Dict, Tuple

import math
import os
import tempfile
from io import BytesIO
import subprocess

import pydot
from torch_geometric.data import Data
from PIL import Image, ImageDraw, ImageFont

from src.utils.features import edge_datarate_idx, edge_delay_idx, edge_lu_idx, edge_queue_rel_load_idx
from src.utils.logging import LOG_CONSOLE_FUNCTION
from src.utils.visualization.vis_utils import get_node_color, hex_to_rgb_tuple, feat_to_alpha_value, angle_to_x_axis


# ====== Functions for both observation and action visualization =====


def _rescale_node_pos(node_pos: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    """
    Linearly rescales the node positions so that they fit into the range [-10, 10].
    Uses the same scaling for both x and y coordinates.
    """
    min_x = min(pos[0] for pos in node_pos.values())
    max_x = max(pos[0] for pos in node_pos.values())
    min_y = min(pos[1] for pos in node_pos.values())
    max_y = max(pos[1] for pos in node_pos.values())
    max_node_pos = 1.5 * math.sqrt(len(node_pos))  # max expected node position before scaling (cheap heuristic)

    if max_x == min_x:
        # All nodes have the same x position, scale y only

        if max_y == min_y:
            # If all nodes have the same position, return a default position
            return {node_id: (0, 0) for node_id in node_pos}

        scale_y = 2 * max_node_pos / (max_y - min_y)
        return {node_id: (0, scale_y * (pos[1] - min_y) - max_node_pos) for node_id, pos in node_pos.items()}

    elif max_y == min_y:
        # All nodes have the same y position, scale x only
        scale_x = 2 * max_node_pos / (max_x - min_x)
        return {node_id: (scale_x * (pos[0] - min_x) - max_node_pos, 0) for node_id, pos in node_pos.items()}

    # General case: scale both x and y coordinates
    scale_x = 2 * max_node_pos / (max_x - min_x)
    scale_y = 2 * max_node_pos / (max_y - min_y)
    scale = min(scale_x, scale_y)

    return {node_id: (scale * (pos[0] - min_x) - max_node_pos, scale * (pos[1] - min_y) - max_node_pos)
            for node_id, pos in node_pos.items()}


def _agent_id_to_node_label(agent_id: int):
    return f"<<FONT POINT-SIZE=\"24\"><b>{agent_id}</b></FONT>>"


def _img_to_edge_label(img_filename: str) -> str:
    return f"<<TABLE border='0'><TR><TD><IMG SRC=\"{img_filename}\"/></TD></TR></TABLE>>"


def _create_dot_vis(nodes,
                    edges,
                    vis_name: str,
                    directed: bool,
                    edge_label_dist) -> Image.Image:
    """
    Creates a DOT graph using the given information, renders it and returns a numpy RGB array.
    :return: A numpy array containing the rendered graph as an RGB-PNG image
    """
    LOG_CONSOLE_FUNCTION()

    graph = pydot.Dot(vis_name,
                      graph_type="digraph" if directed else "graph",
                      label=vis_name,
                      labelloc="t",
                      fontname="Helvetica-Bold",
                      layout="neato")

    graph.set_node_defaults(shape="circle",
                            style="filled",
                            fontname="Courier",
                            penwidth=3)

    graph.set_edge_defaults(fontname="Courier",
                            penwidth=3,
                            arrowsize=0.8,
                            color="gray60",
                            labeldistance=edge_label_dist,
                            labelangle=0)

    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge)

    p = subprocess.run(['dot', "-Tpng"],
                       stdout=subprocess.PIPE,
                       input=graph.to_string().encode('utf-8')
                       )
    return Image.open(BytesIO(p.stdout)).convert("RGB")


# ===== Observation Graph Visualization =====


def _load_to_color(load: float, free_color: str = "gray60") -> str:
    """
    Returns a color string for edges/nodes based on the load value (between 0 and 1).
    """
    if load < 0.005:
        return free_color
    elif load >= 0.995:
        return "red2"

    if load < 0.5:
        load_color = "orange2"
    elif load < 0.8:
        load_color = "OrangeRed1"
    else:
        load_color = "red2"

    return f"{load_color};{load:.2f}:{free_color}"


def _link_config_to_label_img(datarate, delay, angle) -> Image:
    """
    Renders the "<dr> @ <delay> ms" string into a rotated RGBA PIL.Image,
    cropping off any extra transparent border.
    """
    # 1) Build the text string
    if datarate >= 1e9:
        dr_str = f"{round(datarate / 1e9)} Gbps"
    elif datarate >= 1e6:
        dr_str = f"{round(datarate / 1e6)} Mbps"
    elif datarate >= 1e3:
        dr_str = f"{round(datarate / 1e3)} Kbps"
    else:
        dr_str = f"{datarate} bps"
    text = f"{dr_str} @ {round(delay)} ms"

    # 2) Try a real monospace TTF; fall back to PIL default.
    for face in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Courier.ttf"):
        try:
            font = ImageFont.truetype(face, 12)
            break
        except IOError:
            font = None
    if font is None:
        font = ImageFont.load_default()

    # 3) Measure the text
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0

    # 4) Render it with a tiny padding
    pad = 2
    img = Image.new("RGBA", (w + pad*2, h + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), text, font=font, fill="blue")

    # 5) Rotate, expand, then crop off the empty border
    rot = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    bbox = rot.getbbox()
    if bbox:
        rot = rot.crop(bbox)

    return rot


def offset_points(p1, p2, offset):
    """
    Given two 2D points p1 and p2, return two points p1o and p2o that
    lie `offset` units from p1 toward p2, and from p2 toward p1, respectively.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Vector from p1 to p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        raise ValueError("p1 and p2 must be different points")

    # Unit direction vector
    ux, uy = dx / dist, dy / dist

    # Offset points
    p1o = (x1 + ux * offset, y1 + uy * offset)
    p2o = (x2 - ux * offset, y2 - uy * offset)

    return p1o, p2o


def square_scaling(angle):
    """
    Returns: A scaling factor for the square node size based on the angle. For square nodes, graphviz treats
    width/height as values for the non-rotated bounding box, so we need to scale the actual square size so that
    rotation does not change the size of the square.
    """
    return abs(math.cos(angle * math.pi / 180)) + abs(math.sin(angle * math.pi / 180))


def visualize_obs_graph(t: int, obs_graph: Data, graph_node_pos: dict) -> Image.Image:
    """
    Given a Pytorch geometric observation graph, assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given Pytorch geometric
    observation graph as an RGB-PNG image.
    """
    LOG_CONSOLE_FUNCTION()
    nodes, edges = [], []
    rescaled_node_pos = _rescale_node_pos(graph_node_pos)

    # nodes with agent id
    for i, agent_id in enumerate(obs_graph.agent_ids.tolist()):
        i_x, i_y = rescaled_node_pos[i]  # uses INNER node index
        node = pydot.Node(str(i),  # uses INNER node index
                          label=_agent_id_to_node_label(agent_id),  # uses OUTER agent index
                          fillcolor='white',
                          pos=f"{i_x},{i_y}!"
                          )
        nodes.append(node)

    # edges with features and queue load indicators
    with tempfile.TemporaryDirectory() as td:

        source_idx, dest_idx = obs_graph.edge_index.tolist()
        datarates = obs_graph.edge_attr[:, edge_datarate_idx].tolist()
        delays = obs_graph.edge_attr[:, edge_delay_idx].tolist()
        lus = obs_graph.edge_attr[:, edge_lu_idx].tolist()
        queue_loads = obs_graph.edge_attr[:, edge_queue_rel_load_idx].tolist()

        edge_tuples = list(zip(source_idx, dest_idx))
        edge_tuple_dict = {t: i for i, t in enumerate(edge_tuples)}

        while len(edge_tuple_dict) > 0:

            (u, v), edge_id = edge_tuple_dict.popitem()
            edge_id_rev = edge_tuple_dict.pop((v, u))
            pos_u_x, pos_u_y = rescaled_node_pos[u]
            pos_v_x, pos_v_y = rescaled_node_pos[v]
            angle = angle_to_x_axis((pos_v_y - pos_u_y), (pos_v_x - pos_u_x))
            angle_rev = angle - 180

            # Create rotated square nodes next to the nodes that indicate the load of the outgoing queue.
            # square for u is moved by 3 units from u's position in the direction of v (use angle)
            square_size = 0.25
            square_u_pos, square_v_pos = offset_points((pos_u_x, pos_u_y), (pos_v_x, pos_v_y), offset=0.435)

            uv_q_name = f"{u}_queue_to_{v}"
            square_u_scale = square_scaling(angle)
            square_u = pydot.Node(name=uv_q_name,
                                  label="",
                                  shape="square",
                                  style="filled",
                                  fillcolor=_load_to_color(queue_loads[edge_id], "white"),
                                  pos=f"{square_u_pos[0]},{square_u_pos[1]}!",
                                  width=square_size * square_u_scale,
                                  height=square_size * square_u_scale,
                                  penwidth=2,
                                  orientation=angle,
                                  gradientangle=angle,
                                  )
            nodes.append(square_u)

            vu_q_name = f"{v}_queue_to_{u}"
            square_v_scale = square_scaling(angle_rev)
            square_v = pydot.Node(vu_q_name,
                                  label="",
                                  shape="square",
                                  style="filled",
                                  fillcolor=_load_to_color(queue_loads[edge_id_rev], "white"),
                                  pos=f"{square_v_pos[0]},{square_v_pos[1]}!",
                                  width=square_size * square_v_scale,
                                  height=square_size * square_v_scale,
                                  penwidth=2,
                                  orientation=angle_rev,
                                  gradientangle=angle_rev,
                                  )
            nodes.append(square_v)

            # Create edge + reverse with colors based on link utilization
            edges.append(pydot.Edge(uv_q_name, vu_q_name, color=_load_to_color(lus[edge_id])))
            edges.append(pydot.Edge(vu_q_name, uv_q_name, color=_load_to_color(lus[edge_id_rev])))

            # Create transparent nodes with rotated labels for the link configuration.
            # This is a workaround for the fact that, when supplying images as edge labels, graphviz does not
            # support multi-color edges.
            lc_label_angle = angle if pos_v_x >= pos_u_x else angle_rev
            lc_img = _link_config_to_label_img(datarates[edge_id], delays[edge_id], lc_label_angle)
            lc_img_filename = os.path.join(td, f"obs_{u}_{v}.png")
            lc_img.save(lc_img_filename)
            lc_label = _img_to_edge_label(lc_img_filename)
            link_config_node = pydot.Node(name=f"{u}_{v}_link_config",
                                          label=lc_label,
                                          fillcolor="transparent",
                                          shape="plaintext",
                                          pos=f"{(pos_u_x + pos_v_x) / 2},{(pos_u_y + pos_v_y) / 2}!",
                                          )
            nodes.append(link_config_node)

        # create graph and visualize
        dot_vis = _create_dot_vis(nodes,
                                  edges,
                                  f"observation t={t}",
                                  directed=True,
                                  edge_label_dist=6,
                                  )
    return dot_vis


# ===== Action Graph Visualization =====

arrow_H = 16
arrow_W = 12

def edge_dest_cf_to_img(colors_and_feat, angle):
    """
    auxiliary function to create an image containing the routing arrow indicators
    """
    # Create a new image with the exact required size
    image = Image.new('RGBA', (2 * arrow_W * len(colors_and_feat), arrow_H), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    for i, (color, feat) in enumerate(colors_and_feat):
        (r, g, b), a = hex_to_rgb_tuple(color), feat_to_alpha_value(feat)
        draw.polygon([(i*arrow_W, 0), (i*arrow_W, arrow_H), ((i+1)*arrow_W, arrow_H // 2)], fill=(r, g, b, a))

    # rotate
    image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))
    return image


def visualize_action(t: int, ac_graph: Data, graph_node_pos: dict) -> Image.Image:
    """
    Given a Pytorch geometric normalized action graph,
    assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given Pytorch geometric
    normalized action graph as an RGB-PNG image.
    """
    LOG_CONSOLE_FUNCTION()
    rescaled_node_pos = _rescale_node_pos(graph_node_pos)

    # nodes with id and color
    nodes = []
    for i, agent_id in enumerate(ac_graph.agent_ids.tolist()):
        i_x, i_y = rescaled_node_pos[i]  # uses INNER node index
        node = pydot.Node(str(i),  # uses INNER node index
                          label=_agent_id_to_node_label(agent_id),  # uses OUTER agent index
                          fillcolor=get_node_color(agent_id),  # uses OUTER agent index
                          pos=f"{i_x},{i_y}!"
                          )
        nodes.append(node)

    # undirected edges with arrows indicating packet routing decisions per edge
    with tempfile.TemporaryDirectory() as td:

        source_idx, dest_idx = ac_graph.edge_index.tolist()

        edge_tuples = list(zip(source_idx, dest_idx))
        edge_tuple_dict = {t: i for i, t in enumerate(edge_tuples)}

        edges = []
        while len(edge_tuple_dict) > 0:

            edge_labels = dict()

            # pop edge and create label as TAIL label of the edge
            (u, v), edge_id = edge_tuple_dict.popitem()
            pos_u_x, pos_u_y = rescaled_node_pos[u]
            pos_v_x, pos_v_y = rescaled_node_pos[v]
            angle = angle_to_x_axis((pos_v_y - pos_u_y), (pos_v_x - pos_u_x))
            edge_feat = ac_graph.edge_attr[edge_id].tolist()
            edge_dest = [i for i in range(len(edge_feat)) if i != u]
            if len(edge_dest) > 0:
                edge_dest_cf = [(get_node_color(ac_graph.agent_ids[i].item()), edge_feat[i]) for i in edge_dest]
                edge_dest_img = edge_dest_cf_to_img(edge_dest_cf, angle)
                edge_dest_img_filename = os.path.join(td, f"ac_{u}_{v}.png")
                edge_dest_img.save(edge_dest_img_filename)
                edge_labels["taillabel"] = _img_to_edge_label(edge_dest_img_filename)

            # pop reverse edge and create label as HEAD label of the edge
            edge_id_rev = edge_tuple_dict.pop((v, u))
            angle_rev = angle - 180
            edge_feat_rev = ac_graph.edge_attr[edge_id_rev].tolist()
            edge_dest_rev = [i for i in range(len(edge_feat_rev)) if i != v]
            if len(edge_dest_rev) > 0:
                edge_dest_rev_cf = [(get_node_color(ac_graph.agent_ids[i].item()), edge_feat_rev[i]) for i in edge_dest_rev]
                edge_dest_rev_img = edge_dest_cf_to_img(edge_dest_rev_cf, angle_rev)
                edge_dest_rev_img_filename = os.path.join(td, f"ac_{v}_{u}.png")
                edge_dest_rev_img.save(edge_dest_rev_img_filename)
                edge_labels["headlabel"] = _img_to_edge_label(edge_dest_rev_img_filename)

            edges.append(
                pydot.Edge(str(u), str(v), **edge_labels)
            )

        # create graph within the temporary directory's scope
        dot_vis = _create_dot_vis(nodes,
                                  edges,
                                  f"action t={t}",
                                  False,
                                  edge_label_dist=6,
                                  )
    return dot_vis
