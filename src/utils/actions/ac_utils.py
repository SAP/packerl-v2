import torch
from torch import Tensor


def sp_actions_to_ascii(sp_actions: Tensor,
                        num_nodes: int,
                        edge_agent_index: Tensor,
                        randomly_selected_self_idx: Tensor = None) -> str:
    """
    Convert the shortest path actions to an ASCII representation for debugging purposes.

    Args:
        sp_actions: Tensor containing the shortest path actions,
        num_nodes: int, number of nodes in the graph,
        edge_agent_index: Tensor of shape [2, E] containing the edge agent indices,
        randomly_selected_self_idx: Tensor of shape [N,] containing the indices of randomly selected self-destinations.
    """

    if num_nodes > 999:
        raise ValueError("The number of nodes is too large to print nicely.")

    num_edges = edge_agent_index.shape[1]

    sp_actions_print = sp_actions.reshape(num_edges, num_nodes).cpu()
    randomly_selected_in_block = torch.zeros_like(sp_actions_print, dtype=torch.int8)
    if randomly_selected_self_idx is not None:
        randomly_selected_in_block[randomly_selected_self_idx, torch.arange(num_nodes)] = 1

    def to_print_char(x, r):
        if x == 1 and r == 1:
            print_char = "E"
        elif r == 1:
            print_char = "r"
        elif x == 1:
            print_char = "*"
        else:
            print_char = " "
        return print_char.rjust(4)

    def edge_str(src, dst):
        src_str = str(src).rjust(4)
        dst_str = str(dst).rjust(4)
        return f"{src_str}->{dst_str}"  # 4 + 2 + 4 = 10 characters

    def node_str():
        node_str = ""
        for i in range(num_nodes):
            node_str += f"{i:>4}"
        return node_str

    # create table formatting blocks
    v_left = "┃ "
    v_mid = " ┃ "
    v_right = " ┃"

    edge_col_size = len(edge_str(0, 0))
    node_col_size = num_nodes * 4

    top_line = "  " + " " * edge_col_size + " ┏━" + "━" * node_col_size + "━┓"
    mid_line = "┏━" + "━" * edge_col_size + "━╋━" + "━" * node_col_size + "━┫"
    bot_line = "┗━" + "━" * edge_col_size + "━┻━" + "━" * node_col_size + "━┛"

    # create header lines
    first_line = f"sp actions for graph with {num_nodes} nodes and {num_edges} edges"
    underline = "=" * len(first_line)
    explanation = " *: edge selected\n r: random self-destination selected\n E: both selected + random self-destination selected (ERROR)"

    # create table lines
    node_line = "  " + "↓edge/dst→".rjust(edge_col_size) + v_mid + node_str() + v_right
    print_lines = ["", first_line, underline, "", explanation, "", top_line, node_line, mid_line]
    for i in range(num_edges):
        src = edge_agent_index[0][i].item()
        dst = edge_agent_index[1][i].item()
        cur_line = v_left
        cur_line += edge_str(src, dst)
        cur_line += v_mid
        for x, r in zip(sp_actions_print[i], randomly_selected_in_block[i]):
            cur_line += to_print_char(x, r)
        cur_line += v_right
        print_lines.append(cur_line)
    print_lines.append(bot_line)
    print_lines.append("\n")

    return "\n".join(print_lines)
