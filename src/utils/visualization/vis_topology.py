from typing import Tuple, Literal
import io
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image


def visualize_topology(
    G: nx.Graph,
    *,
    figsize: Tuple[float, float] = (7, 7),
    dpi: int = 500,
    node_size: int = 600,
    node_font_size: int = 12,
    node_border_width: float = 1.6,
    edge_font_size: int = 9,
    edge_width: float = 2,
    edge_dash_scale: float = 10,
    node_facecolor: str = "#f2f2f2",  # light background
    node_edgecolor: str = "black",
    edge_display: Literal["label", "stroke"] = "stroke",
) -> Image.Image:
    """
    Visualize an undirected attributed NetworkX graph and return a PIL.Image.Image.

    Nodes
      - Node IDs drawn in bold inside nodes.
      - Node positions from node attribute "pos" = (x, y).

    Edges (edge_display)
      - "label": draw edge labels "{delay}ms @ {datarate/1e6}Mbps" rotated with edge direction,
                 with delay (dark blue) and datarate (dark red) if possible.
      - "stroke": no labels; linewidth scaled proportional to datarate (mean datarate -> edge_width),
                  and dashed style with dash count proportional to delay:
                  dash_count ~= 2 * delay_ms  (e.g., 5ms -> 10 dashes, 10ms -> 20, 2ms -> 4).
    """
    def _fmt_num(x: object, decimals: int = 2) -> str:
        try:
            xf = float(x)  # type: ignore[arg-type]
        except Exception:
            return str(x)
        if math.isfinite(xf) and xf.is_integer():
            return str(int(xf))
        return f"{xf:.{decimals}f}".rstrip("0").rstrip(".")

    # Build positions from node attribute "pos"
    pos = {}
    for n, data in G.nodes(data=True):
        if "pos" not in data:
            raise ValueError(f'Node {n!r} is missing required attribute "pos".')
        p = data["pos"]
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if not (isinstance(p, (tuple, list)) and len(p) == 2):
            raise ValueError(f'Node {n!r} has invalid "pos"={p!r}; expected (x, y).')
        pos[n] = (float(p[0]), float(p[1]))

    if edge_display not in ("label", "stroke"):
        raise ValueError(f"edge_display must be 'label' or 'stroke', got {edge_display!r}.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits with a small padding so strokes/labels don't get clipped
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = 0.12 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    # ---- EDGES ----
    if edge_display == "label":
        # Simple edges (uniform width), labels drawn later
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, edge_color="black")
    else:
        # stroke mode: per-edge width & dashes
        datarates = []
        for u, v, data in G.edges(data=True):
            if "datarate" not in data or "delay" not in data:
                raise ValueError(f"Edge ({u!r}, {v!r}) missing 'delay' or 'datarate' attributes.")
            datarates.append(float(data["datarate"]))
        mean_datarate = sum(datarates) / len(datarates) if datarates else 0.0

        # Need renderer + transforms to compute dash sizes in screen space
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # noqa: F841  (kept symmetrical; useful if you extend)

        for u, v, data in G.edges(data=True):
            (x1, y1) = pos[u]
            (x2, y2) = pos[v]
            dr = float(data["datarate"])
            delay_ms = float(data["delay"])

            # Line width proportional to datarate; mean datarate corresponds to `edge_width`
            if mean_datarate > 0:
                lw = edge_width * (dr / mean_datarate)
            else:
                lw = edge_width
            lw = max(0.2, lw)  # keep visible

            # Dash count proportional to delay, adjusted for datarate-dependent line width
            dash_count = max(1, int(round(edge_dash_scale * delay_ms * (dr / (mean_datarate * edge_width)))))

            # Compute edge length in points to derive dash/gap lengths so the count roughly matches
            p0 = ax.transData.transform((x1, y1))
            p1 = ax.transData.transform((x2, y2))
            edge_len_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            edge_len_pt = edge_len_px * 72.0 / float(fig.dpi)

            # One dash = "on" segment; use equal on/off so dash_count fits approximately
            # total ≈ dash_count * (on + off)  => on=off=edge_len/(2*dash_count)
            base = edge_len_pt / (2.0 * dash_count) if dash_count > 0 else edge_len_pt
            on = max(1.0, base)
            off = max(1.0, base)

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=lw,
                linestyle=(0, (on, off)),
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=1,
            )

    # ---- NODES ----
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_facecolor,
        edgecolors=node_edgecolor,
        linewidths=node_border_width,
    )

    # Node labels (bold)
    node_labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        ax=ax,
        font_size=node_font_size,
        font_weight="bold",
        font_color="black",
    )

    # ---- EDGE LABELS (label mode only) ----
    if edge_display == "label":
        # Prepare edge labels and draw rotated + color-coded segments when possible
        from matplotlib.font_manager import FontProperties

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fp = FontProperties(size=edge_font_size)

        delay_color = "darkblue"
        rate_color = "darkred"

        perp_offset = 0.03 * span  # small perpendicular offset away from edge line

        for u, v, data in G.edges(data=True):
            if "delay" not in data or "datarate" not in data:
                raise ValueError(f"Edge ({u!r}, {v!r}) missing 'delay' or 'datarate' attributes.")

            delay = data["delay"]
            datarate = data["datarate"]
            mbps = float(datarate) / 1e6

            delay_str = f"{_fmt_num(delay)}ms"
            mid_str = " @ "
            rate_str = f"{_fmt_num(mbps)}Mbps"
            full_str = f"{delay_str}{mid_str}{rate_str}"

            (x1, y1) = pos[u]
            (x2, y2) = pos[v]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)

            if length > 0:
                ux, uy = dx / length, dy / length
                angle = math.degrees(math.atan2(dy, dx))
            else:
                ux, uy = 1.0, 0.0
                angle = 0.0

            # Keep text upright and direction consistent
            if angle > 90:
                angle -= 180
                ux, uy = -ux, -uy
            elif angle < -90:
                angle += 180
                ux, uy = -ux, -uy

            # Perpendicular offset
            if length > 0:
                nxp, nyp = -dy / length, dx / length
                mx_off = mx + nxp * perp_offset
                my_off = my + nyp * perp_offset
            else:
                mx_off, my_off = mx, my

            # Single bbox for the whole label (text itself transparent)
            ax.text(
                mx_off,
                my_off,
                full_str,
                fontsize=edge_font_size,
                rotation=angle,
                rotation_mode="anchor",
                ha="center",
                va="center",
                color=(0, 0, 0, 0),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=3,
            )

            # Measure widths in pixels for accurate placement
            w_delay, _, _ = renderer.get_text_width_height_descent(delay_str, fp, ismath=False)
            w_mid, _, _ = renderer.get_text_width_height_descent(mid_str, fp, ismath=False)
            w_rate, _, _ = renderer.get_text_width_height_descent(rate_str, fp, ismath=False)
            w_total = w_delay + w_mid + w_rate

            # Pixels per 1 data unit along edge direction
            p0 = ax.transData.transform((mx_off, my_off))
            p1 = ax.transData.transform((mx_off + ux, my_off + uy))
            px_per_data = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) or 1.0

            left_px = -w_total / 2.0
            c_delay_px = left_px + w_delay / 2.0
            c_mid_px = left_px + w_delay + w_mid / 2.0
            c_rate_px = left_px + w_delay + w_mid + w_rate / 2.0

            def _place(seg: str, center_px: float, color: str) -> None:
                d = center_px / px_per_data
                ax.text(
                    mx_off + ux * d,
                    my_off + uy * d,
                    seg,
                    fontsize=edge_font_size,
                    rotation=angle,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                    color=color,
                    zorder=4,
                )

            _place(delay_str, c_delay_px, delay_color)
            _place(mid_str, c_mid_px, "black")
            _place(rate_str, c_rate_px, rate_color)

    # Render to an RGB PIL image (suitable for saving as JPG)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img
