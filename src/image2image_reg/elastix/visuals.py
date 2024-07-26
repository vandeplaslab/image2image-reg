"""Graph visuals."""

from __future__ import annotations

import typing as ty

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import ArrowStyle

from image2image_reg.enums import NetworkTypes

if ty.TYPE_CHECKING:
    from image2image_reg.workflows import IWsiReg

COLOR_MAP = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]


def draw_registration_nodes(
    workflow: IWsiReg,
    ax: plt.Axes | None,
    node_size: float = 100,
    node_alpha: float = 0.75,
) -> tuple[plt.Figure, plt.Axes]:
    """Create graph.

    Taken from:
    NHPatterson/napari-wsireg/src/napari_wsireg/data/utils/graph.py
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        # fig = ax.get_figure()
        fig = None

    # ax.use_sticky_edges = True
    ax.clear()
    ax.axis("off")

    # Add nodes
    g = nx.DiGraph()
    for idx, modality in enumerate(workflow.modalities):
        if idx > 11:
            color = COLOR_MAP[idx - 11]
        else:
            color = COLOR_MAP[idx]
        g.add_node(modality, color=color, cstyle=f"arc3,rad=-{(idx+1)*0.08+0.05}")

    for src, tgts in workflow.registration_paths.items():
        g.add_edge(src, tgts[0])

    pos = nx.kamada_kawai_layout(g)
    pos = {p: v * 20 for p, v in pos.items()}
    nx.draw_networkx_nodes(
        g, pos, label=g.nodes, node_color=[g.nodes[n]["color"] for n in g.nodes], node_size=node_size, alpha=node_alpha
    )

    for src, tgts in workflow.registration_paths.items():
        src_color = g.nodes[src]["color"]
        cstyle = g.nodes[src]["cstyle"]

        ax.annotate(
            "",
            xy=pos[src],
            xycoords="data",
            xytext=pos[tgts[0]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<-",
                color=src_color,
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                mutation_scale=10,
                connectionstyle=cstyle,
                linewidth=1.5,
            ),
        )
        path_targets = nx.algorithms.shortest_path(g, src, tgts[-1])[1:]

        if len(path_targets) > 1:
            for idx, cont_src in enumerate(path_targets[:-1]):
                current, target = cont_src, path_targets[idx + 1]
                ax.annotate(
                    "",
                    xy=pos[current],
                    xycoords="data",
                    xytext=pos[target],
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="<-",
                        color=src_color,
                        shrinkA=5,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        mutation_aspect=2,
                        connectionstyle=cstyle,
                        linewidth=1,
                    ),
                )
    nx.draw_networkx_labels(g, pos)
    return fig, ax


def draw_workflow(
    workflow: IWsiReg,
    ax: plt.Axes | None = None,
    layout: str | NetworkTypes = "kk",
    node_size: float = 100,
    node_alpha: float = 0.75,
):
    """Draw entire workflow, including attachments."""

    def _nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.clear()
    ax.axis("off")
    ax.set_title("")
    ax.margins(x=0.2, y=0.2)
    g = nx.DiGraph()
    g_layout = nx.DiGraph()

    for idx, modality in enumerate(workflow.modalities):
        rad = (idx + 1) * 0.08 + 0.05
        if idx > 11:
            color = COLOR_MAP[idx - 11]
        else:
            color = COLOR_MAP[idx]

        g.add_node(modality, color=color, cstyle=f"arc3,rad=-{rad}")
        g_layout.add_node(modality, color=color, cstyle=f"arc3,rad=-{rad}")

    for attachment_name, attachment_mod in workflow.attachment_images.items():
        g.add_node(attachment_name, color="green")
        g_layout.add_edge(attachment_name, attachment_mod)

    for shape_set_name, shape_info in workflow.attachment_shapes.items():
        g.add_node(shape_set_name, color="red")
        g_layout.add_edge(shape_set_name, shape_info["attach_to"])

    for shape_set_name, shape_info in workflow.attachment_points.items():
        g.add_node(shape_set_name, color="yellow")
        g_layout.add_edge(shape_set_name, shape_info["attach_to"])

    for source, targets in workflow.registration_paths.items():
        g.add_edge(source, targets[0])
        g_layout.add_edge(source, targets[0])
        for idx, cont_src in enumerate(targets[:-1]):
            current, target = cont_src, targets[idx + 1]
            g_layout.add_edge(current, target)

    if layout == "random":
        pos = nx.random_layout(g_layout)
    elif layout == "kk":
        pos = nx.kamada_kawai_layout(g_layout)
    elif layout == "planar":
        pos = nx.planar_layout(g_layout)
    elif layout == "spring":
        pos = nx.spring_layout(g_layout)
    elif layout == "spectral":
        pos = nx.spectral_layout(g_layout)
    elif layout == "shell":
        pos = nx.shell_layout(g_layout)
    elif layout == "circular":
        pos = nx.circular_layout(g_layout)
    elif layout == "spiral":
        pos = nx.spiral_layout(g_layout)
    elif layout == "arf":
        pos = nx.arf_layout(g_layout)
    else:
        raise ValueError(f"Invalid layout: {layout}")

    # pos = {p: v * 20 for p, v in pos.items()}
    nx.draw_networkx_nodes(
        g,
        pos,
        label=g.nodes,
        node_color=[g.nodes[n]["color"] for n in g.nodes],
        node_size=node_size,
        alpha=node_alpha,
        ax=ax,
    )

    for attachment_name, attachment_mod in workflow.attachment_images.items():
        ax.annotate(
            "",
            xy=pos[attachment_name],
            xycoords="data",
            xytext=pos[attachment_mod],
            textcoords="data",
            arrowprops=dict(
                arrowstyle=ArrowStyle.BracketA(widthA=0.5),
                color="white",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                mutation_scale=10,
                linewidth=1.5,
            ),
        )

    for shape_set_name, shape_info in workflow.attachment_shapes.items():
        ax.annotate(
            "",
            xy=pos[shape_set_name],
            xycoords="data",
            xytext=pos[shape_info["attach_to"]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle=ArrowStyle.BracketA(widthA=0.5),
                color="white",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                mutation_scale=10,
                linewidth=1.5,
            ),
        )

    for shape_set_name, shape_info in workflow.attachment_points.items():
        ax.annotate(
            "",
            xy=pos[shape_set_name],
            xycoords="data",
            xytext=pos[shape_info["attach_to"]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle=ArrowStyle.BracketA(widthA=0.5),
                color="white",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                mutation_scale=10,
                linewidth=1.5,
            ),
        )

    for source, targets in workflow.registration_paths.items():
        src_color = g.nodes[source]["color"]
        cstyle = g.nodes[source]["cstyle"]

        ax.annotate(
            "",
            xy=pos[source],
            xycoords="data",
            xytext=pos[targets[0]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<-",
                color=src_color,
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                mutation_scale=10,
                connectionstyle=cstyle,
                linewidth=1.5,
            ),
        )
        path_targets = nx.algorithms.shortest_path(g, source, targets[-1])[1:]

        if len(path_targets) > 1:
            for idx, cont_src in enumerate(path_targets[:-1]):
                current, target = cont_src, path_targets[idx + 1]
                ax.annotate(
                    "",
                    xy=pos[current],
                    xycoords="data",
                    xytext=pos[target],
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="<-",
                        color=src_color,
                        shrinkA=5,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        mutation_aspect=2,
                        connectionstyle=cstyle,
                        linewidth=1,
                        linestyle="dotted",
                    ),
                )

    if len(pos.keys()) > 2:
        pos_labels = _nudge(pos, 0, -0.075)
    else:
        pos_labels = pos

    nx.draw_networkx_labels(
        g,
        pos_labels,
        ax=ax,
        font_color="white",
        font_size=9.5,
    )
