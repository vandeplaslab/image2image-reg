"""Graph visuals."""

from __future__ import annotations

import typing as ty

import matplotlib.pyplot as plt
import networkx as nx
from koyo.color import get_next_color
from matplotlib.patches import ArrowStyle

from image2image_reg.enums import NetworkTypes

if ty.TYPE_CHECKING:
    from image2image_reg.workflows import ElastixReg


def draw_registration_nodes(
    workflow: ElastixReg,
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
        color = get_next_color(idx)
        g.add_node(modality, color=color, cstyle=f"arc3,rad=-{(idx + 1) * 0.08 + 0.05}")

    for src, tgts in workflow.registration_paths.items():
        g.add_edge(src, tgts[0])

    pos = nx.kamada_kawai_layout(g)
    pos = {p: v * 20 for p, v in pos.items()}
    nx.draw_networkx_nodes(
        g,
        pos,
        label=g.nodes,
        node_color=[g.nodes[n]["color"] for n in g.nodes],
        node_size=node_size,
        alpha=node_alpha,
    )

    # add text to the nodes
    for src, tgts in workflow.registration_paths.items():
        src_color = g.nodes[src]["color"]
        cstyle = g.nodes[src]["cstyle"]

        ax.annotate(
            "",
            xy=pos[src],
            xycoords="data",
            xytext=pos[tgts[0]],
            textcoords="data",
            arrowprops={
                "arrowstyle": "<-",
                "color": src_color,
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "mutation_scale": 10,
                "connectionstyle": cstyle,
                "linewidth": 1.5,
            },
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
                    arrowprops={
                        "arrowstyle": "<-",
                        "color": src_color,
                        "shrinkA": 5,
                        "shrinkB": 5,
                        "patchA": None,
                        "patchB": None,
                        "mutation_aspect": 2,
                        "connectionstyle": cstyle,
                        "linewidth": 1,
                    },
                )
    nx.draw_networkx_labels(g, pos)
    return fig, ax


def draw_workflow(
    workflow: ElastixReg,
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

    ax.clear()
    ax.axis("off")
    ax.set_title("")
    ax.margins(x=0.2, y=0.2)
    g = nx.DiGraph()
    g_layout = nx.DiGraph()

    for idx, modality in enumerate(workflow.modalities):
        rad = (idx + 1) * 0.08 + 0.05
        color = get_next_color(idx)

        g.add_node(modality, color=color, cstyle=f"arc3,rad=-{rad}")
        g_layout.add_node(modality, color=color, cstyle=f"arc3,rad=-{rad}")

    legend = {}
    for attach_to, attached_name in workflow.attachment_images.items():
        g.add_node(attach_to, color="green")
        g_layout.add_edge(attach_to, attached_name)
        legend["attached image"] = {"markerfacecolor": "green", "marker": "o"}

    for shape_set_name, shape_info in workflow.attachment_shapes.items():
        g.add_node(shape_set_name, color="purple")
        g_layout.add_edge(shape_set_name, shape_info["attach_to"])
        legend["attached shape"] = {"markerfacecolor": "purple", "marker": "^"}

    for point_set_name, point_info in workflow.attachment_points.items():
        g.add_node(point_set_name, color="yellow")
        g_layout.add_edge(point_set_name, point_info["attach_to"])
        legend["attached points"] = {"markerfacecolor": "yellow", "marker": "o"}

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

    nx.draw_networkx_nodes(
        g,
        pos,
        label=g.nodes,
        node_color=[g.nodes[n]["color"] for n in g.nodes],
        node_size=node_size,
        alpha=node_alpha,
        ax=ax,
        margins=(0.2, 0.2),
    )
    ax.set_clip_on(False)

    for attach_to, attached_name in workflow.attachment_images.items():
        ax.annotate(
            "",
            xy=pos[attach_to],
            xycoords="data",
            xytext=pos[attached_name],
            textcoords="data",
            arrowprops={
                "arrowstyle": ArrowStyle.BracketA(widthA=0.5),
                "color": "white",
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "mutation_scale": 10,
                "linewidth": 1.5,
            },
            annotation_clip=False,
        )

    for shape_set_name, shape_info in workflow.attachment_shapes.items():
        ax.annotate(
            "",
            xy=pos[shape_set_name],
            xycoords="data",
            xytext=pos[shape_info["attach_to"]],
            textcoords="data",
            arrowprops={
                "arrowstyle": ArrowStyle.BracketA(widthA=0.5),
                "color": "white",
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "mutation_scale": 10,
                "linewidth": 1.5,
            },
            annotation_clip=False,
        )

    for shape_set_name, shape_info in workflow.attachment_points.items():
        ax.annotate(
            "",
            xy=pos[shape_set_name],
            xycoords="data",
            xytext=pos[shape_info["attach_to"]],
            textcoords="data",
            arrowprops={
                "arrowstyle": ArrowStyle.BracketA(widthA=0.5),
                "color": "white",
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "mutation_scale": 10,
                "linewidth": 1.5,
            },
            annotation_clip=False,
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
            arrowprops={
                "arrowstyle": "<-",
                "color": src_color,
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "mutation_scale": 10,
                "connectionstyle": cstyle,
                "linewidth": 1.5,
            },
            annotation_clip=False,
        )
        path_targets = nx.algorithms.shortest_path(g, source, targets[-1])[1:]

        legend["direct modality"] = {"color": "white", "ls": "solid"}
        if len(path_targets) > 1:
            legend["through modality"] = {"color": "white", "ls": "dotted"}
            for idx, cont_src in enumerate(path_targets[:-1]):
                current, target = cont_src, path_targets[idx + 1]
                ax.annotate(
                    "",
                    xy=pos[current],
                    xycoords="data",
                    xytext=pos[target],
                    textcoords="data",
                    arrowprops={
                        "arrowstyle": "<-",
                        "color": src_color,
                        "shrinkA": 5,
                        "shrinkB": 5,
                        "patchA": None,
                        "patchB": None,
                        "mutation_aspect": 2,
                        "connectionstyle": cstyle,
                        "linewidth": 1,
                        "linestyle": "dotted",
                    },
                    annotation_clip=False,
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

    # add legend for node colors
    if legend:
        names, handles = [], []
        for name, kws in legend.items():
            kws.setdefault("color", "white")
            handles.append(plt.Line2D([0], [0], markersize=10, **kws))
            names.append(name)

        ax.legend(
            handles=handles,
            labels=names,
            loc="upper left",
            fontsize="small",
            labelcolor="white",
            frameon=False,
        )
