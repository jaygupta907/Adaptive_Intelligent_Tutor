# rl_tutor/utils.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from . import config

def create_concept_graph():
    """Builds and returns the NetworkX directed graph of topics."""
    graph = nx.DiGraph()
    graph.add_nodes_from(config.TOPICS)
    graph.add_edges_from(config.DEPENDENCIES)
    return graph

def visualize_graph(graph):
    """Saves a visualization of the concept graph to a file."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=0.9)
    nx.draw(graph, pos, with_labels=True, node_size=3500, node_color='skyblue', 
            font_size=9, font_weight='bold', arrows=True, arrowsize=20, 
            width=1.5, edge_color='gray')
    plt.title("Rocket Propulsion Concept Graph", size=16)
    
    plt.savefig(config.GRAPH_IMAGE_PATH)
    plt.close()
    print(f"Concept graph saved to {config.GRAPH_IMAGE_PATH}")

def visualize_inference_path(graph, question_counts):
    """Saves a visualization of the inference path with node colors based on question frequency."""
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=0.9)

    # Determine node colors based on how many times a question was asked
    max_count = max(question_counts.values()) if any(question_counts.values()) else 1
    node_colors = []
    for node in graph.nodes():
        count = question_counts.get(node, 0)
        # Use a log scale to better differentiate low counts
        if count > 0:
            color_value = np.log1p(count) / np.log1p(max_count)
        else:
            color_value = 0.0  # Default color for unasked topics
        node_colors.append(color_value)

    cmap = cm.get_cmap('YlOrRd')  # Yellow -> Orange -> Red colormap

    nx.draw(graph, pos, ax=ax, with_labels=True, node_size=3500,
            node_color=node_colors, cmap=cmap,
            font_size=9, font_weight='bold', arrows=True, arrowsize=20,
            width=1.5, edge_color='gray', vmin=0, vmax=1)

    # Add a color bar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, ticks=range(max_count + 1))
    cbar.set_label('Number of Questions Asked', rotation=270, labelpad=20)

    ax.set_title("Inference Path Heatmap on Concept Graph", size=16)
    plt.savefig(config.INFERENCE_GRAPH_PATH)
    plt.close(fig)
    print(f"Inference path graph saved to {config.INFERENCE_GRAPH_PATH}")

