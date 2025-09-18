# rl_tutor/utils.py

import networkx as nx
import matplotlib.pyplot as plt
from . import config

def create_concept_graph():
    """Creates the directed graph of topic dependencies."""
    graph = nx.DiGraph()
    graph.add_nodes_from(config.TOPICS)
    graph.add_edges_from(config.DEPENDENCIES)
    return graph

def visualize_graph(graph):
    """Saves a visualization of the concept graph."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, k=0.9, iterations=50, seed=42)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue',
            font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    plt.title("Rocket Propulsion Concept Graph", size=15)
    plt.savefig(config.GRAPH_IMAGE_PATH)
    plt.close()
    print(f"Concept graph saved to {config.GRAPH_IMAGE_PATH}")

def visualize_inference_path(graph, question_counts):
    """Saves a heatmap visualization of the topics asked during inference."""
    if not question_counts:
        print("No questions were asked; skipping inference graph generation.")
        return

    max_count = max(question_counts.values())
    node_colors = [question_counts.get(node, 0) for node in graph.nodes()]

    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(graph, k=0.9, iterations=50, seed=42)
    
    cmap = plt.cm.YlOrRd
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=cmap, 
                                   node_size=3500, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, alpha=0.5, arrowsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Number of Questions Asked', rotation=270, labelpad=20)

    ax.set_title("Inference Path Heatmap on Concept Graph", size=20)
    fig.tight_layout()
    plt.savefig(config.INFERENCE_GRAPH_PATH)
    plt.close()

