import networkx as nx

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from copy import deepcopy

from dgu.constants import IS


@dataclass(frozen=True)
class Node:
    label: str


@dataclass(frozen=True)
class IsDstNode(Node):
    src_label: str


@dataclass(frozen=True)
class ExitNode(Node):
    rel_label: str
    dst_label: str


def process_add_triplet_cmd(
    graph: nx.DiGraph,
    timestamp: int,
    src_label: str,
    rel_label: str,
    dst_label: str,
) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    graph = deepcopy(graph)
    events: List[Dict[str, Any]] = []
    # take care of the source node
    src_node: Node
    if src_label == "exit":
        # if the source is an exit, we check if the destination has an exit
        # for the given relation, and if not, add a new node.
        src_node = ExitNode(src_label, rel_label, dst_label)
        if src_node not in graph:
            graph.add_node(src_node)
            events.append({"type": "node-add", "label": src_label})
    else:
        # if it's a regular node, check if a node with the same label exists
        # and add if it doesn't.
        src_node = Node(src_label)
        if src_node not in graph:
            graph.add_node(src_node)
            events.append({"type": "node-add", "label": src_label})

    # take care of the destination node
    dst_node: Node
    if rel_label == IS:
        # if the relation is "is", check if the destination node with the
        # given label exists for the source node with the given label,
        # and add if it doesn't.
        dst_node = IsDstNode(dst_label, src_label)
        if dst_node not in graph:
            graph.add_node(dst_node)
            events.append({"type": "node-add", "label": dst_label})
    else:
        # if it's a regular node, check if a node with the same label exists
        # and add if it doesn't.
        dst_node = Node(dst_label)
        if dst_node not in graph:
            graph.add_node(dst_node)
            events.append({"type": "node-add", "label": dst_label})

    if graph.has_edge(src_node, dst_node):
        # the edge already exists, so we're done
        return events, graph
    # the edge doesn't exist, so add it
    graph.add_edge(src_node, dst_node, label=rel_label, last_update=timestamp)
    node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
    events.append(
        {
            "type": "edge-add",
            "src_id": node_id_map[src_node],
            "dst_id": node_id_map[dst_node],
            "label": rel_label,
        }
    )
    return events, graph


def process_delete_triplet_cmd(
    graph: nx.DiGraph,
    timestamp: int,
    src_label: str,
    rel_label: str,
    dst_label: str,
) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    graph = deepcopy(graph)
    events: List[Dict[str, Any]] = []
    # get the source node
    src_node: Node
    if src_label == "exit":
        src_node = ExitNode(src_label, rel_label, dst_label)
    else:
        src_node = Node(src_label)
    if src_node not in graph:
        # the source node doesn't exist, so return
        return events, graph

    # get the destination node
    dst_node: Node
    if rel_label == IS:
        dst_node = IsDstNode(dst_label, src_label)
    else:
        dst_node = Node(dst_label)
    if dst_node not in graph:
        # the destination node doesn't exist, so return
        return events, graph

    if not graph.has_edge(src_node, dst_node):
        # the edge doesn't exist, so return
        return events, graph

    # delete the edge and add event
    node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
    edge_label = graph.edges[src_node, dst_node]["label"]
    graph.remove_edge(src_node, dst_node)
    events.append(
        {
            "type": "edge-delete",
            "src_id": node_id_map[src_node],
            "dst_id": node_id_map[dst_node],
            "label": edge_label,
        }
    )

    # if there are no edges, delete the nodes
    if graph.degree[dst_node] == 0:
        node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
        graph.remove_node(dst_node)
        events.append(
            {
                "type": "node-delete",
                "node_id": node_id_map[dst_node],
                "label": dst_label,
            }
        )
    if graph.degree[src_node] == 0:
        node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
        graph.remove_node(src_node)
        events.append(
            {
                "type": "node-delete",
                "node_id": node_id_map[src_node],
                "label": src_label,
            }
        )
    return events, graph


def process_triplet_cmd(
    graph: nx.DiGraph, timestamp: int, cmd: str
) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    """
    Update the internal graph based on the given triplet command and return
    corresponding graph events.

    There are four event types: node addtion/deletion and edge addition/deletion.
    Each node event contains the following information:
        {
            "type": "node-{add,delete}",
            "node_id": id for node to be added/deleted from the previous graph,
            "label": label for node to be added/deleted,
            "before_graph": snapshot of the graph before this event,
            "after_graph": snapshot of the graph after this event,
        }
    Each edge event contains the following information:
        {
            "type": "edge-{add,delete}",
            "src_id": id for src node to be added/deleted from the previous graph,
            "dst_id": id for dst node to be added/deleted from the previous graph,
            "label": label for edge to be added/deleted,
            "before_graph": snapshot of the graph before this event,
            "after_graph": snapshot of the graph after this event,
        }
    """
    cmd_type, src_label, dst_label, rel_label = cmd.split(" , ")
    # multi-word relations are joined by underscores, so replace them with spaces
    # so that it's easier to calculate word embeddings down the line.
    rel_label = rel_label.replace("_", " ")
    if cmd_type == "add":
        return process_add_triplet_cmd(
            graph, timestamp, src_label, rel_label.replace("_", " "), dst_label
        )
    elif cmd_type == "delete":
        return process_delete_triplet_cmd(
            graph, timestamp, src_label, rel_label.replace("_", " "), dst_label
        )
    raise ValueError(f"Unknown command {cmd}")
