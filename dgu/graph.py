import networkx as nx

from dataclasses import dataclass
from typing import Dict, List, Any

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from dgu.constants import IS
from dgu.nn.utils import calculate_node_id_offsets


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
) -> List[Dict[str, Any]]:
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
        return events
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
    return events


def process_delete_triplet_cmd(
    graph: nx.DiGraph,
    src_label: str,
    rel_label: str,
    dst_label: str,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    # get the source node
    src_node: Node
    if src_label == "exit":
        src_node = ExitNode(src_label, rel_label, dst_label)
    else:
        src_node = Node(src_label)
    if src_node not in graph:
        # the source node doesn't exist, so return
        return events

    # get the destination node
    dst_node: Node
    if rel_label == IS:
        dst_node = IsDstNode(dst_label, src_label)
    else:
        dst_node = Node(dst_label)
    if dst_node not in graph:
        # the destination node doesn't exist, so return
        return events

    if not graph.has_edge(src_node, dst_node):
        # the edge doesn't exist, so return
        return events

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
    return events


def process_triplet_cmd(
    graph: nx.DiGraph, timestamp: int, cmd: str
) -> List[Dict[str, Any]]:
    """
    Update the internal graph based on the given triplet command and return
    corresponding graph events.

    There are four event types: node addtion/deletion and edge addition/deletion.
    Each node event contains the following information:
        {
            "type": "node-{add,delete}",
            "node_id": id for node to be added/deleted from the previous graph,
            "label": label for node to be added/deleted,
        }
    Each edge event contains the following information:
        {
            "type": "edge-{add,delete}",
            "src_id": id for src node to be added/deleted from the previous graph,
            "dst_id": id for dst node to be added/deleted from the previous graph,
            "label": label for edge to be added/deleted,
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
            graph, src_label, rel_label.replace("_", " "), dst_label
        )
    raise ValueError(f"Unknown command {cmd}")


def data_to_networkx(data: Data, labels: List[str]) -> nx.DiGraph:
    """
    Turn torch_geometric.Data into a networkx graph. Note that there is a bug
    in to_networkx() where it turns an attribute that is a list with one
    element into a scalar, so you do need more than one node and edge for this
    to work properly. This is OK, b/c we use this to compare and a render final
    graph of a game step.
    """
    nx_graph = to_networkx(
        data,
        node_attrs=["x", "node_last_update"],
        edge_attrs=["edge_attr", "edge_last_update"],
    )
    for _, node_data in nx_graph.nodes.data():
        node_data["label"] = labels[node_data["x"]]
    for _, _, edge_data in nx_graph.edges.data():
        edge_data["label"] = labels[edge_data["edge_attr"]]
    return nx_graph


def batch_to_data_list(batch: Batch, batch_size: int) -> List[Data]:
    """
    Split the given batched graph into a list of Data. We have to implement our own
    b/c Batch.to_data_list() doesn't handle batched graphs that have not been created
    using Batch.from_data_list(), which is what we have when we use
    update_batched_graph().
    """
    data_list: List[Data] = []
    node_id_offsets = calculate_node_id_offsets(batch_size, batch.batch)
    for i in range(batch_size):
        # mask for all the nodes that belong to the i'th subgraph
        node_mask = batch.batch == i
        # mask for all the edges that belong to the i'th subgraph
        # use the source node
        edge_mask = batch.batch[batch.edge_index[0]] == i
        subgraph = Data(
            x=batch.x[node_mask],
            edge_index=batch.edge_index[:, edge_mask] - node_id_offsets[i],
            edge_attr=batch.edge_attr[edge_mask],
            node_last_update=batch.node_last_update[node_mask],
            edge_last_update=batch.edge_last_update[edge_mask],
        )
        data_list.append(subgraph)
    return data_list
