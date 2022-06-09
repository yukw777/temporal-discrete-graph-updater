import networkx as nx
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Dict, List, Any, Set

from torch_geometric.data import Data, Batch

from tdgu.constants import FOOD_COLORS, IS, FOOD_KINDS
from tdgu.nn.utils import calculate_node_id_offsets


@dataclass(frozen=True)
class Node:
    label: str


@dataclass(frozen=True)
class DstNode(Node):
    src_label: str


@dataclass(frozen=True)
class ExitNode(Node):
    rel_label: str
    dst_label: str


@dataclass(frozen=True)
class FoodNameNode(Node):
    name: str


@dataclass(frozen=True)
class FoodAdjNode(Node):
    food_name_node: FoodNameNode


def process_add_triplet_cmd(
    graph: nx.DiGraph,
    timestamp: int,
    src_label: str,
    rel_label: str,
    dst_label: str,
    allow_objs_with_same_label: bool,
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
    elif allow_objs_with_same_label and src_label in FOOD_KINDS:
        # we split the node into two nodes: one for the adjective, one for the noun
        src_node = FoodNameNode(FOOD_KINDS[src_label], src_label)
        if src_node not in graph:
            graph.add_node(src_node)
            events.append({"type": "node-add", "label": src_node.label})
            adj = src_label.split()[0]
            adj_node = FoodAdjNode(adj, src_node)
            graph.add_node(adj_node)
            events.append({"type": "node-add", "label": adj})
            graph.add_edge(src_node, adj_node, label=IS, last_update=timestamp)
            node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
            events.append(
                {
                    "type": "edge-add",
                    "src_id": node_id_map[src_node],
                    "dst_id": node_id_map[adj_node],
                    "label": IS,
                }
            )
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
        dst_node = DstNode(dst_label, src_label)
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
    allow_objs_with_same_label: bool,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    # get the source node
    src_node: Node
    if src_label == "exit":
        src_node = ExitNode(src_label, rel_label, dst_label)
    elif allow_objs_with_same_label and src_label in FOOD_KINDS:
        src_node = FoodNameNode(FOOD_KINDS[src_label], src_label)
    else:
        src_node = Node(src_label)
    if src_node not in graph:
        # the source node doesn't exist, so return
        return events

    # get the destination node
    dst_node: Node
    if rel_label == IS:
        dst_node = DstNode(dst_label, src_label)
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
                "label": dst_node.label,
            }
        )
    # if the source node is FoodNameNode and it only has one degree, it's the
    # FoodAdjNode, so delete it.
    if isinstance(src_node, FoodNameNode) and graph.degree[src_node] == 1:
        adj_node = FoodAdjNode(src_label.split()[0], src_node)
        # sanity check
        assert graph.has_edge(src_node, adj_node)
        node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
        graph.remove_edge(src_node, adj_node)
        events.append(
            {
                "type": "edge-delete",
                "src_id": node_id_map[src_node],
                "dst_id": node_id_map[adj_node],
                "label": IS,
            }
        )
        # sanity check
        assert graph.degree[adj_node] == 0
        graph.remove_node(adj_node)
        events.append(
            {
                "type": "node-delete",
                "node_id": node_id_map[adj_node],
                "label": adj_node.label,
            }
        )

    if graph.degree[src_node] == 0:
        node_id_map = {node: node_id for node_id, node in enumerate(graph.nodes)}
        graph.remove_node(src_node)
        events.append(
            {
                "type": "node-delete",
                "node_id": node_id_map[src_node],
                "label": src_node.label,
            }
        )
    return events


def process_triplet_cmd(
    graph: nx.DiGraph,
    timestamp: int,
    cmd: str,
    allow_objs_with_same_label: bool = False,
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
            graph,
            timestamp,
            src_label,
            rel_label.replace("_", " "),
            dst_label,
            allow_objs_with_same_label,
        )
    elif cmd_type == "delete":
        return process_delete_triplet_cmd(
            graph,
            src_label,
            rel_label.replace("_", " "),
            dst_label,
            allow_objs_with_same_label,
        )
    raise ValueError(f"Unknown command {cmd}")


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
        max_node_label_len = (
            batch.node_label_mask.sum(dim=1)[node_mask].max() if node_mask.any() else 0
        )
        max_edge_label_len = (
            batch.edge_label_mask.sum(dim=1)[edge_mask].max() if edge_mask.any() else 0
        )
        subgraph = Data(
            x=F.pad(
                batch.x[node_mask],
                (0, max_node_label_len - batch.x[node_mask].size(1))
                if batch.x.dim() == 2
                else (0, 0, 0, max_node_label_len - batch.x[node_mask].size(1)),
            ),
            node_label_mask=F.pad(
                batch.node_label_mask[node_mask],
                (0, max_node_label_len - batch.node_label_mask[node_mask].size(1)),
            ),
            node_last_update=batch.node_last_update[node_mask],
            edge_index=batch.edge_index[:, edge_mask] - node_id_offsets[i],
            edge_attr=F.pad(
                batch.edge_attr[edge_mask],
                (0, max_edge_label_len - batch.edge_attr[edge_mask].size(1))
                if batch.edge_attr.dim() == 2
                else (0, 0, 0, max_edge_label_len - batch.edge_attr[edge_mask].size(1)),
            ),
            edge_label_mask=F.pad(
                batch.edge_label_mask[edge_mask],
                (0, max_edge_label_len - batch.edge_label_mask[edge_mask].size(1)),
            ),
            edge_last_update=batch.edge_last_update[edge_mask],
        )
        data_list.append(subgraph)
    return data_list


def data_list_to_batch(data_list: List[Data]) -> Batch:
    """
    Inverse of batch_to_data_list(). Can't use Batch.from_data_list() directly due to
    variable length labels.
    """
    max_node_label_len = max(data.x.size(1) for data in data_list)
    max_edge_label_len = max(data.edge_attr.size(1) for data in data_list)
    return Batch.from_data_list(
        [
            Data(
                x=F.pad(
                    data.x,
                    (0, max_node_label_len - data.x.size(1))
                    if data.x.dim() == 2
                    else (0, 0, 0, max_node_label_len - data.x.size(1)),
                ),
                node_label_mask=F.pad(
                    data.node_label_mask,
                    (0, max_node_label_len - data.node_label_mask.size(1)),
                ),
                node_last_update=data.node_last_update,
                edge_index=data.edge_index,
                edge_attr=F.pad(
                    data.edge_attr,
                    (0, max_edge_label_len - data.edge_attr.size(1))
                    if data.edge_attr.dim() == 2
                    else (0, 0, 0, max_edge_label_len - data.edge_attr.size(1)),
                ),
                edge_label_mask=F.pad(
                    data.edge_label_mask,
                    (0, max_edge_label_len - data.edge_label_mask.size(1)),
                ),
                edge_last_update=data.edge_last_update,
            )
            for data in data_list
        ]
    )


def networkx_to_rdf(
    graph: nx.DiGraph, allow_objs_with_same_label: bool = False
) -> Set[str]:
    """
    Turn the given networkx graph into a set of RDF triples.
    """
    rdfs: Set[str] = set()
    for src, dst, edge_data in graph.edges.data():
        src_label = graph.nodes[src]["label"]
        dst_label = graph.nodes[dst]["label"]
        if allow_objs_with_same_label:
            if src_label in FOOD_COLORS and dst_label in FOOD_COLORS[src_label]:
                continue
            elif graph.nodes[src]["label"] in FOOD_COLORS:
                colors: List[str] = []
                for _, food_dst, food_edge_data in graph.out_edges(src, data=True):
                    food_dst_label = graph.nodes[food_dst]["label"]
                    if (
                        food_edge_data["label"] == IS
                        and food_dst_label in FOOD_COLORS[src_label]
                    ):
                        colors.append(food_dst_label)
                if len(colors) == 1:
                    src_label = colors[0] + " " + src_label
        rdfs.add(
            f"{src_label} , {dst_label} ," f" {'_'.join(edge_data['label'].split())}"
        )

    return rdfs


def update_rdf_graph(rdfs: Set[str], graph_cmds: List[str]) -> Set[str]:
    """
    Update the given RDF triple graph using the given graph commands.

    We remove duplicate graph commands while preserving order.

    Since Python 3.7, dict is guaranteed to keep insertion order.
    """
    graph_cmds = list(dict.fromkeys(graph_cmds))
    for cmd in graph_cmds:
        verb, src, dst, relation = cmd.split(" , ")
        rdf = " , ".join([src, dst, relation])
        if verb == "add" and rdf not in rdfs:
            rdfs.add(rdf)
        elif verb == "delete" and rdf in rdfs:
            rdfs.remove(rdf)
    return rdfs
