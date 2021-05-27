import networkx as nx

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Deque, Any, Tuple

from dgu.constants import IS


@dataclass(frozen=True)
class IsDstNode:
    game: str
    walkthrough_step: int
    src_label: str
    dst_label: str


@dataclass(frozen=True)
class ExitNode:
    game: str
    walkthrough_step: int
    rel_label: str
    dst_label: str


@dataclass(frozen=True)
class Node:
    game: str
    walkthrough_step: int
    label: str


@dataclass
class TextWorldGraph:
    def __init__(self) -> None:
        self.is_dst_node_id_map: Dict[IsDstNode, int] = {}
        self.exit_node_id_map: Dict[ExitNode, int] = {}
        self.node_id_map: Dict[Node, int] = {}
        self.removed_node_ids: Deque[int] = deque()
        self.graph = nx.DiGraph()

    def process_add_triplet_cmd(
        self,
        game: str,
        walkthrough_step: int,
        timestamp: int,
        src_label: str,
        rel_label: str,
        dst_label: str,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        # get the source node id
        if src_label == "exit":
            # if the source is an exit, we check if the destination has an exit
            # for the given relation, and if not, add a new node.
            exit_node = ExitNode(game, walkthrough_step, rel_label, dst_label)
            if exit_node not in self.exit_node_id_map:
                src_id = self.add_node(game, walkthrough_step, src_label)
                self.exit_node_id_map[exit_node] = src_id
                events.append(
                    {
                        "type": "node-add",
                        "node_id": src_id,
                        "timestamp": timestamp,
                    }
                )
            else:
                src_id = self.exit_node_id_map[exit_node]
        else:
            # if it's a regular node, check if a node with the same label exists
            # and add if it doesn't.
            node = Node(game, walkthrough_step, src_label)
            if node not in self.node_id_map:
                src_id = self.add_node(game, walkthrough_step, src_label)
                self.node_id_map[node] = src_id
                events.append(
                    {"type": "node-add", "node_id": src_id, "timestamp": timestamp}
                )
            else:
                src_id = self.node_id_map[node]

        # get the destination node id
        if rel_label == IS:
            # if the relation is "is", check if the destination node with the
            # given label exists for the source node with the given label,
            # and add if it doesn't.
            is_dst_node = IsDstNode(game, walkthrough_step, src_label, dst_label)
            if is_dst_node not in self.is_dst_node_id_map:
                dst_id = self.add_node(game, walkthrough_step, dst_label)
                self.is_dst_node_id_map[is_dst_node] = dst_id
                events.append(
                    {"type": "node-add", "node_id": dst_id, "timestamp": timestamp}
                )
            else:
                dst_id = self.is_dst_node_id_map[is_dst_node]
        else:
            # if it's a regular node, check if a node with the same label exists
            # and add if it doesn't.
            node = Node(game, walkthrough_step, dst_label)
            if node not in self.node_id_map:
                dst_id = self.add_node(game, walkthrough_step, dst_label)
                self.node_id_map[node] = dst_id
                events.append(
                    {"type": "node-add", "node_id": dst_id, "timestamp": timestamp}
                )
            else:
                dst_id = self.node_id_map[node]

        if self.graph.has_edge(src_id, dst_id):
            # the edge already exists, so we're done
            return events
        # the edge doesn't exist, so add it
        self.graph.add_edge(src_id, dst_id, label=rel_label)
        events.append(
            {
                "type": "edge-add",
                "src_id": src_id,
                "dst_id": dst_id,
                "timestamp": timestamp,
            }
        )
        return events

    def process_delete_triplet_cmd(
        self,
        game: str,
        walkthrough_step: int,
        timestamp: int,
        src_label: str,
        rel_label: str,
        dst_label: str,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        # get the source node id
        if src_label == "exit":
            src_id = self.exit_node_id_map.get(
                ExitNode(game, walkthrough_step, rel_label, dst_label)
            )
        else:
            src_id = self.node_id_map.get(Node(game, walkthrough_step, src_label))
        if src_id is None:
            # the source node doesn't exist, so return
            return events

        # get the destination node id
        if rel_label == IS:
            dst_id = self.is_dst_node_id_map.get(
                IsDstNode(game, walkthrough_step, src_label, dst_label)
            )
        else:
            dst_id = self.node_id_map.get(Node(game, walkthrough_step, dst_label))
        if dst_id is None:
            # the destination node doesn't exist, so return
            return events

        if not self.graph.has_edge(src_id, dst_id):
            # the edge doesn't exist, continue
            return events

        # delete the edge and add event
        self.graph.remove_edge(src_id, dst_id)
        events.append(
            {
                "type": "edge-delete",
                "src_id": src_id,
                "dst_id": dst_id,
                "timestamp": timestamp,
            }
        )
        # if there are no edges, delete the nodes
        if self.graph.in_degree(dst_id) == 0 and self.graph.out_degree(dst_id) == 0:
            self.remove_node(dst_id)
            if rel_label == IS:
                del self.is_dst_node_id_map[
                    IsDstNode(game, walkthrough_step, src_label, dst_label)
                ]
            else:
                del self.node_id_map[Node(game, walkthrough_step, dst_label)]
            events.append(
                {
                    "type": "node-delete",
                    "node_id": dst_id,
                    "timestamp": timestamp,
                }
            )
        if self.graph.in_degree(src_id) == 0 and self.graph.out_degree(src_id) == 0:
            self.remove_node(src_id)
            if src_label == "exit":
                del self.exit_node_id_map[
                    ExitNode(game, walkthrough_step, rel_label, dst_label)
                ]
            else:
                del self.node_id_map[Node(game, walkthrough_step, src_label)]
            events.append(
                {
                    "type": "node-delete",
                    "node_id": src_id,
                    "timestamp": timestamp,
                }
            )
        return events

    def process_triplet_cmd(
        self, game: str, walkthrough_step: int, timestamp: int, cmd: str
    ) -> List[Dict[str, Any]]:
        """
        Update the internal graph based on the given triplet command and return
        corresponding graph events.
        """
        cmd_type, src_label, dst_label, rel_label = cmd.split(" , ")
        # multi-word relations are joined by underscores, so replace them with spaces
        # so that it's easier to calculate word embeddings down the line.
        rel_label = rel_label.replace("_", " ")
        if cmd_type == "add":
            return self.process_add_triplet_cmd(
                game,
                walkthrough_step,
                timestamp,
                src_label,
                rel_label.replace("_", " "),
                dst_label,
            )
        elif cmd_type == "delete":
            return self.process_delete_triplet_cmd(
                game,
                walkthrough_step,
                timestamp,
                src_label,
                rel_label.replace("_", " "),
                dst_label,
            )
        raise ValueError(f"Unknown command {cmd}")

    def add_node(self, game: str, walkthrough_step: int, label: str) -> int:
        """
        Add a node for the given game, walkthrough step with the given label.
        It initializes the id of the node.
        """
        if len(self.removed_node_ids) > 0:
            node_id = self.removed_node_ids.popleft()
        else:
            node_id = self.graph.order()
        self.graph.add_node(
            node_id, game=game, walkthrough_step=walkthrough_step, label=label
        )
        return node_id

    def remove_node(self, node_id: int) -> None:
        """
        Remove the node with the given id.
        The removed node id is added to removed_node_ids for future use.
        """
        self.graph.remove_node(node_id)
        self.removed_node_ids.append(node_id)

    def get_node_labels(self) -> List[str]:
        """
        Return node labels. If a label is an empty string, the node doesn't exist.
        """
        node_labels = [""] * (self.graph.order() + len(self.removed_node_ids))
        for node_id, data in self.graph.nodes.data():
            node_labels[node_id] = data["label"]
        return node_labels

    def get_edge_labels(self) -> List[Tuple[int, int, str]]:
        return [
            (src_id, dst_id, data["label"])
            for src_id, dst_id, data in self.graph.edges.data()
        ]
