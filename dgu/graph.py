import networkx as nx

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Deque, Any, Tuple, Set

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


class TextWorldGraph:
    def __init__(self) -> None:
        self.is_dst_node_id_map: Dict[IsDstNode, int] = {}
        self.exit_node_id_map: Dict[ExitNode, int] = {}
        self.node_id_map: Dict[Node, int] = {}
        self.removed_node_ids: Deque[int] = deque()
        self.next_node_id = 0
        self.removed_edge_ids: Deque[int] = deque()
        self.next_edge_id = 0
        self._graph = nx.DiGraph()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TextWorldGraph):
            return False
        return (
            self.is_dst_node_id_map == o.is_dst_node_id_map
            and self.exit_node_id_map == o.exit_node_id_map
            and self.node_id_map == o.node_id_map
            and self.removed_node_ids == o.removed_node_ids
            and self.removed_edge_ids == o.removed_edge_ids
            and self.next_edge_id == o.next_edge_id
            and nx.is_isomorphic(self._graph, o._graph)
        )

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
                src_id = self.add_node(
                    src_label, game=game, walkthrough_step=walkthrough_step
                )
                self.exit_node_id_map[exit_node] = src_id
                events.append(
                    {
                        "type": "node-add",
                        "node_id": src_id,
                        "timestamp": timestamp,
                        "label": src_label,
                    }
                )
            else:
                src_id = self.exit_node_id_map[exit_node]
        else:
            # if it's a regular node, check if a node with the same label exists
            # and add if it doesn't.
            node = Node(game, walkthrough_step, src_label)
            if node not in self.node_id_map:
                src_id = self.add_node(
                    src_label, game=game, walkthrough_step=walkthrough_step
                )
                self.node_id_map[node] = src_id
                events.append(
                    {
                        "type": "node-add",
                        "node_id": src_id,
                        "timestamp": timestamp,
                        "label": src_label,
                    }
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
                dst_id = self.add_node(
                    dst_label, game=game, walkthrough_step=walkthrough_step
                )
                self.is_dst_node_id_map[is_dst_node] = dst_id
                events.append(
                    {
                        "type": "node-add",
                        "node_id": dst_id,
                        "timestamp": timestamp,
                        "label": dst_label,
                    }
                )
            else:
                dst_id = self.is_dst_node_id_map[is_dst_node]
        else:
            # if it's a regular node, check if a node with the same label exists
            # and add if it doesn't.
            node = Node(game, walkthrough_step, dst_label)
            if node not in self.node_id_map:
                dst_id = self.add_node(
                    dst_label, game=game, walkthrough_step=walkthrough_step
                )
                self.node_id_map[node] = dst_id
                events.append(
                    {
                        "type": "node-add",
                        "node_id": dst_id,
                        "timestamp": timestamp,
                        "label": dst_label,
                    }
                )
            else:
                dst_id = self.node_id_map[node]

        if (
            self._graph.has_edge(src_id, dst_id)
            and not self._graph[src_id][dst_id]["removed"]
        ):
            # the edge already exists, so we're done
            return events
        # the edge doesn't exist, so add it
        edge_id = self.add_edge(
            src_id, dst_id, rel_label, game=game, walkthrough_step=walkthrough_step
        )
        events.append(
            {
                "type": "edge-add",
                "edge_id": edge_id,
                "src_id": src_id,
                "dst_id": dst_id,
                "timestamp": timestamp,
                "label": rel_label,
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

        if not self._graph.has_edge(src_id, dst_id):
            # the edge doesn't exist, continue
            return events

        # delete the edge and add event
        edge_label = self._graph.edges[src_id, dst_id]["label"]
        edge_id = self.remove_edge(src_id, dst_id)
        events.append(
            {
                "type": "edge-delete",
                "edge_id": edge_id,
                "src_id": src_id,
                "dst_id": dst_id,
                "timestamp": timestamp,
                "label": edge_label,
            }
        )
        # if there are no edges, delete the nodes
        if all(
            self._graph[src][dst]["removed"]
            for src, dst in self._graph.in_edges(dst_id)
        ) and all(
            self._graph[src][dst]["removed"]
            for src, dst in self._graph.out_edges(dst_id)
        ):
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
                    "label": dst_label,
                }
            )
        if all(
            self._graph[src][dst]["removed"]
            for src, dst in self._graph.in_edges(src_id)
        ) and all(
            self._graph[src][dst]["removed"]
            for src, dst in self._graph.out_edges(src_id)
        ):
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
                    "label": src_label,
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

    def add_edge(self, src_id: int, dst_id: int, label: str, **kwargs) -> int:
        """
        Add an edge for the given source node, destination node and associated data,
        then returns the edge ID. The extra keyword arguments are saved as attributes
        of the edge.
        """
        if len(self.removed_edge_ids) > 0:
            edge_id = self.removed_edge_ids.popleft()
        else:
            edge_id = self.next_edge_id
            self.next_edge_id += 1
        self._graph.add_edge(
            src_id, dst_id, id=edge_id, label=label, removed=False, **kwargs
        )
        return edge_id

    def remove_edge(self, src_id: int, dst_id: int) -> int:
        """
        Remove the node with the given source and destination IDs, then return
        the removed edge ID.
        """
        attrs = self._graph[src_id][dst_id]
        edge_id = attrs["id"]
        attrs["removed"] = True
        self._graph.add_edge(src_id, dst_id, **attrs)
        self.removed_edge_ids.append(edge_id)
        return edge_id

    def add_node(self, label: str, **kwargs) -> int:
        """
        Add a node with the given label and initializes it with an ID.
        The extra keyword arguments are saved as attributes of the node.
        """
        if len(self.removed_node_ids) > 0:
            node_id = self.removed_node_ids.popleft()
        else:
            node_id = self.next_node_id
            self.next_node_id += 1
        self._graph.add_node(node_id, label=label, removed=False, **kwargs)
        return node_id

    def remove_node(self, node_id: int) -> None:
        """
        Remove the node with the given id.
        The removed node id is added to removed_node_ids for future use.
        """
        # we actually mark it removed instead of actually removing it.
        # this is to properly handle node-delete events when generating subgraphs
        # for training as we don't want to keep the deleted nodes around when
        # calculating subgraph nodes.
        attrs = self._graph.nodes[node_id]
        attrs["removed"] = True
        self._graph.add_node(node_id, **attrs)
        self.removed_node_ids.append(node_id)

    def get_node_labels(self) -> List[str]:
        """
        Return all the node labels.
        """
        return [data["label"] for _, data in self._graph.nodes.data()]

    def get_edge_labels(self) -> List[Tuple[int, int, str]]:
        return [
            (src_id, dst_id, data["label"])
            for src_id, dst_id, data in self._graph.edges.data()
        ]

    def get_subgraph(
        self, game_walkthrough_set: Set[Tuple[str, int]]
    ) -> Tuple[Set[int], List[int], List[Tuple[int, int]]]:
        """
        Return the node IDs, edge IDs and edge indices for the subgraph
        corresponding to the given set of (game, walkthrough_step)'s.

        game_walkthrough_set: set of (game, walkthrough_step)

        returns: (node_ids, edge_ids, edge_index)
        """

        def filter_node(node):
            game = self._graph.nodes[node]["game"]
            walkthrough_step = self._graph.nodes[node]["walkthrough_step"]
            return (game, walkthrough_step) in game_walkthrough_set

        subgraph_view = nx.subgraph_view(self._graph, filter_node=filter_node)
        edge_ids: List[int] = []
        edge_index: List[Tuple[int, int]] = []
        for src, dst, data in subgraph_view.edges.data():
            edge_ids.append(data["id"])
            edge_index.append((src, dst))
        return set(subgraph_view.nodes()), edge_ids, edge_index

    def process_events(self, events: List[Dict[str, Any]], **kwargs) -> None:
        """
        Process the given graph events and update the graph. Keyword arguments
        are used to set attributes for all the nodes and edges added by the given
        graph events. Useful for specifying the game and walkthrough step.
        """
        for event in events:
            event_type = event["type"]
            if event_type == "node-add":
                self.add_node(event["label"], **kwargs)
            elif event_type == "node-delete":
                self.remove_node(event["node_id"])
            elif event_type == "edge-add":
                self.add_edge(
                    event["src_id"], event["dst_id"], event["label"], **kwargs
                )
            elif event_type == "edge-delete":
                self.remove_edge(event["src_id"], event["dst_id"])
            else:
                raise ValueError(f"Unknown event type: {event_type}")
