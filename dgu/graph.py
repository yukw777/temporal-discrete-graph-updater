import networkx as nx

from typing import Dict


class TextWorldGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_labels: Dict[str, int] = {}

    def add_node(self, label: str) -> None:
        """
        Add a node with the given label. noop if it already exists and
        hasn't been removed. It also initializes the id of the node.
        """
        if self.has_node(label):
            return
        node_id = self.order()
        self.node_labels[label] = node_id
        super().add_node(node_id, label=label, removed=False)

    def remove_node(self, label: str) -> None:
        """
        Remove the node with the given label. noop if it doesn't exist.
        It simply marks the node "removed" so that it is easier to manage
        the node ID space for graph neural network training.
        """
        if not self.has_node(label):
            return
        node_id = self.node_labels[label]
        super().add_node(node_id, label=label, removed=True)

    def has_node(self, label: str) -> bool:
        """
        Check if the graph has a node with the given label that has yet to be removed.
        """
        if label not in self.node_labels:
            return False
        node_id = self.node_labels[label]
        if self.nodes[node_id]["removed"]:
            return False
        return True

    def get_node_id(self, label: str) -> int:
        """
        Return the id of the most recent node with the given label. Raises an exception
        if a node with the label doesn't exist.
        """
        return self.node_labels[label]

    def add_edge(self, src_label: str, edge_label: str, dst_label: str) -> None:
        """
        Add an edge with the given label from the source node to the destination node.

        If either of the source or destination node doesn't exist, add it. If the edge
        already exists, this is a noop.
        """
        if self.has_edge(src_label, dst_label):
            return
        # add src and dst nodes. noop if they exist.
        self.add_node(src_label)
        self.add_node(dst_label)

        # add edge
        src_id = self.get_node_id(src_label)
        dst_id = self.get_node_id(dst_label)
        super().add_edge(src_id, dst_id, label=edge_label)

    def remove_edge(self, src_label: str, dst_label: str) -> None:
        """
        Remove the edge from the source node to the destination node.

        If any of the edge, source node or destination node doesn't exist, it's a noop.
        """
        if not self.has_edge(src_label, dst_label):
            return
        src_id = self.get_node_id(src_label)
        dst_id = self.get_node_id(dst_label)
        super().remove_edge(src_id, dst_id)

    def has_edge(self, src_label: str, dst_label: str) -> bool:
        """
        Check if an edge exists from the source to the destination nodes.
        """
        if not self.has_node(src_label) or not self.has_node(dst_label):
            return False
        src_id = self.get_node_id(src_label)
        dst_id = self.get_node_id(dst_label)
        return super().has_edge(src_id, dst_id)
