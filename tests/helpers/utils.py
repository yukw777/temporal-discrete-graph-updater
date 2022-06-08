import torch
import networkx as nx

from typing import List
from torch_geometric.data import Data, Batch


def increasing_mask(
    batch_size: int, seq_len: int, start_with_zero: bool = False
) -> torch.Tensor:
    """
    Return an increasing mask. Useful for tests. For example:
    batch_size = 3, seq_len = 2, start_with_zero = False
    [
        [1, 0],
        [1, 1],
        [1, 1],
    ]
    batch_size = 3, seq_len = 2, start_with_zero = True
    [
        [0, 0],
        [1, 0],
        [1, 1],
    ]
    """
    data: List[List[int]] = []
    for i in range(batch_size):
        if i < seq_len:
            if start_with_zero:
                data.append([1] * i + [0] * (seq_len - i))
            else:
                data.append([1] * (i + 1) + [0] * (seq_len - i - 1))
        else:
            data.append([1] * seq_len)
    return torch.tensor(data, dtype=torch.float)


class EqualityDiGraph(nx.DiGraph):
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EqualityDiGraph):
            return False

        return self.nodes.data() == o.nodes.data() and list(self.edges.data()) == list(
            o.edges.data()
        )


class EqualityData(Data):
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EqualityData):
            return False

        return (
            self.x.equal(o.x)
            and self.node_label_mask.equal(o.node_label_mask)
            and self.node_last_update.equal(o.node_last_update)
            and self.edge_index.equal(o.edge_index)
            and self.edge_attr.equal(o.edge_attr)
            and self.edge_label_mask.equal(o.edge_label_mask)
            and self.edge_last_update.equal(o.edge_last_update)
        )


class EqualityBatch(Batch):
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EqualityBatch):
            return False

        return (
            self.batch.equal(o.batch)
            and self.x.equal(o.x)
            and self.node_label_mask.equal(o.node_label_mask)
            and self.node_last_update.equal(o.node_last_update)
            and self.edge_index.equal(o.edge_index)
            and self.edge_attr.equal(o.edge_attr)
            and self.edge_label_mask.equal(o.edge_label_mask)
            and self.edge_last_update.equal(o.edge_last_update)
        )
