import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Any, Dict
from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from tdgu.constants import EVENT_TYPE_ID_MAP
from tdgu.preprocessor import SpacyPreprocessor


def masked_mean(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    input: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len)
    output: (batch, hidden_dim)
    """
    return (input * mask.unsqueeze(-1)).sum(dim=1) / (
        mask.float()
        .sum(dim=1, keepdim=True)
        .clamp(min=torch.finfo(input.dtype).tiny)  # clamp to avoid divide by 0
    )


def masked_softmax(
    input: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    input, mask and output all have the same dimensions
    """
    # replace the values to be ignored with the minimum value of the data type
    return (
        F.softmax(
            input.masked_fill(
                ~mask if mask.dtype == torch.bool else mask == 0,
                torch.finfo(input.dtype).min,
            ),
            dim=dim,
        )
        * mask
    )


def masked_log_softmax(
    input: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    input, mask and output all have the same dimensions
    """
    # replace the values to be ignored with the minimum value of the data type
    log_mask = torch.log(mask + torch.finfo(input.dtype).tiny)
    return F.log_softmax(input + log_mask, dim=dim) + log_mask


def compute_masks_from_event_type_ids(
    event_type_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute three boolean masks from the given event type ids tensor.
    1. event mask: masks out pad event
    2. source mask: True if node-delete or edge event
    3. destination mask: True if edge event
    4. label mask: True if node-add or edge-add

    Dimensions of the masks are the same as that of event_type_ids.

    output: {
        "event_mask": event mask,
        "src_mask": source mask,
        "dst_mask": destination mask,
        "label_mask": label mask,
    }
    """
    # only mask out pad
    is_pad_event = event_type_ids == EVENT_TYPE_ID_MAP["pad"]
    # (batch, event_seq_len)
    is_node_add = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
    # (batch, event_seq_len)
    is_node_delete = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
    # (batch, event_seq_len)
    is_edge_add = event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
    # (batch, event_seq_len)
    is_edge_delete = event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
    # (batch, event_seq_len)
    is_edge_event = is_edge_add.logical_or(is_edge_delete)
    # (batch, event_seq_len)

    event_mask = is_pad_event.logical_not()
    # (batch, event_seq_len)
    src_mask = is_node_delete.logical_or(is_edge_event)
    # (batch, event_seq_len)
    dst_mask = is_edge_event
    # (batch, event_seq_len)
    label_mask = is_node_add.logical_or(is_edge_add)
    # (batch, event_seq_len)

    return {
        "event_mask": event_mask,
        "src_mask": src_mask,
        "dst_mask": dst_mask,
        "label_mask": label_mask,
    }


def load_fasttext(
    fname: str, serialized_path: Path, preprocessor: SpacyPreprocessor
) -> nn.Embedding:
    # check if we have already serialized it
    if serialized_path.exists():
        with open(serialized_path, "rb") as serialized:
            return torch.load(serialized)

    with open(fname, "r") as f:
        _, emb_dim = map(int, f.readline().split())

        data = {}
        for line in tqdm(f, desc="loading fasttext embeddings"):
            parts = line.rstrip().split(" ", 1)
            data[parts[0]] = parts[1]
    # embedding for pad is initalized to 0
    # embeddings for OOVs are randomly initialized from N(0, 1)
    emb = nn.Embedding(
        preprocessor.vocab_size, emb_dim, padding_idx=preprocessor.pad_token_id
    )
    for word, i in tqdm(
        preprocessor.get_vocab().items(), desc="constructing word embeddings"
    ):
        if word in data:
            with torch.no_grad():
                emb.weight[i] = torch.tensor(list(map(float, data[word].split())))

    # save it before returning
    torch.save(emb, serialized_path)

    return emb


def find_indices(haystack: torch.Tensor, needle: torch.Tensor) -> torch.Tensor:
    return (haystack.unsqueeze(-1) == needle).transpose(0, 1).nonzero(as_tuple=True)[1]


def pad_batch_seq_of_seq(
    batch_seq_of_seq: List[List[List[Any]]],
    max_len_outer: int,
    max_len_inner: int,
    outer_padding_value: Any,
    inner_padding_value: Any,
) -> torch.Tensor:
    """
    batch_seq_of_seq: unpadded batch of sequence of sequences
    max_len_outer: max length of the outer sequence
    max_len_inner: max length of the inner sequence
    outer_padding_value: value to pad the outer sequence
    inner_padding_value: value to pad the inner sequence

    output: padded batch of sequence of sequences
    """
    return torch.stack(
        [
            pad_sequence(
                [
                    torch.tensor(seq)
                    for seq in seq_of_seq
                    + [[outer_padding_value] * max_len_inner]
                    * (max_len_outer - len(seq_of_seq))
                ],
                batch_first=True,
                padding_value=inner_padding_value,
            )
            for seq_of_seq in batch_seq_of_seq
        ]
    )


def get_edge_index_co_occurrence_matrix(
    edge_index_a: torch.Tensor, edge_index_b: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the co-occurrence matrix between two edge index matrices.

    edge_index_a: (2, num_edge_a)
    edge_index_b: (2, num_edge_b)

    output: (num_edge_a, num_edge_b)
    """
    return torch.all(edge_index_a.t().unsqueeze(-1) == edge_index_b, dim=1)


def index_edge_attr(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """
    Return edge attributes corresponding to the given indices. We assume that there are
    no duplicate edges in edge_index. Attributes for non-existent edges are filled with
    0.

    edge_index: (2, num_edge)
    edge_attr: (num_edge, *)
    indices: (2, num_indexed_edge)

    output: (num_indexed_edge, *)
    """
    # calculate the co-occurrence matrix between edge_index and indices
    co_occur = get_edge_index_co_occurrence_matrix(indices, edge_index)
    # (num_indexed_edge, num_edge)

    # calculate positions of the matching indices
    pos = co_occur.nonzero()[:, 1]
    # (num_indexed_edge)

    # create an empty
    out = torch.zeros(
        indices.size(1),
        *edge_attr.size()[1:],
        device=edge_attr.device,
        dtype=edge_attr.dtype
    )
    out[co_occur.any(1)] = edge_attr[pos]
    return out
    # (num_indexed_edge, *)


def calculate_node_id_offsets(batch_size: int, batch: torch.Tensor) -> torch.Tensor:
    """
    Calculate the node id offsets for turning subgraph node IDs into a batched
    graph node IDs.

    batch_size: scalar
    batch: (num_node)

    output: (batch_size)
    """
    subgraph_size_cumsum = batch.bincount().cumsum(0)
    return F.pad(
        subgraph_size_cumsum, (1, batch_size - subgraph_size_cumsum.size(0) - 1)
    )


def update_batched_graph(
    batched_graph: Batch,
    event_type_ids: torch.Tensor,
    event_src_ids: torch.Tensor,
    event_dst_ids: torch.Tensor,
    event_embeddings: torch.Tensor,
    event_timestamps: torch.Tensor,
) -> Batch:
    """
    Update the given batched graph based on the given batch of graph events. All
    the events are assumed to be valid, and each event in the batch is assuemd to
    be applied only to the corresponding graph in the batched graph.

    batched_graph: diagonally stacked graph BEFORE the given graph events: Batch(
        batch: (num_node)
        x: (num_node, *)
        node_last_update: (num_node, 2)
        edge_index: (2, num_edge)
        edge_attr: (num_edge, *)
        edge_last_update: (num_edge, 2)
    )
    event_type_ids: (batch)
    event_src_ids: (batch)
    event_dst_ids: (batch)
    event_embeddings: (batch, *)
    event_timestamps: (batch, 2)

    output: updated batch of graphs
    """
    # translate src_ids and dst_ids to batched versions
    node_id_offsets = calculate_node_id_offsets(
        event_type_ids.size(0), batched_graph.batch
    )
    # (batch)
    batch_event_src_ids = event_src_ids + node_id_offsets
    # (batch)

    # take care of the nodes
    # collect node add events
    node_add_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
    # (batch)
    added_x = event_embeddings[node_add_event_mask]
    # (num_added_node, *)
    added_node_batch = node_add_event_mask.nonzero().squeeze(-1)
    # (num_added_node)
    added_node_last_update = event_timestamps[node_add_event_mask]
    # (num_added_node, 2)

    # collect node delete events
    node_delete_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
    # (batch)
    node_delete_node_ids = batch_event_src_ids[node_delete_event_mask]
    # (batch)
    delete_node_mask = torch.ones(
        batched_graph.num_nodes,
        dtype=torch.bool,
        device=node_delete_node_ids.device,
    ).index_fill(0, node_delete_node_ids, False)
    # (num_node)

    batch_size = event_type_ids.size(0)
    new_batch = update_node_features(
        batched_graph.batch,
        batched_graph.batch,
        delete_node_mask,
        added_node_batch,
        added_node_batch,
        batch_size,
    )
    # (num_node-num_deleted_node+num_added_node)

    new_x = update_node_features(
        batched_graph.x,
        batched_graph.batch,
        delete_node_mask,
        added_x,
        added_node_batch,
        batch_size,
    )
    # (num_node-num_deleted_node+num_added_node, *)

    new_node_last_update = update_node_features(
        batched_graph.node_last_update,
        batched_graph.batch,
        delete_node_mask,
        added_node_last_update,
        added_node_batch,
        batch_size,
    )
    # (num_node-num_deleted_node+num_added_node)

    # update edge_index based on the added and deleted nodes
    updated_edge_index = update_edge_index(
        batched_graph.edge_index,
        batched_graph.batch,
        delete_node_mask,
        node_add_event_mask,
    )
    # (2, num_edge)

    # now, take care of the edges
    # NOTE: we don't need to update edge event node IDs b/c they wouldn't have
    # changed b/c those subgraphs have edge events, not node events.

    # collect edge add events
    edge_add_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
    # (batch)
    # filter out duplicated added edges
    new_node_id_offsets = calculate_node_id_offsets(event_type_ids.size(0), new_batch)
    # (batch)
    new_batch_event_src_ids = event_src_ids + new_node_id_offsets
    # (batch)
    new_batch_event_dst_ids = event_dst_ids + new_node_id_offsets
    # (batch)
    batch_added_edge_index_with_dups = torch.stack(
        [
            new_batch_event_src_ids[edge_add_event_mask],
            new_batch_event_dst_ids[edge_add_event_mask],
        ]
    )
    # (2, num_added_edge_with_dups)
    delete_dup_added_edge_mask = (
        get_edge_index_co_occurrence_matrix(
            batch_added_edge_index_with_dups, updated_edge_index
        )
        .any(1)
        .logical_not()
    )
    # (num_added_edge_with_dups)
    added_edge_index = torch.stack(
        [
            new_batch_event_src_ids[edge_add_event_mask][delete_dup_added_edge_mask],
            new_batch_event_dst_ids[edge_add_event_mask][delete_dup_added_edge_mask],
        ]
    )
    # (num_added_edge)
    added_edge_attr = event_embeddings[edge_add_event_mask][delete_dup_added_edge_mask]
    # (num_added_edge, *)
    added_edge_last_update = event_timestamps[edge_add_event_mask][
        delete_dup_added_edge_mask
    ]
    # (num_added_edge, 2)

    # collect edge delete events
    edge_delete_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
    # (batch)
    deleted_edge_co_occur = get_edge_index_co_occurrence_matrix(
        updated_edge_index,
        torch.stack(
            [
                new_batch_event_src_ids[edge_delete_event_mask],
                new_batch_event_dst_ids[edge_delete_event_mask],
            ]
        ),
    )
    # (num_edge, num_deleted_edge)
    deleted_edge_mask = torch.ones(
        batched_graph.num_edges,
        dtype=torch.bool,
        device=deleted_edge_co_occur.device,
    ).masked_fill(deleted_edge_co_occur.any(1), False)
    # (num_edge)

    # get the new edge index by removing deleted edges and adding new ones
    new_edge_index = torch.cat(
        [updated_edge_index[:, deleted_edge_mask], added_edge_index],
        dim=1,
    )
    # (2, num_edge-num_deleted_edge+num_added_edge)
    new_edge_attr = torch.cat(
        [batched_graph.edge_attr[deleted_edge_mask], added_edge_attr]
    )
    # (num_edge-num_deleted_edge+num_added_edge, *)
    new_edge_last_update = torch.cat(
        [batched_graph.edge_last_update[deleted_edge_mask], added_edge_last_update]
    )
    # (num_edge-num_deleted_edge+num_added_edge)

    return Batch(
        batch=new_batch,
        x=new_x,
        node_last_update=new_node_last_update,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        edge_last_update=new_edge_last_update,
    )


def update_node_features(
    node_features: torch.Tensor,
    batch: torch.Tensor,
    delete_mask: torch.Tensor,
    added_features: torch.Tensor,
    added_batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Update the given node features using delete mask and added features.
    We first delete the features using the delete mask, then append new
    features for each graph in the batch.

    node_features: (num_node, *)
    batch: (num_node)
    delete_mask: (num_node)
    added_features: (num_added_node, *)
    added_batch: (num_added_node)
    batch_size: scalar

    output: (num_node-num_deleted_node+num_added_node, *)
    """
    batchified_features, batchified_features_mask = to_dense_batch(
        node_features[delete_mask], batch=batch[delete_mask], batch_size=batch_size
    )
    # batchified_features: (batch, max_subgraph_num_node, *)
    # batchified_features_mask: (batch, max_subgraph_num_node)
    (
        batchified_added_features,
        batchified_added_features_mask,
    ) = to_dense_batch(added_features, batch=added_batch, batch_size=batch_size)
    # batchified_added_features: (batch, max_num_added_node, *)
    # batchified_added_features_mask: (batch, max_num_added_node)
    new_features = torch.cat(
        [batchified_features, batchified_added_features], dim=1
    ).flatten(end_dim=1)
    new_features_mask = torch.cat(
        [batchified_features_mask, batchified_added_features_mask], dim=1
    ).flatten(end_dim=1)

    return new_features[new_features_mask]
    # (num_node-num_deleted_node+num_added_node, *)


def update_edge_index(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    delete_node_mask: torch.Tensor,
    node_add_event_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Update the given edge indices based on the added and deleted nodes.
    Note that only deleted nodes can affect the subgraph indices as nodes
    are added to the back.

    edge_index: (2, num_edge)
    batch: (num_node)
    delete_node_mask: (num_node)
    node_add_event_mask: (batch)

    output: (2, num_edge)
    """
    # subtract cumulative sum of the number of deleted nodes
    deleted_node_edge_index = (
        edge_index - torch.cumsum(~delete_node_mask, 0)[edge_index]
    )
    # (2, num_edge)

    # figure out the offsets caused by added nodes
    added_node_offsets = F.pad(node_add_event_mask.cumsum(0), (1, -1))
    # (batch)

    return deleted_node_edge_index + added_node_offsets[batch[edge_index]]
    # (2, num_edge)


class PositionalEncoder(nn.Module):
    """
    Return positional encodings for the given positions. This is the tensor2tensor
    implementation of the positional encoding, which is slightly different
    from the one used by the original Transformer paper.
    Specifically, there are 2 key differences:
    1. Sine and cosine values are concatenated rather than interweaved.
    2. The divisor is calculated differently
        ((d_model (or channels) // 2) -1 vs. d_model)
    There are no material differences between positional encoding implementations.
    The important point is that you use the same implementation throughout. The
    original GATA code uses this version. I've cleaned up the implementation a bit,
    including a small optimization that caches all the positional encodings, which
    was shown in the PyTorch Transformer tutorial
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    def __init__(
        self,
        channels: int,
        max_len: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1e4,
    ) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        num_timescales = channels // 2
        log_timescale_increment = torch.log(
            torch.tensor(max_timescale / min_timescale)
        ) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * -log_timescale_increment
        ).unsqueeze(0)
        scaled_time = position * inv_timescales
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).view(
            max_len, channels
        )
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (*)

        output: (*, channels)
        """
        return self.pe[positions]  # type: ignore
