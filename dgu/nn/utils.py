import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List, Any
from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.preprocessor import SpacyPreprocessor


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
    return F.softmax(
        input.masked_fill(
            mask if mask.dtype == torch.bool else mask == 0,
            torch.finfo(input.dtype).min,
        ),
        dim=dim,
    )


def compute_masks_from_event_type_ids(
    event_type_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute three masks from the given event type ids tensor.
    1. event mask: masks out pad event.
    2. source mask: masks out special events, as well as node-add
        as it is adding a new node.
    3. destination mask: masks out special events and node events
        as only edge events have destination nodes.

    Dimensions of the masks are the same as that of event_type_ids.

    output: (event_mask, src_mask, dst_mask)
    """
    # only mask out pad
    is_pad_event = event_type_ids == EVENT_TYPE_ID_MAP["pad"]
    # (batch, event_seq_len)
    is_node_delete = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
    # (batch, event_seq_len)
    is_edge_event = torch.logical_or(
        event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
        event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
    )
    # (batch, event_seq_len)

    event_mask = torch.logical_not(is_pad_event).float()
    # (batch, event_seq_len)
    src_mask = torch.logical_or(is_node_delete, is_edge_event).float()
    # (batch, event_seq_len)
    dst_mask = is_edge_event.float()
    # (batch, event_seq_len)

    return event_mask, src_mask, dst_mask


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
        len(preprocessor.word_to_id_dict), emb_dim, padding_idx=preprocessor.pad_id
    )
    for word, i in tqdm(
        preprocessor.word_to_id_dict.items(), desc="constructing word embeddings"
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
