import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.preprocessor import SpacyPreprocessor


def masked_mean(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    input: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len)
    output: (batch, hidden_dim)
    """
    return (input * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def masked_softmax(
    input: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    input, mask and output all have the same dimensions
    """
    # replace the values to be ignored with negative infinity
    return F.softmax(input.masked_fill(mask == 0, float("-inf")), dim=dim)


def compute_masks_from_event_type_ids(
    event_type_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute three masks from the given event type ids tensor.
    1. event mask: masks out special events like start, end and pad
    2. source mask: masks out special events, as well as node-add
        as it is adding a new node.
    3. destination mask: masks out special events and node events
        as only edge events have destination nodes.

    Dimensions of the masks are the same as that of event_type_ids.

    output: (event_mask, src_mask, dst_mask)
    """
    is_special_event = torch.logical_or(
        torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["pad"],
            event_type_ids == EVENT_TYPE_ID_MAP["start"],
        ),
        event_type_ids == EVENT_TYPE_ID_MAP["end"],
    )
    # (batch, event_seq_len)
    is_node_delete = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
    # (batch, event_seq_len)
    is_edge_event = torch.logical_or(
        event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
        event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
    )
    # (batch, event_seq_len)

    event_mask = torch.logical_not(is_special_event).float()
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
