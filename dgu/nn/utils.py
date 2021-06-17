import torch
import torch.nn.functional as F

from typing import Optional, List, Tuple

from dgu.constants import EVENT_TYPES


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

    event_type_ids: (event_seq_len)

    output:
        event_mask: (event_seq_len)
        src_mask: (event_seq_len)
        dst_mask: (event_seq_len)
    """
    event_mask: List[float] = []
    src_mask: List[float] = []
    dst_mask: List[float] = []
    for event_type_id in event_type_ids.tolist():
        event = EVENT_TYPES[event_type_id]
        if event in {"pad", "start", "end"}:
            # special events
            event_mask.append(0.0)
            src_mask.append(0.0)
            dst_mask.append(0.0)
        elif event in {"node-add", "node-delete"}:
            # node events
            if event == "node-add":
                src_mask.append(0.0)
            else:
                src_mask.append(1.0)
            event_mask.append(1.0)
            dst_mask.append(0.0)
        elif event in {"edge-add", "edge-delete"}:
            # edge events
            event_mask.append(1.0)
            src_mask.append(1.0)
            dst_mask.append(1.0)
        else:
            raise ValueError(f"Unknown event: {event_type_id}: {event}")

    device = event_type_ids.device
    return (
        torch.tensor(event_mask, device=device),
        torch.tensor(src_mask, device=device),
        torch.tensor(dst_mask, device=device),
    )
