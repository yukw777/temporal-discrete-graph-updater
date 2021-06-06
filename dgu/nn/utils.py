import torch
import torch.nn.functional as F

from typing import Optional


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
