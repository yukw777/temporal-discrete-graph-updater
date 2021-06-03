import torch

from typing import List


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
