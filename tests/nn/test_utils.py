import torch
import torch.nn.functional as F

from dgu.nn.utils import masked_mean, masked_softmax


def test_masked_mean():
    batched_input = torch.tensor(
        [
            [
                [1, 2, 300],
                [300, 100, 200],
                [3, 4, 100],
            ],
            [
                [300, 100, 200],
                [6, 2, 300],
                [10, 4, 100],
            ],
        ]
    ).float()
    batched_mask = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
        ]
    ).float()
    assert masked_mean(batched_input, batched_mask).equal(
        torch.tensor(
            [
                [2, 3, 200],
                [8, 3, 200],
            ]
        ).float()
    )


def test_masked_softmax():
    batched_input = torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float()
    batched_mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float()
    batched_output = masked_softmax(batched_input, batched_mask, dim=1)

    # compare the result from masked_softmax with regular softmax with filtered values
    for input, mask, output in zip(batched_input, batched_mask, batched_output):
        assert output[output != 0].equal(F.softmax(input[mask == 1], dim=0))
