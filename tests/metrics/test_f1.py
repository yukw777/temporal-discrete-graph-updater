import pytest
import torch

from dgu.metrics import F1


@pytest.mark.parametrize(
    "batch_preds,batch_targets,batch_expected",
    [
        ([[[]]], [[[]]], [1.0]),
        ([[["add , n0 , n1 , r"]]], [[[]]], [0.0]),
        ([[[]]], [[["add , n0 , n1 , r"]]], [0.0]),
        ([[["add , n0 , n1 , r"]]], [[["add , n0 , n1 , r"]]], [1.0]),
        ([[["add", "n0", "n1", "r"]]], [[["add", "n0", "n1", "r"]]], [1.0]),
        ([[["add , n0 , n1 , r"]]], [[["delete , n0 , n1 , r"]]], [0.0]),
        ([[["add", "n0", "n1", "r"]]], [[["delete", "n2", "n3", "r0"]]], [0.0]),
        (
            [
                [
                    ["add , n0 , n1 , r0", "delete , n2 , n3 , r1"],
                    ["delete , n0 , n1 , r0"],
                ],
                [
                    ["add , n0 , n1 , r0", "delete , n2 , n3 , r1"],
                    ["delete , n0 , n1 , r0"],
                    [],
                    ["add , n0 , n1 , r0", "delete , n2 , n3 , r1"],
                ],
            ],
            [
                [
                    ["add , n0 , n1 , r0", "delete , n2 , n3 , r1"],
                    ["add , n4 , n5 , r2"],
                ],
                [
                    ["delete , n2 , n3 , r1", "add , n0 , n1 , r0"],
                    ["delete , n0 , n1 , r2"],
                    [],
                    ["add , n0 , n1 , r0", "delete , n2 , n3 , r2"],
                ],
            ],
            [0.5, 3.5 / 6],
        ),
        (
            [
                [
                    ["a", "b", "c", "d"],
                    ["e", "f", "g", "h"],
                ],
                [
                    ["add", "n0", "n1", "r0", "add", "n2", "n3", "r1"],
                    ["delete", "n0", "n1", "r0"],
                    [],
                    ["add", "n0", "n1", "r0", "delete", "n2", "n3", "r1"],
                ],
            ],
            [
                [
                    ["c", "d", "b", "a"],
                    ["g", "h", "i", "j"],
                ],
                [
                    ["delete", "n2", "n3", "r1", "add", "n0", "n1", "r0"],
                    ["delete", "n0", "n1", "r2"],
                    [],
                    ["add", "n0", "n1", "r0", "add", "n2", "n3", "r2"],
                ],
            ],
            [0.75, 0.8125],
        ),
    ],
)
def test_f1(batch_preds, batch_targets, batch_expected):
    f1 = F1()
    for preds, targets, expected in zip(batch_preds, batch_targets, batch_expected):
        f1.update(preds, targets)
        assert f1.compute().isclose(torch.tensor(expected))
