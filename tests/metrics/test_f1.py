import pytest
import torch

from dgu.metrics import F1


@pytest.mark.parametrize(
    "batch_preds,batch_targets,expected",
    [
        ([[]], [[]], torch.tensor(1.0)),
        ([["add , n0 , n1 , r"]], [[]], torch.tensor(0.0)),
        ([[]], [["add , n0 , n1 , r"]], torch.tensor(0.0)),
        ([["add , n0 , n1 , r"]], [["add , n0 , n1 , r"]], torch.tensor(1.0)),
        ([["add", "n0", "n1", "r"]], [["add", "n0", "n1", "r"]], torch.tensor(1.0)),
        ([["add , n0 , n1 , r"]], [["delete , n0 , n1 , r"]], torch.tensor(0.0)),
        ([["add", "n0", "n1", "r"]], [["delete", "n2", "n3", "r0"]], torch.tensor(0.0)),
        (
            [
                [
                    "add , n0 , n1 , r0",
                    "delete , n2 , n3 , r1",
                    "delete , n0 , n1 , r0",
                    "add , n0 , n1 , r0",
                    "delete , n2 , n3 , r1",
                ],
                [
                    "add , n0 , n1 , r0",
                    "delete , n2 , n3 , r1",
                    "delete , n0 , n1 , r0",
                ],
            ],
            [
                [
                    "delete , n2 , n3 , r1",
                    "add , n0 , n1 , r0",
                    "delete , n0 , n1 , r2",
                    "add , n0 , n1 , r0",
                    "delete , n2 , n3 , r2",
                ],
                ["add , n0 , n1 , r0", "delete , n2 , n3 , r1", "add , n4 , n5 , r2"],
            ],
            torch.tensor(
                (
                    2 * 2 / 3 * 2 / 4 / (2 / 3 + 2 / 4)
                    + (2 * 2 / 3 * 2 / 3 / (2 / 3 + 2 / 3))
                )
                / 2
            ),
        ),
        (
            [
                [
                    "add",
                    "n0",
                    "n1",
                    "r0",
                    "<sep>",
                    "add",
                    "n2",
                    "n3",
                    "r1",
                    "<sep>",
                    "delete",
                    "n0",
                    "n1",
                    "r0",
                    "<sep>",
                    "add",
                    "n0",
                    "n1",
                    "r0",
                    "<sep>",
                    "delete",
                    "n2",
                    "n3",
                    "r1",
                ],
                ["a", "b", "c", "d", "<sep>", "e", "f", "g", "h"],
            ],
            [
                [
                    "delete",
                    "n2",
                    "n3",
                    "r1",
                    "<sep>",
                    "add",
                    "n0",
                    "n1",
                    "r0",
                    "<sep>",
                    "delete",
                    "n0",
                    "n1",
                    "r2",
                    "<sep",
                    "add",
                    "n0",
                    "n1",
                    "r0",
                    "<sep>",
                    "add",
                    "n2",
                    "n3",
                    "r2",
                ],
                ["c", "d", "b", "a", "<sep>", "g", "h", "i", "j"],
            ],
            torch.tensor(
                (
                    (2 * 9 / 9 * 9 / 11 / (9 / 9 + 9 / 11))
                    + (2 * 7 / 9 * 7 / 9 / (7 / 9 + 7 / 9))
                )
                / 2
            ),
        ),
    ],
)
def test_f1(batch_preds, batch_targets, expected):
    f1 = F1()
    f1.update(batch_preds, batch_targets)
    assert f1.compute().equal(expected)
