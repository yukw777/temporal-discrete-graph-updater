import pytest
import torch

from tdgu.metrics import F1, DynamicGraphNodeF1


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


@pytest.mark.parametrize(
    "preds,target,expected",
    [
        (torch.zeros(0, 0), torch.zeros(0, 0).int(), torch.tensor(0)),
        (torch.zeros(1, 0), torch.zeros(1, 0).int(), torch.tensor(0)),
        (torch.ones(1, 1), torch.zeros(1).int(), torch.tensor(0)),
        (
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]).float(),
            torch.tensor([1, 1, 2]),
            torch.tensor((0 + 2 / 3 + 1) / 3),
        ),
    ],
)
def test_dynamic_graph_node_f1(
    preds: torch.Tensor, target: torch.Tensor, expected: torch.Tensor
) -> None:
    f1 = DynamicGraphNodeF1()
    f1.update(preds, target)
    assert f1.compute().equal(expected)
