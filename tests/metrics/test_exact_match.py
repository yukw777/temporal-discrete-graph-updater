import pytest

from dgu.metrics import ExactMatch


@pytest.mark.parametrize(
    "batch_preds,batch_targets,batch_expected",
    [
        ([[[]]], [[[]]], [1]),
        ([[["add , n0 , n1 , r"]]], [[[]]], [0]),
        ([[[]]], [[["add , n0 , n1 , r"]]], [0]),
        ([[["add , n0 , n1 , r"]]], [[["add , n0 , n1 , r"]]], [1]),
        ([[["add", "n0", "n1", "r"]]], [[["add", "n0", "n1", "r"]]], [1]),
        ([[["add , n0 , n1 , r"]]], [[["delete , n0 , n1 , r"]]], [0]),
        ([[["add", "n0", "n1", "r"]]], [[["delete", "n2", "n3", "r0"]]], [0]),
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
                    ["add", "n0", "n1", "r0", "add", "n2", "n3", "r1"],
                    ["delete", "n0", "n1", "r0"],
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
                    ["add", "n0", "n1", "r0", "delete", "n2", "n3", "r1"],
                    ["add", "n4", "n5", "r2"],
                ],
                [
                    ["delete", "n2", "n3", "r1", "add", "n0", "n1", "r0"],
                    ["delete", "n0", "n1", "r2"],
                    [],
                    ["add", "n0", "n1", "r0", "add", "n2", "n3", "r2"],
                ],
            ],
            [0.5, (1 + 0 + 1 + 3 / 4 + 1 + 6 / 8) / 6],
        ),
    ],
)
def test_exact_match(batch_preds, batch_targets, batch_expected):
    em = ExactMatch()
    for preds, targets, expected in zip(batch_preds, batch_targets, batch_expected):
        em.update(preds, targets)
        assert em.compute() == expected
