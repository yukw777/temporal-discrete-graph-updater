import pytest
import torch
import torch.nn.functional as F

from pathlib import Path

from dgu.nn.utils import (
    masked_mean,
    masked_softmax,
    compute_masks_from_event_type_ids,
    load_fasttext,
    find_indices,
    pad_batch_seq_of_seq,
)
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.preprocessor import PAD, UNK, SpacyPreprocessor


@pytest.mark.parametrize(
    "batched_input,batched_mask,expected",
    [
        (torch.rand(3, 0, 8), torch.ones(3, 0), torch.zeros(3, 8)),
        (torch.rand(3, 0, 8), torch.zeros(3, 0), torch.zeros(3, 8)),
        (torch.rand(3, 1, 8), torch.zeros(3, 1), torch.zeros(3, 8)),
        (
            torch.tensor(
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
            ).float(),
            torch.tensor(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ).float(),
            torch.tensor(
                [
                    [2, 3, 200],
                    [8, 3, 200],
                ]
            ).float(),
        ),
    ],
)
def test_masked_mean(batched_input, batched_mask, expected):
    assert masked_mean(batched_input, batched_mask).equal(expected)


def test_masked_softmax():
    batched_input = torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float()
    batched_mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float()
    batched_output = masked_softmax(batched_input, batched_mask, dim=1)

    # compare the result from masked_softmax with regular softmax with filtered values
    for input, mask, output in zip(batched_input, batched_mask, batched_output):
        assert output[output != 0].equal(F.softmax(input[mask == 1], dim=0))


@pytest.mark.parametrize(
    "event_type_ids,expected_event_mask,expected_src_mask,expected_dst_mask",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ]
                ]
            ),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ]
                ]
            ),
            torch.ones(1, 2),
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, 2),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.ones(1, 2),
            torch.ones(1, 2),
            torch.ones(1, 2),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ]
                ]
            ),
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_compute_masks_from_event_type_ids(
    event_type_ids, expected_event_mask, expected_src_mask, expected_dst_mask
):
    (
        event_mask,
        src_mask,
        dst_mask,
    ) = compute_masks_from_event_type_ids(event_type_ids)
    assert event_mask.equal(expected_event_mask)
    assert src_mask.equal(expected_src_mask)
    assert dst_mask.equal(expected_dst_mask)


def test_load_fasttext(tmpdir):
    serialized_path = Path(tmpdir / "word-emb.pt")
    assert not serialized_path.exists()
    preprocessor = SpacyPreprocessor([PAD, UNK, "my", "name", "is", "peter"])
    emb = load_fasttext("tests/data/test-fasttext.vec", serialized_path, preprocessor)
    word_ids, _ = preprocessor.preprocess_tokenized(
        [
            ["hi", "there", "what's", "your", "name"],
            ["my", "name", "is", "peter"],
        ]
    )
    embedded = emb(word_ids)
    # OOVs
    assert embedded[0, :4].equal(
        emb(torch.tensor(preprocessor.unk_id)).unsqueeze(0).expand(4, -1)
    )
    # name
    assert embedded[0, 4].equal(emb(torch.tensor(3)))
    # my name is peter
    assert embedded[1, :4].equal(emb(torch.tensor([2, 3, 4, 5])))
    # pad, should be zero
    assert embedded[1, 4].equal(torch.zeros(300))

    with open(serialized_path, "rb") as f:
        assert emb.weight.equal(torch.load(f).weight)


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        (
            torch.tensor([], dtype=torch.long),
            torch.tensor([1, 2]),
            torch.tensor([], dtype=torch.long),
        ),
        (
            torch.tensor([0, 1, 2, 5, 7], dtype=torch.long),
            torch.tensor([1, 7]),
            torch.tensor([1, 4], dtype=torch.long),
        ),
        (
            torch.tensor([3, 2, 5, 6, 8], dtype=torch.long),
            torch.tensor([6, 2]),
            torch.tensor([3, 1], dtype=torch.long),
        ),
    ],
)
def test_find_indices(haystack, needle, expected):
    assert find_indices(haystack, needle).equal(expected)


@pytest.mark.parametrize(
    "batch_seq_of_seq,max_len_outer,max_len_inner,outer_padding_value,"
    "inner_padding_value,expected",
    [
        (
            [[[0, 1], [1, 2]], [[3, 4, 5], [6, 7, 8, 9], [10, 11]]],
            3,
            4,
            0,
            0,
            torch.tensor(
                [
                    [[0, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]],
                    [[3, 4, 5, 0], [6, 7, 8, 9], [10, 11, 0, 0]],
                ]
            ),
        ),
        (
            [[[0, 1], [1, 2]], [[3, 4, 5], [6, 7, 8, 9], [10, 11]]],
            3,
            4,
            -1,
            -2,
            torch.tensor(
                [
                    [[0, 1, -2, -2], [1, 2, -2, -2], [-1, -1, -1, -1]],
                    [[3, 4, 5, -2], [6, 7, 8, 9], [10, 11, -2, -2]],
                ]
            ),
        ),
        (
            [
                [[(0, 1)], [(2, 3), (4, 5)]],
                [
                    [(6, 7), (8, 9)],
                    [(10, 11), (12, 13), (13, 14)],
                    [(15, 16), (0, 1), (0, 2), (0, 3)],
                ],
            ],
            3,
            4,
            (0, 0),
            0,
            torch.tensor(
                [
                    [
                        [[0, 1], [0, 0], [0, 0], [0, 0]],
                        [[2, 3], [4, 5], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    [
                        [[6, 7], [8, 9], [0, 0], [0, 0]],
                        [[10, 11], [12, 13], [13, 14], [0, 0]],
                        [[15, 16], [0, 1], [0, 2], [0, 3]],
                    ],
                ]
            ),
        ),
        (
            [
                [[(0, 1)], [(2, 3), (4, 5)]],
                [
                    [(6, 7), (8, 9)],
                    [(10, 11), (12, 13), (13, 14)],
                    [(15, 16), (0, 1), (0, 2), (0, 3)],
                ],
            ],
            3,
            4,
            (-1, -1),
            -2,
            torch.tensor(
                [
                    [
                        [[0, 1], [-2, -2], [-2, -2], [-2, -2]],
                        [[2, 3], [4, 5], [-2, -2], [-2, -2]],
                        [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                    ],
                    [
                        [[6, 7], [8, 9], [-2, -2], [-2, -2]],
                        [[10, 11], [12, 13], [13, 14], [-2, -2]],
                        [[15, 16], [0, 1], [0, 2], [0, 3]],
                    ],
                ]
            ),
        ),
    ],
)
def test_pad_batch_seq_of_seq(
    batch_seq_of_seq,
    max_len_outer,
    max_len_inner,
    outer_padding_value,
    inner_padding_value,
    expected,
):
    assert pad_batch_seq_of_seq(
        batch_seq_of_seq,
        max_len_outer,
        max_len_inner,
        outer_padding_value,
        inner_padding_value,
    ).equal(expected)
