import pytest
import torch
import torch.nn.functional as F

from dgu.nn.utils import (
    masked_mean,
    masked_softmax,
    compute_masks_from_event_type_ids,
    load_fasttext,
)
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.preprocessor import PAD, UNK, SpacyPreprocessor


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
            torch.zeros(1, 3),
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
            torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
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
            torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0]]),
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


def test_load_fasttext():
    preprocessor = SpacyPreprocessor([PAD, UNK, "my", "name", "is", "peter"])
    emb = load_fasttext("tests/data/test-fasttext.vec", preprocessor)
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
