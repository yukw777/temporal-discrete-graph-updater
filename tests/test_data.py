import pytest
import json
import torch
import pickle
import shutil

from pathlib import Path
from unittest.mock import MagicMock

from dgu.data import (
    TemporalDataBatchSampler,
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
    TWCmdGenTemporalDataCollator,
    read_label_vocab_files,
    TWCmdGenTemporalTextualInput,
    TWCmdGenTemporalBatch,
    TWCmdGenTemporalGraphicalInput,
)
from dgu.preprocessor import SpacyPreprocessor


@pytest.fixture
def tw_cmd_gen_datamodule(tmpdir):
    # copy test files to tmpdir so that the serialized files would be saved there
    shutil.copy2("tests/data/test_data.json", tmpdir)
    return TWCmdGenTemporalDataModule(
        tmpdir / "test_data.json",
        1,
        tmpdir / "test_data.json",
        1,
        tmpdir / "test_data.json",
        1,
        "vocabs/word_vocab.txt",
        "vocabs/node_vocab.txt",
        "vocabs/relation_vocab.txt",
        10,
        10,
    )


@pytest.fixture
def tw_cmd_gen_collator():
    return TWCmdGenTemporalDataCollator(
        10,
        10,
        SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt"),
        read_label_vocab_files("vocabs/node_vocab.txt", "vocabs/relation_vocab.txt"),
    )


@pytest.mark.parametrize(
    "batch_size,event_seq_lens,expected_batches",
    [
        (1, [1], [[0]]),
        (
            3,
            [1, 3, 6, 5],
            [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14]],
        ),
    ],
)
def test_temporal_data_batch_sampler(batch_size, event_seq_lens, expected_batches):
    sampler = TemporalDataBatchSampler(batch_size, event_seq_lens)
    assert list(iter(sampler)) == expected_batches
    assert len(sampler) == len(expected_batches)


def test_tw_cmd_gen_dataset_init():
    dataset = TWCmdGenTemporalDataset("tests/data/test_data.json")
    expected_dataset = []
    with open("tests/data/preprocessed_test_data.jsonl") as f:
        for line in f:
            expected_dataset.append(json.loads(line))

    assert len(dataset) == len(expected_dataset)
    for data, expected_data in zip(dataset, expected_dataset):
        assert data == expected_data


@pytest.mark.parametrize(
    "obs,prev_actions,expected",
    [
        (
            ["you are hungry ! let 's cook a delicious meal ."],
            ["drop knife"],
            TWCmdGenTemporalTextualInput(
                obs_word_ids=torch.tensor(
                    [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                ),
                obs_mask=torch.ones(1, 11),
                prev_action_word_ids=torch.tensor([[257, 404]]),
                prev_action_mask=torch.ones(1, 2),
            ),
        ),
        (
            [
                "you are hungry ! let 's cook a delicious meal .",
                "you take the knife from the table .",
            ],
            ["drop knife", "take knife from table"],
            TWCmdGenTemporalTextualInput(
                obs_word_ids=torch.tensor(
                    [
                        [769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21],
                        [769, 663, 676, 404, 315, 676, 661, 21, 0, 0, 0],
                    ]
                ),
                obs_mask=torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                prev_action_word_ids=torch.tensor(
                    [[257, 404, 0, 0], [663, 404, 315, 661]]
                ),
                prev_action_mask=torch.tensor(
                    [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
                ),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_textual_inputs(
    tw_cmd_gen_collator, obs, prev_actions, expected
):
    assert tw_cmd_gen_collator.collate_textual_inputs(obs, prev_actions) == expected


@pytest.mark.parametrize(
    "batch_step,expected",
    [
        (
            [
                {
                    "event_seq": [
                        {
                            "type": "node-add",
                            "timestamp": 0,
                            "label": "player",
                        },
                        {
                            "type": "node-add",
                            "timestamp": 0,
                            "label": "open",
                        },
                        {
                            "type": "edge-add",
                            "timestamp": 0,
                            "label": "is",
                        },
                    ],
                },
            ],
            {
                "tgt_event_type_ids": torch.tensor([[1, 3, 3, 5]]),
                "groundtruth_event_type_ids": torch.tensor([[3, 3, 5, 2]]),
                "tgt_event_timestamps": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                "tgt_event_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
                "tgt_event_src_mask": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                "tgt_event_dst_mask": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                "groundtruth_event_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
                "groundtruth_event_src_mask": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
                "groundtruth_event_dst_mask": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
                "tgt_event_label_ids": torch.tensor([[0, 1, 7, 101]]),
                "groundtruth_event_label_ids": torch.tensor([[1, 7, 101, 0]]),
            },
        ),
        (
            [
                {
                    "event_seq": [
                        {
                            "type": "node-add",
                            "timestamp": 0,
                            "label": "player",
                        },
                        {
                            "type": "node-add",
                            "timestamp": 0,
                            "label": "open",
                        },
                        {
                            "type": "edge-add",
                            "timestamp": 0,
                            "label": "is",
                        },
                    ]
                },
                {
                    "event_seq": [
                        {
                            "type": "node-add",
                            "timestamp": 1,
                            "label": "player",
                        },
                        {
                            "type": "node-add",
                            "timestamp": 2,
                            "label": "open",
                        },
                        {
                            "type": "edge-add",
                            "timestamp": 3,
                            "label": "is",
                        },
                        {
                            "type": "node-delete",
                            "timestamp": 3,
                            "label": "player",
                        },
                        {
                            "type": "node-delete",
                            "timestamp": 4,
                            "label": "open",
                        },
                        {
                            "type": "edge-delete",
                            "timestamp": 5,
                            "label": "is",
                        },
                    ]
                },
                {},
            ],
            {
                "tgt_event_type_ids": torch.tensor(
                    [
                        [1, 3, 3, 5, 0, 0, 0],
                        [1, 3, 3, 5, 4, 4, 6],
                        [1, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_type_ids": torch.tensor(
                    [
                        [3, 3, 5, 2, 0, 0, 0],
                        [3, 3, 5, 4, 4, 6, 2],
                        [2, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                "tgt_event_timestamps": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "tgt_event_mask": torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "tgt_event_src_mask": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "tgt_event_dst_mask": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "groundtruth_event_mask": torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "groundtruth_event_src_mask": torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "groundtruth_event_dst_mask": torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "tgt_event_label_ids": torch.tensor(
                    [
                        [0, 1, 7, 101, 0, 0, 0],
                        [0, 1, 7, 101, 1, 7, 101],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_label_ids": torch.tensor(
                    [
                        [1, 7, 101, 0, 0, 0, 0],
                        [1, 7, 101, 1, 7, 101, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_non_graphical_inputs(
    tw_cmd_gen_collator, batch_step, expected
):
    results = tw_cmd_gen_collator.collate_non_graphical_inputs(batch_step)
    for k in [
        "tgt_event_type_ids",
        "groundtruth_event_type_ids",
        "tgt_event_timestamps",
        "tgt_event_mask",
        "tgt_event_src_mask",
        "tgt_event_dst_mask",
        "groundtruth_event_mask",
        "groundtruth_event_src_mask",
        "groundtruth_event_dst_mask",
        "tgt_event_label_ids",
        "groundtruth_event_label_ids",
    ]:
        assert results[k].equal(expected[k])


@pytest.mark.parametrize(
    "worker_info,batch_step,expected",
    [
        (
            None,
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [],
                }
            ],
            {
                "node_ids": [torch.tensor([[0]])],
                "edge_ids": [torch.tensor([[0]])],
                "edge_index": [torch.tensor([[[0], [0]]])],
                "edge_timestamps": [torch.tensor([[0.0]])],
                "tgt_event_src_ids": torch.tensor([[0]]),
                "tgt_event_dst_ids": torch.tensor([[0]]),
                "tgt_event_edge_ids": torch.tensor([[0]]),
                "groundtruth_event_src_ids": torch.tensor([[0]]),
                "groundtruth_event_dst_ids": torch.tensor([[0]]),
            },
        ),
        (
            (1, 2),
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [],
                }
            ],
            {
                "node_ids": [torch.tensor([[0]])],
                "edge_ids": [torch.tensor([[0]])],
                "edge_index": [torch.tensor([[[0], [0]]])],
                "edge_timestamps": [torch.tensor([[0.0]])],
                "tgt_event_src_ids": torch.tensor([[0]]),
                "tgt_event_dst_ids": torch.tensor([[0]]),
                "tgt_event_edge_ids": torch.tensor([[0]]),
                "groundtruth_event_src_ids": torch.tensor([[0]]),
                "groundtruth_event_dst_ids": torch.tensor([[0]]),
            },
        ),
        (
            None,
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 2,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 1,
                            "dst_id": 2,
                            "timestamp": 2,
                            "label": "e0",
                        },
                    ],
                }
            ],
            {
                "node_ids": [
                    torch.tensor([[0]]),
                    torch.tensor([[0, 1]]),
                    torch.tensor([[0, 1, 2]]),
                    torch.tensor([[0, 1, 2]]),
                ],
                "edge_ids": [
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0, 1]]),
                ],
                "edge_index": [
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0, 1], [0, 2]]]),
                ],
                "edge_timestamps": [
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0, 2.0]]),
                ],
                "tgt_event_src_ids": torch.tensor([[0, 1, 2, 1]]),
                "tgt_event_dst_ids": torch.tensor([[0, 0, 0, 2]]),
                "tgt_event_edge_ids": torch.tensor([[0, 0, 0, 1]]),
                "groundtruth_event_src_ids": torch.tensor([[0, 0, 1, 0]]),
                "groundtruth_event_dst_ids": torch.tensor([[0, 0, 2, 0]]),
            },
        ),
        (
            (1, 2),
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 2,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 1,
                            "dst_id": 2,
                            "timestamp": 2,
                            "label": "e0",
                        },
                    ],
                }
            ],
            {
                "node_ids": [
                    torch.tensor([[0]]),
                    torch.tensor([[0, 6]]),
                    torch.tensor([[0, 6, 7]]),
                    torch.tensor([[0, 6, 7]]),
                ],
                "edge_ids": [
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0, 6]]),
                ],
                "edge_index": [
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0], [0]]]),
                    torch.tensor([[[0, 6], [0, 7]]]),
                ],
                "edge_timestamps": [
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0]]),
                    torch.tensor([[2.0, 2.0]]),
                ],
                "tgt_event_src_ids": torch.tensor([[0, 6, 7, 6]]),
                "tgt_event_dst_ids": torch.tensor([[0, 0, 0, 7]]),
                "tgt_event_edge_ids": torch.tensor([[0, 0, 0, 6]]),
                "groundtruth_event_src_ids": torch.tensor([[0, 0, 1, 0]]),
                "groundtruth_event_dst_ids": torch.tensor([[0, 0, 2, 0]]),
            },
        ),
        (
            None,
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 1,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 1,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 1,
                            "dst_id": 2,
                            "timestamp": 1,
                            "label": "e0",
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 1,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 3,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 4,
                            "timestamp": 2,
                            "label": "n1",
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 1,
                    "event_seq": [
                        {
                            "type": "edge-add",
                            "edge_id": 2,
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 2,
                            "label": "e0",
                        },
                        {
                            "type": "edge-delete",
                            "edge_id": 2,
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 2,
                            "label": "e0",
                        },
                        {
                            "type": "node-delete",
                            "node_id": 3,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-delete",
                            "node_id": 4,
                            "timestamp": 2,
                            "label": "n1",
                        },
                    ],
                },
                {},
            ],
            {
                "node_ids": [
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 1, 0], [0, 3, 0], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 1, 2], [0, 3, 4], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 1, 2], [0, 0, 0], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 3, 4], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 3, 4], [0, 0, 0]]),
                ],
                "edge_ids": [
                    torch.tensor([[0], [0], [0], [0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 2], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 2], [0, 0]]),
                    torch.tensor([[0, 1], [0, 0], [0, 2], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 2], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 2], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 2], [0, 0]]),
                ],
                "edge_index": [
                    torch.tensor([[[0], [0]], [[0], [0]], [[0], [0]], [[0], [0]]]),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 1], [0, 2]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 3], [0, 4]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                ],
                "edge_timestamps": [
                    torch.tensor([[1.0], [2.0], [2.0], [0.0]]),
                    torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                ],
                "tgt_event_src_ids": torch.tensor(
                    [
                        [0, 1, 2, 1, 0],
                        [0, 3, 4, 0, 0],
                        [0, 3, 3, 3, 4],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "tgt_event_dst_ids": torch.tensor(
                    [
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0],
                        [0, 4, 4, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "tgt_event_edge_ids": torch.tensor(
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 2, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_src_ids": torch.tensor(
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_dst_ids": torch.tensor(
                    [
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            },
        ),
        (
            (1, 2),
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 1,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 1,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 1,
                            "dst_id": 2,
                            "timestamp": 1,
                            "label": "e0",
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 1,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 3,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 4,
                            "timestamp": 2,
                            "label": "n1",
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 1,
                    "event_seq": [
                        {
                            "type": "edge-add",
                            "edge_id": 2,
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 2,
                            "label": "e0",
                        },
                        {
                            "type": "edge-delete",
                            "edge_id": 2,
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 2,
                            "label": "e0",
                        },
                        {
                            "type": "node-delete",
                            "node_id": 3,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-delete",
                            "node_id": 4,
                            "timestamp": 2,
                            "label": "n1",
                        },
                    ],
                },
                {},
            ],
            {
                "node_ids": [
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 6, 0], [0, 8, 0], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 6, 7], [0, 8, 9], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 6, 7], [0, 0, 0], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 8, 9], [0, 0, 0]]),
                    torch.tensor([[0, 0, 0], [0, 0, 0], [0, 8, 9], [0, 0, 0]]),
                ],
                "edge_ids": [
                    torch.tensor([[0], [0], [0], [0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 7], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 7], [0, 0]]),
                    torch.tensor([[0, 6], [0, 0], [0, 7], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 7], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 7], [0, 0]]),
                    torch.tensor([[0, 0], [0, 0], [0, 7], [0, 0]]),
                ],
                "edge_index": [
                    torch.tensor([[[0], [0]], [[0], [0]], [[0], [0]], [[0], [0]]]),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 6], [0, 7]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 8], [0, 9]],
                            [[0, 0], [0, 0]],
                        ]
                    ),
                ],
                "edge_timestamps": [
                    torch.tensor([[1.0], [2.0], [2.0], [0.0]]),
                    torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                    torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]),
                ],
                "tgt_event_src_ids": torch.tensor(
                    [
                        [0, 6, 7, 6, 0],
                        [0, 8, 9, 0, 0],
                        [0, 8, 8, 8, 9],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "tgt_event_dst_ids": torch.tensor(
                    [
                        [0, 0, 0, 7, 0],
                        [0, 0, 0, 0, 0],
                        [0, 9, 9, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "tgt_event_edge_ids": torch.tensor(
                    [
                        [0, 0, 0, 6, 0],
                        [0, 0, 0, 0, 0],
                        [0, 7, 7, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_src_ids": torch.tensor(
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 2, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "groundtruth_event_dst_ids": torch.tensor(
                    [
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            },
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_graphical_inputs(
    tw_cmd_gen_collator, worker_info, batch_step, expected
):
    tw_cmd_gen_collator.init_worker_id_space(worker_info)
    results = tw_cmd_gen_collator.collate_graphical_inputs(batch_step)
    for k in [
        "node_ids",
        "edge_ids",
        "edge_index",
        "edge_timestamps",
    ]:
        for el, expected_el in zip(results[k], expected[k]):
            assert el.equal(expected_el)
    for k in [
        "tgt_event_src_ids",
        "tgt_event_dst_ids",
        "tgt_event_edge_ids",
        "groundtruth_event_src_ids",
        "groundtruth_event_dst_ids",
    ]:
        assert results[k].equal(expected[k])


def test_read_label_vocab_files():
    label_id_map = read_label_vocab_files(
        "tests/data/test_node_vocab.txt", "tests/data/test_relation_vocab.txt"
    )
    assert label_id_map == {
        "": 0,
        "player": 1,
        "inventory": 2,
        "chopped": 3,
        "in": 4,
        "is": 5,
    }


@pytest.mark.parametrize(
    "path,expected",
    [
        ("data-path/data.json", Path("data-path/data.pickle")),
        ("/abs/data-path/data.json", Path("/abs/data-path/data.pickle")),
    ],
)
def test_tw_cmd_gen_datamodule_get_serialized_path(path, expected):
    assert TWCmdGenTemporalDataModule.get_serialized_path(path) == expected


def test_tw_cmd_gen_datamodule_serialize_dataset(tmpdir):
    original_dataset = TWCmdGenTemporalDataset("tests/data/test_data.json")

    serialized_path = tmpdir / "test_data.pickle"
    assert not serialized_path.exists()
    TWCmdGenTemporalDataModule.serialize_dataset(
        "tests/data/test_data.json", serialized_path
    )
    with open(serialized_path, "rb") as f:
        assert original_dataset == pickle.load(f)


@pytest.mark.parametrize(
    "worker_info,events,expected",
    [
        (
            None,
            [
                ("g1", 0, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 2}),
                (
                    "g1",
                    0,
                    {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 2},
                ),
            ],
            [
                (
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1}, {0: 0})},
                ),
                (
                    [3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0})},
                ),
                (
                    [3, 4, 5, 6, 7, 8, 9],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1})},
                ),
            ],
        ),
        (
            None,
            [
                ("g1", 0, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 2}),
                (
                    "g1",
                    0,
                    {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 2},
                ),
                ("g1", 1, {"type": "node-add", "node_id": 3}),
                ("g1", 1, {"type": "node-add", "node_id": 4}),
                (
                    "g1",
                    1,
                    {"type": "edge-add", "edge_id": 2, "src_id": 3, "dst_id": 4},
                ),
            ],
            [
                (
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1}, {0: 0})},
                ),
                (
                    [3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0})},
                ),
                (
                    [3, 4, 5, 6, 7, 8, 9],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1})},
                ),
                (
                    [4, 5, 6, 7, 8, 9],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1}),
                        ("g1", 1): ({0: 0, 3: 3}, {0: 0}),
                    },
                ),
                (
                    [5, 6, 7, 8, 9],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1}),
                        ("g1", 1): ({0: 0, 3: 3, 4: 4}, {0: 0}),
                    },
                ),
                (
                    [5, 6, 7, 8, 9],
                    [3, 4, 5, 6, 7, 8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 1, 2: 2}, {0: 0, 1: 1}),
                        ("g1", 1): ({0: 0, 3: 3, 4: 4}, {0: 0, 2: 2}),
                    },
                ),
            ],
        ),
        (
            (1, 2),
            [
                ("g1", 0, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 2}),
                (
                    "g1",
                    0,
                    {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 2},
                ),
            ],
            [
                (
                    [7, 8, 9],
                    [6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6}, {0: 0})},
                ),
                (
                    [8, 9],
                    [6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0})},
                ),
                (
                    [8, 9],
                    [7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0, 1: 6})},
                ),
            ],
        ),
        (
            (1, 2),
            [
                ("g1", 0, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 2}),
                (
                    "g1",
                    0,
                    {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 2},
                ),
                ("g1", 1, {"type": "node-add", "node_id": 3}),
                ("g1", 1, {"type": "node-add", "node_id": 4}),
                (
                    "g1",
                    1,
                    {"type": "edge-add", "edge_id": 2, "src_id": 3, "dst_id": 4},
                ),
            ],
            [
                (
                    [7, 8, 9],
                    [6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6}, {0: 0})},
                ),
                (
                    [8, 9],
                    [6, 7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0})},
                ),
                (
                    [8, 9],
                    [7, 8, 9],
                    {("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0, 1: 6})},
                ),
                (
                    [9],
                    [7, 8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0, 1: 6}),
                        ("g1", 1): ({0: 0, 3: 8}, {0: 0}),
                    },
                ),
                (
                    [],
                    [7, 8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0, 1: 6}),
                        ("g1", 1): ({0: 0, 3: 8, 4: 9}, {0: 0}),
                    },
                ),
                (
                    [],
                    [8, 9],
                    {
                        ("g1", 0): ({0: 0, 1: 6, 2: 7}, {0: 0, 1: 6}),
                        ("g1", 1): ({0: 0, 3: 8, 4: 9}, {0: 0, 2: 7}),
                    },
                ),
            ],
        ),
    ],
)
def test_tw_cmd_gen_collator_allocate_worker_ids(
    tw_cmd_gen_collator, worker_info, events, expected
):
    def check():
        for (game, walkthrough_step, event), (
            expected_unused_node_ids,
            expected_unused_edge_ids,
            expected_allocated_id_map,
        ) in zip(events, expected):
            tw_cmd_gen_collator.update_subgraph(game, walkthrough_step, event)
            tw_cmd_gen_collator.allocate_worker_ids(game, walkthrough_step)
            assert (
                list(tw_cmd_gen_collator.unused_worker_node_ids)
                == expected_unused_node_ids
            )
            assert (
                list(tw_cmd_gen_collator.unused_worker_edge_ids)
                == expected_unused_edge_ids
            )
            assert (
                tw_cmd_gen_collator.allocated_global_worker_id_map
                == expected_allocated_id_map
            )

    tw_cmd_gen_collator.init_worker_id_space(worker_info)
    check()
    tw_cmd_gen_collator.init_worker_id_space(worker_info)
    check()


@pytest.mark.parametrize(
    "events,expected_local_node_ids,expected_local_edges",
    [
        (
            [("g0", 0, {"type": "node-add", "node_id": 1})],
            {("g0", 0): {0, 1}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 1}),
                ("g0", 0, {"type": "node-delete", "node_id": 1}),
            ],
            {("g0", 0): {0, 1}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2}),
                ("g0", 1, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 3}),
                ("g0", 1, {"type": "node-add", "node_id": 4}),
                ("g1", 0, {"type": "node-add", "node_id": 5}),
                ("g0", 1, {"type": "node-delete", "node_id": 1}),
                ("g1", 0, {"type": "node-delete", "node_id": 3}),
            ],
            {("g0", 0): {0, 2}, ("g0", 1): {0, 1, 4}, ("g1", 0): {0, 3, 5}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2}),
                ("g0", 1, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 3}),
                ("g0", 1, {"type": "node-add", "node_id": 4}),
                ("g1", 0, {"type": "node-add", "node_id": 5}),
                ("g0", 1, {"type": "node-delete", "node_id": 1}),
                ("g1", 0, {"type": "node-delete", "node_id": 3}),
                ("g0", 1, {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 4}),
                ("g1", 0, {"type": "edge-add", "edge_id": 2, "src_id": 5, "dst_id": 3}),
            ],
            {("g0", 0): {0, 2}, ("g0", 1): {0, 1, 4}, ("g1", 0): {0, 3, 5}},
            {("g0", 1): {0: (0, 0), 1: (1, 4)}, ("g1", 0): {0: (0, 0), 2: (5, 3)}},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2}),
                ("g0", 1, {"type": "node-add", "node_id": 1}),
                ("g1", 0, {"type": "node-add", "node_id": 3}),
                ("g0", 1, {"type": "node-add", "node_id": 4}),
                ("g1", 0, {"type": "node-add", "node_id": 5}),
                ("g0", 1, {"type": "node-delete", "node_id": 1}),
                ("g1", 0, {"type": "node-delete", "node_id": 3}),
                ("g0", 1, {"type": "edge-add", "edge_id": 1, "src_id": 1, "dst_id": 4}),
                ("g1", 0, {"type": "edge-add", "edge_id": 2, "src_id": 5, "dst_id": 3}),
                (
                    "g1",
                    0,
                    {"type": "edge-delete", "edge_id": 2, "src_id": 5, "dst_id": 3},
                ),
            ],
            {("g0", 0): {0, 2}, ("g0", 1): {0, 1, 4}, ("g1", 0): {0, 3, 5}},
            {("g0", 1): {0: (0, 0), 1: (1, 4)}, ("g1", 0): {0: (0, 0), 2: (5, 3)}},
        ),
    ],
)
def test_tw_cmd_gen_collator_update_subgraph(
    tw_cmd_gen_collator, events, expected_local_node_ids, expected_local_edges
):
    tw_cmd_gen_collator.init_worker_id_space(None)
    for game, walkthrough_step, event in events:
        tw_cmd_gen_collator.update_subgraph(game, walkthrough_step, event)
    assert tw_cmd_gen_collator.global_node_ids == expected_local_node_ids
    assert tw_cmd_gen_collator.global_edges == expected_local_edges


@pytest.mark.parametrize(
    "max_node_id,max_edge_id,worker_info,expected_unused_worker_node_ids,"
    "expected_unused_worker_edge_ids",
    [
        (10, 10, None, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (12, 16, (0, 4), [1, 2, 3], [1, 2, 3, 4]),
        (12, 16, (1, 4), [4, 5, 6], [5, 6, 7, 8]),
        (12, 16, (2, 4), [7, 8, 9], [9, 10, 11, 12]),
        (12, 16, (3, 4), [10, 11], [13, 14, 15]),
    ],
)
def test_tw_cmd_gen_collator_init_worker_id_space(
    max_node_id,
    max_edge_id,
    worker_info,
    expected_unused_worker_node_ids,
    expected_unused_worker_edge_ids,
):
    collator = TWCmdGenTemporalDataCollator(
        max_node_id,
        max_edge_id,
        SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt"),
        {},
    )
    assert not collator.worker_id_space_initialized
    with pytest.raises(AttributeError):
        getattr(collator, "unused_worker_node_ids")
    with pytest.raises(AttributeError):
        getattr(collator, "unused_worker_edge_ids")
    with pytest.raises(AttributeError):
        getattr(collator, "allocated_global_worker_id_map")
    collator.init_worker_id_space(worker_info)
    assert collator.worker_id_space_initialized
    assert list(collator.unused_worker_node_ids) == expected_unused_worker_node_ids
    assert list(collator.unused_worker_edge_ids) == expected_unused_worker_edge_ids
    assert collator.allocated_global_worker_id_map == {}


@pytest.mark.parametrize(
    "worker_info,batch,expected",
    [
        (
            None,
            [
                [
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you are hungry ! "
                        "let 's cook a delicious meal .",
                        "previous_action": "drop knife",
                        "timestamp": 2,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 1,
                                "timestamp": 2,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 2,
                                "timestamp": 2,
                                "label": "kitchen",
                            },
                            {
                                "type": "edge-add",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 2,
                                "label": "in",
                            },
                        ],
                    }
                ]
            ],
            TWCmdGenTemporalBatch(
                data=(
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                            ),
                            obs_mask=torch.ones(1, 11),
                            prev_action_word_ids=torch.tensor([[257, 404]]),
                            prev_action_mask=torch.ones(1, 2),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[1]]),
                                tgt_event_src_ids=torch.tensor([[0]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[0]]),
                                groundtruth_event_type_ids=torch.tensor([[3]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[3]]),
                                tgt_event_src_ids=torch.tensor([[1]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[1]]),
                                groundtruth_event_type_ids=torch.tensor([[3]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[3]]),
                                tgt_event_src_ids=torch.tensor([[2]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[14]]),
                                groundtruth_event_type_ids=torch.tensor([[5]]),
                                groundtruth_event_src_ids=torch.tensor([[1]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2]]),
                                edge_ids=torch.tensor([[0, 1]]),
                                edge_index=torch.tensor([[[0, 1], [0, 2]]]),
                                edge_timestamps=torch.tensor([[2.0, 2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[5]]),
                                tgt_event_src_ids=torch.tensor([[1]]),
                                tgt_event_src_mask=torch.tensor([[1.0]]),
                                tgt_event_dst_ids=torch.tensor([[2]]),
                                tgt_event_dst_mask=torch.tensor([[1.0]]),
                                tgt_event_edge_ids=torch.tensor([[1]]),
                                tgt_event_label_ids=torch.tensor([[100]]),
                                groundtruth_event_type_ids=torch.tensor([[2]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                        ),
                    ),
                )
            ),
        ),
        (
            (1, 2),
            [
                [
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you are hungry ! "
                        "let 's cook a delicious meal .",
                        "previous_action": "drop knife",
                        "timestamp": 2,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 1,
                                "timestamp": 2,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 2,
                                "timestamp": 2,
                                "label": "kitchen",
                            },
                            {
                                "type": "edge-add",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 2,
                                "label": "in",
                            },
                        ],
                    }
                ]
            ],
            TWCmdGenTemporalBatch(
                data=(
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                            ),
                            obs_mask=torch.ones(1, 11),
                            prev_action_word_ids=torch.tensor([[257, 404]]),
                            prev_action_mask=torch.ones(1, 2),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[1]]),
                                tgt_event_src_ids=torch.tensor([[0]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[0]]),
                                groundtruth_event_type_ids=torch.tensor([[3]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[3]]),
                                tgt_event_src_ids=torch.tensor([[6]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[1]]),
                                groundtruth_event_type_ids=torch.tensor([[3]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[3]]),
                                tgt_event_src_ids=torch.tensor([[7]]),
                                tgt_event_src_mask=torch.tensor([[0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0]]),
                                tgt_event_label_ids=torch.tensor([[14]]),
                                groundtruth_event_type_ids=torch.tensor([[5]]),
                                groundtruth_event_src_ids=torch.tensor([[1]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7]]),
                                edge_ids=torch.tensor([[0, 6]]),
                                edge_index=torch.tensor([[[0, 6], [0, 7]]]),
                                edge_timestamps=torch.tensor([[2.0, 2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
                                tgt_event_mask=torch.tensor([[1.0]]),
                                tgt_event_type_ids=torch.tensor([[5]]),
                                tgt_event_src_ids=torch.tensor([[6]]),
                                tgt_event_src_mask=torch.tensor([[1.0]]),
                                tgt_event_dst_ids=torch.tensor([[7]]),
                                tgt_event_dst_mask=torch.tensor([[1.0]]),
                                tgt_event_edge_ids=torch.tensor([[6]]),
                                tgt_event_label_ids=torch.tensor([[100]]),
                                groundtruth_event_type_ids=torch.tensor([[2]]),
                                groundtruth_event_src_ids=torch.tensor([[0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0]]),
                                groundtruth_event_mask=torch.tensor([[1.0]]),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            None,
            [
                [
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you are hungry ! "
                        "let 's cook a delicious meal .",
                        "previous_action": "drop knife",
                        "timestamp": 2,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 1,
                                "timestamp": 2,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 2,
                                "timestamp": 2,
                                "label": "kitchen",
                            },
                            {
                                "type": "edge-add",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 2,
                                "label": "in",
                            },
                        ],
                    },
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 3,
                        "event_seq": [
                            {
                                "type": "edge-delete",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 3,
                                "label": "in",
                            },
                            {
                                "type": "node-delete",
                                "node_id": 1,
                                "timestamp": 3,
                                "label": "player",
                            },
                            {
                                "type": "node-delete",
                                "node_id": 2,
                                "timestamp": 3,
                                "label": "kitchen",
                            },
                        ],
                    },
                ],
                [
                    {
                        "game": "g2",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 1,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 3,
                                "timestamp": 1,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 4,
                                "timestamp": 1,
                                "label": "kitchen",
                            },
                        ],
                    },
                ],
            ],
            TWCmdGenTemporalBatch(
                (
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [
                                    [769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21],
                                    [769, 663, 676, 404, 315, 676, 661, 21, 0, 0, 0],
                                ]
                            ),
                            obs_mask=torch.tensor(
                                [
                                    [
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                    ],
                                    [
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    ],
                                ]
                            ),
                            prev_action_word_ids=torch.tensor(
                                [[257, 404, 0, 0], [663, 404, 315, 661]]
                            ),
                            prev_action_mask=torch.tensor(
                                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
                            ),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0], [0]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[1], [1]]),
                                tgt_event_src_ids=torch.tensor([[0], [0]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[3], [3]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1], [1]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1], [0, 3]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[3], [3]]),
                                tgt_event_src_ids=torch.tensor([[1], [3]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[1], [1]]),
                                groundtruth_event_type_ids=torch.tensor([[3], [3]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14], [14]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 3, 4]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[3], [3]]),
                                tgt_event_src_ids=torch.tensor([[2], [4]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[14], [14]]),
                                groundtruth_event_type_ids=torch.tensor([[5], [2]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[2.0, 2.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[5], [0]]),
                                tgt_event_src_ids=torch.tensor([[1], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[2], [0]]),
                                tgt_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[1], [0]]),
                                tgt_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                        ),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [
                                    [769, 663, 676, 404, 315, 676, 661, 21],
                                    [2, 3, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            obs_mask=torch.tensor(
                                [
                                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            prev_action_word_ids=torch.tensor(
                                [[663, 404, 315, 661], [2, 3, 0, 0]]
                            ),
                            prev_action_mask=torch.tensor(
                                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
                            ),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[1], [1]]),
                                tgt_event_src_ids=torch.tensor([[0], [0]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[6], [2]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[6], [0]]),
                                tgt_event_src_ids=torch.tensor([[1], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[2], [0]]),
                                tgt_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[1], [0]]),
                                tgt_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[4], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[4], [0]]),
                                tgt_event_src_ids=torch.tensor([[1], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[4], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[4], [0]]),
                                tgt_event_src_ids=torch.tensor([[2], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[14], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                        ),
                    ),
                )
            ),
        ),
        (
            (1, 2),
            [
                [
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you are hungry ! "
                        "let 's cook a delicious meal .",
                        "previous_action": "drop knife",
                        "timestamp": 2,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 1,
                                "timestamp": 2,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 2,
                                "timestamp": 2,
                                "label": "kitchen",
                            },
                            {
                                "type": "edge-add",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 2,
                                "label": "in",
                            },
                        ],
                    },
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 3,
                        "event_seq": [
                            {
                                "type": "edge-delete",
                                "edge_id": 1,
                                "src_id": 1,
                                "dst_id": 2,
                                "timestamp": 3,
                                "label": "in",
                            },
                            {
                                "type": "node-delete",
                                "node_id": 1,
                                "timestamp": 3,
                                "label": "player",
                            },
                            {
                                "type": "node-delete",
                                "node_id": 2,
                                "timestamp": 3,
                                "label": "kitchen",
                            },
                        ],
                    },
                ],
                [
                    {
                        "game": "g2",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 1,
                        "event_seq": [
                            {
                                "type": "node-add",
                                "node_id": 3,
                                "timestamp": 1,
                                "label": "player",
                            },
                            {
                                "type": "node-add",
                                "node_id": 4,
                                "timestamp": 1,
                                "label": "kitchen",
                            },
                        ],
                    },
                ],
            ],
            TWCmdGenTemporalBatch(
                data=(
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [
                                    [769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21],
                                    [769, 663, 676, 404, 315, 676, 661, 21, 0, 0, 0],
                                ]
                            ),
                            obs_mask=torch.tensor(
                                [
                                    [
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                    ],
                                    [
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    ],
                                ]
                            ),
                            prev_action_word_ids=torch.tensor(
                                [[257, 404, 0, 0], [663, 404, 315, 661]]
                            ),
                            prev_action_mask=torch.tensor(
                                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
                            ),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0], [0]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[1], [1]]),
                                tgt_event_src_ids=torch.tensor([[0], [0]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[3], [3]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1], [1]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6], [0, 8]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[3], [3]]),
                                tgt_event_src_ids=torch.tensor([[6], [8]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[1], [1]]),
                                groundtruth_event_type_ids=torch.tensor([[3], [3]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14], [14]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 8, 9]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[3], [3]]),
                                tgt_event_src_ids=torch.tensor([[7], [9]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[14], [14]]),
                                groundtruth_event_type_ids=torch.tensor([[5], [2]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 6], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 6], [0, 7]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[2.0, 2.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[5], [0]]),
                                tgt_event_src_ids=torch.tensor([[6], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[7], [0]]),
                                tgt_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[6], [0]]),
                                tgt_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                        ),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(
                            obs_word_ids=torch.tensor(
                                [
                                    [769, 663, 676, 404, 315, 676, 661, 21],
                                    [2, 3, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            obs_mask=torch.tensor(
                                [
                                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            prev_action_word_ids=torch.tensor(
                                [[663, 404, 315, 661], [2, 3, 0, 0]]
                            ),
                            prev_action_mask=torch.tensor(
                                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
                            ),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 6], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 6], [0, 7]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [1.0]]),
                                tgt_event_type_ids=torch.tensor([[1], [1]]),
                                tgt_event_src_ids=torch.tensor([[0], [0]]),
                                tgt_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[6], [2]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [1.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 6], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 6], [0, 7]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[6], [0]]),
                                tgt_event_src_ids=torch.tensor([[6], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[7], [0]]),
                                tgt_event_dst_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[6], [0]]),
                                tgt_event_label_ids=torch.tensor([[100], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[4], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 6], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 6], [0, 7]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[4], [0]]),
                                tgt_event_src_ids=torch.tensor([[6], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[1], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[4], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[14], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_ids=torch.tensor([[0, 6, 7], [0, 0, 0]]),
                                edge_ids=torch.tensor([[0, 6], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 6], [0, 7]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
                                tgt_event_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_type_ids=torch.tensor([[4], [0]]),
                                tgt_event_src_ids=torch.tensor([[7], [0]]),
                                tgt_event_src_mask=torch.tensor([[1.0], [0.0]]),
                                tgt_event_dst_ids=torch.tensor([[0], [0]]),
                                tgt_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                tgt_event_edge_ids=torch.tensor([[0], [0]]),
                                tgt_event_label_ids=torch.tensor([[14], [0]]),
                                groundtruth_event_type_ids=torch.tensor([[2], [0]]),
                                groundtruth_event_src_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_src_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_dst_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_dst_mask=torch.tensor([[0.0], [0.0]]),
                                groundtruth_event_label_ids=torch.tensor([[0], [0]]),
                                groundtruth_event_mask=torch.tensor([[1.0], [0.0]]),
                            ),
                        ),
                    ),
                )
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_call(
    monkeypatch, tw_cmd_gen_collator, worker_info, batch, expected
):
    if worker_info is not None:
        mock_worker_info = MagicMock()
        mock_worker_info.id = worker_info[0]
        mock_worker_info.num_workers = worker_info[1]
        monkeypatch.setattr("dgu.data.get_worker_info", lambda: mock_worker_info)
    tw_cmd_gen_collator.init_worker_id_space(worker_info)
    assert tw_cmd_gen_collator(batch) == expected


@pytest.mark.parametrize(
    "textual",
    [
        TWCmdGenTemporalTextualInput(
            obs_word_ids=torch.randint(10, (1, 15)),
            obs_mask=torch.randint(2, (1, 15)).float(),
            prev_action_word_ids=torch.randint(10, (1, 3)),
            prev_action_mask=torch.randint(2, (1, 3)).float(),
        ),
        TWCmdGenTemporalTextualInput(
            obs_word_ids=torch.randint(10, (3, 10)),
            obs_mask=torch.randint(2, (3, 10)).float(),
            prev_action_word_ids=torch.randint(10, (3, 5)),
            prev_action_mask=torch.randint(2, (3, 5)).float(),
        ),
    ],
)
def test_tw_cmd_gen_temporal_textual_input_to(textual):
    # just check that we're creating a correct copy
    assert textual.to("cpu") == textual


@pytest.mark.parametrize(
    "graphical",
    [
        TWCmdGenTemporalGraphicalInput(
            node_ids=torch.randint(10, (3, 10)),
            edge_ids=torch.randint(15, (3, 15)),
            edge_index=torch.randint(15, (3, 2, 15)),
            edge_timestamps=torch.rand(3, 15),
            tgt_event_timestamps=torch.rand(3, 6),
            tgt_event_mask=torch.randint(2, (3, 6)).float(),
            tgt_event_type_ids=torch.randint(7, (3, 6)),
            tgt_event_src_ids=torch.randint(10, (3, 6)),
            tgt_event_src_mask=torch.randint(2, (3, 6)).float(),
            tgt_event_dst_ids=torch.randint(10, (3, 6)),
            tgt_event_dst_mask=torch.randint(2, (3, 6)).float(),
            tgt_event_edge_ids=torch.randint(15, (3, 6)),
            tgt_event_label_ids=torch.randint(20, (3, 6)),
            groundtruth_event_type_ids=torch.randint(7, (3, 6)),
            groundtruth_event_src_ids=torch.randint(10, (3, 6)),
            groundtruth_event_src_mask=torch.randint(2, (3, 6)).float(),
            groundtruth_event_dst_ids=torch.randint(10, (3, 6)),
            groundtruth_event_dst_mask=torch.randint(2, (3, 6)).float(),
            groundtruth_event_label_ids=torch.randint(20, (3, 6)),
            groundtruth_event_mask=torch.randint(2, (3, 6)).float(),
        )
    ],
)
def test_tw_cmd_gen_temporal_graphical_input_to(graphical):
    # just check that we're creating a correct copy
    assert graphical.to("cpu") == graphical


@pytest.mark.parametrize(
    "batch",
    [
        TWCmdGenTemporalBatch(
            data=(
                (
                    TWCmdGenTemporalTextualInput(
                        obs_word_ids=torch.randint(10, (1, 15)),
                        obs_mask=torch.randint(2, (1, 15)).float(),
                        prev_action_word_ids=torch.randint(10, (1, 3)),
                        prev_action_mask=torch.randint(2, (1, 3)).float(),
                    ),
                    (
                        TWCmdGenTemporalGraphicalInput(
                            node_ids=torch.randint(10, (3, 10)),
                            edge_ids=torch.randint(15, (3, 15)),
                            edge_index=torch.randint(15, (3, 2, 15)),
                            edge_timestamps=torch.rand(3, 15),
                            tgt_event_timestamps=torch.rand(3, 6),
                            tgt_event_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_type_ids=torch.randint(7, (3, 6)),
                            tgt_event_src_ids=torch.randint(10, (3, 6)),
                            tgt_event_src_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_dst_ids=torch.randint(10, (3, 6)),
                            tgt_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_edge_ids=torch.randint(15, (3, 6)),
                            tgt_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_type_ids=torch.randint(7, (3, 6)),
                            groundtruth_event_src_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_src_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_dst_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_mask=torch.randint(2, (3, 6)).float(),
                        ),
                    ),
                ),
                (
                    TWCmdGenTemporalTextualInput(
                        obs_word_ids=torch.randint(10, (3, 10)),
                        obs_mask=torch.randint(2, (3, 10)).float(),
                        prev_action_word_ids=torch.randint(10, (3, 5)),
                        prev_action_mask=torch.randint(2, (3, 5)).float(),
                    ),
                    (
                        TWCmdGenTemporalGraphicalInput(
                            node_ids=torch.randint(10, (3, 10)),
                            edge_ids=torch.randint(15, (3, 15)),
                            edge_index=torch.randint(15, (3, 2, 15)),
                            edge_timestamps=torch.rand(3, 15),
                            tgt_event_timestamps=torch.rand(3, 6),
                            tgt_event_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_type_ids=torch.randint(7, (3, 6)),
                            tgt_event_src_ids=torch.randint(10, (3, 6)),
                            tgt_event_src_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_dst_ids=torch.randint(10, (3, 6)),
                            tgt_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_edge_ids=torch.randint(15, (3, 6)),
                            tgt_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_type_ids=torch.randint(7, (3, 6)),
                            groundtruth_event_src_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_src_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_dst_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_mask=torch.randint(2, (3, 6)).float(),
                        ),
                        TWCmdGenTemporalGraphicalInput(
                            node_ids=torch.randint(10, (3, 10)),
                            edge_ids=torch.randint(15, (3, 15)),
                            edge_index=torch.randint(15, (3, 2, 15)),
                            edge_timestamps=torch.rand(3, 15),
                            tgt_event_timestamps=torch.rand(3, 6),
                            tgt_event_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_type_ids=torch.randint(7, (3, 6)),
                            tgt_event_src_ids=torch.randint(10, (3, 6)),
                            tgt_event_src_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_dst_ids=torch.randint(10, (3, 6)),
                            tgt_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            tgt_event_edge_ids=torch.randint(15, (3, 6)),
                            tgt_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_type_ids=torch.randint(7, (3, 6)),
                            groundtruth_event_src_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_src_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_dst_ids=torch.randint(10, (3, 6)),
                            groundtruth_event_dst_mask=torch.randint(2, (3, 6)).float(),
                            groundtruth_event_label_ids=torch.randint(20, (3, 6)),
                            groundtruth_event_mask=torch.randint(2, (3, 6)).float(),
                        ),
                    ),
                ),
            )
        )
    ],
)
def test_tw_cmd_gen_temporal_batch_to(batch):
    # just check that we're creating a correct copy
    assert batch.to("cpu") == batch
