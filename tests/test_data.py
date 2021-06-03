import pytest
import json
import torch

from dgu.data import (
    TemporalDataBatchSampler,
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
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


@pytest.fixture
def tw_cmd_gen_datamodule():
    return TWCmdGenTemporalDataModule(
        "tests/data/test_data.json",
        1,
        1,
        "tests/data/test_data.json",
        1,
        1,
        "tests/data/test_data.json",
        1,
        1,
        "vocabs/word_vocab.txt",
    )


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            [
                {
                    "observation": "you are hungry !",
                    "previous_action": "drop flour",
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 0,
                            "timestamp": 0,
                            "label": "player",
                        },
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 0,
                            "label": "hungry",
                        },
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 0,
                            "label": "is",
                        },
                    ],
                }
            ],
            {
                "obs_word_ids": torch.tensor([[769, 122, 377, 5]]),
                "obs_mask": torch.ones(1, 4),
                "prev_action_word_ids": torch.tensor([[257, 305]]),
                "prev_action_mask": torch.ones(1, 2),
                "event_type_ids": torch.tensor([2, 2, 4]),
                "event_timestamps": torch.tensor([0.0, 0.0, 0.0]),
                "event_src_ids": torch.tensor([0, 1, 0]),
                "event_src_mask": torch.tensor([0.0, 0.0, 1.0]),
                "event_dst_ids": torch.tensor([0, 0, 1]),
                "event_dst_mask": torch.tensor([0.0, 0.0, 1.0]),
                "event_label_word_ids": torch.tensor([[530], [377], [396]]),
                "event_label_mask": torch.ones(3, 1),
            },
        ),
        (
            [
                {
                    "observation": "you are hungry !",
                    "previous_action": "drop flour",
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 0,
                            "timestamp": 0,
                            "label": "player",
                        },
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 0,
                            "label": "hungry",
                        },
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 0,
                            "label": "is",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 0,
                            "label": "door",
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 0,
                            "timestamp": 0,
                            "label": "west of",
                        },
                    ],
                },
                {
                    "observation": "you put the flour on the sofa",
                    "previous_action": "put flour on sofa",
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 3,
                            "timestamp": 1,
                            "label": "flour",
                        },
                        {
                            "type": "node-add",
                            "node_id": 4,
                            "timestamp": 1,
                            "label": "sofa",
                        },
                        {
                            "type": "edge-add",
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 1,
                            "label": "on",
                        },
                    ],
                },
                {
                    "observation": "you take the flour",
                    "previous_action": "take flour",
                    "event_seq": [
                        {
                            "type": "edge-add",
                            "src_id": 3,
                            "dst_id": 0,
                            "timestamp": 2,
                            "label": "in",
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 2,
                            "label": "on",
                        },
                    ],
                },
            ],
            {
                "obs_word_ids": torch.tensor(
                    [
                        [769, 122, 377, 5, 0, 0, 0],
                        [769, 546, 676, 305, 494, 676, 620],
                        [769, 663, 676, 305, 0, 0, 0],
                    ]
                ),
                "obs_mask": torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "prev_action_word_ids": torch.tensor(
                    [[257, 305, 0, 0], [546, 305, 494, 620], [663, 305, 0, 0]]
                ),
                "prev_action_mask": torch.tensor(
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0],
                    ]
                ),
                "event_type_ids": torch.tensor([2, 2, 4, 2, 4, 2, 2, 4, 4, 5]),
                "event_timestamps": torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]
                ),
                "event_src_ids": torch.tensor([0, 1, 0, 2, 2, 3, 4, 3, 3, 3]),
                "event_src_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                ),
                "event_dst_ids": torch.tensor([0, 0, 1, 0, 0, 0, 0, 4, 0, 4]),
                "event_dst_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                ),
                "event_label_word_ids": torch.tensor(
                    [
                        [530, 0],
                        [377, 0],
                        [396, 0],
                        [251, 0],
                        [742, 486],
                        [305, 0],
                        [620, 0],
                        [494, 0],
                        [382, 0],
                        [494, 0],
                    ]
                ),
                "event_label_mask": torch.tensor(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 0.0],
                    ]
                ),
            },
        ),
    ],
)
def test_tw_cmd_gen_datamodule_collate(tw_cmd_gen_datamodule, batch, expected):
    collated = tw_cmd_gen_datamodule.collate(batch)
    assert collated["obs_word_ids"].equal(expected["obs_word_ids"])
    assert collated["obs_mask"].equal(expected["obs_mask"])
    assert collated["prev_action_word_ids"].equal(expected["prev_action_word_ids"])
    assert collated["prev_action_mask"].equal(expected["prev_action_mask"])
    assert collated["event_type_ids"].equal(expected["event_type_ids"])
    assert collated["event_timestamps"].equal(expected["event_timestamps"])
    assert collated["event_src_ids"].equal(expected["event_src_ids"])
    assert collated["event_dst_ids"].equal(expected["event_dst_ids"])
    assert collated["event_dst_mask"].equal(expected["event_dst_mask"])
    assert collated["event_label_word_ids"].equal(expected["event_label_word_ids"])
    assert collated["event_label_mask"].equal(expected["event_label_mask"])
