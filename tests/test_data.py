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
        "vocabs/node_vocab.txt",
        "vocabs/relation_vocab.txt",
    )


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            [
                {
                    "game": "tw-cooking-recipe1+cook+open+go6-XPXQs56DSgePt1mK.z8",
                    "walkthrough_step": 0,
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
                            "label": "open",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 0,
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
                "subgraph_node_ids": torch.tensor([0, 1, 2, 3]),
                "subgraph_edge_ids": torch.tensor([0, 1, 2]),
                "tgt_event_timestamps": torch.tensor([0.0, 0.0, 0.0, 0.0]),
                "tgt_event_mask": torch.tensor([0.0, 1.0, 1.0, 1.0]),
                "tgt_event_type_ids": torch.tensor([1, 3, 3, 5]),
                "tgt_event_src_ids": torch.tensor([0, 0, 1, 0]),
                "tgt_event_src_mask": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "tgt_event_dst_ids": torch.tensor([0, 0, 0, 1]),
                "tgt_event_dst_mask": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "tgt_event_edge_ids": torch.tensor([0, 0, 0, 0]),
                "tgt_event_label_ids": torch.tensor([0, 1, 7, 101]),
                "groundtruth_event_type_ids": torch.tensor([3, 3, 5, 2]),
                "groundtruth_event_src_ids": torch.tensor([0, 1, 0, 0]),
                "groundtruth_event_src_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
                "groundtruth_event_dst_ids": torch.tensor([0, 0, 1, 0]),
                "groundtruth_event_dst_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
                "groundtruth_event_label_ids": torch.tensor([1, 7, 101, 0]),
            },
        ),
        (
            [
                {
                    "game": "tw-cooking-recipe1+cook+open+go6-XPXQs56DSgePt1mK.z8",
                    "walkthrough_step": 0,
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
                            "label": "open",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 0,
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 0,
                            "label": "is",
                        },
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 0,
                            "label": "front door",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 2,
                            "dst_id": 0,
                            "timestamp": 0,
                            "label": "west of",
                        },
                    ],
                },
                {
                    "game": "tw-cooking-recipe1+take1+cook+cut+open-"
                    "x9BZfZmVUZ5Ks1m1.z8",
                    "walkthrough_step": 0,
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
                            "edge_id": 2,
                            "src_id": 3,
                            "dst_id": 4,
                            "timestamp": 1,
                            "label": "on",
                        },
                    ],
                },
                {
                    "game": "tw-cooking-recipe3+cook+go9-MrgKh1g1iOlEunVo.z8",
                    "walkthrough_step": 1,
                    "observation": "you take the flour",
                    "previous_action": "take flour",
                    "event_seq": [
                        {
                            "type": "edge-add",
                            "edge_id": 3,
                            "src_id": 3,
                            "dst_id": 0,
                            "timestamp": 2,
                            "label": "in",
                        },
                        {
                            "type": "edge-delete",
                            "edge_id": 2,
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
                "subgraph_node_ids": torch.tensor(
                    [
                        0,
                        1,
                        2,
                        3,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                        34,
                        35,
                        36,
                        37,
                        38,
                        39,
                    ]
                ),
                "subgraph_edge_ids": torch.tensor(
                    [
                        0,
                        1,
                        2,
                        15,
                        16,
                        17,
                        18,
                        19,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                        33,
                        34,
                    ]
                ),
                "tgt_event_timestamps": torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]
                ),
                "tgt_event_mask": torch.tensor(
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
                "tgt_event_type_ids": torch.tensor([1, 3, 3, 5, 3, 5, 3, 3, 5, 5, 6]),
                "tgt_event_src_ids": torch.tensor([0, 0, 1, 0, 2, 2, 3, 4, 3, 3, 3]),
                "tgt_event_src_mask": torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                ),
                "tgt_event_dst_ids": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 4]),
                "tgt_event_dst_mask": torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                ),
                "tgt_event_edge_ids": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 2, 3, 2]),
                "tgt_event_label_ids": torch.tensor(
                    [0, 1, 7, 101, 45, 105, 43, 82, 102, 100, 102]
                ),
                "groundtruth_event_type_ids": torch.tensor(
                    [3, 3, 5, 3, 5, 3, 3, 5, 5, 6, 2]
                ),
                "groundtruth_event_src_ids": torch.tensor(
                    [0, 1, 0, 2, 2, 3, 4, 3, 3, 3, 0]
                ),
                "groundtruth_event_src_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
                ),
                "groundtruth_event_dst_ids": torch.tensor(
                    [0, 0, 1, 0, 0, 0, 0, 4, 0, 4, 0]
                ),
                "groundtruth_event_dst_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
                ),
                "groundtruth_event_label_ids": torch.tensor(
                    [1, 7, 101, 45, 105, 43, 82, 102, 100, 102, 0]
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_tw_cmd_gen_datamodule_collate(tw_cmd_gen_datamodule, stage, batch, expected):
    tw_cmd_gen_datamodule.setup()
    collated = tw_cmd_gen_datamodule.collate(stage, batch)
    assert collated["obs_word_ids"].equal(expected["obs_word_ids"])
    assert collated["obs_mask"].equal(expected["obs_mask"])
    assert collated["prev_action_word_ids"].equal(expected["prev_action_word_ids"])
    assert collated["prev_action_mask"].equal(expected["prev_action_mask"])
    assert collated["subgraph_node_ids"].equal(expected["subgraph_node_ids"])
    assert collated["subgraph_edge_ids"].equal(expected["subgraph_edge_ids"])
    assert collated["tgt_event_timestamps"].equal(expected["tgt_event_timestamps"])
    assert collated["tgt_event_mask"].equal(expected["tgt_event_mask"])
    assert collated["tgt_event_type_ids"].equal(expected["tgt_event_type_ids"])
    assert collated["tgt_event_src_ids"].equal(expected["tgt_event_src_ids"])
    assert collated["tgt_event_src_mask"].equal(expected["tgt_event_src_mask"])
    assert collated["tgt_event_dst_ids"].equal(expected["tgt_event_dst_ids"])
    assert collated["tgt_event_dst_mask"].equal(expected["tgt_event_dst_mask"])
    assert collated["tgt_event_edge_ids"].equal(expected["tgt_event_edge_ids"])
    assert collated["tgt_event_label_ids"].equal(expected["tgt_event_label_ids"])
    assert collated["groundtruth_event_type_ids"].equal(
        expected["groundtruth_event_type_ids"]
    )
    assert collated["groundtruth_event_src_ids"].equal(
        expected["groundtruth_event_src_ids"]
    )
    assert collated["groundtruth_event_src_mask"].equal(
        expected["groundtruth_event_src_mask"]
    )
    assert collated["groundtruth_event_dst_ids"].equal(
        expected["groundtruth_event_dst_ids"]
    )
    assert collated["groundtruth_event_dst_mask"].equal(
        expected["groundtruth_event_dst_mask"]
    )
    assert collated["groundtruth_event_label_ids"].equal(
        expected["groundtruth_event_label_ids"]
    )


def test_read_label_vocab_files():
    label_id_map = TWCmdGenTemporalDataModule.read_label_vocab_files(
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
