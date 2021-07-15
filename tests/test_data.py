import pytest
import json
import torch
import pickle
import shutil

from pathlib import Path

from dgu.graph import TextWorldGraph
from dgu.data import (
    TemporalDataBatchSampler,
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
)


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
    "batch,expected",
    [
        (
            [
                {
                    "observation": "you are hungry ! let 's cook a delicious meal .",
                    "previous_action": "drop knife",
                },
            ],
            {
                "obs_word_ids": torch.tensor(
                    [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                ),
                "obs_mask": torch.ones(1, 11),
                "prev_action_word_ids": torch.tensor([[257, 404]]),
                "prev_action_mask": torch.ones(1, 2),
            },
        ),
        (
            [
                {
                    "observation": "you are hungry ! let 's cook a delicious meal .",
                    "previous_action": "drop knife",
                },
                {
                    "observation": "you take the knife from the table .",
                    "previous_action": "take knife from table",
                },
            ],
            {
                "obs_word_ids": torch.tensor(
                    [
                        [769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21],
                        [769, 663, 676, 404, 315, 676, 661, 21, 0, 0, 0],
                    ]
                ),
                "obs_mask": torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "prev_action_word_ids": torch.tensor(
                    [[257, 404, 0, 0], [663, 404, 315, 661]]
                ),
                "prev_action_mask": torch.tensor(
                    [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
                ),
            },
        ),
    ],
)
def test_collate_textual_inputs(tw_cmd_gen_datamodule, batch, expected):
    results = tw_cmd_gen_datamodule.collate_textual_inputs(batch)
    for k in ["obs_word_ids", "obs_mask", "prev_action_word_ids", "prev_action_mask"]:
        assert results[k].equal(expected[k])


@pytest.mark.parametrize(
    "batch,expected",
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
                }
            ],
            {
                "tgt_event_type_ids": torch.tensor([1, 3, 3, 5]),
                "groundtruth_event_type_ids": torch.tensor([3, 3, 5, 2]),
                "tgt_event_timestamps": torch.tensor([0.0, 0.0, 0.0, 0.0]),
                "tgt_event_mask": torch.tensor([0.0, 1.0, 1.0, 1.0]),
                "tgt_event_src_mask": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "tgt_event_dst_mask": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "groundtruth_event_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "groundtruth_event_src_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
                "groundtruth_event_dst_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
                "tgt_event_label_ids": torch.tensor([0, 1, 7, 101]),
                "groundtruth_event_label_ids": torch.tensor([1, 7, 101, 0]),
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
                    ],
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
                    ],
                },
            ],
            {
                "tgt_event_type_ids": torch.tensor([1, 3, 3, 5, 3, 3, 5, 4, 4, 6]),
                "groundtruth_event_type_ids": torch.tensor(
                    [3, 3, 5, 3, 3, 5, 4, 4, 6, 2]
                ),
                "tgt_event_timestamps": torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
                ),
                "tgt_event_mask": torch.tensor(
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
                "tgt_event_src_mask": torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
                ),
                "tgt_event_dst_mask": torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
                ),
                "groundtruth_event_mask": torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                ),
                "groundtruth_event_src_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                ),
                "groundtruth_event_dst_mask": torch.tensor(
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
                ),
                "tgt_event_label_ids": torch.tensor(
                    [0, 1, 7, 101, 1, 7, 101, 1, 7, 101]
                ),
                "groundtruth_event_label_ids": torch.tensor(
                    [1, 7, 101, 1, 7, 101, 1, 7, 101, 0]
                ),
            },
        ),
    ],
)
def test_collate_non_graphical_inputs(tw_cmd_gen_datamodule, batch, expected):
    results = tw_cmd_gen_datamodule.collate_non_graphical_inputs(batch)
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
    "prev_batch,batch,expected",
    [
        (
            [{"game": "g0", "walkthrough_step": 0, "event_seq": []}],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 0,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 2,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 0,
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 2,
                            "label": "e0",
                        },
                    ],
                }
            ],
            {
                "node_ids": torch.tensor([0, 1]),
                "edge_ids": torch.tensor([0]),
                "edge_index": torch.tensor([[0], [1]]),
                "edge_timestamps": torch.tensor([2.0]),
                "tgt_event_src_ids": torch.tensor([0, 0, 1, 0]),
                "tgt_event_dst_ids": torch.tensor([0, 0, 0, 1]),
                "tgt_event_edge_ids": torch.tensor([0, 0, 0, 0]),
                "groundtruth_event_src_ids": torch.tensor([0, 1, 0, 0]),
                "groundtruth_event_dst_ids": torch.tensor([0, 0, 1, 0]),
            },
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 0,
                            "timestamp": 1,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 1,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 0,
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 1,
                            "label": "e0",
                        },
                    ],
                }
            ],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 2,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 3,
                            "timestamp": 2,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 2,
                            "dst_id": 3,
                            "timestamp": 2,
                            "label": "e0",
                        },
                    ],
                }
            ],
            {
                "node_ids": torch.tensor([0, 1, 2, 3]),
                "edge_ids": torch.tensor([0, 1]),
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "edge_timestamps": torch.tensor([2.0, 2.0]),
                "tgt_event_src_ids": torch.tensor([0, 2, 3, 2]),
                "tgt_event_dst_ids": torch.tensor([0, 0, 0, 3]),
                "tgt_event_edge_ids": torch.tensor([0, 0, 0, 1]),
                "groundtruth_event_src_ids": torch.tensor([2, 3, 2, 0]),
                "groundtruth_event_dst_ids": torch.tensor([0, 0, 3, 0]),
            },
        ),
        (
            [
                {
                    "game": "g0",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 0,
                            "timestamp": 1,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 1,
                            "timestamp": 1,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 0,
                            "src_id": 0,
                            "dst_id": 1,
                            "timestamp": 1,
                            "label": "e0",
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {
                            "type": "node-add",
                            "node_id": 2,
                            "timestamp": 1,
                            "label": "n0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 3,
                            "timestamp": 1,
                            "label": "n1",
                        },
                        {
                            "type": "edge-add",
                            "edge_id": 1,
                            "src_id": 2,
                            "dst_id": 3,
                            "timestamp": 1,
                            "label": "e0",
                        },
                        {
                            "type": "node-add",
                            "node_id": 4,
                            "timestamp": 1,
                            "label": "n2",
                        },
                        {
                            "type": "node-add",
                            "node_id": 5,
                            "timestamp": 1,
                            "label": "n3",
                        },
                        {
                            "type": "node-add",
                            "node_id": 6,
                            "timestamp": 1,
                            "label": "n4",
                        },
                    ],
                },
            ],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "event_seq": [
                        {
                            "type": "edge-add",
                            "edge_id": 2,
                            "src_id": 4,
                            "dst_id": 5,
                            "timestamp": 2,
                            "label": "e1",
                        },
                        {
                            "type": "node-delete",
                            "node_id": 6,
                            "timestamp": 1,
                            "label": "n4",
                        },
                    ],
                },
            ],
            {
                "node_ids": torch.tensor([2, 3, 4, 5, 6]),
                "edge_ids": torch.tensor([1, 2]),
                "edge_index": torch.tensor([[0, 2], [1, 3]]),
                "edge_timestamps": torch.tensor([2.0, 2.0]),
                "tgt_event_src_ids": torch.tensor([0, 4, 6]),
                "tgt_event_dst_ids": torch.tensor([0, 5, 0]),
                "tgt_event_edge_ids": torch.tensor([0, 2, 0]),
                "groundtruth_event_src_ids": torch.tensor([2, 4, 0]),
                "groundtruth_event_dst_ids": torch.tensor([3, 0, 0]),
            },
        ),
    ],
)
def test_collate_graphical_inputs(tw_cmd_gen_datamodule, prev_batch, batch, expected):
    # process the previous batch to set up
    g = TextWorldGraph()
    for prev_ex in prev_batch:
        g.process_events(
            prev_ex["event_seq"],
            game=prev_ex["game"],
            walkthrough_step=prev_ex["walkthrough_step"],
        )
    tw_cmd_gen_datamodule.calculate_subgraph_maps(g, prev_batch)

    results = tw_cmd_gen_datamodule.collate_graphical_inputs(g, batch)
    for k in [
        "node_ids",
        "edge_ids",
        "edge_index",
        "edge_timestamps",
        "tgt_event_src_ids",
        "tgt_event_dst_ids",
        "tgt_event_edge_ids",
        "groundtruth_event_src_ids",
        "groundtruth_event_dst_ids",
    ]:
        assert results[k].equal(expected[k])


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


def test_tw_cmd_gen_datamodule_calc_subgraph_maps(tw_cmd_gen_datamodule):
    # first with one game
    graph = TextWorldGraph()
    for _ in range(5):
        src_id = graph.add_node("n1", game="g1", walkthrough_step=0)
        dst_id = graph.add_node("n2", game="g1", walkthrough_step=0)
        graph.add_edge(src_id, dst_id, "e1", game="g1", walkthrough_step=0)

    node_id_map, edge_id_map = tw_cmd_gen_datamodule.calculate_subgraph_maps(
        graph, [{"game": "g1", "walkthrough_step": 0}]
    )
    assert node_id_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    assert edge_id_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    # now with two new games
    for _ in range(2):
        src_id = graph.add_node("n1", game="g1", walkthrough_step=1)
        dst_id = graph.add_node("n2", game="g1", walkthrough_step=1)
        graph.add_edge(src_id, dst_id, "e1", game="g1", walkthrough_step=1)
    for _ in range(2):
        src_id = graph.add_node("n1", game="g2", walkthrough_step=0)
        dst_id = graph.add_node("n2", game="g2", walkthrough_step=0)
        graph.add_edge(src_id, dst_id, "e1", game="g2", walkthrough_step=0)

    node_id_map, edge_id_map = tw_cmd_gen_datamodule.calculate_subgraph_maps(
        graph,
        [{"game": "g1", "walkthrough_step": 1}, {"game": "g2", "walkthrough_step": 0}],
    )
    assert node_id_map == {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7}
    assert edge_id_map == {5: 5, 6: 6, 7: 7, 8: 8}

    # one old game, one new game
    src_id = graph.add_node("n1", game="g2", walkthrough_step=0)
    dst_id = graph.add_node("n2", game="g2", walkthrough_step=0)
    graph.add_edge(src_id, dst_id, "e1", game="g2", walkthrough_step=0)
    graph.remove_edge(16, 17)
    graph.remove_node(17)
    for _ in range(2):
        src_id = graph.add_node("n1", game="g3", walkthrough_step=0)
        dst_id = graph.add_node("n2", game="g3", walkthrough_step=0)
        graph.add_edge(src_id, dst_id, "e1", game="g1", walkthrough_step=0)

    node_id_map, edge_id_map = tw_cmd_gen_datamodule.calculate_subgraph_maps(
        graph,
        [{"game": "g2", "walkthrough_step": 0}, {"game": "g3", "walkthrough_step": 0}],
    )
    assert node_id_map == {
        20: 0,
        21: 1,
        22: 2,
        23: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 8,
        19: 9,
    }
    assert edge_id_map == {7: 7, 8: 8, 9: 9}
