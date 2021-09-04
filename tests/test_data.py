import pytest
import json
import torch
import shutil

from dgu.data import (
    TemporalDataBatchSampler,
    TWCmdGenTemporalGraphData,
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
    TWCmdGenTemporalDataCollator,
    read_label_vocab_files,
    TWCmdGenTemporalTextualInput,
    TWCmdGenTemporalBatch,
    TWCmdGenTemporalGraphicalInput,
)
from dgu.preprocessor import SpacyPreprocessor
from dgu.graph import Node, IsDstNode, ExitNode
from dgu.constants import EVENT_TYPE_ID_MAP

from utils import EqualityDiGraph


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
    "event_src_index,event_dst_index,timestamp,before_graph,after_graph,label_id_map,expected",
    [
        (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            1,
            EqualityDiGraph(),
            EqualityDiGraph(),
            {},
            TWCmdGenTemporalGraphData(
                x=torch.empty(0, dtype=torch.long),
                node_memory_update_index=torch.empty(0, dtype=torch.long),
                node_memory_update_mask=torch.empty(0, dtype=torch.bool),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.empty(0, dtype=torch.long),
                event_dst_index=torch.empty(0, dtype=torch.long),
            ),
        ),
        (
            torch.tensor(2),
            torch.tensor(1),
            2,
            EqualityDiGraph(),
            EqualityDiGraph({Node("player"): {}}),
            {"player": 5},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5]),
                node_memory_update_index=torch.empty(0, dtype=torch.long),
                node_memory_update_mask=torch.empty(0, dtype=torch.bool),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.tensor(2),
                event_dst_index=torch.tensor(1),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            4,
            EqualityDiGraph({Node("player"): {}}),
            EqualityDiGraph({Node("player"): {}, Node("kitchen"): {}}),
            {"player": 5, "kitchen": 1},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 1]),
                node_memory_update_index=torch.tensor([0]),
                node_memory_update_mask=torch.tensor([True]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(0),
            torch.tensor(1),
            2,
            EqualityDiGraph({Node("player"): {}, Node("kitchen"): {}}),
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 0}}}
            ),
            {"player": 5, "kitchen": 1, "in": 2},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 1]),
                node_memory_update_index=torch.tensor([0, 1]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([0.0]),
                edge_timestamps=torch.tensor([2.0]),
                event_src_index=torch.tensor(0),
                event_dst_index=torch.tensor(1),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            3,
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 0}
                    }
                }
            ),
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 0}
                    },
                    ExitNode("exit", "west of", "kitchen"): {},
                }
            ),
            {"exit": 5, "kitchen": 1, "east of": 2},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 5, 1]),
                node_memory_update_index=torch.tensor([0, 2]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.tensor([[0], [2]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([0.0]),
                edge_timestamps=torch.tensor([3.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            5,
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 0}
                    },
                    ExitNode("exit", "west of", "kitchen"): {},
                }
            ),
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 0}
                    },
                    ExitNode("exit", "west of", "kitchen"): {
                        Node("kitchen"): {"label": "west of", "last_update": 1}
                    },
                }
            ),
            {"exit": 5, "kitchen": 1, "east of": 2, "west of": 3},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 5, 1]),
                node_memory_update_index=torch.tensor([0, 1, 2]),
                node_memory_update_mask=torch.tensor([True, True, True]),
                edge_index=torch.tensor([[0, 1], [2, 2]]),
                edge_attr=torch.tensor([2, 3]),
                edge_last_update=torch.tensor([0.0, 1.0]),
                edge_timestamps=torch.tensor([5.0, 5.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            3,
            EqualityDiGraph(
                {
                    Node("steak"): {
                        IsDstNode("cooked", "steak"): {
                            "label": "is",
                            "last_update": 1,
                        }
                    }
                }
            ),
            EqualityDiGraph(
                {
                    Node("steak"): {
                        IsDstNode("cooked", "steak"): {
                            "label": "is",
                            "last_update": 1,
                        }
                    },
                    IsDstNode("delicious", "steak"): {},
                }
            ),
            {"steak": 5, "cooked": 1, "is": 2, "delicious": 3},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 3, 1]),
                node_memory_update_index=torch.tensor([0, 2]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.tensor([[0], [2]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([1.0]),
                edge_timestamps=torch.tensor([3.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            2,
            EqualityDiGraph(
                {
                    Node("steak"): {
                        IsDstNode("cooked", "steak"): {
                            "label": "is",
                            "last_update": 1,
                        }
                    },
                    IsDstNode("delicious", "steak"): {},
                }
            ),
            EqualityDiGraph(
                {
                    Node("steak"): {
                        IsDstNode("cooked", "steak"): {
                            "label": "is",
                            "last_update": 1,
                        },
                        IsDstNode("delicious", "steak"): {
                            "label": "is",
                            "last_update": 2,
                        },
                    }
                }
            ),
            {"steak": 5, "cooked": 1, "is": 2, "delicious": 3},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 1, 3]),
                node_memory_update_index=torch.tensor([0, 2, 1]),
                node_memory_update_mask=torch.tensor([True, True, True]),
                edge_index=torch.tensor([[0, 0], [1, 2]]),
                edge_attr=torch.tensor([2, 2]),
                edge_last_update=torch.tensor([1.0, 2.0]),
                edge_timestamps=torch.tensor([2.0, 2.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            3,
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 2}}}
            ),
            EqualityDiGraph({Node("player"): {}, Node("kitchen"): {}}),
            {"player": 5, "kitchen": 1, "in": 2},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 1]),
                node_memory_update_index=torch.tensor([0, 1]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            4,
            EqualityDiGraph({Node("player"): {}, Node("kitchen"): {}}),
            EqualityDiGraph({Node("player"): {}}),
            {"player": 5, "kitchen": 1, "in": 2},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5]),
                node_memory_update_index=torch.tensor([0, 0]),
                node_memory_update_mask=torch.tensor([True, False]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            5,
            EqualityDiGraph({Node("player"): {}}),
            EqualityDiGraph(),
            {"player": 5, "kitchen": 1, "in": 2},
            TWCmdGenTemporalGraphData(
                x=torch.empty(0, dtype=torch.long),
                node_memory_update_index=torch.tensor([0]),
                node_memory_update_mask=torch.tensor([False]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            4,
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 2}
                    },
                    ExitNode("exit", "west of", "kitchen"): {
                        Node("kitchen"): {"label": "west of", "last_update": 2}
                    },
                }
            ),
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 2}
                    },
                    ExitNode("exit", "west of", "kitchen"): {},
                }
            ),
            {"exit": 5, "kitchen": 1, "east of": 2, "west of": 3},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 5, 1]),
                node_memory_update_index=torch.tensor([0, 1, 2]),
                node_memory_update_mask=torch.tensor([True, True, True]),
                edge_index=torch.tensor([[0], [2]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([2.0]),
                edge_timestamps=torch.tensor([4.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
        (
            torch.tensor(3),
            torch.tensor(2),
            5,
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 2}
                    },
                    ExitNode("exit", "west of", "kitchen"): {},
                }
            ),
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 2}
                    }
                }
            ),
            {"exit": 5, "kitchen": 1, "east of": 2, "west of": 3},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([5, 1]),
                node_memory_update_index=torch.tensor([0, 0, 1]),
                node_memory_update_mask=torch.tensor([True, False, True]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([2.0]),
                edge_timestamps=torch.tensor([5.0]),
                event_src_index=torch.tensor(3),
                event_dst_index=torch.tensor(2),
            ),
        ),
    ],
)
def test_tw_cmd_gen_temporal_graph_data_from_graph_event(
    event_src_index,
    event_dst_index,
    timestamp,
    before_graph,
    after_graph,
    label_id_map,
    expected,
):
    data = TWCmdGenTemporalGraphData.from_graph_event(
        event_src_index,
        event_dst_index,
        timestamp,
        before_graph,
        after_graph,
        label_id_map,
    )
    assert data.x.equal(expected.x)
    assert data.node_memory_update_index.equal(expected.node_memory_update_index)
    assert data.node_memory_update_mask.equal(expected.node_memory_update_mask)
    assert data.edge_index.equal(expected.edge_index)
    assert data.edge_attr.equal(expected.edge_attr)
    assert data.edge_last_update.equal(expected.edge_last_update)
    assert data.event_src_index.equal(expected.event_src_index)
    assert data.event_dst_index.equal(expected.event_dst_index)


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
        "groundtruth_event_mask",
        "groundtruth_event_src_mask",
        "groundtruth_event_dst_mask",
        "tgt_event_label_ids",
        "groundtruth_event_label_ids",
    ]:
        assert results[k].equal(expected[k])


@pytest.mark.parametrize(
    "batch_graphs,batch_step,expected",
    [
        (
            [EqualityDiGraph()],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": [],
                }
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.empty(0, dtype=torch.long),
                    node_memory_update_index=torch.empty(0).long(),
                    node_memory_update_mask=torch.empty(0).bool(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.empty(0, dtype=torch.long),
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([0]),
                    tgt_event_timestamps=torch.tensor([0.0]),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0]),
                    groundtruth_event_label_ids=torch.tensor([0]),
                    groundtruth_event_mask=torch.tensor([1.0]),
                ),
            ],
        ),
        (
            [EqualityDiGraph()],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                }
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1]),
                    node_memory_update_index=torch.empty(0).long(),
                    node_memory_update_mask=torch.empty(0).bool(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0]),
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([0]),
                    tgt_event_timestamps=torch.tensor([0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0]),
                    groundtruth_event_label_ids=torch.tensor([1]),
                    groundtruth_event_mask=torch.tensor([1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0]),
                    node_memory_update_mask=torch.tensor([True]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([1]),
                    tgt_event_timestamps=torch.tensor([2.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0]),
                    groundtruth_event_label_ids=torch.tensor([14]),
                    groundtruth_event_mask=torch.tensor([1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0, 1]),
                    node_memory_update_mask=torch.tensor([True, True]),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_label_ids=torch.tensor([100]),
                    edge_last_update=torch.tensor([2.0]),
                    edge_timestamps=torch.tensor([2.0]),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([14]),
                    tgt_event_timestamps=torch.tensor([2.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([1.0]),
                    groundtruth_event_dst_ids=torch.tensor([1]),
                    groundtruth_event_dst_mask=torch.tensor([1.0]),
                    groundtruth_event_label_ids=torch.tensor([100]),
                    groundtruth_event_mask=torch.tensor([1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0, 1]),
                    node_memory_update_mask=torch.tensor([True, True]),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_label_ids=torch.tensor([100]),
                    edge_last_update=torch.tensor([2.0]),
                    edge_timestamps=torch.tensor([2.0]),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([1]),
                    tgt_event_label_ids=torch.tensor([100]),
                    tgt_event_timestamps=torch.tensor([2.0]),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0]),
                    groundtruth_event_label_ids=torch.tensor([0]),
                    groundtruth_event_mask=torch.tensor([1.0]),
                ),
            ],
        ),
        (
            [EqualityDiGraph(), EqualityDiGraph()],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 4,
                    "target_commands": ["add , player , livingroom , in"],
                },
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 1]),
                    node_memory_update_index=torch.empty(0).long(),
                    node_memory_update_mask=torch.empty(0).bool(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                    ),
                    # these are [0, 1] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "start".
                    tgt_event_src_ids=torch.tensor([0, 1]),
                    tgt_event_dst_ids=torch.tensor([0, 1]),
                    tgt_event_label_ids=torch.tensor([0, 0]),
                    tgt_event_timestamps=torch.tensor([0.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([1, 1]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14, 1, 16]),
                    node_memory_update_index=torch.tensor([0, 2]),
                    node_memory_update_mask=torch.tensor([True, True]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0, 0, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    # these are [0, 2] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([0, 2]),
                    tgt_event_label_ids=torch.tensor([1, 1]),
                    tgt_event_timestamps=torch.tensor([2.0, 4.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([14, 16]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14, 1, 16]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                    node_memory_update_mask=torch.tensor([True, True, True, True]),
                    edge_index=torch.tensor([[0, 2], [1, 3]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([2.0, 4.0]),
                    edge_timestamps=torch.tensor([2.0, 4.0]),
                    batch=torch.tensor([0, 0, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    # these are [0, 2] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([0, 2]),
                    tgt_event_label_ids=torch.tensor([14, 16]),
                    tgt_event_timestamps=torch.tensor([2.0, 4.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([1, 1]),
                    groundtruth_event_dst_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_label_ids=torch.tensor([100, 100]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14, 1, 16]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                    node_memory_update_mask=torch.tensor([True, True, True, True]),
                    edge_index=torch.tensor([[0, 2], [1, 3]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([2.0, 4.0]),
                    edge_timestamps=torch.tensor([2.0, 4.0]),
                    batch=torch.tensor([0, 0, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([1, 3]),
                    tgt_event_label_ids=torch.tensor([100, 100]),
                    tgt_event_timestamps=torch.tensor([2.0, 4.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["end"], EVENT_TYPE_ID_MAP["end"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([0, 0]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
            ],
        ),
        (
            [EqualityDiGraph(), EqualityDiGraph()],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                },
                {},
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1]),
                    node_memory_update_index=torch.empty(0).long(),
                    node_memory_update_mask=torch.empty(0).bool(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                    ),
                    # these are [0, 1] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "start".
                    tgt_event_src_ids=torch.tensor([0, 1]),
                    tgt_event_dst_ids=torch.tensor([0, 1]),
                    tgt_event_label_ids=torch.tensor([0, 0]),
                    tgt_event_timestamps=torch.tensor([0.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["end"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([1, 0]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0]),
                    node_memory_update_mask=torch.tensor([True]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_label_ids=torch.empty(0).long(),
                    edge_last_update=torch.empty(0),
                    edge_timestamps=torch.empty(0),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["end"]]
                    ),
                    # these are [0, 2] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([0, 2]),
                    tgt_event_label_ids=torch.tensor([1, 0]),
                    tgt_event_timestamps=torch.tensor([2.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["pad"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([14, 0]),
                    groundtruth_event_mask=torch.tensor([1.0, 0.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0, 1]),
                    node_memory_update_mask=torch.tensor([True, True]),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_label_ids=torch.tensor([100]),
                    edge_last_update=torch.tensor([2.0]),
                    edge_timestamps=torch.tensor([2.0]),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["pad"]]
                    ),
                    # these are [0, 2] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([0, 2]),
                    tgt_event_label_ids=torch.tensor([14, 0]),
                    tgt_event_timestamps=torch.tensor([2.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["pad"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([1, 0]),
                    groundtruth_event_dst_mask=torch.tensor([1.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([100, 0]),
                    groundtruth_event_mask=torch.tensor([1.0, 0.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([1, 14]),
                    node_memory_update_index=torch.tensor([0, 1]),
                    node_memory_update_mask=torch.tensor([True, True]),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_label_ids=torch.tensor([100]),
                    edge_last_update=torch.tensor([2.0]),
                    edge_timestamps=torch.tensor([2.0]),
                    batch=torch.tensor([0, 0]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["pad"]]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([1, 2]),
                    tgt_event_label_ids=torch.tensor([100, 0]),
                    tgt_event_timestamps=torch.tensor([2.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["end"], EVENT_TYPE_ID_MAP["pad"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([0, 0]),
                    groundtruth_event_mask=torch.tensor([1.0, 0.0]),
                ),
            ],
        ),
        (
            [
                EqualityDiGraph(
                    {
                        Node("banana"): {
                            Node("livingroom"): {"label": "in", "last_update": 2}
                        }
                    }
                ),
                EqualityDiGraph(
                    {
                        Node("pork chop"): {
                            Node("refrigerator"): {"label": "in", "last_update": 0}
                        }
                    }
                ),
            ],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 3,
                    "target_commands": [
                        "add , player , kitchen , in",
                        "add , chicken leg , kitchen , in",
                        "delete , player , kitchen , in",
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 1,
                    "target_commands": [
                        "add , chicken leg , kitchen , in",
                        "add , player , livingroom , in",
                        "delete , chicken leg , kitchen , in",
                    ],
                },
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([27, 16, 1, 64, 75, 34]),
                    node_memory_update_index=torch.tensor([0, 1, 3, 4]),
                    node_memory_update_mask=torch.tensor([True, True, True, True]),
                    edge_index=torch.tensor([[0, 3], [1, 4]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([2.0, 0.0]),
                    edge_timestamps=torch.tensor([3.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                    ),
                    # these are [0, 3] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "start".
                    tgt_event_src_ids=torch.tensor([0, 3]),
                    tgt_event_dst_ids=torch.tensor([0, 3]),
                    tgt_event_label_ids=torch.tensor([0, 0]),
                    tgt_event_timestamps=torch.tensor([0.0, 0.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([1, 34]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([27, 16, 1, 14, 64, 75, 34, 14]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 4, 5, 6]),
                    node_memory_update_mask=torch.tensor(
                        [True, True, True, True, True, True]
                    ),
                    edge_index=torch.tensor([[0, 4], [1, 5]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([2.0, 0.0]),
                    edge_timestamps=torch.tensor([3.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    # these are [0, 4] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "start".
                    tgt_event_src_ids=torch.tensor([0, 4]),
                    tgt_event_dst_ids=torch.tensor([0, 4]),
                    tgt_event_label_ids=torch.tensor([1, 34]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([14, 14]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([27, 16, 1, 14, 64, 75, 34, 14]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                    node_memory_update_mask=torch.tensor(
                        [True, True, True, True, True, True, True, True]
                    ),
                    edge_index=torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    # these are [0, 2] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 4]),
                    tgt_event_dst_ids=torch.tensor([0, 4]),
                    tgt_event_label_ids=torch.tensor([14, 14]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([2, 2]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([3, 3]),
                    groundtruth_event_dst_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_label_ids=torch.tensor([100, 100]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([27, 16, 1, 14, 34, 64, 75, 34, 14, 1]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]),
                    node_memory_update_mask=torch.tensor(
                        [True, True, True, True, True, True, True, True]
                    ),
                    edge_index=torch.tensor([[0, 2, 5, 7], [1, 3, 6, 8]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 7]),
                    tgt_event_dst_ids=torch.tensor([3, 8]),
                    tgt_event_label_ids=torch.tensor([100, 100]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([34, 1]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor(
                        [27, 16, 1, 14, 34, 64, 75, 34, 14, 1, 16]
                    ),
                    node_memory_update_index=torch.tensor(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    ),
                    node_memory_update_mask=torch.tensor(
                        [True, True, True, True, True, True, True, True, True, True]
                    ),
                    edge_index=torch.tensor([[0, 2, 4, 5, 7], [1, 3, 3, 6, 8]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 3.0, 0.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    # these are [0, 5] instead of [0, 0] due to mini-batching
                    # it doesn't matter as these IDs are not used due to the fact
                    # that the event type is "node-add".
                    tgt_event_src_ids=torch.tensor([0, 5]),
                    tgt_event_dst_ids=torch.tensor([0, 5]),
                    tgt_event_label_ids=torch.tensor([34, 1]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_src_ids=torch.tensor([4, 0]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([3, 0]),
                    groundtruth_event_dst_mask=torch.tensor([1.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([100, 16]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor(
                        [27, 16, 1, 14, 34, 64, 75, 34, 14, 1, 16]
                    ),
                    node_memory_update_index=torch.tensor(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    ),
                    node_memory_update_mask=torch.tensor(
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ]
                    ),
                    edge_index=torch.tensor([[0, 4, 5, 7, 9], [1, 3, 6, 8, 10]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    tgt_event_src_ids=torch.tensor([4, 5]),
                    tgt_event_dst_ids=torch.tensor([3, 5]),
                    tgt_event_label_ids=torch.tensor([100, 16]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    groundtruth_event_src_ids=torch.tensor([2, 4]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([3, 5]),
                    groundtruth_event_dst_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_label_ids=torch.tensor([100, 100]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor(
                        [27, 16, 14, 34, 64, 75, 34, 14, 1, 16]
                    ),
                    node_memory_update_index=torch.tensor(
                        [0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
                    ),
                    node_memory_update_mask=torch.tensor(
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ]
                    ),
                    edge_index=torch.tensor([[0, 3, 4, 8], [1, 2, 5, 9]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 8]),
                    tgt_event_dst_ids=torch.tensor([3, 9]),
                    tgt_event_label_ids=torch.tensor([100, 100]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    groundtruth_event_src_ids=torch.tensor([2, 2]),
                    groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 3]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 1.0]),
                    groundtruth_event_label_ids=torch.tensor([1, 100]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([27, 16, 14, 34, 64, 75, 34, 1, 16]),
                    # The deleted node is assigned 4 instead of 0
                    # b/c of graph mini-batching. The mask takes care of it,
                    # so it's fine.
                    node_memory_update_index=torch.tensor(
                        [0, 1, 2, 3, 4, 5, 6, 4, 7, 8]
                    ),
                    node_memory_update_mask=torch.tensor(
                        [True, True, True, True, True, True, True, False, True, True]
                    ),
                    edge_index=torch.tensor([[0, 3, 4, 7], [1, 2, 5, 8]]),
                    edge_label_ids=torch.tensor([100, 100, 100, 100]),
                    edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                    edge_timestamps=torch.tensor([3.0, 3.0, 1.0, 1.0]),
                    batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 6]),
                    tgt_event_dst_ids=torch.tensor([0, 7]),
                    tgt_event_label_ids=torch.tensor([1, 100]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 3]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([0, 14]),
                    groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([64, 75, 1, 16]),
                    node_memory_update_index=torch.tensor([0, 0, 0, 0, 0, 1, 0, 2, 3]),
                    node_memory_update_mask=torch.tensor(
                        [False, False, False, False, True, True, False, True, True]
                    ),
                    edge_index=torch.tensor([[0, 2], [1, 3]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([0.0, 1.0]),
                    edge_timestamps=torch.tensor([1.0, 1.0]),
                    batch=torch.tensor([1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 3]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_ids=torch.tensor([0, 14]),
                    tgt_event_timestamps=torch.tensor([3.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 2]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 1.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([0, 34]),
                    groundtruth_event_mask=torch.tensor([0.0, 1.0]),
                ),
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=torch.tensor([64, 75, 1, 16]),
                    node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                    node_memory_update_mask=torch.tensor([True, True, True, True]),
                    edge_index=torch.tensor([[0, 2], [1, 3]]),
                    edge_label_ids=torch.tensor([100, 100]),
                    edge_last_update=torch.tensor([0.0, 1.0]),
                    edge_timestamps=torch.tensor([1.0, 1.0]),
                    batch=torch.tensor([1, 1, 1, 1]),
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_ids=torch.tensor([0, 34]),
                    tgt_event_timestamps=torch.tensor([0.0, 1.0]),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["end"],
                        ]
                    ),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                    groundtruth_event_label_ids=torch.tensor([0, 0]),
                    groundtruth_event_mask=torch.tensor([0.0, 1.0]),
                ),
            ],
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_graphical_inputs(
    tw_cmd_gen_collator, batch_graphs, batch_step, expected
):
    assert (
        tw_cmd_gen_collator.collate_graphical_inputs(batch_graphs, batch_step)
        == expected
    )


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
    "events,expected",
    [
        (
            [
                ("g1", 0, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-add", "node_id": 2, "label": "n2"}),
                (
                    "g1",
                    0,
                    {
                        "type": "edge-add",
                        "edge_id": 1,
                        "src_id": 1,
                        "dst_id": 2,
                        "label": "e1",
                    },
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
            [
                ("g1", 0, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-add", "node_id": 2, "label": "n2"}),
                (
                    "g1",
                    0,
                    {
                        "type": "edge-add",
                        "edge_id": 1,
                        "src_id": 1,
                        "dst_id": 2,
                        "label": "e1",
                    },
                ),
                ("g1", 1, {"type": "node-add", "node_id": 3, "label": "n1"}),
                ("g1", 1, {"type": "node-add", "node_id": 4, "label": "n2"}),
                (
                    "g1",
                    1,
                    {
                        "type": "edge-add",
                        "edge_id": 2,
                        "src_id": 3,
                        "dst_id": 4,
                        "label": "e1",
                    },
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
    ],
)
def test_tw_cmd_gen_collator_allocate_ids(tw_cmd_gen_collator, events, expected):
    def check():
        for (game, walkthrough_step, event), (
            expected_unused_node_ids,
            expected_unused_edge_ids,
            expected_allocated_id_map,
        ) in zip(events, expected):
            tw_cmd_gen_collator.update_subgraph(game, walkthrough_step, event)
            tw_cmd_gen_collator.allocate_ids(game, walkthrough_step)
            assert list(tw_cmd_gen_collator.unused_node_ids) == expected_unused_node_ids
            assert list(tw_cmd_gen_collator.unused_edge_ids) == expected_unused_edge_ids
            assert tw_cmd_gen_collator.allocated_id_map == expected_allocated_id_map

    tw_cmd_gen_collator.init_id_space()
    check()
    tw_cmd_gen_collator.init_id_space()
    check()


@pytest.mark.parametrize(
    "events,expected_node_ids,expected_edges",
    [
        (
            [("g0", 0, {"type": "node-add", "node_id": 1, "label": "n1"})],
            {("g0", 0): {0, 1}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g0", 0, {"type": "node-delete", "node_id": 1, "label": "n1"}),
            ],
            {("g0", 0): {0, 1}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2, "label": "n2"}),
                ("g0", 1, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-add", "node_id": 3, "label": "n3"}),
                ("g0", 1, {"type": "node-add", "node_id": 4, "label": "n4"}),
                ("g1", 0, {"type": "node-add", "node_id": 5, "label": "n5"}),
                ("g0", 1, {"type": "node-delete", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-delete", "node_id": 3, "label": "n3"}),
            ],
            {("g0", 0): {0, 2}, ("g0", 1): {0, 1, 4}, ("g1", 0): {0, 3, 5}},
            {},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2, "label": "n2"}),
                ("g0", 1, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-add", "node_id": 3, "label": "n3"}),
                ("g0", 1, {"type": "node-add", "node_id": 4, "label": "n4"}),
                ("g1", 0, {"type": "node-add", "node_id": 5, "label": "n5"}),
                ("g0", 1, {"type": "node-delete", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-delete", "node_id": 3, "label": "n3"}),
                (
                    "g0",
                    1,
                    {
                        "type": "edge-add",
                        "edge_id": 1,
                        "src_id": 1,
                        "dst_id": 4,
                        "label": "e1",
                    },
                ),
                (
                    "g1",
                    0,
                    {
                        "type": "edge-add",
                        "edge_id": 2,
                        "src_id": 5,
                        "dst_id": 3,
                        "label": "e2",
                    },
                ),
            ],
            {("g0", 0): {0, 2}, ("g0", 1): {0, 1, 4}, ("g1", 0): {0, 3, 5}},
            {("g0", 1): {0: (0, 0), 1: (1, 4)}, ("g1", 0): {0: (0, 0), 2: (5, 3)}},
        ),
        (
            [
                ("g0", 0, {"type": "node-add", "node_id": 2, "label": "n2"}),
                ("g0", 1, {"type": "node-add", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-add", "node_id": 3, "label": "n3"}),
                ("g0", 1, {"type": "node-add", "node_id": 4, "label": "n4"}),
                ("g1", 0, {"type": "node-add", "node_id": 5, "label": "n5"}),
                ("g0", 1, {"type": "node-delete", "node_id": 1, "label": "n1"}),
                ("g1", 0, {"type": "node-delete", "node_id": 3, "label": "n3"}),
                (
                    "g0",
                    1,
                    {
                        "type": "edge-add",
                        "edge_id": 1,
                        "src_id": 1,
                        "dst_id": 4,
                        "label": "e1",
                    },
                ),
                (
                    "g1",
                    0,
                    {
                        "type": "edge-add",
                        "edge_id": 2,
                        "src_id": 5,
                        "dst_id": 3,
                        "label": "e2",
                    },
                ),
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
    tw_cmd_gen_collator, events, expected_node_ids, expected_edges
):
    for game, walkthrough_step, event in events:
        tw_cmd_gen_collator.update_subgraph(game, walkthrough_step, event)
    assert tw_cmd_gen_collator.node_ids == expected_node_ids
    assert tw_cmd_gen_collator.edges == expected_edges


@pytest.mark.parametrize(
    "max_node_id,max_edge_id,expected_unused_node_ids,expected_unused_edge_ids",
    [
        (5, 7, [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
        (10, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_tw_cmd_gen_collator_init_id_space(
    max_node_id, max_edge_id, expected_unused_node_ids, expected_unused_edge_ids
):
    collator = TWCmdGenTemporalDataCollator(
        max_node_id,
        max_edge_id,
        SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt"),
        {},
    )
    collator.init_id_space()
    assert list(collator.unused_node_ids) == expected_unused_node_ids
    assert list(collator.unused_edge_ids) == expected_unused_edge_ids
    assert collator.allocated_id_map == {}


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
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
                        "commands": ["add , player , kitchen , in"],
                    },
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
                                node_labels=[{0: "<pad>"}],
                                node_mask=torch.tensor([[0.0]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0]]),
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
                                node_labels=[{0: "<pad>", 1: "player"}],
                                node_mask=torch.tensor([[0.0, 1.0]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
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
                                node_labels=[{0: "<pad>", 1: "player", 2: "kitchen"}],
                                node_mask=torch.tensor([[0, 1.0, 1.0]]),
                                edge_ids=torch.tensor([[0]]),
                                edge_index=torch.tensor([[[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
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
                                node_labels=[{0: "<pad>", 1: "player", 2: "kitchen"}],
                                node_mask=torch.tensor([[0, 1.0, 1.0]]),
                                edge_ids=torch.tensor([[0, 1]]),
                                edge_index=torch.tensor([[[0, 1], [0, 2]]]),
                                edge_timestamps=torch.tensor([[2.0, 2.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0]]),
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
                        (("add , player , kitchen , in",),),
                    ),
                )
            ),
        ),
        (
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
                        "commands": ["add , player , kitchen , in"],
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
                        "commands": ["delete , player , kitchen , in"],
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
                        "commands": ["add , player , kitchen , in"],
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
                                node_labels=[{0: "<pad>"}, {0: "<pad>"}],
                                node_mask=torch.tensor([[0.0], [0.0]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player"},
                                    {0: "<pad>", 3: "player"},
                                ],
                                node_mask=torch.tensor([[0, 1.0], [0, 1.0]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>", 3: "player", 4: "kitchen"},
                                ],
                                node_mask=torch.tensor([[0, 1.0, 1.0], [0, 1.0, 1.0]]),
                                edge_ids=torch.tensor([[0], [0]]),
                                edge_index=torch.tensor([[[0], [0]], [[0], [0]]]),
                                edge_timestamps=torch.tensor([[2.0], [1.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [1.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>"},
                                ],
                                node_mask=torch.tensor(
                                    [[0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                                ),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[2.0, 2.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[2.0], [0.0]]),
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
                        (
                            ("add , player , kitchen , in",),
                            ("add , player , kitchen , in",),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>"},
                                ],
                                node_mask=torch.tensor(
                                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                                ),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[0.0], [0.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>"},
                                ],
                                node_mask=torch.tensor(
                                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                                ),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>"},
                                ],
                                node_mask=torch.tensor(
                                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                                ),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
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
                                node_labels=[
                                    {0: "<pad>", 1: "player", 2: "kitchen"},
                                    {0: "<pad>"},
                                ],
                                node_mask=torch.tensor(
                                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                                ),
                                edge_ids=torch.tensor([[0, 1], [0, 0]]),
                                edge_index=torch.tensor(
                                    [[[0, 1], [0, 2]], [[0, 0], [0, 0]]]
                                ),
                                edge_timestamps=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
                                tgt_event_timestamps=torch.tensor([[3.0], [0.0]]),
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
                        (("delete , player , kitchen , in",), ()),
                    ),
                )
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_call(tw_cmd_gen_collator, batch, expected):
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
                    (("",),),
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
                    (("",), ("",)),
                ),
            )
        )
    ],
)
def test_tw_cmd_gen_temporal_batch_to(batch):
    # just check that we're creating a correct copy
    assert batch.to("cpu") == batch


@pytest.mark.parametrize(
    "batch,split_size,expected",
    [
        (
            TWCmdGenTemporalBatch(
                data=(
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (TWCmdGenTemporalGraphicalInput(),),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                )
            ),
            2,
            [
                TWCmdGenTemporalBatch(
                    data=(
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalTextualInput(),
                            (TWCmdGenTemporalGraphicalInput(),),
                            (("",),),
                        ),
                    )
                ),
                TWCmdGenTemporalBatch(
                    data=(
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                    ),
                ),
            ],
        ),
        (
            TWCmdGenTemporalBatch(
                data=(
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (TWCmdGenTemporalGraphicalInput(),),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalTextualInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                )
            ),
            3,
            [
                TWCmdGenTemporalBatch(
                    data=(
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalTextualInput(),
                            (TWCmdGenTemporalGraphicalInput(),),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                    )
                ),
                TWCmdGenTemporalBatch(
                    data=(
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalTextualInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                    ),
                ),
            ],
        ),
    ],
)
def test_tw_cmd_gen_temporal_batch_split(batch, split_size, expected):
    assert batch.split(split_size) == expected


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
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="pin_memory() doesn't work on Github Actions"
)
def test_tw_cmd_gen_temporal_textual_input_pin_memory(textual):
    # just check that we're creating a correct copy
    pinned = textual.pin_memory()
    assert pinned == textual
    assert pinned.obs_word_ids.is_pinned()
    assert pinned.obs_mask.is_pinned()
    assert pinned.prev_action_word_ids.is_pinned()
    assert pinned.prev_action_mask.is_pinned()


@pytest.mark.parametrize(
    "graphical",
    [
        TWCmdGenTemporalGraphicalInput(
            node_ids=torch.randint(10, (3, 10)),
            edge_ids=torch.randint(15, (3, 15)),
            edge_index=torch.randint(15, (3, 2, 15)),
            edge_timestamps=torch.rand(3, 15),
            tgt_event_timestamps=torch.rand(3, 6),
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
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="pin_memory() doesn't work on Github Actions"
)
def test_tw_cmd_gen_temporal_graphical_input_pin_memory(graphical):
    # just check that we're creating a correct copy
    pinned = graphical.pin_memory()
    assert pinned == graphical
    assert pinned.node_ids.is_pinned()
    assert pinned.edge_ids.is_pinned()
    assert pinned.edge_index.is_pinned()
    assert pinned.edge_timestamps.is_pinned()
    assert pinned.tgt_event_timestamps.is_pinned()
    assert pinned.tgt_event_type_ids.is_pinned()
    assert pinned.tgt_event_src_ids.is_pinned()
    assert pinned.tgt_event_src_mask.is_pinned()
    assert pinned.tgt_event_dst_ids.is_pinned()
    assert pinned.tgt_event_dst_mask.is_pinned()
    assert pinned.tgt_event_edge_ids.is_pinned()
    assert pinned.tgt_event_label_ids.is_pinned()
    assert pinned.groundtruth_event_type_ids.is_pinned()
    assert pinned.groundtruth_event_src_ids.is_pinned()
    assert pinned.groundtruth_event_src_mask.is_pinned()
    assert pinned.groundtruth_event_dst_ids.is_pinned()
    assert pinned.groundtruth_event_dst_mask.is_pinned()
    assert pinned.groundtruth_event_label_ids.is_pinned()
    assert pinned.groundtruth_event_mask.is_pinned()


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
                    (("",),),
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
                    (("",),),
                ),
            )
        )
    ],
)
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="pin_memory() doesn't work on Github Actions"
)
def test_tw_cmd_gen_temporal_batch_pin_memory(batch):
    # just check that we're creating a correct copy
    pinned = batch.pin_memory()
    assert pinned == batch
    for textual, graphicals, _ in pinned.data:
        assert textual.obs_word_ids.is_pinned()
        assert textual.obs_mask.is_pinned()
        assert textual.prev_action_word_ids.is_pinned()
        assert textual.prev_action_mask.is_pinned()
        for graphical in graphicals:
            assert graphical.node_ids.is_pinned()
            assert graphical.edge_ids.is_pinned()
            assert graphical.edge_index.is_pinned()
            assert graphical.edge_timestamps.is_pinned()
            assert graphical.tgt_event_timestamps.is_pinned()
            assert graphical.tgt_event_type_ids.is_pinned()
            assert graphical.tgt_event_src_ids.is_pinned()
            assert graphical.tgt_event_src_mask.is_pinned()
            assert graphical.tgt_event_dst_ids.is_pinned()
            assert graphical.tgt_event_dst_mask.is_pinned()
            assert graphical.tgt_event_edge_ids.is_pinned()
            assert graphical.tgt_event_label_ids.is_pinned()
            assert graphical.groundtruth_event_type_ids.is_pinned()
            assert graphical.groundtruth_event_src_ids.is_pinned()
            assert graphical.groundtruth_event_src_mask.is_pinned()
            assert graphical.groundtruth_event_dst_ids.is_pinned()
            assert graphical.groundtruth_event_dst_mask.is_pinned()
            assert graphical.groundtruth_event_label_ids.is_pinned()
            assert graphical.groundtruth_event_mask.is_pinned()
