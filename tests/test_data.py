import pytest
import json
import torch
import shutil
import networkx as nx

from dgu.data import (
    TWCmdGenTemporalGraphData,
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
    TWCmdGenTemporalDataCollator,
    read_label_vocab_files,
    TWCmdGenTemporalStepInput,
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
    _, label_id_map = read_label_vocab_files(
        "vocabs/node_vocab.txt", "vocabs/relation_vocab.txt"
    )
    return TWCmdGenTemporalDataCollator(
        SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt"), label_id_map
    )


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
    "before_graph,after_graph,label_id_map,expected",
    [
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
        (
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
            ),
        ),
    ],
)
def test_tw_cmd_gen_temporal_graph_data_from_graph_event(
    before_graph, after_graph, label_id_map, expected
):
    data = TWCmdGenTemporalGraphData.from_graph_event(
        before_graph, after_graph, label_id_map
    )
    assert data.x.equal(expected.x)
    assert data.node_memory_update_index.equal(expected.node_memory_update_index)
    assert data.node_memory_update_mask.equal(expected.node_memory_update_mask)
    assert data.edge_index.equal(expected.edge_index)
    assert data.edge_attr.equal(expected.edge_attr)
    assert data.edge_last_update.equal(expected.edge_last_update)


@pytest.mark.parametrize(
    "before_graph,before_graph_node_attrs,after_graph,after_graph_node_attrs,expected",
    [
        (
            EqualityDiGraph(),
            {},
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
            ),
        ),
        (
            EqualityDiGraph(),
            {},
            EqualityDiGraph({"n0": {}}),
            {"n0": {"label": "player", "label_id": 1}},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([1]),
                node_memory_update_index=torch.empty(0, dtype=torch.long),
                node_memory_update_mask=torch.empty(0, dtype=torch.bool),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
            ),
        ),
        (
            EqualityDiGraph({"n0": {}}),
            {"n0": {"label": "player", "label_id": 1}},
            EqualityDiGraph({"n0": {}, "n1": {}}),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            TWCmdGenTemporalGraphData(
                x=torch.tensor([1, 2]),
                node_memory_update_index=torch.tensor([0]),
                node_memory_update_mask=torch.tensor([True]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
            ),
        ),
        (
            EqualityDiGraph({"n0": {}, "n1": {}}),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            EqualityDiGraph(
                {"n0": {"n1": {"label": "in", "label_id": 4, "last_update": 0}}}
            ),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            TWCmdGenTemporalGraphData(
                x=torch.tensor([1, 2]),
                node_memory_update_index=torch.tensor([0, 1]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([4]),
                edge_last_update=torch.tensor([0.0]),
                edge_timestamps=torch.tensor([2.0]),
            ),
        ),
        (
            EqualityDiGraph(
                {"n0": {"n1": {"label": "in", "label_id": 4, "last_update": 0}}}
            ),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            EqualityDiGraph({"n0": {}, "n1": {}}),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            TWCmdGenTemporalGraphData(
                x=torch.tensor([1, 2]),
                node_memory_update_index=torch.tensor([0, 1]),
                node_memory_update_mask=torch.tensor([True, True]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
            ),
        ),
        (
            EqualityDiGraph({"n0": {}, "n1": {}}),
            {
                "n0": {"label": "player", "label_id": 1},
                "n1": {"label": "inventory", "label_id": 2},
            },
            EqualityDiGraph({"n0": {}}),
            {"n0": {"label": "player", "label_id": 1}},
            TWCmdGenTemporalGraphData(
                x=torch.tensor([1]),
                node_memory_update_index=torch.tensor([0, 0]),
                node_memory_update_mask=torch.tensor([True, False]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
            ),
        ),
        (
            EqualityDiGraph({"n0": {}}),
            {"n0": {"label": "player", "label_id": 1}},
            EqualityDiGraph(),
            {},
            TWCmdGenTemporalGraphData(
                x=torch.empty(0, dtype=torch.long),
                node_memory_update_index=torch.tensor([0]),
                node_memory_update_mask=torch.tensor([False]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0),
                edge_timestamps=torch.empty(0),
            ),
        ),
    ],
)
def test_tw_cmd_gen_temporal_graph_data_from_decoded_graph_event(
    before_graph, before_graph_node_attrs, after_graph, after_graph_node_attrs, expected
):
    nx.set_node_attributes(after_graph, after_graph_node_attrs)
    nx.set_node_attributes(before_graph, before_graph_node_attrs)

    data = TWCmdGenTemporalGraphData.from_decoded_graph_event(before_graph, after_graph)
    assert data.x.equal(expected.x)
    assert data.node_memory_update_index.equal(expected.node_memory_update_index)
    assert data.node_memory_update_mask.equal(expected.node_memory_update_mask)
    assert data.edge_index.equal(expected.edge_index)
    assert data.edge_attr.equal(expected.edge_attr)
    assert data.edge_last_update.equal(expected.edge_last_update)


@pytest.mark.parametrize(
    "obs,prev_actions,timestamps,mask,expected",
    [
        (
            ["you are hungry ! let 's cook a delicious meal ."],
            ["drop knife"],
            [2],
            [True],
            TWCmdGenTemporalStepInput(
                obs_word_ids=torch.tensor(
                    [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                ),
                obs_mask=torch.ones(1, 11),
                prev_action_word_ids=torch.tensor([[257, 404]]),
                prev_action_mask=torch.ones(1, 2),
                timestamps=torch.tensor([2.0]),
                mask=torch.tensor([True]),
            ),
        ),
        (
            [
                "you are hungry ! let 's cook a delicious meal .",
                "you take the knife from the table .",
            ],
            ["drop knife", "take knife from table"],
            [2, 3],
            [True, True],
            TWCmdGenTemporalStepInput(
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
                timestamps=torch.tensor([2.0, 3.0]),
                mask=torch.tensor([True, True]),
            ),
        ),
        (
            [
                "you are hungry ! let 's cook a delicious meal .",
                "<bos> <eos>",
            ],
            ["drop knife", "<bos> <eos>"],
            [2, 3],
            [True, False],
            TWCmdGenTemporalStepInput(
                obs_word_ids=torch.tensor(
                    [
                        [769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21],
                        [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                obs_mask=torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                prev_action_word_ids=torch.tensor([[257, 404], [2, 3]]),
                prev_action_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                timestamps=torch.tensor([2.0, 3.0]),
                mask=torch.tensor([True, False]),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_step_inputs(
    tw_cmd_gen_collator, obs, prev_actions, timestamps, mask, expected
):
    assert (
        tw_cmd_gen_collator.collate_step_inputs(obs, prev_actions, timestamps, mask)
        == expected
    )


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
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.empty(0, dtype=torch.long),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.empty(0, dtype=torch.long),
                        tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_ids=torch.tensor([0]),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([0.0]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([0.0]),
                        groundtruth_event_label_ids=torch.tensor([0]),
                        groundtruth_event_mask=torch.tensor([1.0]),
                    ),
                ],
                [EqualityDiGraph()],
            ),
        ),
        (
            [
                EqualityDiGraph(
                    {
                        Node("player"): {
                            Node("kitchen"): {"label": "in", "last_update": 2}
                        }
                    }
                )
            ],
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": [],
                }
            ],
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.tensor([1, 14]),
                        node_memory_update_index=torch.tensor([0, 1]),
                        node_memory_update_mask=torch.tensor([True, True]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_label_ids=torch.tensor([100]),
                        edge_last_update=torch.tensor([2.0]),
                        batch=torch.tensor([0, 0]),
                        tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_ids=torch.tensor([0]),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([0.0]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([0.0]),
                        groundtruth_event_label_ids=torch.tensor([0]),
                        groundtruth_event_mask=torch.tensor([1.0]),
                    ),
                ],
                [
                    EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 2}
                            }
                        }
                    )
                ],
            ),
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
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.empty(0).long(),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.empty(0).long(),
                        tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_ids=torch.tensor([0]),
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
                        node_label_ids=torch.tensor([1]),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_ids=torch.tensor([1]),
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
                        node_memory_update_index=torch.tensor([0]),
                        node_memory_update_mask=torch.tensor([True]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0, 0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_ids=torch.tensor([14]),
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
                        batch=torch.tensor([0, 0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([1]),
                        tgt_event_label_ids=torch.tensor([100]),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([0.0]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([0.0]),
                        groundtruth_event_label_ids=torch.tensor([0]),
                        groundtruth_event_mask=torch.tensor([1.0]),
                    ),
                ],
                [
                    EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 2}
                            }
                        }
                    )
                ],
            ),
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
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.empty(0).long(),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.empty(0).long(),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([0, 0]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        groundtruth_event_src_ids=torch.tensor([0, 0]),
                        groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                        groundtruth_event_dst_ids=torch.tensor([0, 0]),
                        groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                        groundtruth_event_label_ids=torch.tensor([1, 1]),
                        groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                    ),
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.tensor([1, 1]),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([1, 1]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
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
                        node_memory_update_index=torch.tensor([0, 2]),
                        node_memory_update_mask=torch.tensor([True, True]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0, 0, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([14, 16]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["edge-add"],
                            ]
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
                        batch=torch.tensor([0, 0, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["edge-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([1, 1]),
                        tgt_event_label_ids=torch.tensor([100, 100]),
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
                [
                    EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 2}
                            }
                        }
                    ),
                    EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("livingroom"): {"label": "in", "last_update": 4}
                            }
                        }
                    ),
                ],
            ),
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
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.empty(0).long(),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.empty(0).long(),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([0, 0]),
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
                        node_label_ids=torch.tensor([1]),
                        node_memory_update_index=torch.empty(0).long(),
                        node_memory_update_mask=torch.empty(0).bool(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["end"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([1, 0]),
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
                        node_memory_update_index=torch.tensor([0]),
                        node_memory_update_mask=torch.tensor([True]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_label_ids=torch.empty(0).long(),
                        edge_last_update=torch.empty(0),
                        batch=torch.tensor([0, 0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["pad"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([14, 0]),
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
                        batch=torch.tensor([0, 0]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["pad"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([1, 0]),
                        tgt_event_label_ids=torch.tensor([100, 0]),
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
                [
                    EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 2}
                            }
                        }
                    ),
                    EqualityDiGraph(),
                ],
            ),
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
            (
                [
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.tensor([27, 16, 64, 75]),
                        node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                        node_memory_update_mask=torch.tensor([True, True, True, True]),
                        edge_index=torch.tensor([[0, 2], [1, 3]]),
                        edge_label_ids=torch.tensor([100, 100]),
                        edge_last_update=torch.tensor([2.0, 0.0]),
                        batch=torch.tensor([0, 0, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([0, 0]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        groundtruth_event_src_ids=torch.tensor([0, 0]),
                        groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                        groundtruth_event_dst_ids=torch.tensor([0, 0]),
                        groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                        groundtruth_event_label_ids=torch.tensor([1, 34]),
                        groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                    ),
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids=torch.tensor([27, 16, 1, 64, 75, 34]),
                        node_memory_update_index=torch.tensor([0, 1, 3, 4]),
                        node_memory_update_mask=torch.tensor([True, True, True, True]),
                        edge_index=torch.tensor([[0, 3], [1, 4]]),
                        edge_label_ids=torch.tensor([100, 100]),
                        edge_last_update=torch.tensor([2.0, 0.0]),
                        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([1, 34]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
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
                        node_memory_update_index=torch.tensor([0, 1, 2, 4, 5, 6]),
                        node_memory_update_mask=torch.tensor(
                            [True, True, True, True, True, True]
                        ),
                        edge_index=torch.tensor([[0, 4], [1, 5]]),
                        edge_label_ids=torch.tensor([100, 100]),
                        edge_last_update=torch.tensor([2.0, 0.0]),
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([14, 14]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["edge-add"],
                            ]
                        ),
                        groundtruth_event_src_ids=torch.tensor([2, 2]),
                        groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                        groundtruth_event_dst_ids=torch.tensor([3, 3]),
                        groundtruth_event_dst_mask=torch.tensor([1.0, 1.0]),
                        groundtruth_event_label_ids=torch.tensor([100, 100]),
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
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["edge-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([2, 2]),
                        tgt_event_dst_ids=torch.tensor([3, 3]),
                        tgt_event_label_ids=torch.tensor([100, 100]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
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
                            [27, 16, 1, 14, 34, 64, 75, 34, 14, 1]
                        ),
                        node_memory_update_index=torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]),
                        node_memory_update_mask=torch.tensor(
                            [True, True, True, True, True, True, True, True]
                        ),
                        edge_index=torch.tensor([[0, 2, 5, 7], [1, 3, 6, 8]]),
                        edge_label_ids=torch.tensor([100, 100, 100, 100]),
                        edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([34, 1]),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
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
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        ),
                        node_memory_update_mask=torch.tensor(
                            [True, True, True, True, True, True, True, True, True, True]
                        ),
                        edge_index=torch.tensor([[0, 2, 4, 5, 7], [1, 3, 3, 6, 8]]),
                        edge_label_ids=torch.tensor([100, 100, 100, 100, 100]),
                        edge_last_update=torch.tensor([2.0, 3.0, 3.0, 0.0, 1.0]),
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-add"],
                                EVENT_TYPE_ID_MAP["node-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([4, 0]),
                        tgt_event_dst_ids=torch.tensor([3, 0]),
                        tgt_event_label_ids=torch.tensor([100, 16]),
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
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["edge-delete"],
                                EVENT_TYPE_ID_MAP["edge-add"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([2, 4]),
                        tgt_event_dst_ids=torch.tensor([3, 5]),
                        tgt_event_label_ids=torch.tensor([100, 100]),
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
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-delete"],
                                EVENT_TYPE_ID_MAP["edge-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([2, 2]),
                        tgt_event_dst_ids=torch.tensor([0, 3]),
                        tgt_event_label_ids=torch.tensor([1, 100]),
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
                        node_label_ids=torch.tensor(
                            [27, 16, 14, 34, 64, 75, 34, 1, 16]
                        ),
                        # The deleted node is assigned 4 instead of 0
                        # b/c of graph mini-batching. The mask takes care of it,
                        # so it's fine.
                        node_memory_update_index=torch.tensor(
                            [0, 1, 2, 3, 4, 5, 6, 4, 7, 8]
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
                                False,
                                True,
                                True,
                            ]
                        ),
                        edge_index=torch.tensor([[0, 3, 4, 7], [1, 2, 5, 8]]),
                        edge_label_ids=torch.tensor([100, 100, 100, 100]),
                        edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["end"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 3]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([0, 14]),
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
                        node_label_ids=torch.tensor([27, 16, 14, 34, 64, 75, 1, 16]),
                        node_memory_update_index=torch.tensor(
                            [0, 1, 2, 3, 4, 5, 4, 6, 7]
                        ),
                        node_memory_update_mask=torch.tensor(
                            [True, True, True, True, True, True, False, True, True]
                        ),
                        edge_index=torch.tensor([[0, 3, 4, 6], [1, 2, 5, 7]]),
                        edge_label_ids=torch.tensor([100, 100, 100, 100]),
                        edge_last_update=torch.tensor([2.0, 3.0, 0.0, 1.0]),
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["pad"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 2]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_ids=torch.tensor([0, 34]),
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
                [
                    EqualityDiGraph(
                        {
                            Node("banana"): {
                                Node("livingroom"): {"label": "in", "last_update": 2}
                            },
                            Node("chicken leg"): {
                                Node("kitchen"): {"label": "in", "last_update": 3}
                            },
                        }
                    ),
                    EqualityDiGraph(
                        {
                            Node("pork chop"): {
                                Node("refrigerator"): {"label": "in", "last_update": 0}
                            },
                            Node("player"): {
                                Node("livingroom"): {"label": "in", "last_update": 1}
                            },
                        }
                    ),
                ],
            ),
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
    labels, label_id_map = read_label_vocab_files(
        "tests/data/test_node_vocab.txt", "tests/data/test_relation_vocab.txt"
    )
    assert labels == ["", "player", "inventory", "chopped", "in", "is"]
    assert label_id_map == {
        "": 0,
        "player": 1,
        "inventory": 2,
        "chopped": 3,
        "in": 4,
        "is": 5,
    }


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
                        "target_commands": ["add , player , kitchen , in"],
                    },
                ]
            ],
            TWCmdGenTemporalBatch(
                ids=(("g1", 0),),
                data=(
                    (
                        TWCmdGenTemporalStepInput(
                            obs_word_ids=torch.tensor(
                                [[769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21]]
                            ),
                            obs_mask=torch.ones(1, 11),
                            prev_action_word_ids=torch.tensor([[257, 404]]),
                            prev_action_mask=torch.ones(1, 2),
                            timestamps=torch.tensor([2.0]),
                            mask=torch.tensor([True]),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.empty(0).long(),
                                node_memory_update_index=torch.empty(0).long(),
                                node_memory_update_mask=torch.empty(0).bool(),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.empty(0).long(),
                                tgt_event_type_ids=torch.tensor(
                                    [EVENT_TYPE_ID_MAP["start"]]
                                ),
                                tgt_event_src_ids=torch.tensor([0]),
                                tgt_event_dst_ids=torch.tensor([0]),
                                tgt_event_label_ids=torch.tensor([0]),
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
                                node_label_ids=torch.tensor([1]),
                                node_memory_update_index=torch.empty(0).long(),
                                node_memory_update_mask=torch.empty(0).bool(),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.tensor([0]),
                                tgt_event_type_ids=torch.tensor(
                                    [EVENT_TYPE_ID_MAP["node-add"]]
                                ),
                                tgt_event_src_ids=torch.tensor([0]),
                                tgt_event_dst_ids=torch.tensor([0]),
                                tgt_event_label_ids=torch.tensor([1]),
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
                                node_memory_update_index=torch.tensor([0]),
                                node_memory_update_mask=torch.tensor([True]),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.tensor([0, 0]),
                                tgt_event_type_ids=torch.tensor(
                                    [EVENT_TYPE_ID_MAP["node-add"]]
                                ),
                                tgt_event_src_ids=torch.tensor([0]),
                                tgt_event_dst_ids=torch.tensor([0]),
                                tgt_event_label_ids=torch.tensor([14]),
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
                                batch=torch.tensor([0, 0]),
                                tgt_event_type_ids=torch.tensor(
                                    [EVENT_TYPE_ID_MAP["edge-add"]]
                                ),
                                tgt_event_src_ids=torch.tensor([0]),
                                tgt_event_dst_ids=torch.tensor([1]),
                                tgt_event_label_ids=torch.tensor([100]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [EVENT_TYPE_ID_MAP["end"]]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0]),
                                groundtruth_event_src_mask=torch.tensor([0.0]),
                                groundtruth_event_dst_ids=torch.tensor([0]),
                                groundtruth_event_dst_mask=torch.tensor([0.0]),
                                groundtruth_event_label_ids=torch.tensor([0]),
                                groundtruth_event_mask=torch.tensor([1.0]),
                            ),
                        ),
                        (("add , player , kitchen , in",),),
                    ),
                ),
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
                        "target_commands": ["add , player , kitchen , in"],
                    },
                    {
                        "game": "g1",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 3,
                        "target_commands": ["delete , player , kitchen , in"],
                    },
                ],
                [
                    {
                        "game": "g2",
                        "walkthrough_step": 0,
                        "observation": "you take the knife from the table .",
                        "previous_action": "take knife from table",
                        "timestamp": 1,
                        "target_commands": ["add , player , kitchen , in"],
                    },
                ],
            ],
            TWCmdGenTemporalBatch(
                ids=(("g1", 0), ("g2", 0)),
                data=(
                    (
                        TWCmdGenTemporalStepInput(
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
                            timestamps=torch.tensor([2.0, 1.0]),
                            mask=torch.tensor([True, True]),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.empty(0).long(),
                                node_memory_update_index=torch.empty(0).long(),
                                node_memory_update_mask=torch.empty(0).bool(),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.empty(0).long(),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["start"],
                                        EVENT_TYPE_ID_MAP["start"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([0, 0]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-add"],
                                        EVENT_TYPE_ID_MAP["node-add"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0, 0]),
                                groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_dst_ids=torch.tensor([0, 0]),
                                groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_label_ids=torch.tensor([1, 1]),
                                groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 1]),
                                node_memory_update_index=torch.empty(0).long(),
                                node_memory_update_mask=torch.empty(0).bool(),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.tensor([0, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-add"],
                                        EVENT_TYPE_ID_MAP["node-add"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([1, 1]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-add"],
                                        EVENT_TYPE_ID_MAP["node-add"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0, 0]),
                                groundtruth_event_src_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_dst_ids=torch.tensor([0, 0]),
                                groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_label_ids=torch.tensor([14, 14]),
                                groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 14, 1, 14]),
                                node_memory_update_index=torch.tensor([0, 2]),
                                node_memory_update_mask=torch.tensor([True, True]),
                                edge_index=torch.empty(2, 0).long(),
                                edge_label_ids=torch.empty(0).long(),
                                edge_last_update=torch.empty(0),
                                batch=torch.tensor([0, 0, 1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-add"],
                                        EVENT_TYPE_ID_MAP["node-add"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([14, 14]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["edge-add"],
                                        EVENT_TYPE_ID_MAP["edge-add"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0, 0]),
                                groundtruth_event_src_mask=torch.tensor([1.0, 1.0]),
                                groundtruth_event_dst_ids=torch.tensor([1, 1]),
                                groundtruth_event_dst_mask=torch.tensor([1.0, 1.0]),
                                groundtruth_event_label_ids=torch.tensor([100, 100]),
                                groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 14, 1, 14]),
                                node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                                node_memory_update_mask=torch.tensor(
                                    [True, True, True, True]
                                ),
                                edge_index=torch.tensor([[0, 2], [1, 3]]),
                                edge_label_ids=torch.tensor([100, 100]),
                                edge_last_update=torch.tensor([2.0, 1.0]),
                                batch=torch.tensor([0, 0, 1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["edge-add"],
                                        EVENT_TYPE_ID_MAP["edge-add"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([1, 1]),
                                tgt_event_label_ids=torch.tensor([100, 100]),
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
                        ),
                        (
                            ("add , player , kitchen , in",),
                            ("add , player , kitchen , in",),
                        ),
                    ),
                    (
                        TWCmdGenTemporalStepInput(
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
                            timestamps=torch.tensor([3.0, 0.0]),
                            mask=torch.tensor([True, False]),
                        ),
                        (
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 14, 1, 14]),
                                node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                                node_memory_update_mask=torch.tensor(
                                    [True, True, True, True]
                                ),
                                edge_index=torch.tensor([[0, 2], [1, 3]]),
                                edge_label_ids=torch.tensor([100, 100]),
                                edge_last_update=torch.tensor([2.0, 1.0]),
                                batch=torch.tensor([0, 0, 1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["start"],
                                        EVENT_TYPE_ID_MAP["start"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([0, 0]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["edge-delete"],
                                        EVENT_TYPE_ID_MAP["end"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0, 0]),
                                groundtruth_event_src_mask=torch.tensor([1.0, 0.0]),
                                groundtruth_event_dst_ids=torch.tensor([1, 0]),
                                groundtruth_event_dst_mask=torch.tensor([1.0, 0.0]),
                                groundtruth_event_label_ids=torch.tensor([100, 0]),
                                groundtruth_event_mask=torch.tensor([1.0, 1.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 14, 1, 14]),
                                node_memory_update_index=torch.tensor([0, 1, 2, 3]),
                                node_memory_update_mask=torch.tensor(
                                    [True, True, True, True]
                                ),
                                edge_index=torch.tensor([[2], [3]]),
                                edge_label_ids=torch.tensor([100]),
                                edge_last_update=torch.tensor([1.0]),
                                batch=torch.tensor([0, 0, 1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["edge-delete"],
                                        EVENT_TYPE_ID_MAP["end"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([1, 0]),
                                tgt_event_label_ids=torch.tensor([100, 0]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-delete"],
                                        EVENT_TYPE_ID_MAP["pad"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([1, 0]),
                                groundtruth_event_src_mask=torch.tensor([1.0, 0.0]),
                                groundtruth_event_dst_ids=torch.tensor([0, 0]),
                                groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_label_ids=torch.tensor([14, 0]),
                                groundtruth_event_mask=torch.tensor([1.0, 0.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 1, 14]),
                                node_memory_update_index=torch.tensor([0, 0, 1, 2]),
                                node_memory_update_mask=torch.tensor(
                                    [True, False, True, True]
                                ),
                                edge_index=torch.tensor([[1], [2]]),
                                edge_label_ids=torch.tensor([100]),
                                edge_last_update=torch.tensor([1.0]),
                                batch=torch.tensor([0, 1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-delete"],
                                        EVENT_TYPE_ID_MAP["pad"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([1, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([14, 0]),
                                groundtruth_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-delete"],
                                        EVENT_TYPE_ID_MAP["pad"],
                                    ]
                                ),
                                groundtruth_event_src_ids=torch.tensor([0, 0]),
                                groundtruth_event_src_mask=torch.tensor([1.0, 0.0]),
                                groundtruth_event_dst_ids=torch.tensor([0, 0]),
                                groundtruth_event_dst_mask=torch.tensor([0.0, 0.0]),
                                groundtruth_event_label_ids=torch.tensor([1, 0]),
                                groundtruth_event_mask=torch.tensor([1.0, 0.0]),
                            ),
                            TWCmdGenTemporalGraphicalInput(
                                node_label_ids=torch.tensor([1, 14]),
                                node_memory_update_index=torch.tensor([0, 0, 1]),
                                node_memory_update_mask=torch.tensor(
                                    [False, True, True]
                                ),
                                edge_index=torch.tensor([[0], [1]]),
                                edge_label_ids=torch.tensor([100]),
                                edge_last_update=torch.tensor([1.0]),
                                batch=torch.tensor([1, 1]),
                                tgt_event_type_ids=torch.tensor(
                                    [
                                        EVENT_TYPE_ID_MAP["node-delete"],
                                        EVENT_TYPE_ID_MAP["pad"],
                                    ]
                                ),
                                tgt_event_src_ids=torch.tensor([0, 0]),
                                tgt_event_dst_ids=torch.tensor([0, 0]),
                                tgt_event_label_ids=torch.tensor([1, 0]),
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
                        ),
                        (("delete , player , kitchen , in",), ()),
                    ),
                ),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_call(tw_cmd_gen_collator, batch, expected):
    assert tw_cmd_gen_collator(batch) == expected


@pytest.mark.parametrize(
    "batch,split_size,expected",
    [
        (
            TWCmdGenTemporalBatch(
                ids=(("g0", 1),),
                data=(
                    (
                        TWCmdGenTemporalStepInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
                        (TWCmdGenTemporalGraphicalInput(),),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                ),
            ),
            2,
            [
                TWCmdGenTemporalBatch(
                    ids=(("g0", 1),),
                    data=(
                        (
                            TWCmdGenTemporalStepInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalStepInput(),
                            (TWCmdGenTemporalGraphicalInput(),),
                            (("",),),
                        ),
                    ),
                ),
                TWCmdGenTemporalBatch(
                    ids=(("g0", 1),),
                    data=(
                        (
                            TWCmdGenTemporalStepInput(),
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
                ids=(("g0", 1),),
                data=(
                    (
                        TWCmdGenTemporalStepInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
                        (TWCmdGenTemporalGraphicalInput(),),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
                        (
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                            TWCmdGenTemporalGraphicalInput(),
                        ),
                        (("",),),
                    ),
                    (
                        TWCmdGenTemporalStepInput(),
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
            3,
            [
                TWCmdGenTemporalBatch(
                    ids=(("g0", 1),),
                    data=(
                        (
                            TWCmdGenTemporalStepInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalStepInput(),
                            (TWCmdGenTemporalGraphicalInput(),),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalStepInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                    ),
                ),
                TWCmdGenTemporalBatch(
                    ids=(("g0", 1),),
                    data=(
                        (
                            TWCmdGenTemporalStepInput(),
                            (
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                                TWCmdGenTemporalGraphicalInput(),
                            ),
                            (("",),),
                        ),
                        (
                            TWCmdGenTemporalStepInput(),
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
