import pytest
import json
import torch
import shutil

from dgu.data import (
    TWCmdGenTemporalDataset,
    TWCmdGenTemporalDataModule,
    TWCmdGenTemporalDataCollator,
    read_label_vocab_files,
    TWCmdGenTemporalStepInput,
    TWCmdGenTemporalBatch,
    TWCmdGenTemporalGraphicalInput,
)
from dgu.preprocessor import SpacyPreprocessor
from dgu.constants import EVENT_TYPE_ID_MAP


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
    "batch_step,expected",
    [
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": [],
                    "graph_events": [],
                }
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([0]),
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
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": [],
                    "graph_events": [],
                }
            ],
            [
                TWCmdGenTemporalGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_ids=torch.tensor([0]),
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
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                    ],
                }
            ],
            [
                TWCmdGenTemporalGraphicalInput(
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
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
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
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
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
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([1]),
                    tgt_event_label_ids=torch.tensor([100]),
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
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 4,
                    "target_commands": ["add , player , livingroom , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "livingroom"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                    ],
                },
            ],
            [
                TWCmdGenTemporalGraphicalInput(
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
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                    ],
                },
                {},
            ],
            [
                TWCmdGenTemporalGraphicalInput(
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
        ),
        (
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
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 3, "label": "in"},
                        {"type": "node-add", "label": "chicken leg"},
                        {"type": "edge-add", "src_id": 4, "dst_id": 3, "label": "in"},
                        {
                            "type": "edge-delete",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "in",
                        },
                        {"type": "node-delete", "node_id": 2, "label": "player"},
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
                    "graph_events": [
                        {"type": "node-add", "label": "chicken leg"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 3, "label": "in"},
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "livingroom"},
                        {"type": "edge-add", "src_id": 4, "dst_id": 5, "label": "in"},
                        {
                            "type": "edge-delete",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "in",
                        },
                        {"type": "node-delete", "node_id": 3, "label": "kitchen"},
                        {"type": "node-delete", "node_id": 2, "label": "chicken leg"},
                    ],
                },
            ],
            [
                TWCmdGenTemporalGraphicalInput(
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
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_graphical_inputs(
    tw_cmd_gen_collator, batch_step, expected
):
    assert tw_cmd_gen_collator.collate_graphical_inputs(batch_step) == expected


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
                        "graph_events": [
                            {"type": "node-add", "label": "player"},
                            {"type": "node-add", "label": "kitchen"},
                            {
                                "type": "edge-add",
                                "src_id": 0,
                                "dst_id": 1,
                                "label": "in",
                            },
                        ],
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
                        "graph_events": [
                            {"type": "node-add", "label": "player"},
                            {"type": "node-add", "label": "kitchen"},
                            {
                                "type": "edge-add",
                                "src_id": 0,
                                "dst_id": 1,
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
                        "target_commands": ["delete , player , kitchen , in"],
                        "graph_events": [
                            {
                                "type": "edge-delete",
                                "src_id": 0,
                                "dst_id": 1,
                                "label": "in",
                            },
                            {"type": "node-delete", "node_id": 1, "label": "kitchen"},
                            {"type": "node-delete", "node_id": 0, "label": "player"},
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
                        "target_commands": ["add , player , kitchen , in"],
                        "graph_events": [
                            {"type": "node-add", "label": "player"},
                            {"type": "node-add", "label": "kitchen"},
                            {
                                "type": "edge-add",
                                "src_id": 0,
                                "dst_id": 1,
                                "label": "in",
                            },
                        ],
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
