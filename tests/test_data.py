import pytest
import json
import torch

from torch_geometric.data import Batch
from unittest.mock import Mock

from tdgu.data import (
    TWCmdGenGraphEventDataset,
    TWCmdGenGraphEventDataCollator,
    read_label_vocab_files,
    TWCmdGenGraphEventStepInput,
    TWCmdGenGraphEventBatch,
    TWCmdGenGraphEventGraphicalInput,
    sort_target_commands,
    TWCmdGenGraphEventFreeRunDataset,
)
from tdgu.preprocessor import SpacyPreprocessor
from tdgu.constants import EVENT_TYPE_ID_MAP


@pytest.fixture
def tw_cmd_gen_collator():
    return TWCmdGenGraphEventDataCollator(
        SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    )


@pytest.mark.parametrize("sort_cmds", [True, False])
def test_tw_cmd_gen_dataset_init(sort_cmds):
    dataset = TWCmdGenGraphEventDataset(
        "tests/data/test_data.json", sort_commands=sort_cmds
    )
    expected_dataset = []
    expected_dataset_filename = (
        "tests/data/preprocessed_sorted_test_data.jsonl"
        if sort_cmds
        else "tests/data/preprocessed_test_data.jsonl"
    )
    with open(expected_dataset_filename) as f:
        for line in f:
            expected_dataset.append(json.loads(line))

    assert len(dataset) == len(expected_dataset)
    for data, expected_data in zip(dataset, expected_dataset):
        assert data == expected_data


def test_tw_cmd_gen_dataset_init_allow_objs_with_same_label():
    dataset = TWCmdGenGraphEventDataset(
        "tests/data/test_data_same_label_obj.json", allow_objs_with_same_label=True
    )
    expected_dataset = []
    with open("tests/data/preprocessed_objs_with_same_label_test_data.jsonl") as f:
        for line in f:
            expected_dataset.append(json.loads(line))

    assert len(dataset) == len(expected_dataset)
    for data, expected_data in zip(dataset, expected_dataset):
        assert data == expected_data


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sort_cmds", [True, False])
def test_tw_cmd_gen_free_run_dataset_init(batch_size, sort_cmds):
    dataset = TWCmdGenGraphEventFreeRunDataset(
        "tests/data/test_data.json", batch_size, sort_commands=sort_cmds
    )
    expected_dataset = []
    expected_dataset_filename = (
        f"tests/data/preprocessed_sorted_test_free_run_data_batch_{batch_size}.jsonl"
        if sort_cmds
        else f"tests/data/preprocessed_test_free_run_data_batch_{batch_size}.jsonl"
    )
    with open(expected_dataset_filename) as f:
        for line in f:
            expected_dataset.append(json.loads(line))

    assert len(dataset) == 5  # number of walkthroughs
    generated_dataset = list(iter(dataset))
    assert len(generated_dataset) == len(expected_dataset)
    for data, expected_data in zip(generated_dataset, expected_dataset):
        assert data == list(map(tuple, expected_data))


@pytest.mark.parametrize("shuffle", [True, False])
def test_tw_cmd_gen_free_run_dataset_shuffle(monkeypatch, shuffle):
    mock_shuffle = Mock()
    monkeypatch.setattr("random.shuffle", mock_shuffle)
    dataset = TWCmdGenGraphEventFreeRunDataset(
        "tests/data/test_data.json", 3, shuffle=shuffle
    )
    list(iter(dataset))
    if shuffle:
        mock_shuffle.assert_called_once_with(dataset.walkthrough_example_ids)
    else:
        mock_shuffle.assert_not_called()


@pytest.mark.parametrize(
    "tgt_cmds,expected",
    [
        # add always before delete
        (
            ["delete , n0 , n1 , r0", "add , n0 , n1 , r0"],
            ["add , n0 , n1 , r0", "delete , n0 , n1 , r0"],
        ),
        # relations with player as a source before others
        (
            ["add , cookbook , table , on", "add , player , livingroom , in"],
            ["add , player , livingroom , in", "add , cookbook , table , on"],
        ),
        # relations with player as a destination before others
        (
            [
                "add , cookbook , table , on",
                "add , player , livingroom , in",
                "add , knife , player , in",
            ],
            [
                "add , player , livingroom , in",
                "add , knife , player , in",
                "add , cookbook , table , on",
            ],
        ),
        # room connections before others
        (
            ["add , cookbook , table , on", "add , exit , livingroom , west_of"],
            ["add , exit , livingroom , west_of", "add , cookbook , table , on"],
        ),
        (
            ["add , cookbook , table , on", "add , exit , livingroom , east_of"],
            ["add , exit , livingroom , east_of", "add , cookbook , table , on"],
        ),
        (
            ["add , cookbook , table , on", "add , exit , livingroom , north_of"],
            ["add , exit , livingroom , north_of", "add , cookbook , table , on"],
        ),
        (
            ["add , cookbook , table , on", "add , exit , livingroom , south_of"],
            ["add , exit , livingroom , south_of", "add , cookbook , table , on"],
        ),
        # recipe before others
        (
            ["add , cookbook , table , on", "add , potato , recipe , part_of"],
            ["add , potato , recipe , part_of", "add , cookbook , table , on"],
        ),
        # two args relations
        (
            ["add , potato , cooked , is", "add , cookbook , table , on"],
            ["add , cookbook , table , on", "add , potato , cooked , is"],
        ),
        # skip one arg requirement relations as it's part of TWO_ARGS_RELATIONS
        # reverse alphabetical, destination node
        (
            ["add , potato , stove , on", "add , potato , table , on"],
            ["add , potato , table , on", "add , potato , stove , on"],
        ),
        # reverse alphabetical, source node
        (
            ["add , cookbook , table , on", "add , potato , table , on"],
            ["add , potato , table , on", "add , cookbook , table , on"],
        ),
        (
            [
                "add , cookbook , table , on",
                "add , counter , kitchen , at",
                "add , fridge , closed , is",
                "add , fridge , kitchen , at",
                "add , oven , kitchen , at",
                "add , player , kitchen , at",
                "add , purple potato , counter , on",
                "add , purple potato , sliced , is",
                "add , red potato , counter , on",
                "add , red potato , uncut , is",
                "add , stove , kitchen , at",
                "add , table , kitchen , at",
            ],
            [
                "add , player , kitchen , at",
                "add , cookbook , table , on",
                "add , table , kitchen , at",
                "add , stove , kitchen , at",
                "add , oven , kitchen , at",
                "add , fridge , kitchen , at",
                "add , counter , kitchen , at",
                "add , red potato , counter , on",
                "add , purple potato , counter , on",
                "add , red potato , uncut , is",
                "add , purple potato , sliced , is",
                "add , fridge , closed , is",
            ],
        ),
        (
            [
                "add , fridge , open , is",
                "add , parsley , fridge , in",
                "add , parsley , uncut , is",
                "add , red onion , fridge , in",
                "add , red onion , raw , is",
                "add , red onion , uncut , is",
                "add , white onion , fridge , in",
                "add , white onion , raw , is",
                "add , white onion , sliced , is",
                "delete , fridge , closed , is",
            ],
            [
                "add , white onion , fridge , in",
                "add , red onion , fridge , in",
                "add , parsley , fridge , in",
                "add , red onion , uncut , is",
                "add , parsley , uncut , is",
                "add , white onion , sliced , is",
                "add , white onion , raw , is",
                "add , red onion , raw , is",
                "add , fridge , open , is",
                "delete , fridge , closed , is",
            ],
        ),
    ],
)
def test_sort_target_commands(tgt_cmds, expected):
    assert sort_target_commands(tgt_cmds) == expected


@pytest.mark.parametrize(
    "obs,prev_actions,timestamps,expected",
    [
        (
            ["you are hungry ! let 's cook a delicious meal ."],
            ["drop knife"],
            [2],
            TWCmdGenGraphEventStepInput(
                obs_word_ids=torch.tensor(
                    [[2, 769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21, 3]]
                ),
                obs_mask=torch.ones(1, 13).bool(),
                prev_action_word_ids=torch.tensor([[2, 257, 404, 3]]),
                prev_action_mask=torch.ones(1, 4).bool(),
                timestamps=torch.tensor([2]),
            ),
        ),
        (
            [
                "you are hungry ! let 's cook a delicious meal .",
                "you take the knife from the table .",
            ],
            ["drop knife", "take knife from table"],
            [2, 3],
            TWCmdGenGraphEventStepInput(
                obs_word_ids=torch.tensor(
                    [
                        [2, 769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21, 3],
                        [2, 769, 663, 676, 404, 315, 676, 661, 21, 3, 0, 0, 0],
                    ]
                ),
                obs_mask=torch.tensor([[True] * 13, [True] * 10 + [False] * 3]),
                prev_action_word_ids=torch.tensor(
                    [[2, 257, 404, 3, 0, 0], [2, 663, 404, 315, 661, 3]]
                ),
                prev_action_mask=torch.tensor(
                    [[True, True, True, True, False, False], [True] * 6]
                ),
                timestamps=torch.tensor([2, 3]),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_step_inputs(
    tw_cmd_gen_collator, obs, prev_actions, timestamps, expected
):
    assert (
        tw_cmd_gen_collator.collate_step_inputs(obs, prev_actions, timestamps)
        == expected
    )


@pytest.mark.parametrize(
    "batch,initial_batched_graph,expected",
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
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.empty(1, 0).long(),
                    tgt_event_label_mask=torch.empty(1, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor([[2, 3]]),
                    groundtruth_event_label_mask=torch.tensor([False]),
                    prev_batched_graph=Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
            ),
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
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                node_label_mask=torch.ones(2, 3).bool(),
                node_last_update=torch.tensor([[0, 0], [0, 1]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([[2, 382, 3]]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[0, 2]]),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.empty(1, 0).long(),
                    tgt_event_label_mask=torch.empty(1, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor([[2, 3]]),
                    groundtruth_event_label_mask=torch.tensor([False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[0, 0], [0, 1]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=torch.tensor([[2, 382, 3]]),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[0, 2]]),
                    ),
                ),
            ),
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
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.empty(1, 0).long(),
                    tgt_event_label_mask=torch.empty(1, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2, 530]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 530, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True]),
                    prev_batched_graph=Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 530, 3]]),
                    tgt_event_label_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2, 402]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 402, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True]),
                    prev_batched_graph=Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 402, 3]]),
                    tgt_event_label_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([True]),
                    groundtruth_event_dst_ids=torch.tensor([1]),
                    groundtruth_event_dst_mask=torch.tensor([True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2, 382]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[2, 530, 3]]),
                        node_label_mask=torch.ones(1, 3).bool(),
                        node_last_update=torch.tensor([[2, 0]]),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([1]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor([[2, 3]]),
                    groundtruth_event_label_mask=torch.tensor([False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[2, 0], [2, 1]]),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , chicken leg , kitchen , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "chicken leg"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 1, "label": "in"},
                    ],
                }
            ],
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                node_label_mask=torch.ones(2, 3).bool(),
                node_last_update=torch.tensor([[1, 0], [1, 1]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([[2, 382, 3]]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[1, 2]]),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.empty(1, 0).long(),
                    tgt_event_label_mask=torch.empty(1, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["node-add"]]
                    ),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 192, 415]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 4).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 192, 415, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[1, 0], [1, 1]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=torch.tensor([[2, 382, 3]]),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    tgt_event_src_ids=torch.tensor([0]),
                    tgt_event_dst_ids=torch.tensor([0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 192, 415, 3]]),
                    tgt_event_label_mask=torch.ones(1, 4).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["edge-add"]]
                    ),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([2]),
                    groundtruth_event_src_mask=torch.tensor([True]),
                    groundtruth_event_dst_ids=torch.tensor([1]),
                    groundtruth_event_dst_mask=torch.tensor([True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2, 382]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[1, 0], [1, 1]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=torch.tensor([[2, 382, 3]]),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    tgt_event_src_ids=torch.tensor([2]),
                    tgt_event_dst_ids=torch.tensor([1]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(1, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    groundtruth_event_mask=torch.tensor([True]),
                    groundtruth_event_src_ids=torch.tensor([0]),
                    groundtruth_event_src_mask=torch.tensor([False]),
                    groundtruth_event_dst_ids=torch.tensor([0]),
                    groundtruth_event_dst_mask=torch.tensor([False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor([[2, 3]]),
                    groundtruth_event_label_mask=torch.tensor([False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0]),
                        x=torch.tensor(
                            [[2, 530, 3, 0], [2, 402, 3, 0], [2, 192, 415, 3]]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                            ]
                        ),
                        node_last_update=torch.tensor([[1, 0], [1, 1], [2, 0]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=torch.tensor([[2, 382, 3]]),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[1, 2]]),
                    ),
                ),
            ),
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
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                        {"type": "node-add", "label": "chicken leg"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 1, "label": "in"},
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                        },
                        {"type": "node-delete", "node_id": 0, "label": "player"},
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
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "livingroom"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 3, "label": "in"},
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                        },
                        {"type": "node-delete", "node_id": 1, "label": "kitchen"},
                        {"type": "node-delete", "node_id": 0, "label": "chicken leg"},
                    ],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.empty(2, 0).long(),
                    tgt_event_label_mask=torch.empty(2, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 530, 3], [2, 2, 192, 415]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 530, 3, 0], [2, 192, 415, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 530, 3, 0], [2, 192, 415, 3]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 402], [2, 2, 402]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 402, 3], [2, 402, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 402, 3], [2, 402, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([1, 1]),
                    groundtruth_event_dst_mask=torch.tensor([True, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[2, 530, 3, 0], [2, 192, 415, 3]]),
                        node_label_mask=torch.tensor(
                            [[True, True, True, False], [True, True, True, True]]
                        ),
                        node_last_update=torch.tensor([[3, 0], [1, 0]]),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([1, 1]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 192, 415], [2, 2, 530, 3]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, True, True], [True, True, True, False]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 192, 415, 3], [2, 530, 3, 0]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor([[3, 0], [3, 1], [1, 0], [1, 1]]),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 192, 415, 3], [2, 530, 3, 0]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, True, True], [True, True, True, False]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([2, 0]),
                    groundtruth_event_src_mask=torch.tensor([True, False]),
                    groundtruth_event_dst_ids=torch.tensor([1, 0]),
                    groundtruth_event_dst_mask=torch.tensor([True, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 425]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 425, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor([[3, 0], [3, 1], [1, 0], [1, 1]]),
                        edge_index=torch.tensor([[0, 2], [1, 3]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 0]),
                    tgt_event_dst_ids=torch.tensor([1, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 425, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 2]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([1, 3]),
                    groundtruth_event_dst_mask=torch.tensor([True, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [3, 3], [1, 0], [1, 1], [1, 3]]
                        ),
                        edge_index=torch.tensor([[0, 3], [1, 4]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 2]),
                    tgt_event_dst_ids=torch.tensor([1, 3]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 1]),
                    groundtruth_event_dst_mask=torch.tensor([False, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 530], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 530, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [3, 3], [1, 0], [1, 1], [1, 3], [1, 4]]
                        ),
                        edge_index=torch.tensor([[0, 3, 2], [1, 4, 1]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(3, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2], [3, 4]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 1]),
                    tgt_event_label_word_ids=torch.tensor([[2, 530, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 1]),
                    groundtruth_event_src_mask=torch.tensor([False, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 3], [2, 2, 402]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3, 0], [2, 402, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [3, 3], [1, 0], [1, 1], [1, 3], [1, 4]]
                        ),
                        edge_index=torch.tensor([[3, 2, 5], [4, 1, 6]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(3, 3).bool(),
                        edge_last_update=torch.tensor([[1, 2], [3, 4], [1, 5]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 1]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 3, 0], [2, 402, 3]]),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([False, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 3, 0], [2, 2, 192, 415]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, False, False], [True] * 4]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3, 0, 0], [2, 192, 415, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 1], [3, 3], [1, 0], [1, 1], [1, 3], [1, 4]]
                        ),
                        edge_index=torch.tensor([[1, 4], [0, 5]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 4], [1, 5]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 3, 0, 0], [2, 192, 415, 3]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, False, False], [True] * 4]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["end"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([False, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2], [2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3], [2, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 1], [3, 3], [1, 0], [1, 3], [1, 4]]
                        ),
                        edge_index=torch.tensor([[1, 3], [0, 4]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 4], [1, 5]]),
                    ),
                ),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 4,
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
                    "timestamp": 2,
                    "target_commands": [
                        "add , chicken leg , kitchen , in",
                        "add , player , livingroom , in",
                        "delete , chicken leg , kitchen , in",
                    ],
                    "graph_events": [
                        {"type": "node-add", "label": "chicken leg"},
                        {"type": "node-add", "label": "kitchen"},
                        {"type": "edge-add", "src_id": 3, "dst_id": 4, "label": "in"},
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "livingroom"},
                        {"type": "edge-add", "src_id": 5, "dst_id": 6, "label": "in"},
                        {
                            "type": "edge-delete",
                            "src_id": 3,
                            "dst_id": 4,
                            "label": "in",
                        },
                        {"type": "node-delete", "node_id": 4, "label": "kitchen"},
                        {"type": "node-delete", "node_id": 3, "label": "chicken leg"},
                    ],
                },
            ],
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1]),
                x=torch.tensor(
                    [
                        [2, 402, 3, 0],
                        [2, 192, 415, 3],
                        [2, 192, 415, 3],
                        [2, 530, 3, 0],
                        [2, 425, 3, 0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, True, True, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor([[3, 0], [3, 1], [1, 0], [1, 1], [1, 3]]),
                edge_index=torch.tensor([[1, 3], [0, 4]]),
                edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                edge_label_mask=torch.ones(2, 3).bool(),
                edge_last_update=torch.tensor([[3, 2], [1, 2]]),
            ),
            (
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.empty(2, 0).long(),
                    tgt_event_label_mask=torch.empty(2, 0).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 530, 3], [2, 2, 192, 415]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 530, 3, 0], [2, 192, 415, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [1, 0], [1, 1], [1, 3]]
                        ),
                        edge_index=torch.tensor([[1, 3], [0, 4]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 530, 3, 0], [2, 192, 415, 3]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 402], [2, 2, 402]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 402, 3], [2, 402, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [1, 0], [1, 1], [1, 3]]
                        ),
                        edge_index=torch.tensor([[1, 3], [0, 4]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 402, 3], [2, 402, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([2, 3]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([3, 4]),
                    groundtruth_event_dst_mask=torch.tensor([True, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [[3, 0], [3, 1], [4, 0], [1, 0], [1, 1], [1, 3], [2, 0]]
                        ),
                        edge_index=torch.tensor([[1, 4], [0, 5]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 3]),
                    tgt_event_dst_ids=torch.tensor([3, 4]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 192, 415], [2, 2, 530, 3]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, True, True], [True, True, True, False]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 192, 415, 3], [2, 530, 3, 0]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 0],
                                [4, 1],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 5], [0, 6]]),
                        edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                        edge_label_mask=torch.ones(2, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 0]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 192, 415, 3], [2, 530, 3, 0]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, True, True], [True, True, True, False]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([4, 0]),
                    groundtruth_event_src_mask=torch.tensor([True, False]),
                    groundtruth_event_dst_ids=torch.tensor([3, 0]),
                    groundtruth_event_dst_mask=torch.tensor([True, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 425]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 425, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([True, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 0],
                                [4, 1],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 5, 2, 7], [0, 6, 3, 8]]),
                        edge_attr=torch.tensor(
                            [[2, 382, 3], [2, 382, 3], [2, 382, 3], [2, 382, 3]]
                        ),
                        edge_label_mask=torch.ones(4, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2], [4, 2], [2, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([4, 0]),
                    tgt_event_dst_ids=torch.tensor([3, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 425, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([2, 5]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([3, 6]),
                    groundtruth_event_dst_mask=torch.tensor([True, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 382], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 382, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, True]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 0],
                                [4, 1],
                                [4, 3],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                                [2, 3],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 6, 2, 8], [0, 7, 3, 9]]),
                        edge_attr=torch.tensor(
                            [[2, 382, 3], [2, 382, 3], [2, 382, 3], [2, 382, 3]]
                        ),
                        edge_label_mask=torch.ones(4, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2], [4, 2], [2, 2]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 5]),
                    tgt_event_dst_ids=torch.tensor([3, 6]),
                    tgt_event_label_word_ids=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([2, 3]),
                    groundtruth_event_src_mask=torch.tensor([True, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 4]),
                    groundtruth_event_dst_mask=torch.tensor([False, True]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 530], [2, 2, 382]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 530, 3], [2, 382, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 0],
                                [4, 1],
                                [4, 3],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                                [2, 3],
                                [2, 4],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 6, 2, 8, 4], [0, 7, 3, 9, 3]]),
                        edge_attr=torch.tensor(
                            [
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                            ]
                        ),
                        edge_label_mask=torch.ones(5, 3).bool(),
                        edge_last_update=torch.tensor(
                            [[3, 2], [1, 2], [4, 2], [2, 2], [4, 4]]
                        ),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-delete"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([2, 3]),
                    tgt_event_dst_ids=torch.tensor([0, 4]),
                    tgt_event_label_word_ids=torch.tensor([[2, 530, 3], [2, 382, 3]]),
                    tgt_event_label_mask=torch.ones(2, 3).bool(),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([True, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 4]),
                    groundtruth_event_src_mask=torch.tensor([False, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 3], [2, 2, 402]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3, 0], [2, 402, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 0],
                                [4, 1],
                                [4, 3],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                                [2, 3],
                                [2, 4],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 6, 8, 4, 10], [0, 7, 9, 3, 11]]),
                        edge_attr=torch.tensor(
                            [
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                                [2, 382, 3],
                            ]
                        ),
                        edge_label_mask=torch.ones(5, 3).bool(),
                        edge_last_update=torch.tensor(
                            [[3, 2], [1, 2], [2, 2], [4, 4], [2, 5]]
                        ),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["end"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 4]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor([[2, 3, 0], [2, 402, 3]]),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([False, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 3]),
                    groundtruth_event_src_mask=torch.tensor([False, True]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor(
                        [[2, 2, 3, 0], [2, 2, 192, 415]]
                    ),
                    groundtruth_event_label_tgt_mask=torch.tensor(
                        [[True, True, False, False], [True] * 4]
                    ),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3, 0, 0], [2, 192, 415, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 1],
                                [4, 3],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 1],
                                [2, 3],
                                [2, 4],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 5, 3, 9], [0, 6, 2, 10]]),
                        edge_attr=torch.tensor(
                            [[2, 382, 3], [2, 382, 3], [2, 382, 3], [2, 382, 3]]
                        ),
                        edge_label_mask=torch.ones(4, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2], [4, 4], [2, 5]]),
                    ),
                ),
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    tgt_event_src_ids=torch.tensor([0, 3]),
                    tgt_event_dst_ids=torch.tensor([0, 0]),
                    tgt_event_label_word_ids=torch.tensor(
                        [[2, 3, 0, 0], [2, 192, 415, 3]]
                    ),
                    tgt_event_label_mask=torch.tensor(
                        [[True, True, False, False], [True] * 4]
                    ),
                    groundtruth_event_type_ids=torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["pad"],
                            EVENT_TYPE_ID_MAP["end"],
                        ]
                    ),
                    groundtruth_event_mask=torch.tensor([False, True]),
                    groundtruth_event_src_ids=torch.tensor([0, 0]),
                    groundtruth_event_src_mask=torch.tensor([False, False]),
                    groundtruth_event_dst_ids=torch.tensor([0, 0]),
                    groundtruth_event_dst_mask=torch.tensor([False, False]),
                    groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2], [2, 2]]),
                    groundtruth_event_label_tgt_mask=torch.ones(2, 2).bool(),
                    groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                        [[2, 3], [2, 3]]
                    ),
                    groundtruth_event_label_mask=torch.tensor([False, False]),
                    prev_batched_graph=Batch(
                        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 402, 3, 0],
                                [2, 192, 415, 3],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                                [2, 192, 415, 3],
                                [2, 530, 3, 0],
                                [2, 425, 3, 0],
                            ]
                        ),
                        node_label_mask=torch.tensor(
                            [
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                                [True, True, True, True],
                                [True, True, True, False],
                                [True, True, True, False],
                            ]
                        ),
                        node_last_update=torch.tensor(
                            [
                                [3, 0],
                                [3, 1],
                                [4, 1],
                                [4, 3],
                                [1, 0],
                                [1, 1],
                                [1, 3],
                                [2, 0],
                                [2, 3],
                                [2, 4],
                            ]
                        ),
                        edge_index=torch.tensor([[1, 5, 3, 8], [0, 6, 2, 9]]),
                        edge_attr=torch.tensor(
                            [[2, 382, 3], [2, 382, 3], [2, 382, 3], [2, 382, 3]]
                        ),
                        edge_label_mask=torch.ones(4, 3).bool(),
                        edge_last_update=torch.tensor([[3, 2], [1, 2], [4, 4], [2, 5]]),
                    ),
                ),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_graphical_input_seq(
    tw_cmd_gen_collator, batch, initial_batched_graph, expected
):
    collated = tw_cmd_gen_collator.collate_graphical_input_seq(
        batch, initial_batched_graph
    )
    assert collated == expected


def test_read_label_vocab_files():
    labels, label_id_map = read_label_vocab_files(
        "tests/data/test_node_vocab.txt", "tests/data/test_relation_vocab.txt"
    )
    assert labels == ["", "player", "inventory", "chopped", "in", "is", "east of"]
    assert label_id_map == {
        "": 0,
        "player": 1,
        "inventory": 2,
        "chopped": 3,
        "in": 4,
        "is": 5,
        "east of": 6,
    }


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": [],
                    "prev_graph_events": [],
                }
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                    "prev_graph_events": [
                        {"type": "node-add", "label": "player", "timestamp": [2, 0]},
                        {"type": "node-add", "label": "kitchen", "timestamp": [3, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [3, 1],
                        },
                    ],
                }
            ],
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                node_label_mask=torch.ones(2, 3).bool(),
                node_last_update=torch.tensor([[2, 0], [3, 0]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([[2, 382, 3]]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[3, 1]]),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 2,
                    "target_commands": ["add , player , kitchen , in"],
                    "prev_graph_events": [
                        {"type": "node-add", "label": "player", "timestamp": [1, 3]},
                        {"type": "node-add", "label": "kitchen", "timestamp": [2, 1]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [3, 0],
                        },
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "timestamp": 4,
                    "target_commands": ["add , player , livingroom , in"],
                    "prev_graph_events": [
                        {"type": "node-add", "label": "player", "timestamp": [4, 2]},
                        {
                            "type": "node-add",
                            "label": "livingroom",
                            "timestamp": [5, 1],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [6, 2],
                        },
                    ],
                },
            ],
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[2, 530, 3], [2, 402, 3], [2, 530, 3], [2, 425, 3]]),
                node_label_mask=torch.ones(4, 3).bool(),
                node_last_update=torch.tensor([[1, 3], [2, 1], [4, 2], [5, 1]]),
                edge_index=torch.tensor([[0, 2], [1, 3]]),
                edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                edge_label_mask=torch.ones(2, 3).bool(),
                edge_last_update=torch.tensor([[3, 0], [6, 2]]),
            ),
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
                    "prev_graph_events": [
                        {"type": "node-add", "label": "player", "timestamp": [1, 0]},
                        {"type": "node-add", "label": "kitchen", "timestamp": [2, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [3, 0],
                        },
                        {
                            "type": "node-add",
                            "label": "chicken leg",
                            "timestamp": [3, 1],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [4, 0],
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [5, 0],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 0,
                            "label": "player",
                            "timestamp": [5, 1],
                        },
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
                    "prev_graph_events": [
                        {
                            "type": "node-add",
                            "label": "chicken leg",
                            "timestamp": [2, 0],
                        },
                        {"type": "node-add", "label": "kitchen", "timestamp": [3, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [4, 0],
                        },
                        {"type": "node-add", "label": "player", "timestamp": [5, 0]},
                        {
                            "type": "node-add",
                            "label": "livingroom",
                            "timestamp": [6, 0],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "in",
                            "timestamp": [7, 0],
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [7, 1],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 1,
                            "label": "kitchen",
                            "timestamp": [8, 0],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 0,
                            "label": "chicken leg",
                            "timestamp": [8, 1],
                        },
                    ],
                },
            ],
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor(
                    [[2, 402, 3, 0], [2, 192, 415, 3], [2, 530, 3, 0], [2, 425, 3, 0]]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, True, True, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor([[2, 0], [3, 1], [5, 0], [6, 0]]),
                edge_index=torch.tensor([[1, 2], [0, 3]]),
                edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                edge_label_mask=torch.ones(2, 3).bool(),
                edge_last_update=torch.tensor([[4, 0], [7, 0]]),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_collate_prev_graph_events(
    tw_cmd_gen_collator, batch, expected
):
    collated = tw_cmd_gen_collator.collate_prev_graph_events(batch)

    assert collated.batch.equal(expected.batch)
    assert collated.x.equal(expected.x)
    assert collated.node_last_update.equal(expected.node_last_update)
    assert collated.edge_index.equal(expected.edge_index)
    assert collated.edge_attr.equal(expected.edge_attr)
    assert collated.edge_last_update.equal(expected.edge_last_update)


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "random_step": 0,
                    "observation": "you are hungry ! " "let 's cook a delicious meal .",
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
                    "prev_graph_events": [],
                },
            ],
            TWCmdGenGraphEventBatch(
                ids=(("g1", 0, 0),),
                step_input=TWCmdGenGraphEventStepInput(
                    obs_word_ids=torch.tensor(
                        [[2, 769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21, 3]]
                    ),
                    obs_mask=torch.ones(1, 13).bool(),
                    prev_action_word_ids=torch.tensor([[2, 257, 404, 3]]),
                    prev_action_mask=torch.ones(1, 4).bool(),
                    timestamps=torch.tensor([2]),
                ),
                initial_batched_graph=Batch(
                    batch=torch.empty(0, dtype=torch.long),
                    x=torch.empty(0, 0, dtype=torch.long),
                    node_label_mask=torch.empty(0, 0).bool(),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, 0).long(),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                graphical_input_seq=(
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.empty(1, 0).long(),
                        tgt_event_label_mask=torch.empty(1, 0).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 530]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 530, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.empty(0, dtype=torch.long),
                            x=torch.empty(0, 0, dtype=torch.long),
                            node_label_mask=torch.empty(0, 0).bool(),
                            node_last_update=torch.empty(0, 2, dtype=torch.long),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=torch.empty(0, 0).long(),
                            edge_label_mask=torch.empty(0, 0).bool(),
                            edge_last_update=torch.empty(0, 2, dtype=torch.long),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.tensor([[2, 530, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 402]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 402, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.empty(0, dtype=torch.long),
                            x=torch.empty(0, 0, dtype=torch.long),
                            node_label_mask=torch.empty(0, 0).bool(),
                            node_last_update=torch.empty(0, 2, dtype=torch.long),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=torch.empty(0, 0).long(),
                            edge_label_mask=torch.empty(0, 0).bool(),
                            edge_last_update=torch.empty(0, 2, dtype=torch.long),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.tensor([[2, 402, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([True]),
                        groundtruth_event_dst_ids=torch.tensor([1]),
                        groundtruth_event_dst_mask=torch.tensor([True]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 382]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 382, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0]),
                            x=torch.tensor([[2, 530, 3]]),
                            node_label_mask=torch.ones(1, 3).bool(),
                            node_last_update=torch.tensor([[2, 0]]),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=torch.empty(0, 0).long(),
                            edge_label_mask=torch.empty(0, 0).bool(),
                            edge_last_update=torch.empty(0, 2, dtype=torch.long),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([1]),
                        tgt_event_label_word_ids=torch.tensor([[2, 382, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0]),
                            x=torch.tensor([[2, 530, 3], [2, 402, 3]]),
                            node_label_mask=torch.ones(2, 3).bool(),
                            node_last_update=torch.tensor([[2, 0], [2, 1]]),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=torch.empty(0, 0).long(),
                            edge_label_mask=torch.empty(0, 0).bool(),
                            edge_last_update=torch.empty(0, 2, dtype=torch.long),
                        ),
                    ),
                ),
                graph_commands=(("add , player , kitchen , in",),),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "random_step": 0,
                    "observation": "you are hungry ! " "let 's cook a delicious meal .",
                    "previous_action": "drop knife",
                    "timestamp": 9,
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
                    "prev_graph_events": [
                        {
                            "type": "node-add",
                            "label": "chicken leg",
                            "timestamp": [2, 0],
                        },
                        {"type": "node-add", "label": "kitchen", "timestamp": [3, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [4, 0],
                        },
                        {"type": "node-add", "label": "player", "timestamp": [5, 0]},
                        {
                            "type": "node-add",
                            "label": "livingroom",
                            "timestamp": [6, 0],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "in",
                            "timestamp": [7, 0],
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [7, 1],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 1,
                            "label": "kitchen",
                            "timestamp": [8, 0],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 0,
                            "label": "chicken leg",
                            "timestamp": [8, 1],
                        },
                    ],
                },
            ],
            TWCmdGenGraphEventBatch(
                ids=(("g1", 0, 0),),
                step_input=TWCmdGenGraphEventStepInput(
                    obs_word_ids=torch.tensor(
                        [[2, 769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21, 3]]
                    ),
                    obs_mask=torch.ones(1, 13).bool(),
                    prev_action_word_ids=torch.tensor([[2, 257, 404, 3]]),
                    prev_action_mask=torch.ones(1, 4).bool(),
                    timestamps=torch.tensor([9]),
                ),
                initial_batched_graph=Batch(
                    batch=torch.tensor([0, 0]),
                    x=torch.tensor([[2, 530, 3], [2, 425, 3]]),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([[5, 0], [6, 0]]),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=torch.tensor([[2, 382, 3]]),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[7, 0]]),
                ),
                graphical_input_seq=(
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.empty(1, 0).long(),
                        tgt_event_label_mask=torch.empty(1, 0).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 530]]
                        ),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 530, 3]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0]),
                            x=torch.tensor([[2, 530, 3], [2, 425, 3]]),
                            node_label_mask=torch.ones(2, 3).bool(),
                            node_last_update=torch.tensor([[5, 0], [6, 0]]),
                            edge_index=torch.tensor([[0], [1]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[7, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.tensor([[2, 530, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 402]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 402, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0]),
                            x=torch.tensor([[2, 530, 3], [2, 425, 3]]),
                            node_label_mask=torch.ones(2, 3).bool(),
                            node_last_update=torch.tensor([[5, 0], [6, 0]]),
                            edge_index=torch.tensor([[0], [1]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[7, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["node-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([0]),
                        tgt_event_label_word_ids=torch.tensor([[2, 402, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([True]),
                        groundtruth_event_dst_ids=torch.tensor([1]),
                        groundtruth_event_dst_mask=torch.tensor([True]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 382]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 382, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 0]),
                            x=torch.tensor([[2, 530, 3], [2, 425, 3], [2, 530, 3]]),
                            node_label_mask=torch.ones(3, 3).bool(),
                            node_last_update=torch.tensor([[5, 0], [6, 0], [9, 0]]),
                            edge_index=torch.tensor([[0], [1]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[7, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["edge-add"]]
                        ),
                        tgt_event_src_ids=torch.tensor([0]),
                        tgt_event_dst_ids=torch.tensor([1]),
                        tgt_event_label_word_ids=torch.tensor([[2, 382, 3]]),
                        tgt_event_label_mask=torch.ones(1, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_mask=torch.tensor([True]),
                        groundtruth_event_src_ids=torch.tensor([0]),
                        groundtruth_event_src_mask=torch.tensor([False]),
                        groundtruth_event_dst_ids=torch.tensor([0]),
                        groundtruth_event_dst_mask=torch.tensor([False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor([[2, 2]]),
                        groundtruth_event_label_tgt_mask=torch.ones(1, 2).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 0, 0]),
                            x=torch.tensor(
                                [[2, 530, 3], [2, 425, 3], [2, 530, 3], [2, 402, 3]]
                            ),
                            node_label_mask=torch.ones(4, 3).bool(),
                            node_last_update=torch.tensor(
                                [[5, 0], [6, 0], [9, 0], [9, 1]]
                            ),
                            edge_index=torch.tensor([[0], [1]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[7, 0]]),
                        ),
                    ),
                ),
                graph_commands=(("add , player , kitchen , in",),),
            ),
        ),
        (
            [
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "random_step": 1,
                    "observation": "you are hungry ! " "let 's cook a delicious meal .",
                    "previous_action": "drop knife",
                    "timestamp": 6,
                    "target_commands": ["add , player , kitchen , in"],
                    "graph_events": [
                        {"type": "node-add", "label": "player"},
                        {"type": "node-add", "label": "kitchen"},
                    ],
                    "prev_graph_events": [
                        {"type": "node-add", "label": "player", "timestamp": [1, 0]},
                        {"type": "node-add", "label": "kitchen", "timestamp": [2, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [3, 0],
                        },
                        {
                            "type": "node-add",
                            "label": "chicken leg",
                            "timestamp": [3, 1],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [4, 0],
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [5, 0],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 0,
                            "label": "player",
                            "timestamp": [5, 1],
                        },
                    ],
                },
                {
                    "game": "g2",
                    "walkthrough_step": 2,
                    "random_step": 3,
                    "observation": "you take the knife from the table .",
                    "previous_action": "take knife from table",
                    "timestamp": 9,
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
                    "prev_graph_events": [
                        {
                            "type": "node-add",
                            "label": "chicken leg",
                            "timestamp": [2, 0],
                        },
                        {"type": "node-add", "label": "kitchen", "timestamp": [3, 0]},
                        {
                            "type": "edge-add",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [4, 0],
                        },
                        {"type": "node-add", "label": "player", "timestamp": [5, 0]},
                        {
                            "type": "node-add",
                            "label": "livingroom",
                            "timestamp": [6, 0],
                        },
                        {
                            "type": "edge-add",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "in",
                            "timestamp": [7, 0],
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "in",
                            "timestamp": [7, 1],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 1,
                            "label": "kitchen",
                            "timestamp": [8, 0],
                        },
                        {
                            "type": "node-delete",
                            "node_id": 0,
                            "label": "chicken leg",
                            "timestamp": [8, 1],
                        },
                    ],
                },
            ],
            TWCmdGenGraphEventBatch(
                ids=(("g1", 0, 1), ("g2", 2, 3)),
                step_input=TWCmdGenGraphEventStepInput(
                    obs_word_ids=torch.tensor(
                        [
                            [2, 769, 122, 377, 5, 416, 12, 215, 94, 237, 441, 21, 3],
                            [2, 769, 663, 676, 404, 315, 676, 661, 21, 3, 0, 0, 0],
                        ]
                    ),
                    obs_mask=torch.tensor([[True] * 13, [True] * 10 + [False] * 3]),
                    prev_action_word_ids=torch.tensor(
                        [[2, 257, 404, 3, 0, 0], [2, 663, 404, 315, 661, 3]]
                    ),
                    prev_action_mask=torch.tensor(
                        [[True, True, True, True, False, False], [True] * 6]
                    ),
                    timestamps=torch.tensor([6, 9]),
                ),
                initial_batched_graph=Batch(
                    batch=torch.tensor([0, 0, 1, 1]),
                    x=torch.tensor(
                        [
                            [2, 402, 3, 0],
                            [2, 192, 415, 3],
                            [2, 530, 3, 0],
                            [2, 425, 3, 0],
                        ]
                    ),
                    node_label_mask=torch.tensor(
                        [
                            [True, True, True, False],
                            [True, True, True, True],
                            [True, True, True, False],
                            [True, True, True, False],
                        ]
                    ),
                    node_last_update=torch.tensor([[2, 0], [3, 1], [5, 0], [6, 0]]),
                    edge_index=torch.tensor([[1, 2], [0, 3]]),
                    edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                    edge_label_mask=torch.ones(2, 3).bool(),
                    edge_last_update=torch.tensor([[4, 0], [7, 0]]),
                ),
                graphical_input_seq=(
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["start"],
                                EVENT_TYPE_ID_MAP["start"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_word_ids=torch.empty(2, 0).long(),
                        tgt_event_label_mask=torch.empty(2, 0).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["edge-delete"],
                            ]
                        ),
                        groundtruth_event_mask=torch.tensor([True, True]),
                        groundtruth_event_src_ids=torch.tensor([0, 0]),
                        groundtruth_event_src_mask=torch.tensor([False, True]),
                        groundtruth_event_dst_ids=torch.tensor([0, 1]),
                        groundtruth_event_dst_mask=torch.tensor([False, True]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 530], [2, 2, 382]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 530, 3], [2, 382, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True, False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 1, 1]),
                            x=torch.tensor(
                                [
                                    [2, 402, 3, 0],
                                    [2, 192, 415, 3],
                                    [2, 530, 3, 0],
                                    [2, 425, 3, 0],
                                ]
                            ),
                            node_label_mask=torch.tensor(
                                [
                                    [True, True, True, False],
                                    [True, True, True, True],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                ]
                            ),
                            node_last_update=torch.tensor(
                                [[2, 0], [3, 1], [5, 0], [6, 0]]
                            ),
                            edge_index=torch.tensor([[1, 2], [0, 3]]),
                            edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                            edge_label_mask=torch.ones(2, 3).bool(),
                            edge_last_update=torch.tensor([[4, 0], [7, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["edge-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 1]),
                        tgt_event_label_word_ids=torch.tensor(
                            [[2, 530, 3], [2, 382, 3]]
                        ),
                        tgt_event_label_mask=torch.ones(2, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        groundtruth_event_mask=torch.tensor([True, True]),
                        groundtruth_event_src_ids=torch.tensor([0, 1]),
                        groundtruth_event_src_mask=torch.tensor([False, True]),
                        groundtruth_event_dst_ids=torch.tensor([0, 0]),
                        groundtruth_event_dst_mask=torch.tensor([False, False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 402], [2, 2, 402]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.ones(2, 3).bool(),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 402, 3], [2, 402, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([True, False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 1, 1]),
                            x=torch.tensor(
                                [
                                    [2, 402, 3, 0],
                                    [2, 192, 415, 3],
                                    [2, 530, 3, 0],
                                    [2, 425, 3, 0],
                                ]
                            ),
                            node_label_mask=torch.tensor(
                                [
                                    [True, True, True, False],
                                    [True, True, True, True],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                ]
                            ),
                            node_last_update=torch.tensor(
                                [[2, 0], [3, 1], [5, 0], [6, 0]]
                            ),
                            edge_index=torch.tensor([[1, 2], [0, 3]]),
                            edge_attr=torch.tensor([[2, 382, 3], [2, 382, 3]]),
                            edge_label_mask=torch.ones(2, 3).bool(),
                            edge_last_update=torch.tensor([[4, 0], [7, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["node-add"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 1]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_word_ids=torch.tensor(
                            [[2, 402, 3], [2, 402, 3]]
                        ),
                        tgt_event_label_mask=torch.ones(2, 3).bool(),
                        groundtruth_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["end"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        groundtruth_event_mask=torch.tensor([True, True]),
                        groundtruth_event_src_ids=torch.tensor([0, 0]),
                        groundtruth_event_src_mask=torch.tensor([False, True]),
                        groundtruth_event_dst_ids=torch.tensor([0, 0]),
                        groundtruth_event_dst_mask=torch.tensor([False, False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2, 3], [2, 2, 530]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.tensor(
                            [[True, True, False], [True, True, True]]
                        ),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 3, 0], [2, 530, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([False, False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 0, 1, 1]),
                            x=torch.tensor(
                                [
                                    [2, 402, 3, 0],
                                    [2, 192, 415, 3],
                                    [2, 530, 3, 0],
                                    [2, 530, 3, 0],
                                    [2, 425, 3, 0],
                                ]
                            ),
                            node_label_mask=torch.tensor(
                                [
                                    [True, True, True, False],
                                    [True, True, True, True],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                ]
                            ),
                            node_last_update=torch.tensor(
                                [[2, 0], [3, 1], [6, 0], [5, 0], [6, 0]]
                            ),
                            edge_index=torch.tensor([[1], [0]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[4, 0]]),
                        ),
                    ),
                    TWCmdGenGraphEventGraphicalInput(
                        tgt_event_type_ids=torch.tensor(
                            [
                                EVENT_TYPE_ID_MAP["end"],
                                EVENT_TYPE_ID_MAP["node-delete"],
                            ]
                        ),
                        tgt_event_src_ids=torch.tensor([0, 0]),
                        tgt_event_dst_ids=torch.tensor([0, 0]),
                        tgt_event_label_word_ids=torch.tensor([[2, 3, 0], [2, 530, 3]]),
                        tgt_event_label_mask=torch.tensor(
                            [[True, True, False], [True, True, True]]
                        ),
                        groundtruth_event_type_ids=torch.tensor(
                            [EVENT_TYPE_ID_MAP["pad"], EVENT_TYPE_ID_MAP["end"]]
                        ),
                        groundtruth_event_mask=torch.tensor([False, True]),
                        groundtruth_event_src_ids=torch.tensor([0, 0]),
                        groundtruth_event_src_mask=torch.tensor([False, False]),
                        groundtruth_event_dst_ids=torch.tensor([0, 0]),
                        groundtruth_event_dst_mask=torch.tensor([False, False]),
                        groundtruth_event_label_tgt_word_ids=torch.tensor(
                            [[2, 2], [2, 2]]
                        ),
                        groundtruth_event_label_tgt_mask=torch.tensor(
                            [[True, True], [True, True]]
                        ),
                        groundtruth_event_label_groundtruth_word_ids=torch.tensor(
                            [[2, 3], [2, 3]]
                        ),
                        groundtruth_event_label_mask=torch.tensor([False, False]),
                        prev_batched_graph=Batch(
                            batch=torch.tensor([0, 0, 0, 0, 1]),
                            x=torch.tensor(
                                [
                                    [2, 402, 3, 0],
                                    [2, 192, 415, 3],
                                    [2, 530, 3, 0],
                                    [2, 402, 3, 0],
                                    [2, 530, 3, 0],
                                ]
                            ),
                            node_label_mask=torch.tensor(
                                [
                                    [True, True, True, False],
                                    [True, True, True, True],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                    [True, True, True, False],
                                ]
                            ),
                            node_last_update=torch.tensor(
                                [[2, 0], [3, 1], [6, 0], [6, 1], [5, 0]]
                            ),
                            edge_index=torch.tensor([[1], [0]]),
                            edge_attr=torch.tensor([[2, 382, 3]]),
                            edge_label_mask=torch.ones(1, 3).bool(),
                            edge_last_update=torch.tensor([[4, 0]]),
                        ),
                    ),
                ),
                graph_commands=(
                    ("add , player , kitchen , in",),
                    ("delete , player , kitchen , in",),
                ),
            ),
        ),
    ],
)
def test_tw_cmd_gen_collator_call(tw_cmd_gen_collator, batch, expected):
    assert tw_cmd_gen_collator(batch) == expected
