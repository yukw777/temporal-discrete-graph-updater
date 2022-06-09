import pytest
import torch
import torch.nn.functional as F
import shutil

from utils import EqualityData, EqualityBatch

from tdgu.train.self_supervised import ObsGenSelfSupervisedTDGU
from tdgu.data import TWCmdGenGraphEventStepInput, TWCmdGenGraphEventDataCollator


@pytest.fixture
def obs_gen_self_supervised_tdgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return ObsGenSelfSupervisedTDGU(
        pretrained_word_embedding_path=f"{tmp_path}/test-fasttext.vec",
        word_vocab_path="tests/data/test_word_vocab.txt",
    )


@pytest.mark.parametrize(
    "init_game_id_to_step_data_graph,batch,expected_step_input,expected_batched_graph,"
    "expected_game_id_to_step_data_graph",
    [
        (
            {},
            [
                (
                    0,
                    {
                        "observation": "my name is peter",
                        "previous_action": "chopped east",
                        "timestamp": 0,
                    },
                ),
                (
                    1,
                    {
                        "observation": "player is chopped",
                        "previous_action": "player",
                        "timestamp": 0,
                    },
                ),
            ],
            TWCmdGenGraphEventStepInput(
                obs_word_ids=torch.tensor([[2, 5, 6, 7, 8, 3], [2, 13, 7, 15, 3, 0]]),
                obs_mask=torch.tensor([[True] * 6, [True] * 5 + [False]]),
                prev_action_word_ids=torch.tensor([[2, 15, 16, 3], [2, 13, 3, 0]]),
                prev_action_mask=torch.tensor([[True] * 4, [True] * 3 + [False]]),
                timestamps=torch.tensor([0, 0]),
            ),
            EqualityBatch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, 17),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0, 17),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            {
                0: (
                    {
                        "observation": "my name is peter",
                        "previous_action": "chopped east",
                        "timestamp": 0,
                    },
                    EqualityData(
                        x=torch.empty(0, 0, 17),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0, 17),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                1: (
                    {
                        "observation": "player is chopped",
                        "previous_action": "player",
                        "timestamp": 0,
                    },
                    EqualityData(
                        x=torch.empty(0, 0, 17),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0, 17),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
            },
        ),
        (
            {
                0: ({}, EqualityData()),
                1: (
                    {
                        "observation": "east of inventory is player",
                        "previous_action": "to east",
                        "timestamp": 2,
                    },
                    EqualityData(
                        x=F.one_hot(
                            torch.tensor([[2, 13, 3], [2, 14, 3]]), num_classes=17
                        ),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[1, 2], [0, 0]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[0, 1]]),
                    ),
                ),
                2: ({}, EqualityData()),
            },
            [
                (
                    1,
                    {
                        "observation": "player east is peter",
                        "previous_action": "inventory",
                        "timestamp": 3,
                    },
                ),
                (
                    3,
                    {
                        "observation": "my name is peter",
                        "previous_action": "chopped east",
                        "timestamp": 0,
                    },
                ),
                (
                    4,
                    {
                        "observation": "player is chopped",
                        "previous_action": "player",
                        "timestamp": 0,
                    },
                ),
            ],
            TWCmdGenGraphEventStepInput(
                obs_word_ids=torch.tensor(
                    [[2, 13, 16, 7, 8, 3], [2, 5, 6, 7, 8, 3], [2, 13, 7, 15, 3, 0]]
                ),
                obs_mask=torch.tensor([[True] * 6, [True] * 6, [True] * 5 + [False]]),
                prev_action_word_ids=torch.tensor(
                    [[2, 14, 3, 0], [2, 15, 16, 3], [2, 13, 3, 0]]
                ),
                prev_action_mask=torch.tensor(
                    [[True] * 3 + [False], [True] * 4, [True] * 3 + [False]]
                ),
                timestamps=torch.tensor([3, 0, 0]),
            ),
            EqualityBatch(
                batch=torch.tensor([0, 0]),
                x=F.one_hot(
                    torch.tensor([[2, 13, 3], [2, 14, 3]]), num_classes=17
                ).float(),
                node_label_mask=torch.ones(2, 3).bool(),
                node_last_update=torch.tensor([[1, 2], [0, 0]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float(),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[0, 1]]),
            ),
            {
                1: (
                    {
                        "observation": "player east is peter",
                        "previous_action": "inventory",
                        "timestamp": 3,
                    },
                    EqualityData(
                        x=F.one_hot(
                            torch.tensor([[2, 13, 3], [2, 14, 3]]), num_classes=17
                        ),
                        node_label_mask=torch.ones(2, 3).bool(),
                        node_last_update=torch.tensor([[1, 2], [0, 0]]),
                        edge_index=torch.tensor([[0], [1]]),
                        edge_attr=F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17),
                        edge_label_mask=torch.ones(1, 3).bool(),
                        edge_last_update=torch.tensor([[0, 1]]),
                    ),
                ),
                3: (
                    {
                        "observation": "my name is peter",
                        "previous_action": "chopped east",
                        "timestamp": 0,
                    },
                    EqualityData(
                        x=torch.empty(0, 0, 17),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0, 17),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
                4: (
                    {
                        "observation": "player is chopped",
                        "previous_action": "player",
                        "timestamp": 0,
                    },
                    EqualityData(
                        x=torch.empty(0, 0, 17),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0, 17),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                ),
            },
        ),
    ],
)
def test_prepare_greedy_decode_input(
    monkeypatch,
    obs_gen_self_supervised_tdgu,
    init_game_id_to_step_data_graph,
    batch,
    expected_step_input,
    expected_batched_graph,
    expected_game_id_to_step_data_graph,
):
    monkeypatch.setattr("tdgu.train.self_supervised.Data", EqualityData)
    monkeypatch.setattr("tdgu.train.self_supervised.Batch", EqualityBatch)
    monkeypatch.setattr("tdgu.graph.Data", EqualityData)
    monkeypatch.setattr("tdgu.graph.Batch", EqualityBatch)

    obs_gen_self_supervised_tdgu.game_id_to_step_data_graph = (
        init_game_id_to_step_data_graph
    )

    (
        step_input,
        batched_graph,
    ) = obs_gen_self_supervised_tdgu.prepare_greedy_decode_input(
        batch, TWCmdGenGraphEventDataCollator(obs_gen_self_supervised_tdgu.preprocessor)
    )
    assert step_input == expected_step_input
    assert batched_graph == expected_batched_graph
    assert (
        obs_gen_self_supervised_tdgu.game_id_to_step_data_graph
        == expected_game_id_to_step_data_graph
    )
