import pytest
import shutil
import torch
import random

from torch_geometric.data.batch import Data, Batch

from tdgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP
from tdgu.train.supervised import SupervisedTDGU, UncertaintyWeightedLoss


@pytest.fixture()
def supervised_tdgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return SupervisedTDGU(
        text_encoder_conf={
            "pretrained_word_embedding_path": f"{tmp_path}/test-fasttext.vec",
            "word_vocab_path": "tests/data/test_word_vocab.txt",
        }
    )


@pytest.mark.parametrize("num_word", [10, 20])
@pytest.mark.parametrize("label_len", [2, 3])
@pytest.mark.parametrize("num_node", [5, 12])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("label", [True, False])
@pytest.mark.parametrize("dst", [True, False])
@pytest.mark.parametrize("src", [True, False])
def test_uncertainty_weighted_loss(
    src, dst, label, batch, num_node, label_len, num_word
):
    groundtruth_event_src_mask = torch.zeros(batch).bool()
    if src:
        groundtruth_event_src_mask[
            random.sample(range(batch), k=random.choice(range(1, batch + 1)))
        ] = True
    groundtruth_event_dst_mask = torch.zeros(batch).bool()
    if dst:
        groundtruth_event_dst_mask[
            random.sample(range(batch), k=random.choice(range(1, batch + 1)))
        ] = True
    groundtruth_event_label_mask = torch.zeros(batch).bool()
    if label:
        groundtruth_event_label_mask[
            random.sample(range(batch), k=random.choice(range(1, batch + 1)))
        ] = True
    assert UncertaintyWeightedLoss()(
        torch.rand(batch, len(EVENT_TYPES)),
        torch.randint(len(EVENT_TYPES), (batch,)),
        torch.rand(batch, num_node),
        torch.randint(num_node, (batch,)),
        torch.rand(batch, num_node),
        torch.randint(num_node, (batch,)),
        torch.randint(2, (batch, num_node)).bool(),
        torch.rand(batch, label_len, num_word),
        torch.randint(num_word, (batch, label_len)),
        torch.randint(2, (batch, label_len)).bool(),
        torch.randint(2, (batch,)).bool(),
        groundtruth_event_src_mask,
        groundtruth_event_dst_mask,
        groundtruth_event_label_mask,
    ).size() == (batch,)


@pytest.mark.parametrize(
    "event_type_logits,groundtruth_event_type_ids,event_src_logits,"
    "groundtruth_event_src_ids,event_dst_logits,groundtruth_event_dst_ids,"
    "batch_node_mask,event_label_logits,groundtruth_event_label_word_ids,"
    "groundtruth_event_label_word_mask,groundtruth_event_mask,"
    "groundtruth_event_src_mask,groundtruth_event_dst_mask,"
    "groundtruth_event_label_mask,expected_event_type_f1,"
    "expected_src_node_f1,expected_dst_node_f1,expected_label_f1",
    [
        (
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[False] * 7]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([False]),
            1,
            0,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[False] * 7]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([False]),
            0,
            0,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[False] * 7]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([True]),
            1,
            0,
            0,
            1,
        ),
        (
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[False] * 7]),
            torch.tensor(
                [[[0] * 1 + [1] + [0] * 15, [0] * 2 + [1] + [0] * 14]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([True]),
            0,
            0,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 0, 0, 1, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([1]),
            torch.tensor([[1, 0, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[True, True, True] + [False] * 4]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            1,
            1 / 7,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[True, True, True] + [False] * 4]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            0,
            0,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 0, 0, 0, 1, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([2]),
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).float(),
            torch.tensor([3]),
            torch.tensor([[True, True, True, True, True, False, False]]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            1,
            1 / 7,
            1 / 7,
            1,
        ),
        (
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[True, True, True, True, True, False, False]]),
            torch.tensor(
                [[[0] * 1 + [1] + [0] * 15, [0] * 2 + [1] + [0] * 14]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            0,
            0,
            0,
            0,
        ),
        (
            torch.tensor([[0, 0, 0, 0, 0, 0, 1]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([2]),
            torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).float(),
            torch.tensor([3]),
            torch.tensor([[True, True, True, True, True, False, False]]),
            torch.tensor(
                [[[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            1,
            1 / 7,
            1 / 7,
            0,
        ),
        (
            torch.tensor([[0, 0, 1, 0, 0, 0, 0]]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0]]).float(),
            torch.tensor([0]),
            torch.tensor([[True, True, True, True, True, False, False]]),
            torch.tensor(
                [[[0] * 1 + [1] + [0] * 15, [0] * 2 + [1] + [0] * 14]]
            ).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            0,
            0,
            0,
            0,
        ),
        (
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                ]
            ).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                ]
            ).float(),
            torch.tensor([3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0]),
            torch.tensor(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                ]
            ).float(),
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            torch.tensor(
                [
                    [True] * 8,
                    [True] * 8,
                    [False] * 8,
                    [False] * 8,
                    [False] * 8,
                    [False] * 8,
                    [True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False, False],
                ]
            ),
            torch.tensor(
                [
                    [[0] * 2 + [1] + [0] * 14, [0] * 3 + [1] + [0] * 13],
                    [[0] * 3 + [1] + [0] * 13, [0] * 7 + [1] + [0] * 9],
                    [[0] * 5 + [1] + [0] * 11, [0] * 4 + [1] + [0] * 12],
                    [[0] * 2 + [1] + [0] * 14, [0] * 4 + [1] + [0] * 12],
                    [[0] * 3 + [1] + [0] * 13, [0] * 6 + [1] + [0] * 10],
                    [[0] * 6 + [1] + [0] * 10, [0] * 5 + [1] + [0] * 11],
                    [[0] * 7 + [1] + [0] * 9, [0] * 3 + [1] + [0] * 13],
                    [[0] * 5 + [1] + [0] * 11, [0] * 3 + [1] + [0] * 13],
                    [[0] * 4 + [1] + [0] * 12, [0] * 2 + [1] + [0] * 14],
                    [[0] * 5 + [1] + [0] * 11, [0] * 6 + [1] + [0] * 10],
                    [[0] * 3 + [1] + [0] * 13, [0] * 5 + [1] + [0] * 11],
                    [[0] * 4 + [1] + [0] * 12, [0] * 3 + [1] + [0] * 13],
                ]
            ).float(),
            torch.tensor(
                [
                    [2, 3],
                    [3, 7],
                    [5, 4],
                    [2, 4],
                    [3, 6],
                    [6, 5],
                    [7, 3],
                    [5, 3],
                    [4, 2],
                    [5, 6],
                    [3, 5],
                    [4, 3],
                ]
            ),
            torch.ones(12, 2).bool(),
            torch.tensor(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ]
            ),
            torch.tensor(
                [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                ]
            ),
            torch.tensor(
                [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                ]
            ),
            torch.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                ]
            ),
            1,
            3 / 8,
            1 / 8,
            1,
        ),
    ],
)
@pytest.mark.parametrize("log_prefix", ["val", "test"])
def test_supervised_tdgu_calculate_f1s(
    supervised_tdgu,
    log_prefix,
    event_type_logits,
    groundtruth_event_type_ids,
    event_src_logits,
    groundtruth_event_src_ids,
    event_dst_logits,
    groundtruth_event_dst_ids,
    batch_node_mask,
    event_label_logits,
    groundtruth_event_label_word_ids,
    groundtruth_event_label_word_mask,
    groundtruth_event_mask,
    groundtruth_event_src_mask,
    groundtruth_event_dst_mask,
    groundtruth_event_label_mask,
    expected_event_type_f1,
    expected_src_node_f1,
    expected_dst_node_f1,
    expected_label_f1,
):
    supervised_tdgu.calculate_f1s(
        event_type_logits,
        groundtruth_event_type_ids,
        event_src_logits,
        groundtruth_event_src_ids,
        event_dst_logits,
        groundtruth_event_dst_ids,
        batch_node_mask,
        event_label_logits,
        groundtruth_event_label_word_ids,
        groundtruth_event_label_word_mask,
        groundtruth_event_mask,
        groundtruth_event_src_mask,
        groundtruth_event_dst_mask,
        groundtruth_event_label_mask,
        log_prefix,
    )
    assert (
        getattr(supervised_tdgu, log_prefix + "_event_type_f1").compute()
        == expected_event_type_f1
    )
    if groundtruth_event_src_mask.any():
        assert (
            getattr(supervised_tdgu, log_prefix + "_src_node_f1").compute()
            == expected_src_node_f1
        )
    if groundtruth_event_dst_mask.any():
        assert (
            getattr(supervised_tdgu, log_prefix + "_dst_node_f1").compute()
            == expected_dst_node_f1
        )
    if groundtruth_event_label_mask.any():
        assert (
            getattr(supervised_tdgu, log_prefix + "_label_f1").compute()
            == expected_label_f1
        )


@pytest.mark.parametrize(
    "event_type_ids,src_ids,dst_ids,batch_label_word_ids,batch_label_mask,"
    "batched_graph,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[13, 3]]),
            torch.ones(1, 2).bool(),
            Batch(
                batch=torch.tensor([0]),
                x=torch.tensor([[13, 3]]),
                node_label_mask=torch.ones(1, 2).bool(),
                node_last_update=torch.tensor([[1, 2]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, dtype=torch.long),
            ),
            ([""], [[]]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[13, 3]]),
            torch.ones(1, 2).bool(),
            Batch(
                batch=torch.tensor([0]),
                x=torch.tensor([[13, 3]]),
                node_label_mask=torch.ones(1, 2).bool(),
                node_last_update=torch.tensor([[1, 2]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, dtype=torch.long),
            ),
            ([""], [[]]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[7, 3]]),
            torch.ones(1, 2).bool(),
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[15, 3], [13, 3]]),
                node_label_mask=torch.ones(2, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 3]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, dtype=torch.long),
            ),
            (["add , player , chopped , is"], [["add", "player", "chopped", "is"]]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[7, 3]]),
            torch.ones(1, 2).bool(),
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[15, 3], [13, 3]]),
                node_label_mask=torch.ones(2, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [1, 3]]),
                edge_index=torch.tensor([[1], [0]]),
                edge_attr=torch.tensor([[7, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[1, 4]]),
            ),
            (
                ["delete , player , chopped , is"],
                [["delete", "player", "chopped", "is"]],
            ),
        ),
        (
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            torch.tensor([[7, 3], [3, 0]]),
            torch.tensor([[True, True], [True, False]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                x=torch.tensor([[15, 3], [13, 3], [13, 3], [8, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(6, 2).bool(),
                node_last_update=torch.tensor(
                    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 4], [1, 5]]
                ),
                edge_index=torch.tensor([[4], [5]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[1, 6]]),
            ),
            (
                ["add , player , chopped , is", "delete , player , inventory , in"],
                [
                    ["add", "player", "chopped", "is"],
                    ["delete", "player", "inventory", "in"],
                ],
            ),
        ),
        (
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            torch.tensor([[7, 3], [3, 0]]),
            torch.tensor([[True, True], [True, False]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                x=torch.tensor([[15, 3], [13, 3], [13, 3], [8, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(6, 2).bool(),
                node_last_update=torch.tensor(
                    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 4], [1, 5]]
                ),
                edge_index=torch.tensor([[4], [5]]),
                edge_attr=torch.tensor([[16, 10, 3]]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[1, 6]]),
            ),
            (
                [
                    "add , player , chopped , is",
                    "delete , player , inventory , east_of",
                ],
                [
                    ["add", "player", "chopped", "is"],
                    ["delete", "player", "inventory", "east_of"],
                ],
            ),
        ),
        (
            torch.tensor(
                [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([[13, 3], [3, 0]]),
            torch.tensor([[True, True], [True, False]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                x=torch.tensor([[15, 3], [13, 3], [13, 3], [8, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(6, 2).bool(),
                node_last_update=torch.tensor(
                    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 4], [1, 5]]
                ),
                edge_index=torch.tensor([[4], [5]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[1, 6]]),
            ),
            (
                ["", "delete , player , inventory , in"],
                [[], ["delete", "player", "inventory", "in"]],
            ),
        ),
    ],
)
def test_supervised_tdgu_generate_graph_triples(
    supervised_tdgu,
    event_type_ids,
    src_ids,
    dst_ids,
    batch_label_word_ids,
    batch_label_mask,
    batched_graph,
    expected,
):
    assert (
        supervised_tdgu.generate_graph_triples(
            event_type_ids,
            src_ids,
            dst_ids,
            batch_label_word_ids,
            batch_label_mask,
            batched_graph,
        )
        == expected
    )


@pytest.mark.parametrize(
    "event_type_id_seq,src_id_seq,dst_id_seq,label_word_id_seq,label_mask_seq,"
    "batched_graph_seq,expected",
    [
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-add"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([[3]])],
            [torch.tensor([[True]])],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=torch.tensor([[2, 13, 3]]),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0).long(),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([[3]])],
            [torch.tensor([[True]])],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=torch.tensor([[2, 13, 3]]),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0).long(),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
            [torch.tensor([[7, 3]])],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=torch.tensor([[2, 15, 3], [2, 13, 3]]),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0).long(),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            (
                [["add , player , chopped , is"]],
                [["add", "player", "chopped", "is"]],
            ),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
            [torch.tensor([[3]])],
            [torch.tensor([[True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=torch.tensor([[2, 15, 3], [2, 13, 3]]),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([[0, 0], [0, 1]]),
                    edge_index=torch.tensor([[1], [0]]),
                    edge_attr=torch.tensor([[2, 7, 3]]),
                    edge_label_mask=torch.ones(1, 2).bool(),
                    edge_last_update=torch.tensor([[0, 2]]),
                )
            ],
            (
                [["delete , player , chopped , is"]],
                [["delete", "player", "chopped", "is"]],
            ),
        ),
        (
            [
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-delete"], EVENT_TYPE_ID_MAP["edge-add"]]
                ),
            ],
            [torch.tensor([1, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 3])],
            [torch.tensor([[7, 3], [3, 0]]), torch.tensor([[3, 0], [12, 3]])],
            [
                torch.tensor([[True] * 2, [True, False]]),
                torch.tensor([[True, False], [True] * 2]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=torch.tensor(
                        [
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 13, 3],
                            [2, 8, 3],
                            [2, 13, 3],
                            [2, 14, 3],
                        ]
                    ),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 0], [2, 1], [3, 1], [2, 3], [1, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=torch.tensor([[2, 12, 3]]),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[2, 1]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=torch.tensor(
                        [
                            [2, 13, 3],
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 14, 3],
                        ]
                    ),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=torch.tensor([[2, 7, 3]]),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 1]]),
                ),
            ],
            (
                [
                    ["add , player , chopped , is", "delete , player , chopped , is"],
                    [
                        "delete , player , inventory , in",
                        "add , player , inventory , in",
                    ],
                ],
                [
                    [
                        "add",
                        "player",
                        "chopped",
                        "is",
                        "<sep>",
                        "delete",
                        "player",
                        "chopped",
                        "is",
                    ],
                    [
                        "delete",
                        "player",
                        "inventory",
                        "in",
                        "<sep>",
                        "add",
                        "player",
                        "inventory",
                        "in",
                    ],
                ],
            ),
        ),
        (
            [
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["node-delete"]]
                ),
            ],
            [torch.tensor([0, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 0])],
            [torch.tensor([[7, 3], [3, 0]]), torch.tensor([[7, 3], [3, 0]])],
            [
                torch.tensor([[True] * 2, [True, False]]),
                torch.tensor([[True] * 2, [True, False]]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=torch.tensor(
                        [
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 13, 3],
                            [2, 8, 3],
                            [2, 13, 3],
                            [2, 14, 3],
                        ]
                    ),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 4]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=torch.tensor([[2, 12, 3]]),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 5]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=torch.tensor(
                        [
                            [2, 13, 3],
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 15, 3],
                            [2, 13, 3],
                            [2, 14, 3],
                        ]
                    ),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
                    ),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0).long(),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0, 2).long(),
                ),
            ],
            (
                [["add , player , chopped , is"], ["delete , player , inventory , in"]],
                [
                    ["add", "player", "chopped", "is"],
                    ["delete", "player", "inventory", "in"],
                ],
            ),
        ),
    ],
)
def test_supervised_tdgu_generate_batch_graph_triples_seq(
    supervised_tdgu,
    event_type_id_seq,
    src_id_seq,
    dst_id_seq,
    label_word_id_seq,
    label_mask_seq,
    batched_graph_seq,
    expected,
):
    assert (
        supervised_tdgu.generate_batch_graph_triples_seq(
            event_type_id_seq,
            src_id_seq,
            dst_id_seq,
            label_word_id_seq,
            label_mask_seq,
            batched_graph_seq,
        )
        == expected
    )


@pytest.mark.parametrize(
    "groundtruth_cmd_seq,expected",
    [
        (
            (("add , player , kitchen , in",),),
            [["add", "player", "kitchen", "in"]],
        ),
        (
            (("add , player , kitchen , in", "delete , player , kitchen , in"),),
            [
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ]
            ],
        ),
        (
            (
                ("add , player , kitchen , in", "delete , player , kitchen , in"),
                (
                    "add , player , kitchen , in",
                    "add , table , kitchen , in",
                    "delete , player , kitchen , in",
                ),
            ),
            [
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ],
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "add",
                    "table",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ],
            ],
        ),
    ],
)
def test_tdgu_generate_batch_groundtruth_graph_triple_tokens(
    groundtruth_cmd_seq, expected
):
    assert (
        SupervisedTDGU.generate_batch_groundtruth_graph_triple_tokens(
            groundtruth_cmd_seq
        )
        == expected
    )


@pytest.mark.parametrize(
    "ids,batch_cmds,expected",
    [
        (
            (("g0", 0, 1),),
            [
                (("add , player , kitchen , in", "add , table , kitchen , in"),),
                [["add , player , kitchen , in", "add , fire , kitchen , in"]],
                [["add , player , kitchen , in", "delete , player , kitchen , in"]],
            ],
            [
                (
                    "g0|0|1",
                    "add , player , kitchen , in | add , table , kitchen , in",
                    "add , player , kitchen , in | add , fire , kitchen , in",
                    "add , player , kitchen , in | delete , player , kitchen , in",
                )
            ],
        ),
        (
            (("g0", 0, 1), ("g2", 1, 2)),
            [
                (
                    ("add , player , kitchen , in", "add , table , kitchen , in"),
                    ("add , player , livingroom , in", "add , sofa , livingroom , in"),
                ),
                [
                    ["add , player , kitchen , in", "add , fire , kitchen , in"],
                    ["add , player , livingroom , in", "add , TV , livingroom , in"],
                ],
                [
                    ["add , player , kitchen , in", "delete , player , kitchen , in"],
                    ["add , player , garage , in", "add , sofa , livingroom , in"],
                ],
            ],
            [
                (
                    "g0|0|1",
                    "add , player , kitchen , in | add , table , kitchen , in",
                    "add , player , kitchen , in | add , fire , kitchen , in",
                    "add , player , kitchen , in | delete , player , kitchen , in",
                ),
                (
                    "g2|1|2",
                    "add , player , livingroom , in | add , sofa , livingroom , in",
                    "add , player , livingroom , in | add , TV , livingroom , in",
                    "add , player , garage , in | add , sofa , livingroom , in",
                ),
            ],
        ),
    ],
)
def test_tdgu_generate_predict_table_rows(ids, batch_cmds, expected):
    assert SupervisedTDGU.generate_predict_table_rows(ids, *batch_cmds) == expected


@pytest.mark.parametrize(
    "data,expected_node_attrs,expected_edge_attrs",
    [
        (
            Data(
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            {},
            [],
        ),
        (
            Data(
                x=torch.tensor([[13, 3]]),
                node_label_mask=torch.tensor([[True, True]]),
                node_last_update=torch.tensor([[1, 0]]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            {0: {"node_last_update": [1, 0], "label": "player"}},
            [],
        ),
        (
            Data(
                x=torch.tensor([[13, 3, 0], [15, 14, 3]]),
                node_label_mask=torch.tensor([[True, True, False], [True] * 3]),
                node_last_update=torch.tensor([[1, 0], [1, 1]]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            {
                0: {"node_last_update": [1, 0], "label": "player"},
                1: {"node_last_update": [1, 1], "label": "chopped inventory"},
            },
            [],
        ),
        (
            Data(
                x=torch.tensor([[13, 3, 0], [15, 14, 3]]),
                node_label_mask=torch.tensor([[True, True, False], [True] * 3]),
                node_last_update=torch.tensor([[1, 0], [1, 1]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.tensor([[True, True]]),
                edge_last_update=torch.tensor([[1, 2]]),
            ),
            {
                0: {"node_last_update": [1, 0], "label": "player"},
                1: {"node_last_update": [1, 1], "label": "chopped inventory"},
            },
            [
                (0, 1, {"edge_last_update": [1, 2], "label": "in"}),
            ],
        ),
        (
            Data(
                x=torch.tensor([[13, 3, 0], [15, 14, 3], [8, 3, 0]]),
                node_label_mask=torch.tensor(
                    [[True, True, False], [True] * 3, [True, True, False]]
                ),
                node_last_update=torch.tensor([[1, 0], [1, 1], [1, 2]]),
                edge_index=torch.tensor([[0, 0], [1, 2]]),
                edge_attr=torch.tensor([[12, 11, 3], [7, 3, 0]]),
                edge_label_mask=torch.tensor([[True] * 3, [True, True, False]]),
                edge_last_update=torch.tensor([[1, 2], [2, 2]]),
            ),
            {
                0: {"node_last_update": [1, 0], "label": "player"},
                1: {"node_last_update": [1, 1], "label": "chopped inventory"},
                2: {"node_last_update": [1, 2], "label": "peter"},
            },
            [
                (0, 1, {"edge_last_update": [1, 2], "label": "in a"}),
                (0, 2, {"edge_last_update": [2, 2], "label": "is"}),
            ],
        ),
    ],
)
def test_supervised_tdgu_data_to_networkx(
    supervised_tdgu, data, expected_node_attrs, expected_edge_attrs
):
    nx_graph = supervised_tdgu.data_to_networkx(data)
    assert dict(nx_graph.nodes.data()) == expected_node_attrs
    assert list(nx_graph.edges.data()) == expected_edge_attrs
