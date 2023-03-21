import shutil

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from torch_geometric.nn import TransformerConv
from utils import increasing_mask

from tdgu.constants import EVENT_TYPE_ID_MAP, EVENT_TYPES
from tdgu.data import TWCmdGenGraphEventStepInput
from tdgu.nn.dynamic_gnn import DynamicGNN
from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.nn.text import QANetTextEncoder
from tdgu.nn.utils import compute_masks_from_event_type_ids, masked_softmax


@pytest.fixture()
def tdgu(tmp_path):
    shutil.copy("tests/data/test-fasttext.vec", tmp_path)
    shutil.copy("tests/data/test_word_vocab.txt", tmp_path)
    return TemporalDiscreteGraphUpdater(
        QANetTextEncoder(
            str(tmp_path / "test-fasttext.vec"),
            str(tmp_path / "test_word_vocab.txt"),
            0,
            True,
            1,
            3,
            5,
            8,
            1,
            8,
        ),
        DynamicGNN(TransformerConv, 8, 8, 8, 1, 1),
        8,
        8,
        8,
        8,
        1,
        1,
        8,
        2,
        3,
        0,
        17,
        0.3,
    )


@pytest.mark.parametrize(
    "batch_size,event_type_ids,event_src_ids,event_dst_ids,event_label_len,obs_len,"
    "prev_action_len,batched_graph,groundtruth_event,expected_num_node,"
    "expected_num_edge",
    [
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            3,
            10,
            5,
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
            None,
            0,
            0,
        ),
        (
            4,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                ]
            ),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            3,
            10,
            5,
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
            None,
            0,
            0,
        ),
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            3,
            10,
            5,
            Batch(
                batch=torch.tensor([0, 0, 0]),
                x=torch.randint(6, (3, 2)),
                node_label_mask=torch.randint(2, (3, 2)).bool(),
                node_last_update=torch.randint(10, (3, 2)),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.randint(6, (1, 3)),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.randint(10, (1, 2)),
            ),
            None,
            3,
            1,
        ),
        (
            4,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                ]
            ),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            3,
            10,
            5,
            Batch(
                batch=torch.tensor([0, 0, 1, 2, 2, 3]),
                x=torch.randint(6, (6, 3)),
                node_label_mask=torch.randint(2, (6, 3)).bool(),
                node_last_update=torch.randint(10, (6, 2)),
                edge_index=torch.tensor([[1, 3], [0, 4]]),
                edge_attr=torch.randint(6, (2, 3)),
                edge_label_mask=torch.randint(2, (2, 3)).bool(),
                edge_last_update=torch.randint(10, (2, 2)),
            ),
            None,
            6,
            2,
        ),
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            3,
            10,
            5,
            Batch(
                batch=torch.tensor([0, 0, 0]),
                x=torch.randint(6, (3, 2)),
                node_label_mask=torch.randint(2, (3, 2)).bool(),
                node_last_update=torch.randint(10, (3, 2)),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.randint(6, (1, 3)),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.randint(10, (1, 2)),
            ),
            None,
            4,
            1,
        ),
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            3,
            10,
            5,
            Batch(
                batch=torch.tensor([0, 0, 0]),
                x=torch.randint(6, (3, 2)),
                node_label_mask=torch.randint(2, (3, 2)).bool(),
                node_last_update=torch.randint(10, (3, 2)),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.randint(6, (1, 3)),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.randint(10, (1, 2)),
            ),
            None,
            2,
            1,
        ),
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([2]),
            torch.tensor([6]),
            3,
            10,
            5,
            Batch(
                batch=torch.zeros(9).long(),
                x=torch.randint(6, (9, 3)),
                node_label_mask=torch.randint(2, (9, 3)).bool(),
                node_last_update=torch.randint(10, (9, 2)),
                edge_index=torch.tensor([[2, 1, 8], [0, 3, 6]]),
                edge_attr=torch.randint(6, (3, 2)),
                edge_label_mask=torch.randint(2, (3, 2)).bool(),
                edge_last_update=torch.randint(10, (3, 2)),
            ),
            None,
            9,
            4,
        ),
        (
            1,
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([2]),
            torch.tensor([6]),
            3,
            10,
            5,
            Batch(
                batch=torch.zeros(9).long(),
                x=torch.randint(6, (9, 3)),
                node_label_mask=torch.randint(2, (9, 3)).bool(),
                node_last_update=torch.randint(10, (9, 2)),
                edge_index=torch.tensor([[2, 1, 2, 8], [0, 3, 6, 6]]),
                edge_attr=torch.randint(6, (4, 2)),
                edge_label_mask=torch.randint(2, (4, 2)).bool(),
                edge_last_update=torch.randint(10, (4, 2)),
            ),
            None,
            9,
            3,
        ),
        (
            4,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 1, 0]),
            torch.tensor([0, 0, 0, 0]),
            5,
            12,
            8,
            Batch(
                batch=torch.tensor([1, 1, 1, 2, 2, 3, 3, 3]),
                x=torch.randint(6, (8, 4)),
                node_label_mask=torch.randint(2, (8, 4)).bool(),
                node_last_update=torch.randint(10, (8, 2)),
                edge_index=torch.tensor([[0, 5, 7], [2, 6, 6]]),
                edge_attr=torch.randint(6, (3, 3)),
                edge_label_mask=torch.randint(2, (3, 3)).bool(),
                edge_last_update=torch.randint(10, (3, 2)),
            ),
            None,
            9,
            4,
        ),
        (
            4,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                ]
            ),
            torch.tensor([0, 0, 3, 1]),
            torch.tensor([2, 0, 0, 0]),
            5,
            12,
            8,
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
                x=torch.randint(6, (12, 3)),
                node_label_mask=torch.randint(2, (12, 3)).bool(),
                node_last_update=torch.randint(10, (12, 2)),
                edge_index=torch.tensor([[0, 3, 7, 8], [2, 4, 6, 6]]),
                edge_attr=torch.randint(6, (4, 3)),
                edge_label_mask=torch.randint(2, (4, 3)).bool(),
                edge_last_update=torch.randint(10, (4, 2)),
            ),
            None,
            12,
            4,
        ),
        (
            4,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                ]
            ),
            torch.tensor([0, 0, 3, 1]),
            torch.tensor([2, 0, 0, 0]),
            5,
            12,
            8,
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
                x=torch.randint(6, (12, 3)),
                node_label_mask=torch.randint(2, (12, 3)).bool(),
                node_last_update=torch.randint(10, (12, 2)),
                edge_index=torch.tensor([[0, 3, 7, 8], [2, 4, 6, 6]]),
                edge_attr=torch.randint(6, (4, 3)),
                edge_label_mask=torch.randint(2, (4, 3)).bool(),
                edge_last_update=torch.randint(10, (4, 2)),
            ),
            {
                "groundtruth_event_type_ids": torch.tensor(
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ]
                ),
                "groundtruth_event_mask": torch.tensor([True, True, True, False]),
                "groundtruth_event_src_ids": torch.tensor([0, 2, 0, 1]),
                "groundtruth_event_src_mask": torch.tensor([True, True, False, False]),
                "groundtruth_event_dst_ids": torch.tensor([1, 0, 0, 0]),
                "groundtruth_event_dst_mask": torch.tensor([True, False, False, False]),
                "groundtruth_event_label_tgt_word_ids": torch.randint(6, (4, 3)),
                "groundtruth_event_label_tgt_mask": increasing_mask(4, 3).bool(),
            },
            12,
            4,
        ),
    ],
)
@pytest.mark.parametrize("encoded_textual_input", [True, False])
@pytest.mark.parametrize("prev_input_seq_len", [0, 6, 8])
def test_tdgu_forward(
    tdgu,
    prev_input_seq_len,
    encoded_textual_input,
    batch_size,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_len,
    obs_len,
    prev_action_len,
    batched_graph,
    groundtruth_event,
    expected_num_node,
    expected_num_edge,
):
    encoded_obs = (
        torch.rand(batch_size, obs_len, tdgu.hidden_dim)
        if encoded_textual_input
        else None
    )
    encoded_prev_action = (
        torch.rand(batch_size, prev_action_len, tdgu.hidden_dim)
        if encoded_textual_input
        else None
    )
    # need to have at least one unmasked token, otherwise pack_padded_sequence
    # raises an exception
    obs_mask = torch.cat(
        [
            torch.ones(batch_size, 1).bool(),
            torch.randint(2, (batch_size, obs_len - 1)).bool(),
        ],
        dim=1,
    )
    prev_action_mask = torch.cat(
        [
            torch.ones(batch_size, 1).bool(),
            torch.randint(2, (batch_size, prev_action_len - 1)).bool(),
        ],
        dim=1,
    )
    prev_input_event_emb_seq_mask = (
        None
        if prev_input_seq_len == 0
        else torch.cat(
            [
                torch.ones(batch_size, 1).bool(),
                torch.randint(2, (batch_size, prev_input_seq_len - 1)).bool(),
            ],
            dim=1,
        )
    )
    results = tdgu(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        torch.randint(
            tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
            (batch_size, event_label_len),
        ),
        increasing_mask(batch_size, event_label_len).bool(),
        batched_graph,
        obs_mask,
        prev_action_mask,
        torch.randint(10, (batch_size,)),
        obs_word_ids=None
        if encoded_textual_input
        else torch.randint(
            tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
            (batch_size, obs_len),
        ),
        prev_action_word_ids=None
        if encoded_textual_input
        else torch.randint(
            tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
            (batch_size, prev_action_len),
        ),
        encoded_obs=encoded_obs,
        encoded_prev_action=encoded_prev_action,
        prev_input_event_emb_seq=None
        if prev_input_seq_len == 0
        else torch.rand(
            len(tdgu.graph_event_decoder.dec_blocks),
            batch_size,
            prev_input_seq_len,
            tdgu.graph_event_decoder.hidden_dim,
        ),
        prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
        groundtruth_event=groundtruth_event,
    )
    assert results["event_type_logits"].size() == (batch_size, len(EVENT_TYPES))
    if results["updated_batched_graph"].batch.numel() > 0:
        max_sub_graph_num_node = (
            results["updated_batched_graph"].batch.bincount().max().item()
        )
    else:
        max_sub_graph_num_node = 0
    assert results["event_src_logits"].size() == (batch_size, max_sub_graph_num_node)
    assert results["event_dst_logits"].size() == (batch_size, max_sub_graph_num_node)
    if groundtruth_event is None:
        assert results["decoded_event_label_word_ids"].dim() == 2
        assert results["decoded_event_label_word_ids"].size(0) == batch_size
        assert results["decoded_event_label_mask"].dim() == 2
        assert results["decoded_event_label_mask"].size(0) == batch_size
    else:
        assert results["event_label_logits"].size() == groundtruth_event[
            "groundtruth_event_label_tgt_word_ids"
        ].size() + (tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,)
    assert results["updated_prev_input_event_emb_seq"].size() == (
        len(tdgu.graph_event_decoder.dec_blocks),
        batch_size,
        prev_input_seq_len + 1,
        tdgu.graph_event_decoder.hidden_dim,
    )
    assert results["updated_prev_input_event_emb_seq_mask"].size() == (
        batch_size,
        prev_input_seq_len + 1,
    )
    if encoded_textual_input:
        assert results["encoded_obs"].equal(encoded_obs)
        assert results["encoded_prev_action"].equal(encoded_prev_action)
    else:
        assert results["encoded_obs"].size() == (batch_size, obs_len, tdgu.hidden_dim)
        assert results["encoded_prev_action"].size() == (
            batch_size,
            prev_action_len,
            tdgu.hidden_dim,
        )
    assert results["updated_batched_graph"].batch.size() == (expected_num_node,)
    assert results["updated_batched_graph"].x.dim() == 2
    assert results["updated_batched_graph"].x.size(0) == expected_num_node
    assert results["updated_batched_graph"].node_label_mask.dim() == 2
    assert results["updated_batched_graph"].node_label_mask.size(0) == expected_num_node
    assert results["updated_batched_graph"].node_last_update.size() == (
        expected_num_node,
        2,
    )
    assert results["updated_batched_graph"].edge_index.size() == (2, expected_num_edge)
    assert results["updated_batched_graph"].edge_attr.dim() == 2
    assert results["updated_batched_graph"].edge_attr.size(0) == expected_num_edge
    assert results["updated_batched_graph"].edge_last_update.size() == (
        expected_num_edge,
        2,
    )
    assert results["batch_node_embeddings"].size() == (
        batch_size,
        max_sub_graph_num_node,
        tdgu.hidden_dim,
    )
    assert results["batch_node_mask"].size() == (batch_size, max_sub_graph_num_node)
    assert len(results["self_attn_weights"]) == len(tdgu.graph_event_decoder.dec_blocks)
    for self_attn_weights in results["self_attn_weights"]:
        assert self_attn_weights.size() == (batch_size, 1, prev_input_seq_len + 1)
    assert len(results["obs_graph_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for obs_graph_attn_weights in results["obs_graph_attn_weights"]:
        assert obs_graph_attn_weights.size() == (batch_size, 1, obs_len)
    assert len(results["prev_action_graph_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for prev_action_graph_attn_weights in results["prev_action_graph_attn_weights"]:
        assert prev_action_graph_attn_weights.size() == (batch_size, 1, prev_action_len)
    assert len(results["graph_obs_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for graph_obs_attn_weights in results["graph_obs_attn_weights"]:
        assert graph_obs_attn_weights.size() == (batch_size, 1, max_sub_graph_num_node)
    assert len(results["graph_prev_action_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for graph_prev_action_attn_weights in results["graph_prev_action_attn_weights"]:
        assert graph_prev_action_attn_weights.size() == (
            batch_size,
            1,
            max_sub_graph_num_node,
        )


@pytest.mark.parametrize("tau", [0.1, 0.5, 1])
@pytest.mark.parametrize(
    "batch_size,event_type_ids,event_src_ids,event_dst_ids,event_label_len,obs_len,"
    "prev_action_len,batched_graph,expected_num_node,"
    "expected_num_edge",
    [
        (
            4,
            F.one_hot(
                torch.tensor(
                    [
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ]
                ),
                num_classes=len(EVENT_TYPE_ID_MAP),
            ).float(),
            torch.cat(
                [
                    F.one_hot(torch.tensor([0]), num_classes=4).float(),
                    torch.zeros(1, 4),
                    F.one_hot(torch.tensor([3, 1]), num_classes=4).float(),
                ]
            ),
            torch.cat(
                [
                    F.one_hot(torch.tensor([2]), num_classes=4).float(),
                    torch.zeros(1, 4),
                    F.one_hot(torch.tensor([0]), num_classes=4).float(),
                    torch.zeros(1, 4),
                ]
            ),
            5,
            12,
            8,
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
                x=F.one_hot(torch.randint(6, (12, 3)), num_classes=17).float(),
                node_label_mask=torch.randint(2, (12, 3)).bool(),
                node_last_update=torch.randint(10, (12, 2)),
                edge_index=torch.tensor([[0, 3, 7, 8], [2, 4, 6, 6]]),
                edge_attr=F.one_hot(torch.randint(6, (4, 3)), num_classes=17).float(),
                edge_label_mask=torch.randint(2, (4, 3)).bool(),
                edge_last_update=torch.randint(10, (4, 2)),
            ),
            12,
            4,
        ),
    ],
)
def test_tdgu_gumbel_forward(
    tdgu,
    batch_size,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_len,
    obs_len,
    prev_action_len,
    batched_graph,
    expected_num_node,
    expected_num_edge,
    tau,
):
    prev_input_seq_len = 8

    # need to have at least one unmasked token, otherwise pack_padded_sequence
    # raises an exception
    obs_mask = torch.cat(
        [
            torch.ones(batch_size, 1).bool(),
            torch.randint(2, (batch_size, obs_len - 1)).bool(),
        ],
        dim=1,
    )
    prev_action_mask = torch.cat(
        [
            torch.ones(batch_size, 1).bool(),
            torch.randint(2, (batch_size, prev_action_len - 1)).bool(),
        ],
        dim=1,
    )
    prev_input_event_emb_seq_mask = torch.cat(
        [
            torch.ones(batch_size, 1).bool(),
            torch.randint(2, (batch_size, prev_input_seq_len - 1)).bool(),
        ],
        dim=1,
    )
    results = tdgu(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        F.one_hot(
            torch.randint(
                tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
                (batch_size, event_label_len),
            ),
            num_classes=tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
        ).float(),
        increasing_mask(batch_size, event_label_len).bool(),
        batched_graph,
        obs_mask,
        prev_action_mask,
        torch.randint(10, (batch_size,)),
        obs_word_ids=torch.randint(
            tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
            (batch_size, obs_len),
        ),
        prev_action_word_ids=torch.randint(
            tdgu.event_label_head.pretrained_word_embeddings.num_embeddings,
            (batch_size, prev_action_len),
        ),
        prev_input_event_emb_seq=torch.rand(
            len(tdgu.graph_event_decoder.dec_blocks),
            batch_size,
            prev_input_seq_len,
            tdgu.graph_event_decoder.hidden_dim,
        ),
        prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
        gumbel_greedy_decode=True,
        gumbel_tau=tau,
    )
    assert results["event_type_logits"].size() == (batch_size, len(EVENT_TYPES))
    if results["updated_batched_graph"].batch.numel() > 0:
        max_sub_graph_num_node = (
            results["updated_batched_graph"].batch.bincount().max().item()
        )
    else:
        max_sub_graph_num_node = 0
    assert results["event_src_logits"].size() == (batch_size, max_sub_graph_num_node)
    assert results["event_dst_logits"].size() == (batch_size, max_sub_graph_num_node)
    assert results["decoded_event_label_word_ids"].dim() == 3
    assert results["decoded_event_label_word_ids"].size(0) == batch_size
    assert (
        results["decoded_event_label_word_ids"].size(2)
        == tdgu.event_label_head.pretrained_word_embeddings.num_embeddings
    )
    assert results["decoded_event_label_mask"].dim() == 2
    assert results["decoded_event_label_mask"].size(0) == batch_size
    assert results["updated_prev_input_event_emb_seq"].size() == (
        len(tdgu.graph_event_decoder.dec_blocks),
        batch_size,
        prev_input_seq_len + 1,
        tdgu.graph_event_decoder.hidden_dim,
    )
    assert results["updated_prev_input_event_emb_seq_mask"].size() == (
        batch_size,
        prev_input_seq_len + 1,
    )
    assert results["encoded_obs"].size() == (batch_size, obs_len, tdgu.hidden_dim)
    assert results["encoded_prev_action"].size() == (
        batch_size,
        prev_action_len,
        tdgu.hidden_dim,
    )
    assert results["updated_batched_graph"].batch.size() == (expected_num_node,)
    assert results["updated_batched_graph"].x.dim() == 3
    assert results["updated_batched_graph"].x.size(0) == expected_num_node
    assert (
        results["updated_batched_graph"].x.size(2)
        == tdgu.event_label_head.pretrained_word_embeddings.num_embeddings
    )
    assert results["updated_batched_graph"].node_label_mask.dim() == 2
    assert results["updated_batched_graph"].node_label_mask.size(0) == expected_num_node
    assert results["updated_batched_graph"].node_last_update.size() == (
        expected_num_node,
        2,
    )
    assert results["updated_batched_graph"].edge_index.size() == (2, expected_num_edge)
    assert results["updated_batched_graph"].edge_attr.dim() == 3
    assert results["updated_batched_graph"].edge_attr.size(0) == expected_num_edge
    assert (
        results["updated_batched_graph"].edge_attr.size(2)
        == tdgu.event_label_head.pretrained_word_embeddings.num_embeddings
    )
    assert results["updated_batched_graph"].edge_last_update.size() == (
        expected_num_edge,
        2,
    )
    assert results["batch_node_embeddings"].size() == (
        batch_size,
        max_sub_graph_num_node,
        tdgu.hidden_dim,
    )
    assert results["batch_node_mask"].size() == (batch_size, max_sub_graph_num_node)
    assert len(results["self_attn_weights"]) == len(tdgu.graph_event_decoder.dec_blocks)
    for self_attn_weights in results["self_attn_weights"]:
        assert self_attn_weights.size() == (batch_size, 1, prev_input_seq_len + 1)
    assert len(results["obs_graph_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for obs_graph_attn_weights in results["obs_graph_attn_weights"]:
        assert obs_graph_attn_weights.size() == (batch_size, 1, obs_len)
    assert len(results["prev_action_graph_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for prev_action_graph_attn_weights in results["prev_action_graph_attn_weights"]:
        assert prev_action_graph_attn_weights.size() == (batch_size, 1, prev_action_len)
    assert len(results["graph_obs_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for graph_obs_attn_weights in results["graph_obs_attn_weights"]:
        assert graph_obs_attn_weights.size() == (batch_size, 1, max_sub_graph_num_node)
    assert len(results["graph_prev_action_attn_weights"]) == len(
        tdgu.graph_event_decoder.dec_blocks
    )
    for graph_prev_action_attn_weights in results["graph_prev_action_attn_weights"]:
        assert graph_prev_action_attn_weights.size() == (
            batch_size,
            1,
            max_sub_graph_num_node,
        )


@pytest.mark.parametrize(
    "event_type_ids,event_src_ids,event_dst_ids,event_label_word_ids,event_label_mask,"
    "batch,node_label_word_ids,node_label_mask,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0).long(),
            torch.empty(0, 0).bool(),
            torch.tensor([[0.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[0.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0).long(),
            torch.empty(0, 0).bool(),
            torch.tensor([[3.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[3.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0).long(),
            torch.empty(0, 0).bool(),
            torch.tensor([[4.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[4.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
            torch.empty(0).long(),
            torch.empty(0, 0).long(),
            torch.empty(0, 0).bool(),
            torch.tensor([[5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [6.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [6.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[6.0] * 4 + [5.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            torch.tensor([[4, 3], [5, 3]]),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[7.0] * 4 + [5.0] * 8 + [4.0] * 8 + [6.0] * 8]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[4, 3], [5, 3], [8, 3]]),
            torch.tensor([[True, True], [True, True], [True, True]]),
            torch.tensor([[8.0] * 4 + [8.0] * 8 + [5.0] * 8 + [0.0] * 8]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 0, 2, 1, 1, 0]),
            torch.tensor([1, 0, 0, 0, 2, 0]),
            torch.tensor([[2, 3], [3, 3], [4, 3], [1, 3], [2, 3], [3, 3]]),
            torch.ones(6, 2).bool(),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
            torch.tensor(
                [
                    [1, 3],
                    [2, 3],
                    [1, 3],
                    [3, 3],
                    [2, 3],
                    [4, 3],
                    [2, 3],
                    [1, 3],
                    [3, 3],
                    [4, 3],
                    [3, 3],
                    [4, 3],
                    [2, 3],
                    [3, 3],
                    [4, 3],
                    [2, 3],
                    [1, 3],
                    [4, 3],
                ]
            ),
            torch.ones(18, 2).bool(),
            torch.tensor(
                [
                    [7.0] * 4 + [1.0] * 8 + [2.0] * 8 + [2.0] * 8,
                    [5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [3.0] * 8,
                    [8.0] * 4 + [3.0] * 8 + [2.0] * 8 + [0.0] * 8,
                    [6.0] * 4 + [3.0] * 8 + [0.0] * 8 + [0.0] * 8,
                    [7.0] * 4 + [3.0] * 8 + [4.0] * 8 + [2.0] * 8,
                    [5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [3.0] * 8,
                ]
            ),
        ),
    ],
)
def test_tdgu_get_decoder_input(
    tdgu,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_word_ids,
    event_label_mask,
    batch,
    node_label_word_ids,
    node_label_mask,
    expected,
):
    tdgu.event_type_embeddings = nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0] * 4,
                [3.0] * 4,
                [4.0] * 4,
                [5.0] * 4,
                [6.0] * 4,
                [7.0] * 4,
                [8.0] * 4,
            ]
        )
    )

    class MockTextEncoder(nn.Module):
        def forward(self, word_ids, mask, return_pooled_output):
            return {
                "pooled_output": word_ids[:, 0]
                .unsqueeze(-1)
                .expand(-1, tdgu.hidden_dim)
                .float()
            }

    tdgu.text_encoder = MockTextEncoder()

    assert tdgu.get_decoder_input(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_label_word_ids,
        event_label_mask,
        batch,
        node_label_word_ids,
        node_label_mask,
        compute_masks_from_event_type_ids(event_type_ids),
    ).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_src_ids,event_dst_ids,event_label_word_ids,event_label_mask,"
    "batch,node_label_word_ids,node_label_mask,expected",
    [
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["pad"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 0),
            torch.zeros(1, 0),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0, 12),
            torch.empty(0, 0).bool(),
            torch.tensor([[0.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["pad"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[0.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["start"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 0),
            torch.zeros(1, 0),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0, 12),
            torch.empty(0, 0).bool(),
            torch.tensor([[3.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["start"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[3.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["end"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 0),
            torch.zeros(1, 0),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.empty(0).long(),
            torch.empty(0, 0, 12),
            torch.empty(0, 0).bool(),
            torch.tensor([[4.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["end"]]), num_classes=len(EVENT_TYPES)
            ).float(),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            F.one_hot(torch.tensor([[3]]), num_classes=12).float(),
            torch.tensor([[True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[4.0] * 4 + [0.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                num_classes=len(EVENT_TYPES),
            ).float(),
            torch.zeros(1, 0),
            torch.zeros(1, 0),
            F.one_hot(torch.tensor([[6, 3]]), num_classes=12).float(),
            torch.tensor([[True, True]]),
            torch.empty(0).long(),
            torch.empty(0, 0, 12),
            torch.empty(0, 0).bool(),
            torch.tensor([[5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [6.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                num_classes=len(EVENT_TYPES),
            ).float(),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            F.one_hot(torch.tensor([[6, 3]]), num_classes=12).float(),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [6.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
                num_classes=len(EVENT_TYPES),
            ).float(),
            F.one_hot(torch.tensor([1]), num_classes=2).float(),
            torch.zeros(1, 2),
            F.one_hot(torch.tensor([[6, 3]]), num_classes=12).float(),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[6.0] * 4 + [5.0] * 8 + [0.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                num_classes=len(EVENT_TYPES),
            ).float(),
            F.one_hot(torch.tensor([1]), num_classes=2).float(),
            F.one_hot(torch.tensor([0]), num_classes=2).float(),
            F.one_hot(torch.tensor([[6, 3]]), num_classes=12).float(),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True]]),
            torch.tensor([[7.0] * 4 + [5.0] * 8 + [4.0] * 8 + [6.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
                num_classes=len(EVENT_TYPES),
            ).float(),
            F.one_hot(torch.tensor([2]), num_classes=3).float(),
            F.one_hot(torch.tensor([1]), num_classes=3).float(),
            F.one_hot(torch.tensor([[6, 3]]), num_classes=12).float(),
            torch.tensor([[True, True]]),
            torch.tensor([0, 0, 0]),
            F.one_hot(torch.tensor([[4, 3], [5, 3], [8, 3]]), num_classes=12).float(),
            torch.tensor([[True, True], [True, True], [True, True]]),
            torch.tensor([[8.0] * 4 + [8.0] * 8 + [5.0] * 8 + [0.0] * 8]),
        ),
        (
            F.one_hot(
                torch.tensor(
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["node-add"],
                    ]
                ),
                num_classes=len(EVENT_TYPES),
            ).float(),
            torch.cat(
                [
                    F.one_hot(torch.tensor([0]), num_classes=3).float(),
                    torch.zeros(1, 3),
                    F.one_hot(torch.tensor([2, 1, 1]), num_classes=3).float(),
                    torch.zeros(1, 3),
                ]
            ),
            torch.cat(
                [
                    F.one_hot(torch.tensor([1]), num_classes=3).float(),
                    torch.zeros(1, 3),
                    F.one_hot(torch.tensor([0]), num_classes=3).float(),
                    torch.zeros(1, 3),
                    F.one_hot(torch.tensor([2]), num_classes=3).float(),
                    torch.zeros(1, 3),
                ]
            ),
            F.one_hot(
                torch.tensor([[2, 3], [3, 3], [4, 3], [1, 3], [2, 3], [3, 3]]),
                num_classes=12,
            ).float(),
            torch.ones(6, 2).bool(),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
            F.one_hot(
                torch.tensor(
                    [
                        [1, 3],
                        [2, 3],
                        [1, 3],
                        [3, 3],
                        [2, 3],
                        [4, 3],
                        [2, 3],
                        [1, 3],
                        [3, 3],
                        [4, 3],
                        [3, 3],
                        [4, 3],
                        [2, 3],
                        [3, 3],
                        [4, 3],
                        [2, 3],
                        [1, 3],
                        [4, 3],
                    ]
                ),
                num_classes=12,
            ).float(),
            torch.ones(18, 2).bool(),
            torch.tensor(
                [
                    [7.0] * 4 + [1.0] * 8 + [2.0] * 8 + [2.0] * 8,
                    [5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [3.0] * 8,
                    [8.0] * 4 + [3.0] * 8 + [2.0] * 8 + [0.0] * 8,
                    [6.0] * 4 + [3.0] * 8 + [0.0] * 8 + [0.0] * 8,
                    [7.0] * 4 + [3.0] * 8 + [4.0] * 8 + [2.0] * 8,
                    [5.0] * 4 + [0.0] * 8 + [0.0] * 8 + [3.0] * 8,
                ]
            ),
        ),
    ],
)
def test_tdgu_get_decoder_input_one_hot(
    tdgu,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_word_ids,
    event_label_mask,
    batch,
    node_label_word_ids,
    node_label_mask,
    expected,
):
    tdgu.event_type_embeddings = nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0] * 4,
                [3.0] * 4,
                [4.0] * 4,
                [5.0] * 4,
                [6.0] * 4,
                [7.0] * 4,
                [8.0] * 4,
            ]
        )
    )

    class MockTextEncoder(nn.Module):
        def forward(self, word_ids, mask, return_pooled_output):
            return {
                "pooled_output": word_ids.argmax(-1)[:, 0]
                .unsqueeze(-1)
                .expand(-1, tdgu.hidden_dim)
                .float()
            }

    tdgu.text_encoder = MockTextEncoder()
    assert tdgu.get_decoder_input(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_label_word_ids,
        event_label_mask,
        batch,
        node_label_word_ids,
        node_label_mask,
        compute_masks_from_event_type_ids(event_type_ids.argmax(dim=-1)),
    ).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,src_ids,dst_ids,batch,edge_index,expected",
    [
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.empty(0).long(),
            torch.empty(2, 0).long(),
            torch.tensor([False, False, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0, 1, 1]),
            torch.empty(2, 0).long(),
            torch.tensor([False, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 0, 1, 1]),
            torch.empty(2, 0).long(),
            torch.tensor([False, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor([[1], [2]]),
            torch.tensor([True, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor([[2], [1]]),
            torch.tensor([True, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([3, 5, 2, 0]),
            torch.tensor([0, 0, 4, 1]),
            torch.empty(0).long(),
            torch.empty(2, 0).long(),
            torch.tensor([True, True, True, True]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([3, 5, 2, 0]),
            torch.tensor([0, 0, 4, 1]),
            torch.tensor([3, 3]),
            torch.empty(2, 0).long(),
            torch.tensor([True, True, True, False]),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 1, 2, 3, 3, 3]),
            torch.tensor([[1], [2]]),
            torch.tensor([False, True, False, False]),
        ),
    ],
)
def test_tdgu_filter_invalid_events(
    event_type_ids, src_ids, dst_ids, batch, edge_index, expected
):
    assert TemporalDiscreteGraphUpdater.filter_invalid_events(
        event_type_ids, src_ids, dst_ids, batch, edge_index
    ).equal(expected)


@pytest.mark.parametrize("gumbel_tau", [0.3, 0.5])
@pytest.mark.parametrize(
    "max_event_decode_len,max_label_decode_len,gumbel_greedy_decode,batch,obs_len,"
    "prev_action_len,forward_results,prev_batched_graph,expected_decoded_list",
    [
        (
            10,
            2,
            False,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor([[True]]),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                    "self_attn_weights": [torch.rand(1, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                    "self_attn_weights": [torch.rand(1, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 1)],
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
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2]),  # [end]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                },
            ],
        ),
        (
            2,
            3,
            False,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor([[True]]),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                    "self_attn_weights": [torch.rand(1, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "decoded_event_label_word_ids": torch.tensor([[14, 3]]),
                    "decoded_event_label_mask": torch.tensor([[True, True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                    "self_attn_weights": [torch.rand(1, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add]
                    "event_src_logits": torch.tensor([[0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor([[12, 3]]),
                    "decoded_event_label_mask": torch.tensor([[True, True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[13, 3], [14, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [1, 3]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(1, 2, 8),
                    "batch_node_mask": torch.tensor([[True, True]]),
                    "self_attn_weights": [torch.rand(1, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 2),
                    "event_dst_logits": torch.rand(1, 2),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[13, 3], [14, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [1, 3]]),
                        edge_index=torch.tensor([[1], [0]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[1, 4]]),
                    ),
                    "batch_node_embeddings": torch.zeros(1, 2, 8),
                    "batch_node_mask": torch.tensor([[True, True]]),
                    "self_attn_weights": [torch.rand(1, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[14, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                },
            ],
        ),
        (
            10,
            3,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
            ],
        ),
        (
            2,
            3,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, False]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
            ],
        ),
        (
            10,
            5,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-delete, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
                    ).float(),  # [edge-delete, edge-delete]
                    "event_src_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 0], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0]]
                    ).float(),  # [edge-delete, node-delete]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 5, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [
                            [True, True, True, False, False],
                            [True, True, True, True, True],
                        ]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 5)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 6, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [
                            [True, True, True, False, False, False],
                            [True, True, True, True, True, True],
                        ]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 6)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
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
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor(
                        [0, 6]
                    ),  # [pad, edge-delete]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor(
                        [0, 4]
                    ),  # [pad, node-delete]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
            ],
        ),
        (
            10,
            4,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 3]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 3]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
            ],
        ),
        (
            2,
            5,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
            ],
        ),
        (
            10,
            4,
            True,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [13, 3]]), num_classes=20
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                        ).float(),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[3, 0], [14, 3]]), num_classes=20
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[12, 3], [7, 3]]), num_classes=20
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [14, 3]]), num_classes=20
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3], [12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=F.one_hot(
                    torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                ).float(),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=F.one_hot(torch.tensor([[12, 3]]), num_classes=20).float(),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": F.one_hot(
                        torch.tensor([3, 3]), num_classes=len(EVENT_TYPES)
                    ).float(),  # [node-add, node-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([0, 0]), num_classes=2
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([0, 1]), num_classes=2
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [13, 3]]), num_classes=20
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                        ).float(),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": F.one_hot(
                        torch.tensor([2, 3]), num_classes=len(EVENT_TYPES)
                    ).float(),  # [end, node-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=3
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=3
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[3, 0], [14, 3]]), num_classes=20
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.cat(
                        [
                            torch.zeros(1, len(EVENT_TYPES)),
                            F.one_hot(
                                torch.tensor([5]), num_classes=len(EVENT_TYPES)
                            ).float(),
                        ]
                    ),  # [pad, edge-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=4
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 3]), num_classes=4
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[12, 3], [7, 3]]), num_classes=20
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.cat(
                        [
                            torch.zeros(1, len(EVENT_TYPES)),
                            F.one_hot(
                                torch.tensor([2]), num_classes=len(EVENT_TYPES)
                            ).float(),
                        ]
                    ),  # [pad, end]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=4
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 3]), num_classes=4
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [14, 3]]), num_classes=20
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3], [12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
            ],
        ),
    ],
)
def test_tdgu_greedy_decode(
    tdgu,
    monkeypatch,
    max_event_decode_len,
    max_label_decode_len,
    gumbel_greedy_decode,
    batch,
    obs_len,
    prev_action_len,
    forward_results,
    prev_batched_graph,
    expected_decoded_list,
    gumbel_tau,
):
    # monkeypatch gumbel_softmax to argmax() + F.one_hot()
    # to remove randomness for tests
    def mock_gumbel_softmax(logits, **kwargs):
        return F.one_hot(logits.argmax(-1), num_classes=logits.size(-1)).float()

    monkeypatch.setattr("tdgu.nn.graph_updater.F.gumbel_softmax", mock_gumbel_softmax)

    # monkeypatch masked_gumbel_softmax to masked_softmax() + argmax() + F.one_hot()
    # to remove randomness for tests
    def mock_masked_gumbel_softmax(logits, mask, **kwargs):
        return F.one_hot(
            masked_softmax(logits, mask, dim=-1).argmax(-1), num_classes=logits.size(-1)
        ).float()

    monkeypatch.setattr(
        "tdgu.nn.graph_updater.masked_gumbel_softmax", mock_masked_gumbel_softmax
    )

    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            decoded = forward_results[self.num_calls]
            decoded["encoded_obs"] = torch.rand(batch, obs_len, tdgu.hidden_dim)
            decoded["encoded_prev_action"] = torch.rand(
                batch, prev_action_len, tdgu.hidden_dim
            )
            self.num_calls += 1
            return decoded

    monkeypatch.setattr(tdgu, "forward", MockForward())
    decoded_list = tdgu.greedy_decode(
        TWCmdGenGraphEventStepInput(
            obs_word_ids=torch.randint(tdgu.vocab_size, (batch, obs_len)),
            obs_mask=torch.randint(2, (batch, obs_len)).float(),
            prev_action_word_ids=torch.randint(
                tdgu.vocab_size, (batch, prev_action_len)
            ),
            prev_action_mask=torch.randint(2, (batch, prev_action_len)).float(),
            timestamps=torch.tensor([4.0] * batch),
        ),
        prev_batched_graph,
        max_event_decode_len=max_event_decode_len,
        max_label_decode_len=max_label_decode_len,
        gumbel_greedy_decode=gumbel_greedy_decode,
        gumbel_tau=gumbel_tau,
    )

    assert len(decoded_list) == len(expected_decoded_list)
    for decoded, expected in zip(decoded_list, expected_decoded_list):
        assert decoded["decoded_event_type_ids"].equal(
            expected["decoded_event_type_ids"]
        )
        assert decoded["decoded_event_src_ids"].equal(expected["decoded_event_src_ids"])
        assert decoded["decoded_event_dst_ids"].equal(expected["decoded_event_dst_ids"])
        assert decoded["decoded_event_label_word_ids"].equal(
            expected["decoded_event_label_word_ids"]
        )
        assert decoded["decoded_event_label_mask"].equal(
            expected["decoded_event_label_mask"]
        )
        assert decoded["updated_batched_graph"].batch.equal(
            expected["updated_batched_graph"].batch
        )
        assert decoded["updated_batched_graph"].x.equal(
            expected["updated_batched_graph"].x
        )
        assert decoded["updated_batched_graph"].node_label_mask.equal(
            expected["updated_batched_graph"].node_label_mask
        )
        assert decoded["updated_batched_graph"].node_last_update.equal(
            expected["updated_batched_graph"].node_last_update
        )
        assert decoded["updated_batched_graph"].edge_index.equal(
            expected["updated_batched_graph"].edge_index
        )
        assert decoded["updated_batched_graph"].edge_attr.equal(
            expected["updated_batched_graph"].edge_attr
        )
        assert decoded["updated_batched_graph"].edge_label_mask.equal(
            expected["updated_batched_graph"].edge_label_mask
        )
        assert decoded["updated_batched_graph"].edge_last_update.equal(
            expected["updated_batched_graph"].edge_last_update
        )
        assert decoded["batch_node_embeddings"].equal(expected["batch_node_embeddings"])
        assert decoded["batch_node_mask"].equal(expected["batch_node_mask"])
