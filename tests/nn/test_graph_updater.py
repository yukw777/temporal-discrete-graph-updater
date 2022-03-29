import pytest
import torch
import shutil
import torch.nn as nn
import random

from torch_geometric.data.batch import Data, Batch

from tdgu.nn.graph_updater import (
    TemporalDiscreteGraphUpdater,
    UncertaintyWeightedLoss,
)
from tdgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP
from tdgu.data import TWCmdGenGraphEventStepInput
from tdgu.nn.utils import compute_masks_from_event_type_ids

from utils import increasing_mask


@pytest.fixture()
def tdgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return TemporalDiscreteGraphUpdater(
        pretrained_word_embedding_path=f"{tmp_path}/test-fasttext.vec",
        word_vocab_path="tests/data/test_word_vocab.txt",
    )


@pytest.mark.parametrize("label_len", [2, 4])
@pytest.mark.parametrize("batch", [1, 8])
def test_tdgu_embed_label(tdgu, batch, label_len):
    assert tdgu.embed_label(
        torch.randint(len(tdgu.preprocessor.word_vocab), (batch, label_len)),
        torch.randint(2, (batch, label_len)).bool(),
    ).size() == (
        batch,
        tdgu.hparams.hidden_dim,
    )


@pytest.mark.parametrize(
    "label_word_ids,label_mask,expected",
    [
        (torch.tensor([[13, 3]]), torch.ones(1, 2).bool(), ["player"]),
        (
            torch.tensor([[14, 3, 0, 0, 0], [8, 7, 11, 13, 3]]),
            torch.tensor([[True, True, False, False, False], [True] * 5]),
            ["inventory", "peter is a player"],
        ),
    ],
)
def test_tdgu_decode_label(tdgu, label_word_ids, label_mask, expected):
    assert tdgu.decode_label(label_word_ids, label_mask) == expected


@pytest.mark.parametrize("batch,seq_len", [(1, 10), (8, 24)])
def test_tdgu_encode_text(tdgu, batch, seq_len):
    assert tdgu.full_encoder(
        torch.randint(13, (batch, seq_len)), torch.randint(2, (batch, seq_len)).float()
    ).size() == (batch, seq_len, tdgu.hparams.hidden_dim)


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
        torch.rand(batch_size, obs_len, tdgu.hparams.hidden_dim)
        if encoded_textual_input
        else None
    )
    encoded_prev_action = (
        torch.rand(batch_size, prev_action_len, tdgu.hparams.hidden_dim)
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
        torch.randint(len(tdgu.preprocessor.word_vocab), (batch_size, event_label_len)),
        increasing_mask(batch_size, event_label_len).bool(),
        batched_graph,
        obs_mask,
        prev_action_mask,
        torch.randint(10, (batch_size,)),
        obs_word_ids=None
        if encoded_textual_input
        else torch.randint(tdgu.preprocessor.vocab_size, (batch_size, obs_len)),
        prev_action_word_ids=None
        if encoded_textual_input
        else torch.randint(tdgu.preprocessor.vocab_size, (batch_size, prev_action_len)),
        encoded_obs=encoded_obs,
        encoded_prev_action=encoded_prev_action,
        prev_input_event_emb_seq=None
        if prev_input_seq_len == 0
        else torch.rand(
            tdgu.hparams.graph_event_decoder_num_dec_blocks,
            batch_size,
            prev_input_seq_len,
            tdgu.hparams.graph_event_decoder_hidden_dim,
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
        ].size() + (len(tdgu.preprocessor.word_vocab),)
    assert results["updated_prev_input_event_emb_seq"].size() == (
        tdgu.hparams.graph_event_decoder_num_dec_blocks,
        batch_size,
        prev_input_seq_len + 1,
        tdgu.hparams.graph_event_decoder_hidden_dim,
    )
    assert results["updated_prev_input_event_emb_seq_mask"].size() == (
        batch_size,
        prev_input_seq_len + 1,
    )
    if encoded_textual_input:
        assert results["encoded_obs"].equal(encoded_obs)
        assert results["encoded_prev_action"].equal(encoded_prev_action)
    else:
        assert results["encoded_obs"].size() == (
            batch_size,
            obs_len,
            tdgu.hparams.hidden_dim,
        )
        assert results["encoded_prev_action"].size() == (
            batch_size,
            prev_action_len,
            tdgu.hparams.hidden_dim,
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
    assert results["batch_node_mask"].size() == (batch_size, max_sub_graph_num_node)
    assert len(results["self_attn_weights"]) == len(tdgu.decoder.dec_blocks)
    for self_attn_weights in results["self_attn_weights"]:
        assert self_attn_weights.size() == (batch_size, 1, prev_input_seq_len + 1)
    assert len(results["obs_graph_attn_weights"]) == len(tdgu.decoder.dec_blocks)
    for obs_graph_attn_weights in results["obs_graph_attn_weights"]:
        assert obs_graph_attn_weights.size() == (batch_size, 1, obs_len)
    assert len(results["prev_action_graph_attn_weights"]) == len(
        tdgu.decoder.dec_blocks
    )
    for prev_action_graph_attn_weights in results["prev_action_graph_attn_weights"]:
        assert prev_action_graph_attn_weights.size() == (batch_size, 1, prev_action_len)
    assert len(results["graph_obs_attn_weights"]) == len(tdgu.decoder.dec_blocks)
    for graph_obs_attn_weights in results["graph_obs_attn_weights"]:
        assert graph_obs_attn_weights.size() == (batch_size, 1, max_sub_graph_num_node)
    assert len(results["graph_prev_action_attn_weights"]) == len(
        tdgu.decoder.dec_blocks
    )
    for graph_prev_action_attn_weights in results["graph_prev_action_attn_weights"]:
        assert graph_prev_action_attn_weights.size() == (
            batch_size,
            1,
            max_sub_graph_num_node,
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            1,
            1,
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            1,
            1,
            1,
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
            torch.tensor([[[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]]).float(),
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
            torch.tensor([[[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]).float(),
            torch.tensor([[2, 3]]),
            torch.ones(1, 2).bool(),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            1,
            1,
            1,
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
            torch.tensor([[[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]]).float(),
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
                    [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
                    [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
                    [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
                    [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
                    [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]],
                    [[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]],
                    [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]],
                    [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]],
                    [[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
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
            1,
            1,
            1,
        ),
    ],
)
@pytest.mark.parametrize("log_prefix", ["val", "test"])
def test_tdgu_calculate_f1s(
    tdgu,
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
    tdgu.calculate_f1s(
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
        getattr(tdgu, log_prefix + "_event_type_f1").compute() == expected_event_type_f1
    )
    if groundtruth_event_src_mask.any():
        assert (
            getattr(tdgu, log_prefix + "_src_node_f1").compute() == expected_src_node_f1
        )
    if groundtruth_event_dst_mask.any():
        assert (
            getattr(tdgu, log_prefix + "_dst_node_f1").compute() == expected_dst_node_f1
        )
    if groundtruth_event_label_mask.any():
        assert getattr(tdgu, log_prefix + "_label_f1").compute() == expected_label_f1


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
def test_tdgu_generate_graph_triples(
    tdgu,
    event_type_ids,
    src_ids,
    dst_ids,
    batch_label_word_ids,
    batch_label_mask,
    batched_graph,
    expected,
):
    assert (
        tdgu.generate_graph_triples(
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
                    x=torch.tensor([[13, 3]]),
                    node_label_mask=torch.tensor([[True, True]]),
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
                    x=torch.tensor([[13, 3]]),
                    node_label_mask=torch.tensor([[True, True]]),
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
                    x=torch.tensor([[15, 3], [13, 3]]),
                    node_label_mask=torch.ones(2, 2).bool(),
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
                    x=torch.tensor([[15, 3], [13, 3]]),
                    node_label_mask=torch.ones(2, 2).bool(),
                    node_last_update=torch.tensor([[0, 0], [0, 1]]),
                    edge_index=torch.tensor([[1], [0]]),
                    edge_attr=torch.tensor([[7, 3]]),
                    edge_label_mask=torch.tensor([[True, True]]),
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
                        [[15, 3], [13, 3], [13, 3], [8, 3], [13, 3], [14, 3]]
                    ),
                    node_label_mask=torch.ones(6, 2).bool(),
                    node_last_update=torch.tensor(
                        [[1, 0], [2, 1], [3, 1], [2, 3], [1, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=torch.tensor([[12, 3]]),
                    edge_label_mask=torch.ones(1, 2).bool(),
                    edge_last_update=torch.tensor([[2, 1]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=torch.tensor(
                        [[13, 3], [15, 3], [13, 3], [15, 3], [13, 3], [14, 3]]
                    ),
                    node_label_mask=torch.ones(6, 2).bool(),
                    node_last_update=torch.tensor(
                        [[1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=torch.tensor([[7, 3]]),
                    edge_label_mask=torch.ones(1, 2).bool(),
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
                        [[15, 3], [13, 3], [13, 3], [8, 3], [13, 3], [14, 3]]
                    ),
                    node_label_mask=torch.ones(6, 2).bool(),
                    node_last_update=torch.tensor(
                        [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 4]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=torch.tensor([[12, 3]]),
                    edge_label_mask=torch.ones(1, 2).bool(),
                    edge_last_update=torch.tensor([[1, 5]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=torch.tensor(
                        [[13, 3], [15, 3], [13, 3], [15, 3], [13, 3], [14, 3]]
                    ),
                    node_label_mask=torch.ones(6, 2).bool(),
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
def test_tdgu_generate_batch_graph_triples_seq(
    tdgu,
    event_type_id_seq,
    src_id_seq,
    dst_id_seq,
    label_word_id_seq,
    label_mask_seq,
    batched_graph_seq,
    expected,
):
    assert (
        tdgu.generate_batch_graph_triples_seq(
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
        TemporalDiscreteGraphUpdater.generate_batch_groundtruth_graph_triple_tokens(
            groundtruth_cmd_seq
        )
        == expected
    )


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
    tdgu, event_type_ids, src_ids, dst_ids, batch, edge_index, expected
):
    assert tdgu.filter_invalid_events(
        event_type_ids, src_ids, dst_ids, batch, edge_index
    ).equal(expected)


@pytest.mark.parametrize(
    "max_event_decode_len,batch,obs_len,prev_action_len,forward_results,"
    "prev_batched_graph,expected_decoded_list",
    [
        (
            10,
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
                },
            ],
        ),
        (
            2,
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
                },
            ],
        ),
        (
            10,
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
                },
            ],
        ),
        (
            2,
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
                },
            ],
        ),
        (
            10,
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
                },
            ],
        ),
        (
            10,
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
                },
            ],
        ),
        (
            2,
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
                },
            ],
        ),
    ],
)
def test_tdgu_greedy_decode(
    monkeypatch,
    tdgu,
    max_event_decode_len,
    batch,
    obs_len,
    prev_action_len,
    forward_results,
    prev_batched_graph,
    expected_decoded_list,
):
    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            decoded = forward_results[self.num_calls]
            decoded["encoded_obs"] = torch.rand(batch, obs_len, tdgu.hparams.hidden_dim)
            decoded["encoded_prev_action"] = torch.rand(
                batch, prev_action_len, tdgu.hparams.hidden_dim
            )
            self.num_calls += 1
            return decoded

    monkeypatch.setattr(tdgu, "forward", MockForward())
    monkeypatch.setattr(tdgu.hparams, "max_event_decode_len", max_event_decode_len)
    decoded_list = tdgu.greedy_decode(
        TWCmdGenGraphEventStepInput(
            obs_word_ids=torch.randint(tdgu.preprocessor.vocab_size, (batch, obs_len)),
            obs_mask=torch.randint(2, (batch, obs_len)).float(),
            prev_action_word_ids=torch.randint(
                tdgu.preprocessor.vocab_size, (batch, prev_action_len)
            ),
            prev_action_mask=torch.randint(2, (batch, prev_action_len)).float(),
            timestamps=torch.tensor([4.0] * batch),
        ),
        prev_batched_graph,
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
    assert (
        TemporalDiscreteGraphUpdater.generate_predict_table_rows(ids, *batch_cmds)
        == expected
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
    tdgu.embed_label = (
        lambda x, y: x[:, 0].unsqueeze(-1).expand(-1, tdgu.hparams.hidden_dim).float()
    )
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
def test_tdgu_data_to_networkx(tdgu, data, expected_node_attrs, expected_edge_attrs):
    nx_graph = tdgu.data_to_networkx(data)
    assert dict(nx_graph.nodes.data()) == expected_node_attrs
    assert list(nx_graph.edges.data()) == expected_edge_attrs
