import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data.batch import Batch
from torch_geometric.nn import TransformerConv

from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP
from tdgu.nn.utils import compute_masks_from_event_type_ids
from tdgu.nn.text import QANetTextEncoder
from tdgu.preprocessor import SpacyPreprocessor
from tdgu.nn.graph_event_decoder import TransformerGraphEventDecoder

from utils import increasing_mask


@pytest.fixture()
def tdgu():
    preprocessor = SpacyPreprocessor.load_from_file("tests/data/test_word_vocab.txt")
    text_encoder = QANetTextEncoder(
        nn.Embedding(preprocessor.vocab_size, 300), 1, 3, 5, 8, 1
    )
    return TemporalDiscreteGraphUpdater(
        text_encoder,
        preprocessor,
        TransformerConv,
        8,
        8,
        1,
        1,
        False,
        TransformerGraphEventDecoder(8 + 3 * 8, 8, 1, 1, 8),
        8,
        8,
        8,
        0.3,
    )


@pytest.mark.parametrize("label_len", [2, 4])
@pytest.mark.parametrize("batch", [1, 8])
def test_tdgu_embed_label(tdgu, batch, label_len):
    assert tdgu.embed_label(
        torch.randint(tdgu.preprocessor.vocab_size, (batch, label_len)),
        torch.randint(2, (batch, label_len)).bool(),
    ).size() == (batch, tdgu.hidden_dim)


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
        torch.randint(tdgu.preprocessor.vocab_size, (batch_size, event_label_len)),
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
        ].size() + (len(tdgu.preprocessor.word_vocab),)
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
            torch.randint(tdgu.preprocessor.vocab_size, (batch_size, event_label_len)),
            num_classes=tdgu.preprocessor.vocab_size,
        ).float(),
        increasing_mask(batch_size, event_label_len).bool(),
        batched_graph,
        obs_mask,
        prev_action_mask,
        torch.randint(10, (batch_size,)),
        obs_word_ids=torch.randint(tdgu.preprocessor.vocab_size, (batch_size, obs_len)),
        prev_action_word_ids=torch.randint(
            tdgu.preprocessor.vocab_size, (batch_size, prev_action_len)
        ),
        prev_input_event_emb_seq=torch.rand(
            len(tdgu.graph_event_decoder.dec_blocks),
            batch_size,
            prev_input_seq_len,
            tdgu.graph_event_decoder.hidden_dim,
        ),
        prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
        gumbel_greedy_decode_labels=True,
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
        results["decoded_event_label_word_ids"].size(2) == tdgu.preprocessor.vocab_size
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
    assert results["updated_batched_graph"].x.size(2) == tdgu.preprocessor.vocab_size
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
        == tdgu.preprocessor.vocab_size
    )
    assert results["updated_batched_graph"].edge_last_update.size() == (
        expected_num_edge,
        2,
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


@pytest.mark.parametrize("one_hot", [True, False])
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
    one_hot,
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
    if one_hot:
        tdgu.embed_label = (
            lambda x, y: x.argmax(-1)[:, 0]
            .unsqueeze(-1)
            .expand(-1, tdgu.hidden_dim)
            .float()
        )
        event_label_word_ids = F.one_hot(event_label_word_ids)
        if node_label_word_ids.size(0) == 0:
            node_label_word_ids = torch.empty(
                0, node_label_word_ids.size(1), event_label_word_ids.size(2)
            )
        else:
            node_label_word_ids = F.one_hot(node_label_word_ids)
    else:
        tdgu.embed_label = (
            lambda x, y: x[:, 0].unsqueeze(-1).expand(-1, tdgu.hidden_dim).float()
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
