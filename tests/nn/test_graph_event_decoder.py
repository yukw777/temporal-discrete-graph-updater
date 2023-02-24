import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tdgu.constants import EVENT_TYPES
from tdgu.nn.graph_event_decoder import (
    EventNodeHead,
    EventSequentialLabelHead,
    EventStaticLabelHead,
    EventTypeHead,
    RNNGraphEventDecoder,
    TransformerGraphEventDecoder,
    TransformerGraphEventDecoderBlock,
)


@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
@pytest.mark.parametrize(
    "graph_event_embedding_dim,hidden_dim,autoregressive_embedding_dim,batch",
    [
        (24, 12, 12, 1),
        (128, 64, 64, 8),
    ],
)
def test_event_type_head(
    dropout, graph_event_embedding_dim, hidden_dim, autoregressive_embedding_dim, batch
):
    if dropout == 0.0:
        head = EventTypeHead(
            graph_event_embedding_dim, hidden_dim, autoregressive_embedding_dim
        )
    else:
        head = EventTypeHead(
            graph_event_embedding_dim,
            hidden_dim,
            autoregressive_embedding_dim,
            dropout=dropout,
        )
    logits = head(torch.rand(batch, graph_event_embedding_dim))
    assert logits.size() == (batch, len(EVENT_TYPES))


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
@pytest.mark.parametrize(
    "graph_event_embedding_dim,hidden_dim,autoregressive_embedding_dim,batch",
    [
        (24, 12, 12, 1),
        (128, 64, 64, 8),
    ],
)
def test_event_type_head_get_autoregressive_embedding(
    graph_event_embedding_dim,
    hidden_dim,
    autoregressive_embedding_dim,
    batch,
    dropout,
    one_hot,
):
    if dropout == 0.0:
        head = EventTypeHead(
            graph_event_embedding_dim, hidden_dim, autoregressive_embedding_dim
        )
    else:
        head = EventTypeHead(
            graph_event_embedding_dim,
            hidden_dim,
            autoregressive_embedding_dim,
            dropout=dropout,
        )
    event_type_ids = torch.randint(len(EVENT_TYPES), (batch,))
    if one_hot:
        event_type_ids = F.one_hot(event_type_ids, num_classes=len(EVENT_TYPES)).float()
    autoregressive_embedding = head.get_autoregressive_embedding(
        torch.rand(batch, graph_event_embedding_dim),
        event_type_ids,
        torch.randint(2, (batch,)).bool(),
    )
    assert autoregressive_embedding.size() == (batch, autoregressive_embedding_dim)


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize(
    "node_embedding_dim,autoregressive_embedding_dim,hidden_dim,"
    "key_query_dim,batch,num_node",
    [
        (24, 12, 8, 4, 1, 0),
        (24, 12, 8, 4, 1, 1),
        (24, 12, 8, 4, 5, 0),
        (24, 12, 8, 4, 5, 10),
    ],
)
def test_event_node_head(
    node_embedding_dim,
    autoregressive_embedding_dim,
    hidden_dim,
    key_query_dim,
    batch,
    num_node,
    one_hot,
):
    head = EventNodeHead(
        node_embedding_dim, autoregressive_embedding_dim, hidden_dim, key_query_dim
    )
    logits, key = head(
        torch.rand(batch, autoregressive_embedding_dim),
        torch.rand(batch, num_node, node_embedding_dim),
    )
    node_ids = (
        torch.randint(num_node, (batch,)) if num_node > 0 else torch.zeros(batch).long()
    )
    if one_hot:
        if num_node > 0:
            node_ids = F.one_hot(node_ids, num_classes=num_node).float()
        else:
            node_ids = torch.zeros(batch, 0)
    autoregressive_embedding = head.update_autoregressive_embedding(
        torch.rand(batch, autoregressive_embedding_dim),
        node_ids,
        torch.rand(batch, num_node, node_embedding_dim),
        torch.randint(2, (batch, num_node)).bool(),
        torch.randint(2, (batch,)).bool(),
        key,
    )
    assert logits.size() == (batch, num_node)
    assert key.size() == (batch, num_node, key_query_dim)
    assert autoregressive_embedding.size() == (batch, autoregressive_embedding_dim)


@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
@pytest.mark.parametrize(
    "autoregressive_embedding_dim,hidden_dim,num_label,batch",
    [
        (24, 12, 4, 1),
        (24, 12, 4, 10),
        (48, 24, 8, 24),
    ],
)
def test_event_static_label_head(
    dropout, autoregressive_embedding_dim, hidden_dim, num_label, batch
):
    if dropout == 0.0:
        head = EventStaticLabelHead(autoregressive_embedding_dim, hidden_dim, num_label)
    else:
        head = EventStaticLabelHead(
            autoregressive_embedding_dim, hidden_dim, num_label, dropout=dropout
        )
    label_logits = head(torch.rand(batch, autoregressive_embedding_dim))
    assert label_logits.size() == (batch, num_label)


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("seq_len", [1, 5, 10])
@pytest.mark.parametrize(
    "autoregressive_embedding_dim,hidden_dim,word_embedding_dim,batch",
    [(8, 4, 16, 1), (8, 4, 16, 4)],
)
def test_event_seq_label_head_forward(
    autoregressive_embedding_dim,
    hidden_dim,
    word_embedding_dim,
    batch,
    seq_len,
    one_hot,
):
    vocab_size = 6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    head = EventSequentialLabelHead(
        autoregressive_embedding_dim,
        hidden_dim,
        nn.Embedding(vocab_size, word_embedding_dim, padding_idx=pad_token_id),
        bos_token_id,
        eos_token_id,
        pad_token_id,
    )

    output_tgt_seq = torch.randint(vocab_size, (batch, seq_len))
    if one_hot:
        output_tgt_seq = F.one_hot(output_tgt_seq, num_classes=vocab_size).float()
    seq_mask_list = [[True] * seq_len]
    for _ in range(batch - 1):
        length = random.randrange(1, seq_len) if seq_len > 1 else 1
        seq_mask_list.append([True] * length + [False] * (seq_len - length))
    output_tgt_seq_mask = torch.tensor(seq_mask_list)
    autoregressive_embedding = torch.rand(batch, autoregressive_embedding_dim)
    prev_hidden = torch.rand(batch, hidden_dim)
    # either autoregressive_embedding or prev_hidden has to be provided
    with pytest.raises(ValueError):
        head(output_tgt_seq, output_tgt_seq_mask)

    # only one of autoregressive_embedding or prev_hidden has to be provided
    with pytest.raises(AssertionError):
        head(
            output_tgt_seq,
            output_tgt_seq_mask,
            autoregressive_embedding=autoregressive_embedding,
            prev_hidden=prev_hidden,
        )

    # forward pass with autoregressive_embedding
    output_seq_logits, updated_hidden = head(
        output_tgt_seq,
        output_tgt_seq_mask,
        autoregressive_embedding=autoregressive_embedding,
    )
    assert output_seq_logits.size() == (batch, seq_len, vocab_size)
    assert updated_hidden.size() == (batch, hidden_dim)

    # forward pass with prev_hidden
    output_seq_logits, updated_hidden = head(
        output_tgt_seq, output_tgt_seq_mask, prev_hidden=prev_hidden
    )
    assert output_seq_logits.size() == (batch, seq_len, vocab_size)
    assert updated_hidden.size() == (batch, hidden_dim)


@pytest.mark.parametrize("autoregressive_embedding_dim", [4, 16])
@pytest.mark.parametrize(
    "batch,forward_logits,max_decode_len,expected_decoded,expected_mask",
    [
        (
            1,
            [torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1)],
            None,
            torch.tensor([[2]]),
            torch.tensor([[True]]),
        ),
        (
            1,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            None,
            torch.tensor([[3, 4, 2]]),
            torch.ones(1, 3).bool(),
        ),
        (
            1,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            2,
            torch.tensor([[3, 4]]),
            torch.ones(1, 2).bool(),
        ),
        (
            2,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            None,
            torch.tensor([[3, 4, 2], [4, 2, 0]]),
            torch.tensor([[True, True, True], [True, True, False]]),
        ),
        (
            2,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
                .float()
                .unsqueeze(1),
            ],
            1,
            torch.tensor([[3], [4]]),
            torch.ones(2, 1).bool(),
        ),
    ],
)
def test_event_seq_label_head_greedy_decode(
    batch,
    forward_logits,
    max_decode_len,
    expected_decoded,
    expected_mask,
    autoregressive_embedding_dim,
):
    vocab_size = 6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    head = EventSequentialLabelHead(
        autoregressive_embedding_dim,
        4,
        nn.Embedding(vocab_size, 12, padding_idx=pad_token_id),
        bos_token_id,
        eos_token_id,
        pad_token_id,
    )

    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            logits = forward_logits[self.num_calls]
            self.num_calls += 1
            return logits, torch.rand(batch, 4)

    head.forward = MockForward()

    if max_decode_len is None:
        decoded, mask = head.greedy_decode(
            torch.rand(batch, autoregressive_embedding_dim)
        )
    else:
        decoded, mask = head.greedy_decode(
            torch.rand(batch, autoregressive_embedding_dim),
            max_decode_len=max_decode_len,
        )
    assert decoded.equal(expected_decoded)
    assert mask.equal(expected_mask)


@pytest.mark.parametrize("tau", [0.1, 0.5, 1])
@pytest.mark.parametrize("autoregressive_embedding_dim", [4, 16])
@pytest.mark.parametrize(
    "batch,forward_logits,max_decode_len,expected_decoded,expected_mask",
    [
        (
            1,
            [torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1)],
            None,
            F.one_hot(torch.tensor([[2]]), num_classes=6).float(),
            torch.tensor([[True]]),
        ),
        (
            1,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            None,
            F.one_hot(torch.tensor([[3, 4, 2]]), num_classes=6).float(),
            torch.ones(1, 3).bool(),
        ),
        (
            1,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0]]).float().unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            2,
            F.one_hot(torch.tensor([[3, 4]]), num_classes=6).float(),
            torch.ones(1, 2).bool(),
        ),
        (
            2,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0]]).float().unsqueeze(1),
            ],
            None,
            torch.cat(
                [
                    F.one_hot(torch.tensor([[3, 4, 2]]), num_classes=6),
                    torch.cat(
                        [
                            F.one_hot(torch.tensor([4, 2]), num_classes=6),
                            torch.zeros(1, 6),
                        ]
                    ).unsqueeze(0),
                ]
            ),
            torch.tensor([[True, True, True], [True, True, False]]),
        ),
        (
            2,
            [
                torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])
                .float()
                .unsqueeze(1),
                torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
                .float()
                .unsqueeze(1),
            ],
            1,
            F.one_hot(torch.tensor([[3], [4]]), num_classes=6).float(),
            torch.ones(2, 1).bool(),
        ),
    ],
)
def test_event_seq_label_head_gumbel_greedy_decode(
    monkeypatch,
    batch,
    forward_logits,
    max_decode_len,
    expected_decoded,
    expected_mask,
    autoregressive_embedding_dim,
    tau,
):
    vocab_size = 6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    # monkeypatch gumbel_softmax to argmax() + F.one_hot()
    # to remove randomness for tests
    def mock_gumbel_softmax(logits, **kwargs):
        return F.one_hot(logits.argmax(-1), num_classes=vocab_size).float()

    monkeypatch.setattr(
        "tdgu.nn.graph_event_decoder.F.gumbel_softmax", mock_gumbel_softmax
    )
    head = EventSequentialLabelHead(
        autoregressive_embedding_dim,
        4,
        nn.Embedding(vocab_size, 12, padding_idx=pad_token_id),
        bos_token_id,
        eos_token_id,
        pad_token_id,
    )

    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            logits = forward_logits[self.num_calls]
            self.num_calls += 1
            return logits, torch.rand(batch, 4)

    head.forward = MockForward()

    if max_decode_len is None:
        decoded, mask = head.gumbel_greedy_decode(
            torch.rand(batch, autoregressive_embedding_dim), tau=tau
        )
    else:
        decoded, mask = head.gumbel_greedy_decode(
            torch.rand(batch, autoregressive_embedding_dim),
            max_decode_len=max_decode_len,
            tau=tau,
        )
    assert decoded.equal(expected_decoded)
    assert mask.equal(expected_mask)


@pytest.mark.parametrize(
    "input_dim,aggr_dim,hidden_dim,batch,obs_len,prev_action_len,num_node",
    [(12, 6, 8, 1, 8, 4, 6), (12, 6, 8, 4, 8, 4, 6)],
)
@pytest.mark.parametrize("hidden", [True, False])
def test_rnn_graph_event_decoder(
    hidden, input_dim, aggr_dim, hidden_dim, batch, obs_len, prev_action_len, num_node
):
    decoder = RNNGraphEventDecoder(input_dim, aggr_dim, hidden_dim)
    # need to have at least one unmasked token, otherwise pack_padded_sequence
    # raises an exception
    obs_mask = torch.cat(
        [torch.ones(batch, 1).bool(), torch.randint(2, (batch, obs_len - 1)).bool()],
        dim=1,
    )
    prev_action_mask = torch.cat(
        [
            torch.ones(batch, 1).bool(),
            torch.randint(2, (batch, prev_action_len - 1)).bool(),
        ],
        dim=1,
    )

    assert decoder(
        torch.rand(batch, input_dim),
        torch.rand(batch, obs_len, aggr_dim),
        obs_mask,
        torch.rand(batch, prev_action_len, aggr_dim),
        prev_action_mask,
        torch.rand(batch, num_node, aggr_dim),
        torch.rand(batch, num_node, aggr_dim),
        torch.randint(2, (batch, num_node)).bool(),
        hidden=torch.rand(batch, hidden_dim) if hidden else None,
    ).size() == (batch, hidden_dim)


@pytest.mark.parametrize("prev_input_seq_len", [0, 6, 8])
@pytest.mark.parametrize(
    "aggr_dim,hidden_dim,num_heads,batch,obs_len,prev_action_len,num_node",
    [
        (4, 8, 1, 1, 6, 4, 0),
        (4, 8, 1, 1, 6, 4, 4),
        (4, 8, 1, 8, 6, 4, 0),
        (4, 8, 1, 8, 6, 4, 4),
        (8, 16, 4, 1, 12, 4, 0),
        (8, 16, 4, 1, 12, 4, 8),
        (8, 16, 4, 8, 12, 4, 0),
        (8, 16, 4, 8, 12, 4, 8),
    ],
)
def test_transformer_graph_event_decoder_block(
    aggr_dim,
    hidden_dim,
    num_heads,
    batch,
    obs_len,
    prev_action_len,
    num_node,
    prev_input_seq_len,
):
    block = TransformerGraphEventDecoderBlock(aggr_dim, hidden_dim, num_heads)
    obs_mask = torch.cat(
        [
            torch.ones(batch, 1).bool(),
            torch.randint(2, (batch, obs_len - 1)).bool(),
        ],
        dim=1,
    )
    prev_action_mask = torch.cat(
        [
            torch.ones(batch, 1).bool(),
            torch.randint(2, (batch, prev_action_len - 1)).bool(),
        ],
        dim=1,
    )
    if prev_input_seq_len == 0:
        prev_input_seq_mask = torch.ones(batch, 0).bool()
        input_mask = torch.ones(batch).bool()
    else:
        prev_input_seq_mask = torch.cat(
            [
                torch.ones(batch, 1).bool(),
                torch.randint(2, (batch, prev_input_seq_len - 1)).bool(),
            ],
            dim=1,
        )
        if batch > 1:
            input_mask = torch.cat(
                [torch.tensor([True]), torch.randint(2, (batch - 1,)).bool()]
            )
        else:
            input_mask = torch.ones(batch).bool()
    results = block(
        torch.rand(batch, hidden_dim),
        input_mask,
        torch.rand(batch, obs_len, aggr_dim),
        obs_mask,
        torch.rand(batch, prev_action_len, aggr_dim),
        prev_action_mask,
        torch.rand(batch, num_node, aggr_dim),
        torch.rand(batch, num_node, aggr_dim),
        torch.randint(2, (batch, num_node)).bool(),
        torch.rand(batch, prev_input_seq_len, hidden_dim),
        prev_input_seq_mask,
    )
    assert not results["output"].isnan().any()
    assert results["output"].size() == (batch, hidden_dim)
    assert results["self_attn_weights"].size() == (batch, 1, prev_input_seq_len + 1)
    assert results["obs_graph_attn_weights"].size() == (batch, 1, obs_len)
    assert results["prev_action_graph_attn_weights"].size() == (
        batch,
        1,
        prev_action_len,
    )
    assert results["graph_obs_attn_weights"].size() == (batch, 1, num_node)
    assert results["graph_prev_action_attn_weights"].size() == (batch, 1, num_node)


@pytest.mark.parametrize("prev_input_event_emb_seq_len", [0, 6, 8])
@pytest.mark.parametrize(
    "input_dim,aggr_dim,num_dec_blocks,dec_block_num_heads,hidden_dim,batch,obs_len,"
    "prev_action_len,num_node",
    [
        (4, 8, 1, 1, 8, 1, 6, 4, 0),
        (4, 8, 1, 1, 8, 1, 6, 4, 6),
        (4, 8, 1, 1, 8, 4, 6, 4, 0),
        (4, 8, 1, 1, 8, 4, 6, 4, 6),
        (8, 16, 4, 2, 12, 1, 12, 6, 0),
        (8, 16, 4, 2, 12, 1, 12, 6, 8),
        (8, 16, 4, 2, 12, 4, 12, 6, 0),
        (8, 16, 4, 2, 12, 4, 12, 6, 8),
    ],
)
def test_transformer_graph_event_decoder(
    input_dim,
    aggr_dim,
    num_dec_blocks,
    dec_block_num_heads,
    hidden_dim,
    batch,
    obs_len,
    prev_action_len,
    num_node,
    prev_input_event_emb_seq_len,
):
    decoder = TransformerGraphEventDecoder(
        input_dim, aggr_dim, num_dec_blocks, dec_block_num_heads, hidden_dim
    )
    results, attn = decoder(
        torch.rand(batch, input_dim),
        torch.randint(2, (batch,)).bool(),
        torch.rand(batch, obs_len, aggr_dim),
        torch.randint(2, (batch, obs_len)).bool(),
        torch.rand(batch, prev_action_len, aggr_dim),
        torch.randint(2, (batch, prev_action_len)).bool(),
        torch.rand(batch, num_node, aggr_dim),
        torch.rand(batch, num_node, aggr_dim),
        torch.randint(2, (batch, num_node)).bool(),
        prev_input_event_emb_seq=None
        if prev_input_event_emb_seq_len == 0
        else torch.rand(
            num_dec_blocks, batch, prev_input_event_emb_seq_len, hidden_dim
        ),
        prev_input_event_emb_seq_mask=None
        if prev_input_event_emb_seq_len == 0
        else torch.randint(2, (batch, prev_input_event_emb_seq_len)).bool(),
    )
    assert results["output"].size() == (batch, hidden_dim)
    assert results["updated_prev_input_event_emb_seq"].size() == (
        num_dec_blocks,
        batch,
        prev_input_event_emb_seq_len + 1,
        hidden_dim,
    )
    assert results["updated_prev_input_event_emb_seq_mask"].size() == (
        batch,
        prev_input_event_emb_seq_len + 1,
    )

    assert len(attn["self_attn_weights"]) == num_dec_blocks
    for self_attn_weights in attn["self_attn_weights"]:
        assert self_attn_weights.size() == (batch, 1, prev_input_event_emb_seq_len + 1)
    assert len(attn["obs_graph_attn_weights"]) == num_dec_blocks
    for obs_graph_attn_weights in attn["obs_graph_attn_weights"]:
        assert obs_graph_attn_weights.size() == (batch, 1, obs_len)
    assert len(attn["prev_action_graph_attn_weights"]) == num_dec_blocks
    for prev_action_graph_attn_weights in attn["prev_action_graph_attn_weights"]:
        assert prev_action_graph_attn_weights.size() == (batch, 1, prev_action_len)
    assert len(attn["graph_obs_attn_weights"]) == num_dec_blocks
    for graph_obs_attn_weights in attn["graph_obs_attn_weights"]:
        assert graph_obs_attn_weights.size() == (batch, 1, num_node)
    assert len(attn["graph_prev_action_attn_weights"]) == num_dec_blocks
    for graph_prev_action_attn_weights in attn["graph_prev_action_attn_weights"]:
        assert graph_prev_action_attn_weights.size() == (batch, 1, num_node)
