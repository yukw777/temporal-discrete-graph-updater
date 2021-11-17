import pytest
import torch

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    RNNGraphEventDecoder,
    TransformerGraphEventDecoderBlock,
    TransformerGraphEventDecoder,
)
from dgu.constants import EVENT_TYPES


@pytest.mark.parametrize(
    "graph_event_embedding_dim,hidden_dim,batch", [(24, 12, 1), (128, 64, 8)]
)
def test_event_type_head(graph_event_embedding_dim, hidden_dim, batch):
    head = EventTypeHead(graph_event_embedding_dim, hidden_dim)
    logits = head(torch.rand(batch, graph_event_embedding_dim))
    assert logits.size() == (batch, len(EVENT_TYPES))


@pytest.mark.parametrize(
    "graph_event_embedding_dim,hidden_dim,batch", [(24, 12, 1), (128, 64, 8)]
)
def test_event_type_head_get_autoregressive_embedding(
    graph_event_embedding_dim, hidden_dim, batch
):
    head = EventTypeHead(graph_event_embedding_dim, hidden_dim)
    autoregressive_embedding = head.get_autoregressive_embedding(
        torch.rand(batch, graph_event_embedding_dim),
        torch.randint(len(EVENT_TYPES), (batch,)),
    )
    assert autoregressive_embedding.size() == (batch, graph_event_embedding_dim)


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
):
    head = EventNodeHead(
        node_embedding_dim, autoregressive_embedding_dim, hidden_dim, key_query_dim
    )
    logits, key = head(
        torch.rand(batch, autoregressive_embedding_dim),
        torch.rand(batch, num_node, node_embedding_dim),
    )
    autoregressive_embedding = head.update_autoregressive_embedding(
        torch.rand(batch, autoregressive_embedding_dim),
        torch.randint(num_node, (batch,))
        if num_node > 0
        else torch.zeros(batch).long(),
        torch.rand(batch, num_node, node_embedding_dim),
        key,
    )
    assert logits.size() == (batch, num_node)
    assert key.size() == (batch, num_node, key_query_dim)
    assert autoregressive_embedding.size() == (batch, autoregressive_embedding_dim)


@pytest.mark.parametrize(
    "autoregressive_embedding_dim,hidden_dim,key_query_dim,num_label,"
    "label_embedding_dim,batch",
    [
        (24, 12, 8, 4, 24, 1),
        (24, 12, 8, 4, 24, 10),
        (48, 24, 12, 8, 48, 24),
    ],
)
def test_event_static_label_head(
    autoregressive_embedding_dim,
    hidden_dim,
    key_query_dim,
    num_label,
    label_embedding_dim,
    batch,
):
    head = EventStaticLabelHead(
        autoregressive_embedding_dim, label_embedding_dim, hidden_dim, key_query_dim
    )
    label_logits = head(
        torch.rand(batch, autoregressive_embedding_dim),
        torch.rand(num_label, label_embedding_dim),
    )
    assert label_logits.size() == (batch, num_label)


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

    assert (
        decoder(
            torch.rand(batch, input_dim),
            torch.rand(batch, obs_len, aggr_dim),
            obs_mask,
            torch.rand(batch, prev_action_len, aggr_dim),
            prev_action_mask,
            torch.rand(batch, num_node, aggr_dim),
            torch.rand(batch, num_node, aggr_dim),
            torch.randint(2, (batch, num_node)).bool(),
            hidden=torch.rand(batch, hidden_dim) if hidden else None,
        ).size()
        == (batch, hidden_dim)
    )


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
    output = block(
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
    assert not output.isnan().any()
    assert output.size() == (batch, hidden_dim)


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
    (
        output,
        updated_prev_input_event_emb_seq,
        updated_prev_input_event_emb_seq_mask,
    ) = decoder(
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
    assert output.size() == (batch, hidden_dim)
    assert updated_prev_input_event_emb_seq.size() == (
        num_dec_blocks,
        batch,
        prev_input_event_emb_seq_len + 1,
        hidden_dim,
    )
    assert updated_prev_input_event_emb_seq_mask.size() == (
        batch,
        prev_input_event_emb_seq_len + 1,
    )
