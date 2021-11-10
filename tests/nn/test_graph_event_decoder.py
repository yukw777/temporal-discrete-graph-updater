import pytest
import torch

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    RNNGraphEventDecoder,
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
