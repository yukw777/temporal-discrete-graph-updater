import pytest
import torch

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    StaticLabelGraphEventDecoder,
    RNNGraphEventDecoder,
)
from dgu.constants import EVENT_TYPES


@pytest.mark.parametrize(
    "graph_event_embedding_dim,hidden_dim,batch", [(24, 12, 1), (128, 64, 8)]
)
def test_event_type_head(graph_event_embedding_dim, hidden_dim, batch):
    head = EventTypeHead(graph_event_embedding_dim, hidden_dim)
    logits, autoregressive_embedding = head(
        torch.rand(batch, graph_event_embedding_dim)
    )
    assert logits.size() == (batch, len(EVENT_TYPES))
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
    logits, autoregressive_embedding = head(
        torch.rand(batch, autoregressive_embedding_dim),
        torch.rand(batch, num_node, node_embedding_dim),
    )
    assert logits.size() == (batch, num_node)
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
    "graph_event_embedding_dim,node_embedding_dim,hidden_dim,key_query_dim,"
    "num_label,label_embedding_dim,batch,num_node",
    [
        (36, 24, 12, 8, 4, 24, 1, 4),
        (36, 24, 12, 8, 4, 24, 10, 12),
        (72, 36, 24, 12, 8, 48, 24, 36),
    ],
)
def test_static_label_graph_event_decoder(
    graph_event_embedding_dim,
    node_embedding_dim,
    hidden_dim,
    key_query_dim,
    num_label,
    label_embedding_dim,
    batch,
    num_node,
):
    decoder = StaticLabelGraphEventDecoder(
        graph_event_embedding_dim,
        node_embedding_dim,
        label_embedding_dim,
        hidden_dim,
        key_query_dim,
    )
    results = decoder(
        torch.rand(batch, graph_event_embedding_dim),
        torch.rand(batch, num_node, node_embedding_dim),
        torch.rand(num_label, label_embedding_dim),
    )

    assert results["event_type_logits"].size() == (batch, len(EVENT_TYPES))
    assert results["src_logits"].size() == (batch, num_node)
    assert results["dst_logits"].size() == (batch, num_node)
    assert results["label_logits"].size() == (batch, num_label)


@pytest.mark.parametrize(
    "input_dim,node_embedding_dim,label_embedding_dim,delta_g_dim,"
    "hidden_dim,batch,num_node",
    [(12, 6, 8, 16, 4, 1, 10), (12, 6, 8, 16, 4, 8, 10)],
)
@pytest.mark.parametrize("hidden", [True, False])
def test_rnn_graph_event_decoder(
    hidden,
    input_dim,
    node_embedding_dim,
    label_embedding_dim,
    delta_g_dim,
    hidden_dim,
    batch,
    num_node,
):
    num_label = 10

    decoder = RNNGraphEventDecoder(
        input_dim,
        delta_g_dim,
        hidden_dim,
        StaticLabelGraphEventDecoder(
            hidden_dim, node_embedding_dim, label_embedding_dim, 8, 8
        ),
    )

    results = decoder(
        torch.rand(batch, input_dim),
        torch.rand(batch, delta_g_dim),
        torch.rand(batch, num_node, node_embedding_dim),
        torch.rand(num_label, label_embedding_dim),
        hidden=torch.rand(batch, hidden_dim) if hidden else None,
    )

    assert results["event_type_logits"].size() == (batch, len(EVENT_TYPES))
    assert results["src_logits"].size() == (batch, num_node)
    assert results["dst_logits"].size() == (batch, num_node)
    assert results["label_logits"].size() == (batch, num_label)
    assert results["new_hidden"].size() == (batch, hidden_dim)
