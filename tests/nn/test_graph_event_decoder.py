import pytest
import torch
import torch.nn as nn

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    StaticLabelGraphEventDecoder,
    RNNGraphEventDecoder,
)
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


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
    "event_type_embedding_dim,node_embedding_dim,label_embedding_dim,delta_g_dim,"
    "hidden_dim,batch,num_node",
    [(4, 6, 8, 16, 4, 1, 10), (4, 6, 8, 16, 4, 8, 10)],
)
@pytest.mark.parametrize("hidden", [True, False])
def test_rnn_graph_event_decoder(
    hidden,
    event_type_embedding_dim,
    node_embedding_dim,
    label_embedding_dim,
    delta_g_dim,
    hidden_dim,
    batch,
    num_node,
):
    num_label = 10

    decoder = RNNGraphEventDecoder(
        event_type_embedding_dim,
        node_embedding_dim,
        label_embedding_dim,
        delta_g_dim,
        hidden_dim,
        StaticLabelGraphEventDecoder(
            hidden_dim, node_embedding_dim, label_embedding_dim, 8, 8
        ),
    )

    results = decoder(
        torch.randint(len(EVENT_TYPES), (batch,)),
        torch.randint(num_node, (batch,)),
        torch.randint(num_node, (batch,)),
        torch.randint(num_label, (batch,)),
        torch.rand(batch, delta_g_dim),
        torch.rand(batch, num_node, node_embedding_dim),
        torch.rand(num_label, label_embedding_dim),
        hidden=torch.rand(batch, hidden_dim) if hidden else None,
    )

    assert results["event_type_logits"].size() == (batch, len(EVENT_TYPES))
    assert results["src_logits"].size() == (batch, num_node)
    assert results["dst_logits"].size() == (batch, num_node)
    assert results["label_logits"].size() == (batch, num_label)


@pytest.mark.parametrize(
    "event_type_ids,event_src_ids,event_dst_ids,event_label_ids,event_type_embeddings,"
    "node_embeddings,label_embeddings,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[0.0] * 4, [3.0] * 4]),
            torch.empty(1, 0, 4),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[0.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[0.0] * 4, [3.0] * 4]),
            torch.tensor([[[1.0] * 4, [2.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[0.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[2.0] * 4, [3.0] * 4]),
            torch.empty(1, 0, 4),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[3.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[0.0] * 4, [3.0] * 4]),
            torch.tensor([[[1.0] * 4, [2.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[3.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[0.0] * 4, [2.0] * 4, [3.0] * 4]),
            torch.empty(1, 0, 4),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[3.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[0.0] * 4, [2.0] * 4, [3.0] * 4]),
            torch.tensor([[[1.0] * 4, [2.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[3.0] * 4 + [0.0] * 4 + [0.0] * 4 + [0.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[0.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4]),
            torch.empty(1, 0, 4),
            torch.tensor([[0.0] * 6, [2.0] * 6]),
            torch.tensor([[4.0] * 4 + [0.0] * 4 + [0.0] * 4 + [2.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([[0.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4]),
            torch.tensor([[[1.0] * 4, [2.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6, [3.0] * 6]),
            torch.tensor([[4.0] * 4 + [0.0] * 4 + [0.0] * 4 + [3.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([[0.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4, [5.0] * 4]),
            torch.tensor([[[1.0] * 4, [2.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6, [3.0] * 6]),
            torch.tensor([[5.0] * 4 + [2.0] * 4 + [0.0] * 4 + [3.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor(
                [[0.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4, [5.0] * 4, [6.0] * 4]
            ),
            torch.tensor([[[1.0] * 4, [2.0] * 4, [3.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6, [3.0] * 6]),
            torch.tensor([[6.0] * 4 + [2.0] * 4 + [3.0] * 4 + [2.0] * 6]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor(
                [
                    [0.0] * 4,
                    [2.0] * 4,
                    [3.0] * 4,
                    [4.0] * 4,
                    [5.0] * 4,
                    [6.0] * 4,
                    [7.0] * 4,
                ]
            ),
            torch.tensor([[[1.0] * 4, [2.0] * 4, [3.0] * 4]]),
            torch.tensor([[0.0] * 6, [2.0] * 6, [3.0] * 6]),
            torch.tensor([[7.0] * 4 + [2.0] * 4 + [3.0] * 4 + [2.0] * 6]),
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
            torch.tensor([2, 3, 4, 1, 2, 3]),
            torch.tensor(
                [
                    [0.0] * 4,
                    [2.0] * 4,
                    [3.0] * 4,
                    [4.0] * 4,
                    [5.0] * 4,
                    [6.0] * 4,
                    [7.0] * 4,
                ]
            ),
            torch.tensor(
                [
                    [[2.0] * 4, [1.0] * 4, [3.0] * 4],
                    [[3.0] * 4, [2.0] * 4, [1.0] * 4],
                    [[1.0] * 4, [2.0] * 4, [3.0] * 4],
                    [[3.0] * 4, [1.0] * 4, [2.0] * 4],
                    [[1.0] * 4, [2.0] * 4, [3.0] * 4],
                    [[2.0] * 4, [3.0] * 4, [1.0] * 4],
                ]
            ),
            torch.tensor([[0.0] * 6, [2.0] * 6, [3.0] * 6, [4.0] * 6, [5.0] * 6]),
            torch.tensor(
                [
                    [6.0] * 4 + [2.0] * 4 + [1.0] * 4 + [3.0] * 6,
                    [4.0] * 4 + [0.0] * 4 + [0.0] * 4 + [4.0] * 6,
                    [7.0] * 4 + [3.0] * 4 + [1.0] * 4 + [5.0] * 6,
                    [5.0] * 4 + [1.0] * 4 + [0.0] * 4 + [2.0] * 6,
                    [6.0] * 4 + [2.0] * 4 + [3.0] * 4 + [3.0] * 6,
                    [4.0] * 4 + [0.0] * 4 + [0.0] * 4 + [4.0] * 6,
                ]
            ),
        ),
    ],
)
def test_rnn_graph_event_decoder_get_decoder_input(
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_ids,
    event_type_embeddings,
    node_embeddings,
    label_embeddings,
    expected,
):
    decoder = RNNGraphEventDecoder(
        event_type_embeddings.size(1),
        node_embeddings.size(2),
        label_embeddings.size(1),
        4,
        4,
        StaticLabelGraphEventDecoder(
            4, node_embeddings.size(2), label_embeddings.size(1), 8, 8
        ),
    )
    decoder.event_type_embeddings = nn.Embedding.from_pretrained(
        event_type_embeddings, padding_idx=EVENT_TYPE_ID_MAP["pad"]
    )
    assert decoder.get_decoder_input(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_label_ids,
        node_embeddings,
        label_embeddings,
    ).equal(expected)
