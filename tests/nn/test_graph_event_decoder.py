import pytest
import torch

from itertools import cycle

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    StaticLabelGraphEventDecoder,
    StaticLabelGraphEventEncoder,
)
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


@pytest.mark.parametrize("hidden_dim,batch", [(12, 1), (64, 8)])
def test_event_type_head(hidden_dim, batch):
    head = EventTypeHead(hidden_dim)
    logits, autoregressive_embedding = head(torch.rand(batch, hidden_dim))
    assert logits.size() == (batch, len(EVENT_TYPES))
    assert autoregressive_embedding.size() == (batch, hidden_dim)


@pytest.mark.parametrize(
    "hidden_dim,key_query_dim,batch,num_node",
    [
        (12, 8, 1, 0),
        (12, 8, 1, 1),
        (12, 8, 5, 0),
        (12, 8, 5, 10),
    ],
)
def test_event_node_head(hidden_dim, key_query_dim, batch, num_node):
    head = EventNodeHead(hidden_dim, key_query_dim)
    logits, autoregressive_embedding = head(
        torch.rand(batch, hidden_dim), torch.rand(num_node, hidden_dim)
    )
    assert logits.size() == (batch, num_node)
    assert autoregressive_embedding.size() == (batch, hidden_dim)


@pytest.mark.parametrize(
    "hidden_dim,key_query_dim,num_node_label,num_edge_label,label_embedding_dim,batch",
    [
        (12, 8, 4, 5, 24, 1),
        (12, 8, 4, 5, 24, 10),
        (24, 12, 8, 10, 48, 24),
    ],
)
def test_event_static_label_head(
    hidden_dim,
    key_query_dim,
    num_node_label,
    num_edge_label,
    label_embedding_dim,
    batch,
):
    head = EventStaticLabelHead(
        hidden_dim,
        key_query_dim,
        torch.rand(num_node_label, label_embedding_dim),
        torch.rand(num_edge_label, label_embedding_dim),
    )
    event_types = []
    for i in cycle(range(len(EVENT_TYPES))):
        event_types.append(i)
        if len(event_types) == batch:
            break
    event_type = torch.tensor(event_types)
    label_logits = head(torch.rand(batch, hidden_dim), event_type)
    assert label_logits.size() == (batch, num_node_label + num_edge_label)

    # make sure the logits are masked correctly
    for et, logits in zip(event_type, label_logits):
        if et.item() in {
            EVENT_TYPE_ID_MAP["pad"],
            EVENT_TYPE_ID_MAP["start"],
            EVENT_TYPE_ID_MAP["end"],
        }:
            # pad, start or end so all the logits should be zero
            assert logits.equal(torch.zeros(num_node_label + num_edge_label))
        elif et.item() in {
            EVENT_TYPE_ID_MAP["node-add"],
            EVENT_TYPE_ID_MAP["node-delete"],
        }:
            # node-add or node-delete so the edge logits should be zero
            assert not logits[:num_node_label].equal(torch.zeros(num_node_label))
            assert logits[num_node_label:].equal(torch.zeros(num_edge_label))
        elif et.item() in {
            EVENT_TYPE_ID_MAP["edge-add"],
            EVENT_TYPE_ID_MAP["edge-delete"],
        }:
            # edge-add or edge-delete so the node logits should be zero
            assert logits[:num_node_label].equal(torch.zeros(num_node_label))
            assert not logits[num_node_label:].equal(torch.zeros(num_edge_label))
        else:
            raise ValueError("Unknown event type")


@pytest.mark.parametrize(
    "hidden_dim,key_query_dim,num_node_label,num_edge_label,label_embedding_dim,"
    "batch,num_node",
    [
        (12, 8, 4, 5, 24, 1, 0),
        (12, 8, 4, 5, 24, 1, 4),
        (12, 8, 4, 5, 24, 10, 12),
        (24, 12, 8, 10, 48, 24, 36),
    ],
)
def test_static_label_graph_event_decoder(
    hidden_dim,
    key_query_dim,
    num_node_label,
    num_edge_label,
    label_embedding_dim,
    batch,
    num_node,
):
    decoder = StaticLabelGraphEventDecoder(
        hidden_dim,
        key_query_dim,
        torch.rand(num_node_label, label_embedding_dim),
        torch.rand(num_edge_label, label_embedding_dim),
    )
    results = decoder(torch.rand(batch, hidden_dim), torch.rand(num_node, hidden_dim))

    assert results["event_type_logits"].size() == (batch, len(EVENT_TYPES))
    assert results["src_node_logits"].size() == (batch, num_node)
    assert results["dst_node_logits"].size() == (batch, num_node)
    assert results["label_logits"].size() == (batch, num_node_label + num_edge_label)


@pytest.mark.parametrize(
    "hidden_dim,batch,graph_event_seq_len,num_node,num_label",
    [
        (10, 1, 5, 12, 24),
        (10, 10, 5, 12, 24),
        (24, 16, 10, 24, 48),
    ],
)
def test_static_label_graph_event_encoder(
    hidden_dim, batch, graph_event_seq_len, num_node, num_label
):
    encoder = StaticLabelGraphEventEncoder()
    assert (
        encoder(
            torch.rand(batch, graph_event_seq_len),
            torch.randint(num_node, (batch, graph_event_seq_len)),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.randint(num_node, (batch, graph_event_seq_len)),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.randint(num_label, (batch, graph_event_seq_len)),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.rand(num_node, hidden_dim),
            torch.rand(num_label, hidden_dim),
        ).size()
        == (batch, graph_event_seq_len, 4 * hidden_dim)
    )
