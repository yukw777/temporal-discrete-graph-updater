import pytest
import torch

from itertools import cycle

from dgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
    StaticLabelGraphEventDecoder,
    StaticLabelGraphEventEncoder,
    RNNGraphEventSeq2Seq,
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
        (12, 8, 1, 1),
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
    assert results["src_logits"].size() == (batch, num_node)
    assert results["dst_logits"].size() == (batch, num_node)
    assert results["label_logits"].size() == (batch, num_node_label + num_edge_label)


@pytest.mark.parametrize(
    "hidden_dim,batch,graph_event_seq_len,num_node,num_label,label_embedding_dim",
    [
        (10, 1, 5, 12, 24, 36),
        (10, 10, 5, 12, 24, 36),
        (24, 16, 10, 24, 48, 64),
    ],
)
def test_static_label_graph_event_encoder(
    hidden_dim, batch, graph_event_seq_len, num_node, num_label, label_embedding_dim
):
    encoder = StaticLabelGraphEventEncoder()
    assert (
        encoder(
            torch.rand(batch, graph_event_seq_len),
            torch.randint(num_node, (batch, graph_event_seq_len))
            if num_node > 0
            else torch.zeros(batch, graph_event_seq_len).long(),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.randint(num_node, (batch, graph_event_seq_len))
            if num_node > 0
            else torch.zeros(batch, graph_event_seq_len).long(),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.randint(num_label, (batch, graph_event_seq_len)),
            torch.randint(2, (batch, graph_event_seq_len)).float(),
            torch.rand(num_node, hidden_dim),
            torch.rand(num_label, label_embedding_dim),
        ).size()
        == (batch, graph_event_seq_len, 3 * hidden_dim + label_embedding_dim)
    )


@pytest.mark.parametrize(
    "hidden_dim,key_query_dim,num_node_label,num_edge_label,"
    "label_embedding_dim,batch,obs_seq_len,graph_event_seq_len,num_node,"
    "subgraph_num_node",
    [
        (20, 10, 15, 7, 36, 1, 5, 10, 12, 6),
        (20, 10, 15, 7, 36, 5, 5, 10, 12, 6),
        (36, 24, 25, 10, 48, 8, 20, 18, 36, 24),
    ],
)
def test_rnn_graph_event_seq2seq_teacher_forcing(
    hidden_dim,
    key_query_dim,
    num_node_label,
    num_edge_label,
    label_embedding_dim,
    batch,
    obs_seq_len,
    graph_event_seq_len,
    num_node,
    subgraph_num_node,
):
    seq2seq = RNNGraphEventSeq2Seq(
        hidden_dim,
        100,
        label_embedding_dim,
        StaticLabelGraphEventEncoder(),
        StaticLabelGraphEventDecoder(
            hidden_dim,
            key_query_dim,
            torch.rand(num_node_label, label_embedding_dim),
            torch.rand(num_edge_label, label_embedding_dim),
        ),
    )
    seq2seq.train()
    results = seq2seq(
        torch.rand(batch, obs_seq_len, 4 * hidden_dim),
        torch.rand(num_node, hidden_dim),
        subgraph_node_ids=torch.randint(num_node, (subgraph_num_node,)),
        tgt_event_mask=torch.randint(2, (batch, graph_event_seq_len)).float(),
        tgt_event_type_ids=torch.randint(
            len(EVENT_TYPES), (batch, graph_event_seq_len)
        ),
        tgt_event_src_ids=torch.randint(num_node, (batch, graph_event_seq_len)),
        tgt_event_src_mask=torch.randint(2, (batch, graph_event_seq_len)).float(),
        tgt_event_dst_ids=torch.randint(num_node, (batch, graph_event_seq_len)),
        tgt_event_dst_mask=torch.randint(2, (batch, graph_event_seq_len)).float(),
        tgt_event_label_ids=torch.randint(
            num_node_label + num_edge_label, (batch, graph_event_seq_len)
        ),
    )
    assert results["event_type_logits"].size() == (
        batch,
        graph_event_seq_len,
        len(EVENT_TYPES),
    )
    assert results["src_logits"].size() == (
        batch,
        graph_event_seq_len,
        subgraph_num_node,
    )
    assert results["dst_logits"].size() == (
        batch,
        graph_event_seq_len,
        subgraph_num_node,
    )
    assert results["label_logits"].size() == (
        batch,
        graph_event_seq_len,
        num_node_label + num_edge_label,
    )


@pytest.mark.parametrize(
    "hidden_dim,key_query_dim,num_node_label,num_edge_label,"
    "label_embedding_dim,batch,obs_seq_len,num_node",
    [
        (20, 10, 15, 7, 24, 1, 5, 12),
        (20, 10, 15, 7, 24, 5, 5, 12),
        (36, 24, 25, 10, 48, 8, 20, 36),
    ],
)
def test_rnn_graph_event_seq2seq_greedy_decode(
    hidden_dim,
    key_query_dim,
    num_node_label,
    num_edge_label,
    label_embedding_dim,
    batch,
    obs_seq_len,
    num_node,
):
    seq2seq = RNNGraphEventSeq2Seq(
        hidden_dim,
        100,
        label_embedding_dim,
        StaticLabelGraphEventEncoder(),
        StaticLabelGraphEventDecoder(
            hidden_dim,
            key_query_dim,
            torch.rand(num_node_label, label_embedding_dim),
            torch.rand(num_edge_label, label_embedding_dim),
        ),
    )
    seq2seq.eval()

    results = seq2seq(
        torch.rand(batch, obs_seq_len, 4 * hidden_dim), torch.rand(num_node, hidden_dim)
    )
    assert results["decoded_event_type_ids"].size(0) == batch
    assert results["decoded_event_type_ids"].dtype == torch.int64
    assert results["decoded_src_ids"].size(0) == batch
    assert results["decoded_src_ids"].dtype == torch.int64
    assert results["decoded_dst_ids"].size(0) == batch
    assert results["decoded_dst_ids"].dtype == torch.int64
    assert results["decoded_label_ids"].size(0) == batch
    assert results["decoded_label_ids"].dtype == torch.int64

    assert results["decoded_event_type_ids"].size(1) == results["decoded_src_ids"].size(
        1
    )
    assert results["decoded_src_ids"].size(1) == results["decoded_dst_ids"].size(1)
    assert results["decoded_dst_ids"].size(1) == results["decoded_label_ids"].size(1)
    assert results["decoded_label_ids"].size(1) <= seq2seq.max_decode_len + 1
