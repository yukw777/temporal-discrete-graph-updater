import pytest
import torch
import torch.nn as nn

from dgu.nn.temporal_graph import TemporalGraphNetwork, TransformerConvStack
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


class MockTimeEncoder(nn.Module):
    def forward(self, timestamp):
        return timestamp.unsqueeze(-1).expand(-1, 4) + 3


@pytest.mark.parametrize(
    "event_type_ids,event_type_emb,node_ids,event_embeddings,event_timestamps,"
    "event_mask,memory,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([0]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [3] * 4 + [4] * 4 + [0] * 4 + [3] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([0]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [4] * 4 + [4] * 4 + [0] * 4 + [3] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [4] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 3, 0, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 1, 0, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [9] * 4, [10] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([0, 1, 1, 1, 0]).float(),
            torch.linspace(11, 14, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [4] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [5] * 4 + [13] * 4 + [0] * 4 + [5] * 4 + [9] * 4,
                    [4] * 4 + [14] * 4 + [0] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_node_message(
    event_type_ids,
    event_type_emb,
    node_ids,
    event_embeddings,
    event_timestamps,
    event_mask,
    memory,
    expected,
):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(event_type_emb)

    assert tgn.node_message(
        event_type_ids, node_ids, event_embeddings, event_timestamps, event_mask, memory
    ).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_type_emb,src_ids,dst_ids,event_embeddings,event_timestamps,"
    "event_mask,memory,edge_index,last_update,src_expected,dst_expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[7, 0, 8], [9, 1, 10]]),
            torch.tensor([2, 1, 3]).float(),
            torch.tensor(
                [
                    [5] * 4 + [4] * 4 + [5] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [5] * 4 + [5] * 4 + [4] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[7, 0, 8], [9, 1, 10]]),
            torch.tensor([2, 1, 3]).float(),
            torch.tensor(
                [
                    [6] * 4 + [4] * 4 + [5] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [6] * 4 + [5] * 4 + [4] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[4, 5, 8, 2, 4], [3, 7, 4, 3, 6]]),
            torch.tensor([2, 4, 5, 3, 4]).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [11] * 4 + [12] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [12] * 4 + [11] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[4, 5, 8, 2, 4], [3, 7, 4, 3, 6]]),
            torch.tensor([2, 4, 5, 3, 4]).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [11] * 4 + [12] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [12] * 4 + [11] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 2, 0, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4, [0] * 4]).float(),
            torch.tensor([0, 5, 0, 0]),
            torch.tensor([0, 1, 0, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[4, 5, 8, 1, 4], [3, 7, 4, 2, 6]]),
            torch.tensor([3, 2, 4, 5, 3]).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [10] * 4 + [11] * 4 + [3] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [11] * 4 + [10] * 4 + [3] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([0, 3, 2, 1, 0]),
            torch.tensor([[0] * 4, [8] * 4, [9] * 4, [10] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 5, 6, 0]),
            torch.tensor([0, 1, 1, 1, 0]).float(),
            torch.linspace(11, 14, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[4, 5, 1, 2, 3], [3, 7, 3, 2, 1]]),
            torch.tensor([3, 2, 4, 5, 3]).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [12] * 4 + [14] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [14] * 4 + [12] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [14] * 4 + [12] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [12] * 4 + [14] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_edge_message(
    event_type_ids,
    event_type_emb,
    src_ids,
    dst_ids,
    event_embeddings,
    event_timestamps,
    event_mask,
    memory,
    edge_index,
    last_update,
    src_expected,
    dst_expected,
):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(event_type_emb)

    src_messages, dst_messages = tgn.edge_message(
        event_type_ids,
        src_ids,
        dst_ids,
        event_embeddings,
        event_timestamps,
        event_mask,
        memory,
        edge_index,
        last_update,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "node_features,node_event_type_ids,node_event_node_ids,"
    "node_event_embeddings,expected",
    [
        (
            torch.zeros(0, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.ones(1, 4),
            torch.ones(1, 4),
        ),
        (
            torch.zeros(0, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([2]),
            torch.ones(1, 4),
            torch.tensor([[0] * 4, [0] * 4, [1] * 4]).float(),
        ),
        (
            torch.ones(1, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([0]),
            torch.ones(1, 4),
            torch.ones(1, 4),
        ),
        (
            torch.ones(3, 4),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 2, 0, 0]).long(),
            torch.tensor([0, 2, 3, 0]).float().unsqueeze(-1).expand(-1, 4),
            torch.tensor([[3] * 4, [1] * 4, [2] * 4]).float(),
        ),
        (
            torch.ones(3, 4),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 2, 5, 2, 0, 0]).long(),
            torch.tensor([0, 2, 3, 2, 0, 0]).float().unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [[1] * 4, [1] * 4, [2] * 4, [0] * 4, [0] * 4, [3] * 4]
            ).float(),
        ),
    ],
)
def test_tgn_update_node_features(
    node_features,
    node_event_type_ids,
    node_event_node_ids,
    node_event_embeddings,
    expected,
):
    assert TemporalGraphNetwork.update_node_features(
        node_features, node_event_type_ids, node_event_node_ids, node_event_embeddings
    ).equal(expected)


@pytest.mark.parametrize(
    "edge_features,edge_event_type_ids,edge_event_edge_ids,"
    "edge_event_embeddings,expected",
    [
        (
            torch.zeros(0, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([0]),
            torch.ones(1, 4),
            torch.ones(1, 4),
        ),
        (
            torch.zeros(0, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([2]),
            torch.ones(1, 4),
            torch.tensor([[0] * 4, [0] * 4, [1] * 4]).float(),
        ),
        (
            torch.ones(1, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([0]),
            torch.ones(1, 4),
            torch.ones(1, 4),
        ),
        (
            torch.ones(3, 4),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 2, 0, 0]).long(),
            torch.tensor([0, 2, 3, 0]).float().unsqueeze(-1).expand(-1, 4),
            torch.tensor([[3] * 4, [1] * 4, [2] * 4]).float(),
        ),
        (
            torch.ones(3, 4),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 2, 5, 2, 0, 0]).long(),
            torch.tensor([0, 2, 3, 2, 0, 0]).float().unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [[1] * 4, [1] * 4, [2] * 4, [0] * 4, [0] * 4, [3] * 4]
            ).float(),
        ),
    ],
)
def test_tgn_update_edge_features(
    edge_features,
    edge_event_type_ids,
    edge_event_edge_ids,
    edge_event_embeddings,
    expected,
):
    assert TemporalGraphNetwork.update_edge_features(
        edge_features, edge_event_type_ids, edge_event_edge_ids, edge_event_embeddings
    ).equal(expected)


@pytest.mark.parametrize(
    "edge_last_update,edge_event_type_ids,edge_event_edge_ids,edge_event_timestamps,"
    "expected",
    [
        (
            torch.zeros(0),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([2.0]),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([0]),
            torch.tensor([3.0]),
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([2.0]),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([0]),
            torch.tensor([3.0]),
            torch.tensor([3.0]),
        ),
        (
            torch.tensor([2.0, 3.0]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0]),
            torch.tensor([0.0, 4.0, 5.0, 6.0, 6.0, 0.0]),
            torch.tensor([4.0, 3.0, 6.0, 6.0]),
        ),
    ],
)
def test_tgn_expand_last_update(
    edge_last_update,
    edge_event_type_ids,
    edge_event_edge_ids,
    edge_event_timestamps,
    expected,
):
    assert TemporalGraphNetwork.expand_last_update(
        edge_last_update,
        edge_event_type_ids,
        edge_event_edge_ids,
        edge_event_timestamps,
    ).equal(expected)


@pytest.mark.parametrize(
    "edge_last_update,edge_event_type_ids,edge_event_edge_ids,edge_event_timestamps,"
    "expected",
    [
        (
            torch.tensor([2.0]),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([0]),
            torch.tensor([3.0]),
            torch.tensor([3.0]),
        ),
        (
            torch.tensor([2.0]),
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([0]),
            torch.tensor([3.0]),
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([2.0, 3.0, 6.0, 6.0]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0]),
            torch.tensor([0.0, 4.0, 5.0, 6.0, 6.0, 0.0]),
            torch.tensor([2.0, 5.0, 6.0, 6.0]),
        ),
    ],
)
def test_tgn_update_last_update(
    edge_last_update,
    edge_event_type_ids,
    edge_event_edge_ids,
    edge_event_timestamps,
    expected,
):
    assert TemporalGraphNetwork.update_last_update(
        edge_last_update,
        edge_event_type_ids,
        edge_event_edge_ids,
        edge_event_timestamps,
    ).equal(expected)


@pytest.mark.parametrize(
    "event_type_dim,memory_dim,time_enc_dim,event_embedding_dim,output_dim,"
    "num_node_event,num_edge_event,num_node,num_edge,transformer_conv_num_block,"
    "transformer_conv_num_heads",
    [
        (4, 8, 16, 20, 12, 0, 0, 0, 0, 1, 1),
        (4, 8, 16, 20, 12, 1, 0, 1, 0, 1, 1),
        (4, 8, 16, 20, 12, 0, 1, 2, 1, 1, 1),
        (4, 8, 16, 20, 12, 8, 4, 5, 10, 4, 4),
        (8, 16, 32, 48, 24, 16, 10, 8, 10, 6, 6),
        (8, 16, 32, 48, 24, 16, 10, 8, 10, 8, 8),
    ],
)
def test_tgn_forward(
    event_type_dim,
    memory_dim,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    num_node_event,
    num_edge_event,
    num_node,
    num_edge,
    transformer_conv_num_block,
    transformer_conv_num_heads,
):
    tgn = TemporalGraphNetwork(
        event_type_dim,
        memory_dim,
        time_enc_dim,
        event_embedding_dim,
        output_dim,
        transformer_conv_num_block,
        transformer_conv_num_heads,
    )
    results = tgn(
        torch.randint(len(EVENT_TYPES), (num_node_event,)),
        torch.randint(num_node, (num_node_event,))
        if num_node > 0
        else torch.zeros(num_node_event).long(),
        torch.rand(num_node_event, event_embedding_dim),
        torch.randint(10, (num_node_event,)).float(),
        torch.randint(2, (num_node_event,)).float(),
        torch.randint(len(EVENT_TYPES), (num_edge_event,)),
        torch.randint(num_node, (num_edge_event,))
        if num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.randint(num_node, (num_edge_event,))
        if num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.rand(num_edge_event, event_embedding_dim),
        torch.randint(10, (num_edge_event,)).float(),
        torch.randint(2, (num_edge_event,)).float(),
        torch.rand(num_node, memory_dim),
        torch.rand(num_node, event_embedding_dim),
        torch.randint(num_node, (2, num_edge))
        if num_node > 0
        else torch.zeros(2, num_edge).long(),
        torch.rand(num_edge, event_embedding_dim),
        torch.randint(10, (num_edge,)).float(),
        torch.randint(10, (num_edge,)).float(),
    )
    assert results["node_embeddings"].size() == (num_node, output_dim)
    assert results["updated_memory"].size() == (num_node, memory_dim)


@pytest.mark.parametrize(
    "node_dim,output_dim,num_block,heads,edge_dim,num_node,num_edge",
    [(16, 8, 1, 1, None, 1, 0), (16, 8, 4, 3, 12, 5, 4)],
)
def test_transformer_conv_stack(
    node_dim, output_dim, num_block, heads, edge_dim, num_node, num_edge
):
    stack = TransformerConvStack(
        node_dim, output_dim, num_block, heads=heads, edge_dim=edge_dim
    )
    assert (
        stack(
            torch.rand(num_node, node_dim),
            torch.randint(num_node, (2, num_edge)),
            edge_attr=None if edge_dim is None else torch.rand(num_edge, edge_dim),
        ).size()
        == (num_node, output_dim)
    )


@pytest.mark.parametrize(
    "memory,node_event_type_ids,node_event_node_ids,expected",
    [
        (
            torch.zeros(0, 4),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.zeros(1, 4),
        ),
        (
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([[0] * 4, [2] * 4, [3] * 4]).float(),
        ),
        (
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([5]),
            torch.tensor(
                [[1] * 4, [2] * 4, [3] * 4, [0] * 4, [0] * 4, [0] * 4]
            ).float(),
        ),
        (
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 0, 5]),
            torch.tensor(
                [[0] * 4, [2] * 4, [3] * 4, [0] * 4, [0] * 4, [0] * 4]
            ).float(),
        ),
    ],
)
def test_tgn_expand_memory(memory, node_event_type_ids, node_event_node_ids, expected):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    assert tgn.expand_memory(memory, node_event_type_ids, node_event_node_ids).equal(
        expected
    )
