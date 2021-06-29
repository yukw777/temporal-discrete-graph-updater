import pytest
import torch
import torch.nn as nn

from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.nn.utils import compute_masks_from_event_type_ids
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


@pytest.mark.parametrize(
    "num_nodes,num_edges,event_type_ids,src_ids,dst_ids,event_embeddings,"
    "event_edge_ids,event_timestamps,src_expected,dst_expected",
    [
        (
            5,
            7,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([1, 1, 3]),
            torch.tensor([1, 1, 2]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 2, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.zeros(3, 20),
        ),
        (
            5,
            7,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([1, 1, 3]),
            torch.tensor([1, 2, 2]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 2, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [2] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [2] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            10,
            20,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0, 0, 0]),
            torch.tensor([0, 0, 0, 3, 1, 0, 0, 0]),
            torch.tensor([[i + 1] * 4 for i in range(8)]),
            torch.tensor([0, 0, 0, 2, 4, 0, 0, 0]),
            torch.linspace(1, 8, 8).long(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                    [4] * 4 + [1] * 4 + [0] * 4 + [6] * 4 + [3] * 4,
                    [5] * 4 + [2] * 4 + [3] * 4 + [3] * 4 + [4] * 4,
                    [6] * 4 + [3] * 4 + [1] * 4 + [0] * 4 + [5] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [3] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                    [6] * 4 + [1] * 4 + [3] * 4 + [0] * 4 + [5] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_message(
    num_nodes,
    num_edges,
    event_type_ids,
    src_ids,
    dst_ids,
    event_embeddings,
    event_edge_ids,
    event_timestamps,
    src_expected,
    dst_expected,
):
    tgn = TemporalGraphNetwork(num_nodes, num_edges, 4)
    for i in range(num_nodes):
        tgn.memory[i] = torch.tensor([i] * 4).float()
    for i in range(num_edges):
        tgn.last_update[i] = i * 2

    class MockTimeEncoder(nn.Module):
        def forward(self, timestamp):
            return timestamp.unsqueeze(-1).expand(-1, 4) + 3

    tgn.time_encoder = MockTimeEncoder()

    event_mask, src_mask, dst_mask = compute_masks_from_event_type_ids(event_type_ids)
    src_messages, dst_messages = tgn.message(
        event_type_ids,
        src_ids,
        src_mask,
        dst_ids,
        dst_mask,
        event_embeddings,
        event_mask,
        event_edge_ids,
        event_timestamps,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "messages,ids,expected",
    [
        (torch.ones(1, 4), torch.tensor([0]), torch.ones(1, 4)),
        (
            torch.cat([torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
            torch.tensor([0, 1, 1]),
            torch.cat([torch.ones(1, 4), torch.tensor([[0.5, 0.5, 0.5, 0.5]])]),
        ),
    ],
)
def test_tgn_agg_message(messages, ids, expected):
    tgn = TemporalGraphNetwork(10, 10, 4)
    assert tgn.agg_message(messages, ids).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,src_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([1, 2, 3]).long(),
            torch.linspace(0, 2, 3).unsqueeze(-1).expand(-1, 10),
            torch.zeros(5, 10),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 1, 0]).long(),
            torch.linspace(0, 2, 3).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[0] * 10, [1] * 10, [0] * 10, [0] * 10, [0] * 10]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([1, 1]).long(),
            torch.linspace(0, 1, 2).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[0] * 10, [1] * 10, [0] * 10, [0] * 10, [0] * 10]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0, 0, 0]),
            torch.linspace(0, 7, 8).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[1] * 10, [0] * 10, [0] * 10, [0] * 10, [0] * 10]).float(),
        ),
    ],
)
def test_tgn_update_node_features(event_type_ids, src_ids, event_embeddings, expected):
    tgn = TemporalGraphNetwork(5, 5, 10)
    tgn.update_node_features(event_type_ids, src_ids, event_embeddings)
    assert tgn.node_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([1, 2, 3]).long(),
            torch.linspace(0, 2, 3).unsqueeze(-1).expand(-1, 10),
            torch.zeros(5, 10),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 1, 0]).long(),
            torch.linspace(0, 2, 3).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[0] * 10, [1] * 10, [0] * 10, [0] * 10, [0] * 10]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([1, 1]).long(),
            torch.linspace(0, 1, 2).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[0] * 10, [1] * 10, [0] * 10, [0] * 10, [0] * 10]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0, 0, 0]),
            torch.linspace(0, 7, 8).unsqueeze(-1).expand(-1, 10),
            torch.tensor([[0] * 10, [0] * 10, [3] * 10, [0] * 10, [0] * 10]).float(),
        ),
    ],
)
def test_tgn_update_edge_features(
    event_type_ids, event_edge_ids, event_embeddings, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10)
    tgn.update_edge_features(event_type_ids, event_edge_ids, event_embeddings)
    assert tgn.edge_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_timestamps,expected",
    [
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 0, 0]).long(),
            torch.linspace(0, 3, 4),
            torch.zeros(5),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([1, 2, 3]).long(),
            torch.linspace(0, 2, 3),
            torch.tensor([0, 0, 0, 2, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 1, 0]).long(),
            torch.linspace(0, 2, 3),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([1, 1]).long(),
            torch.linspace(0, 1, 2),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 0, 0, 2, 3, 0, 0, 0]),
            torch.linspace(0, 7, 8),
            torch.tensor([0, 0, 3, 4, 0]).float(),
        ),
    ],
)
def test_tgn_update_last_update(
    event_type_ids, event_edge_ids, event_timestamps, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10)
    tgn.update_last_update(event_type_ids, event_edge_ids, event_timestamps)
    assert tgn.last_update.equal(expected)


@pytest.mark.parametrize(
    "event_seq_len,hidden_dim,num_node,num_edge",
    [
        (3, 5, 4, 8),
        (12, 64, 20, 40),
    ],
)
def test_tgn_forward(event_seq_len, hidden_dim, num_node, num_edge):
    tgn = TemporalGraphNetwork(num_node * 2, num_edge * 2, hidden_dim)
    assert (
        tgn(
            torch.randint(len(EVENT_TYPES), (event_seq_len,)),
            torch.randint(num_node * 2, (event_seq_len,)),
            torch.randint(2, (event_seq_len,)).float(),
            torch.randint(num_node * 2, (event_seq_len,)),
            torch.randint(2, (event_seq_len,)).float(),
            torch.randint(num_edge * 2, (event_seq_len,)),
            torch.rand(event_seq_len, hidden_dim),
            torch.randint(2, (event_seq_len,)).float(),
            torch.randint(10, (event_seq_len,)).float(),
            torch.randint(num_node, (num_node,)),
            torch.randint(num_edge, (num_edge,)),
            torch.randint(num_node, (2, num_edge)),
            torch.tensor(12),
        ).size()
        == (num_node, hidden_dim)
    )
