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
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[1, 1, 3]]),
            torch.tensor([[1, 1, 2]]),
            torch.tensor([[[1] * 4, [2] * 4, [3] * 4]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[1, 2, 3]]).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ]
                ]
            ).float(),
            torch.zeros(1, 3, 20),
        ),
        (
            5,
            7,
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[1, 1, 3]]),
            torch.tensor([[1, 2, 2]]),
            torch.tensor([[[1] * 4, [2] * 4, [3] * 4]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[1, 2, 3]]).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [2] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ]
                ]
            ).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [2] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ]
                ]
            ).float(),
        ),
        (
            10,
            20,
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 3, 0]]),
            torch.tensor([[0, 0, 0, 3, 1, 0]]),
            torch.tensor([[[i + 1] * 4 for i in range(6)]]),
            torch.tensor([[0, 0, 0, 2, 4, 0]]),
            torch.tensor(list(range(1, 7))).unsqueeze(0).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                        [4] * 4 + [1] * 4 + [0] * 4 + [6] * 4 + [3] * 4,
                        [5] * 4 + [2] * 4 + [3] * 4 + [3] * 4 + [4] * 4,
                        [6] * 4 + [3] * 4 + [1] * 4 + [0] * 4 + [5] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ]
                ]
            ).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [3] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                        [6] * 4 + [1] * 4 + [3] * 4 + [0] * 4 + [5] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ]
                ]
            ).float(),
        ),
        (
            10,
            20,
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 3, 0, 0, 0], [0, 4, 5, 6, 7, 8, 9, 0]]),
            torch.tensor([[0, 0, 0, 3, 1, 0, 0, 0], [0, 0, 0, 7, 5, 0, 6, 0]]),
            torch.tensor(
                [[[i + 1] * 4 for i in range(8)], [[i + 2] * 4 for i in range(8)]]
            ),
            torch.tensor([[0, 0, 0, 2, 4, 0, 0, 0], [0, 0, 0, 5, 6, 0, 7, 0]]),
            torch.tensor([list(range(1, 9)), list(range(5, 13))]),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                        [4] * 4 + [1] * 4 + [0] * 4 + [6] * 4 + [3] * 4,
                        [5] * 4 + [2] * 4 + [3] * 4 + [3] * 4 + [4] * 4,
                        [6] * 4 + [3] * 4 + [1] * 4 + [0] * 4 + [5] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ],
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [3] * 4 + [0] * 4 + [0] * 4 + [9] * 4 + [3] * 4,
                        [4] * 4 + [5] * 4 + [0] * 4 + [10] * 4 + [4] * 4,
                        [5] * 4 + [6] * 4 + [7] * 4 + [1] * 4 + [5] * 4,
                        [6] * 4 + [7] * 4 + [5] * 4 + [0] * 4 + [6] * 4,
                        [3] * 4 + [0] * 4 + [0] * 4 + [13] * 4 + [7] * 4,
                        [5] * 4 + [9] * 4 + [6] * 4 + [0] * 4 + [8] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ],
                ]
            ).float(),
            torch.tensor(
                [
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [3] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                        [6] * 4 + [1] * 4 + [3] * 4 + [0] * 4 + [5] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ],
                    [
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [7] * 4 + [6] * 4 + [1] * 4 + [5] * 4,
                        [6] * 4 + [5] * 4 + [7] * 4 + [0] * 4 + [6] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                        [5] * 4 + [6] * 4 + [9] * 4 + [0] * 4 + [8] * 4,
                        [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    ],
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
    tgn = TemporalGraphNetwork(num_nodes, num_edges, 4, 4, 4, 4, 8)
    for i in range(num_nodes):
        tgn.memory[i] = torch.tensor([i] * 4).float()
    for i in range(num_edges):
        tgn.last_update[i] = i * 2

    class MockTimeEncoder(nn.Module):
        def forward(self, timestamp):
            return timestamp.unsqueeze(-1).expand(-1, -1, 4) + 3

    tgn.time_encoder = MockTimeEncoder()

    tgn.event_type_emb = nn.Embedding.from_pretrained(
        torch.tensor(list(range(len(EVENT_TYPES)))).unsqueeze(-1).expand(-1, 4).float()
    )

    _, src_mask, dst_mask = compute_masks_from_event_type_ids(event_type_ids)
    src_messages, dst_messages = tgn.message(
        event_type_ids,
        src_ids,
        src_mask,
        dst_ids,
        dst_mask,
        event_embeddings,
        event_edge_ids,
        event_timestamps,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "messages,ids,expected",
    [
        (torch.ones(1, 1, 4), torch.tensor([0]), torch.ones(1, 1, 4)),
        (
            torch.cat(
                [torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]
            ).unsqueeze(0),
            torch.tensor([[0, 1, 1]]),
            torch.cat(
                [torch.ones(1, 4), torch.tensor([[0.5, 0.5, 0.5, 0.5]])]
            ).unsqueeze(0),
        ),
        (
            torch.stack(
                [
                    torch.cat([torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
                    torch.cat([torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
                ]
            ),
            torch.tensor([[0, 1, 1], [0, 1, 0]]),
            torch.stack(
                [
                    torch.cat([torch.ones(1, 4), torch.tensor([[0.5, 0.5, 0.5, 0.5]])]),
                    torch.cat(
                        [torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.zeros(1, 4)]
                    ),
                ]
            ),
        ),
    ],
)
def test_tgn_agg_message(messages, ids, expected):
    tgn = TemporalGraphNetwork(10, 10, 4, 4, 4, 4, 8)
    assert tgn.agg_message(messages, ids).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,src_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.zeros(5, 16),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-add"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1)
            .expand(-1, -1, 16),
            torch.tensor([[1] * 16, [0] * 16, [0] * 16, [3] * 16, [7] * 16]).float(),
        ),
    ],
)
def test_tgn_update_node_features(event_type_ids, src_ids, event_embeddings, expected):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10)
    tgn.update_node_features(event_type_ids, src_ids, event_embeddings)
    assert tgn.node_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.zeros(5, 16),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1)
            .expand(-1, -1, 16),
            torch.tensor([[0] * 16, [0] * 16, [3] * 16, [5] * 16, [0] * 16]).float(),
        ),
    ],
)
def test_tgn_update_edge_features(
    event_type_ids, event_edge_ids, event_embeddings, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10)
    tgn.update_edge_features(event_type_ids, event_edge_ids, event_embeddings)
    assert tgn.edge_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_timestamps,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]).unsqueeze(-1),
            torch.zeros(5),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1),
            torch.tensor([0, 0, 0, 2, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1),
            torch.tensor([0, 0, 4, 8, 6]).float(),
        ),
    ],
)
def test_tgn_update_last_update(
    event_type_ids, event_edge_ids, event_timestamps, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10)
    tgn.update_last_update(event_type_ids, event_edge_ids, event_timestamps)
    assert tgn.last_update.equal(expected)


@pytest.mark.parametrize(
    "event_type_dim,memory_dim,time_enc_dim,event_embedding_dim,output_dim,"
    "batch,event_seq_len,num_node,num_edge",
    [
        (4, 8, 16, 20, 12, 1, 0, 0, 0),
        (4, 8, 16, 20, 12, 1, 3, 4, 5),
        (4, 8, 16, 20, 12, 8, 5, 10, 20),
        (8, 16, 32, 48, 24, 16, 10, 20, 40),
    ],
)
def test_tgn_forward(
    event_type_dim,
    memory_dim,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    batch,
    event_seq_len,
    num_node,
    num_edge,
):
    tgn = TemporalGraphNetwork(
        100,
        200,
        event_type_dim,
        memory_dim,
        time_enc_dim,
        event_embedding_dim,
        output_dim,
    )
    assert (
        tgn(
            torch.randint(len(EVENT_TYPES), (batch, event_seq_len)),
            torch.randint(num_node, (batch, event_seq_len))
            if num_node > 0
            else torch.zeros((batch, event_seq_len)).long(),
            torch.randint(2, (batch, event_seq_len)).float(),
            torch.randint(num_node, (batch, event_seq_len))
            if num_node > 0
            else torch.zeros(batch, event_seq_len).long(),
            torch.randint(2, (batch, event_seq_len)).float(),
            torch.randint(num_edge, (batch, event_seq_len))
            if num_edge > 0
            else torch.zeros(batch, event_seq_len).long(),
            torch.rand(batch, event_seq_len, event_embedding_dim),
            torch.randint(10, (batch, event_seq_len)).float(),
            torch.randint(num_node, (batch, num_node))
            if num_node > 0
            else torch.zeros(batch, num_node).long(),
            torch.randint(num_edge, (batch, num_edge))
            if num_edge > 0
            else torch.zeros(batch, num_edge).long(),
            torch.randint(num_node, (batch, 2, num_edge))
            if num_node > 0
            else torch.zeros(batch, 2, num_edge).long(),
            torch.randint(10, (batch, num_edge)).float(),
        ).size()
        == (batch, num_node, output_dim)
    )


@pytest.mark.parametrize(
    "edge_index,node_ids,expected",
    [
        (
            torch.tensor([[[0], [0]]]),
            torch.tensor([[0]]),
            torch.tensor([[[0], [0]]]),
        ),
        (
            torch.tensor([[[0, 1], [0, 2]]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[[0, 1], [0, 2]]]),
        ),
        (
            torch.tensor([[[0, 1], [0, 2]]]),
            torch.tensor([[0, 2, 1]]),
            torch.tensor([[[0, 2], [0, 1]]]),
        ),
        (
            torch.tensor(
                [[[0, 5, 7, 5], [0, 9, 10, 4]], [[0, 8, 3, 8], [0, 6, 13, 12]]]
            ),
            torch.tensor([[0, 4, 5, 7, 9, 10], [0, 3, 6, 8, 12, 13]]),
            torch.tensor(
                [[[0, 2, 3, 2], [0, 4, 5, 1]], [[0, 9, 7, 9], [0, 8, 11, 10]]]
            ),
        ),
    ],
)
def test_localize_edge_index(edge_index, node_ids, expected):
    assert TemporalGraphNetwork.localize_edge_index(edge_index, node_ids).equal(
        expected
    )
