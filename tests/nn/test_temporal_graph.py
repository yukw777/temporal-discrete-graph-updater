import pytest
import torch
import torch.nn as nn

from dgu.nn.temporal_graph import TemporalGraphNetwork, TransformerConvStack
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


class MockTimeEncoder(nn.Module):
    def forward(self, timestamp):
        return timestamp.unsqueeze(-1).expand(-1, 4) + 3


@pytest.mark.parametrize(
    "event_type_ids,event_type_emb,src_ids,dst_ids,event_embeddings,event_timestamps,"
    "memory,edge_index,last_update,src_expected,dst_expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2]),
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
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([1, 2, 3]),
            torch.tensor([3, 2, 1]),
            torch.tensor([[8] * 4, [9] * 4, [10] * 4]).float(),
            torch.tensor([4, 5, 6]),
            torch.linspace(11, 14, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([[4, 5, 1, 2, 3], [3, 7, 3, 2, 1]]),
            torch.tensor([3, 2, 4, 5, 3]).float(),
            torch.tensor(
                [
                    [6] * 4 + [12] * 4 + [14] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [14] * 4 + [12] * 4 + [6] * 4 + [10] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [6] * 4 + [14] * 4 + [12] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [12] * 4 + [14] * 4 + [6] * 4 + [10] * 4,
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
        memory,
        edge_index,
        last_update,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "event_type_dim,memory_dim,time_enc_dim,event_embedding_dim,output_dim,"
    "num_edge_event,prev_num_node,num_node,num_edge,"
    "transformer_conv_num_block,transformer_conv_num_heads",
    [
        (4, 8, 16, 20, 12, 0, 0, 0, 0, 1, 1),
        (4, 8, 16, 20, 12, 0, 1, 1, 0, 1, 1),
        (4, 8, 16, 20, 12, 1, 2, 2, 1, 1, 1),
        (4, 8, 16, 20, 12, 4, 8, 5, 10, 4, 4),
        (8, 16, 32, 48, 24, 10, 10, 8, 10, 6, 6),
        (8, 16, 32, 48, 24, 10, 6, 8, 10, 6, 6),
        (8, 16, 32, 48, 24, 10, 12, 8, 10, 8, 8),
    ],
)
def test_tgn_forward(
    event_type_dim,
    memory_dim,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    num_edge_event,
    prev_num_node,
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
    edge_index = (
        torch.randint(num_node, (2, num_edge)).unique(dim=1)
        if num_node > 0 and num_edge > 1
        else torch.zeros(2, num_edge).long()
    )
    results = tgn(
        torch.randint(len(EVENT_TYPES), (num_edge_event,)),
        torch.randint(prev_num_node, (num_edge_event,))
        if prev_num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.randint(prev_num_node, (num_edge_event,))
        if prev_num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.rand(num_edge_event, event_embedding_dim),
        torch.randint(10, (num_edge_event,)).float(),
        torch.rand(prev_num_node, memory_dim),
        torch.randint(num_node, (prev_num_node,))
        if num_node > 0
        else torch.zeros(prev_num_node).long(),
        torch.randint(2, (prev_num_node,), dtype=torch.bool)
        if num_node > 0
        else torch.zeros(prev_num_node).bool(),
        torch.rand(num_node, event_embedding_dim),
        edge_index,
        torch.rand(edge_index.size(1), event_embedding_dim),
        torch.randint(10, (edge_index.size(1),)).float(),
        torch.randint(10, (edge_index.size(1),)).float(),
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


class MockRNN(nn.Module):
    def forward(self, input, hidden):
        return input + hidden


@pytest.mark.parametrize(
    "memory,delete_node_mask,sorted_node_indices,event_node_ids,agg_msgs,expected",
    [
        (
            torch.tensor([[2] * 4, [3] * 4, [4] * 4]).float(),
            torch.tensor([True, True, True]),
            torch.tensor([0, 3, 1, 2]),
            torch.tensor([0, 2, 3]),
            torch.tensor([[1] * 4, [0] * 4, [3] * 4, [5] * 4]).float(),
            torch.tensor([[3] * 4, [0] * 4, [6] * 4, [9] * 4]).float(),
        ),
        (
            torch.tensor([[2] * 4, [3] * 4, [4] * 4]).float(),
            torch.tensor([True, False, True]),
            torch.tensor([0, 2, 1]),
            torch.tensor([0, 2]),
            torch.tensor([[3] * 4, [0] * 4, [5] * 4]).float(),
            torch.tensor([[5] * 4, [0] * 4, [9] * 4]).float(),
        ),
        (
            torch.tensor([[2] * 4, [3] * 4, [4] * 4]).float(),
            torch.tensor([False, False, True]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 1]),
            torch.tensor([[3] * 4, [5] * 4]).float(),
            torch.tensor([[3] * 4, [5] * 4, [4] * 4]).float(),
        ),
    ],
)
def test_tgn_update_memory(
    memory,
    delete_node_mask,
    sorted_node_indices,
    event_node_ids,
    agg_msgs,
    expected,
):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    tgn.rnn = MockRNN()
    assert tgn.update_memory(
        memory, delete_node_mask, sorted_node_indices, event_node_ids, agg_msgs
    ).equal(expected)


@pytest.mark.parametrize(
    "batch_size,batch,expected",
    [
        (1, torch.empty(0).long(), torch.tensor([0])),
        (1, torch.tensor([0, 0, 0]), torch.tensor([0])),
        (3, torch.tensor([0, 1, 1, 2, 2, 2]), torch.tensor([0, 1, 3])),
        (5, torch.tensor([0, 2, 2, 3, 3, 3]), torch.tensor([0, 1, 1, 3, 6])),
    ],
)
def test_tgn_calculate_node_id_offsets(batch_size, batch, expected):
    node_id_offsets = TemporalGraphNetwork.calculate_node_id_offsets(batch_size, batch)
    assert node_id_offsets.equal(expected)


@pytest.mark.parametrize(
    "batched_graphs,event_type_ids,event_src_ids,event_dst_ids,event_embeddings,"
    "event_timestamps,expected_updated_batched_graph,expected_delete_node_mask,"
    "expected_sorted_node_indices",
    [
        (
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 4),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.zeros(3).long(),
            torch.zeros(3).long(),
            torch.zeros(3, 4),
            torch.zeros(3),
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 4),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.empty(0).bool(),
            torch.empty(0).long(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]).float(),
            torch.tensor([3.0, 5.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor([True, True, True, True]),
            torch.tensor([0, 1, 4, 2, 3]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]),
            torch.tensor([2.0, 3.0]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor([True, True, False, True, True]),
            torch.tensor([0, 1, 2, 3]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4]).float(),
            torch.tensor([3.0, 5.0, 4.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor(
                    [[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]
                ).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor([True, True, True, True]),
            torch.tensor([0, 1, 4, 2, 3, 5]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor(
                    [[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]
                ).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[3] * 4, [8] * 4, [9] * 4]).float(),
            torch.tensor([2.0, 3.0, 4.0]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [2] * 4, [1] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor([True, False, True, True, True, True]),
            torch.tensor([0, 1, 2, 3, 5, 4, 6]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [2] * 4, [1] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[10] * 4, [6] * 4, [11] * 4]).float(),
            torch.tensor([5.0, 3.0, 2.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 4,
                        [7] * 4,
                        [9] * 4,
                    ]
                ).float(),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 4]).float(),
                edge_last_update=torch.tensor([2.0]),
            ),
            torch.tensor([True, True, True, True, True, True, True]),
            torch.tensor([0, 1, 7, 2, 3, 4, 5, 6]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 4,
                        [7] * 4,
                        [9] * 4,
                    ]
                ).float(),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 4]).float(),
                edge_last_update=torch.tensor([2.0]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[12] * 4, [1] * 4, [11] * 4]).float(),
            torch.tensor([3.0, 1.0, 6.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 4]).float(),
                edge_last_update=torch.tensor([3.0]),
            ),
            torch.tensor([True, True, True, True, False, True, True, True]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 4]).float(),
                edge_last_update=torch.tensor([3.0]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([2, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([[12] * 4, [13] * 4, [14] * 4]).float(),
            torch.tensor([2.0, 8.0, 7.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3, 6], [4, 5]]),
                edge_attr=torch.tensor([[13] * 4, [14] * 4]).float(),
                edge_last_update=torch.tensor([8.0, 7.0]),
            ),
            torch.tensor([True, True, True, True, True, True, True]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        ),
    ],
)
def test_tgn_update_batched_graph(
    batched_graphs,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_embeddings,
    event_timestamps,
    expected_updated_batched_graph,
    expected_delete_node_mask,
    expected_sorted_node_indices,
):
    (
        updated_batched_graph,
        delete_node_mask,
        sorted_node_indices,
    ) = TemporalGraphNetwork.update_batched_graph(
        batched_graphs,
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_embeddings,
        event_timestamps,
    )
    assert updated_batched_graph.batch.equal(expected_updated_batched_graph.batch)
    assert updated_batched_graph.x.equal(expected_updated_batched_graph.x)
    assert updated_batched_graph.edge_index.equal(
        expected_updated_batched_graph.edge_index
    )
    assert updated_batched_graph.edge_attr.equal(
        expected_updated_batched_graph.edge_attr
    )
    assert updated_batched_graph.edge_last_update.equal(
        expected_updated_batched_graph.edge_last_update
    )
    assert delete_node_mask.equal(expected_delete_node_mask)
    assert sorted_node_indices.equal(expected_sorted_node_indices)
