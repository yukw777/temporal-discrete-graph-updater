import pytest
import torch

from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, GATv2Conv

from tdgu.nn.dynamic_gnn import DynamicGNN, GNNStack, ZeroPositionalEncoder, GNNLayer


@pytest.mark.parametrize(
    "timestamp_enc_dim,event_embedding_dim,output_dim,transformer_conv_num_block,"
    "transformer_conv_num_heads,batched_graph,num_node",
    [
        (
            16,
            20,
            12,
            1,
            1,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 20),
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 20),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            11,
        ),
        (
            16,
            20,
            12,
            1,
            1,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 20),
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 20),
                edge_last_update=torch.randint(10, (3, 2)),
            ),
            11,
        ),
        (
            32,
            48,
            24,
            6,
            8,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 48),
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 48),
                edge_last_update=torch.randint(10, (3, 2)),
            ),
            11,
        ),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
@pytest.mark.parametrize("zero_timestamp_encoder", [True, False])
@pytest.mark.parametrize("gnn_module", [TransformerConv, GATv2Conv])
def test_dgnn_forward(
    gnn_module,
    zero_timestamp_encoder,
    dropout,
    timestamp_enc_dim,
    event_embedding_dim,
    output_dim,
    transformer_conv_num_block,
    transformer_conv_num_heads,
    batched_graph,
    num_node,
):
    if dropout is None:
        dgnn = DynamicGNN(
            gnn_module,
            timestamp_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
            zero_timestamp_encoder=zero_timestamp_encoder,
        )
    else:
        dgnn = DynamicGNN(
            gnn_module,
            timestamp_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
            dropout=dropout,
            zero_timestamp_encoder=zero_timestamp_encoder,
        )
    node_embeddings = dgnn(batched_graph)
    assert node_embeddings.size() == (num_node, output_dim)


@pytest.mark.parametrize(
    "input_dim,output_dim,edge_dim,heads,num_node,num_edge",
    [(8, 16, 8, 1, 1, 0), (8, 16, 8, 3, 5, 4), (8, 16, 4, 3, 6, 2)],
)
@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("gnn_module", [TransformerConv, GATv2Conv])
def test_gnn_layer_forward(
    gnn_module,
    dropout,
    input_dim,
    output_dim,
    edge_dim,
    heads,
    num_node,
    num_edge,
):
    layer = GNNLayer(gnn_module, input_dim, output_dim, edge_dim, heads, dropout)
    assert layer(
        torch.rand(num_node, input_dim),
        torch.randint(num_node, (2, num_edge)),
        torch.rand(num_edge, edge_dim),
    ).size() == (num_node, heads * output_dim)


@pytest.mark.parametrize(
    "node_dim,edge_dim,output_dim,num_block,heads,num_node,num_edge",
    [(16, 16, 8, 1, 1, 1, 0), (16, 12, 8, 4, 3, 5, 4)],
)
@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("gnn_module", [TransformerConv, GATv2Conv])
def test_gnn_stack_forward(
    gnn_module,
    dropout,
    node_dim,
    edge_dim,
    output_dim,
    num_block,
    heads,
    num_node,
    num_edge,
):
    stack = GNNStack(
        gnn_module, node_dim, edge_dim, output_dim, num_block, heads, dropout
    )
    assert stack(
        torch.rand(num_node, node_dim),
        torch.randint(num_node, (2, num_edge)),
        torch.rand(num_edge, edge_dim),
    ).size() == (num_node, output_dim)


@pytest.mark.parametrize(
    "channels,input_size", [(8, (2,)), (8, (2, 3)), (16, (2,)), (16, (2, 3))]
)
def test_zero_positional_encoder(channels, input_size):
    enc = ZeroPositionalEncoder(channels)
    encoded = enc(torch.rand(*input_size))
    assert encoded.size() == input_size + (channels,)
    assert torch.all(encoded == 0)
