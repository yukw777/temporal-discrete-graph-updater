import torch
import torch.nn as nn

from typing import Optional
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Batch

from dgu.nn.utils import PositionalEncoder


class TransformerConvStack(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int,
        num_block: int,
        heads: int = 1,
        edge_dim: Optional[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stack = nn.ModuleList(
            [
                TransformerConv(node_dim, output_dim, edge_dim=edge_dim, heads=heads)
                if i == 0
                else TransformerConv(
                    node_dim + heads * output_dim,
                    output_dim,
                    edge_dim=edge_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for i in range(num_block)
            ]
        )
        self.linear = nn.Linear(heads * output_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (num_node, node_dim)
        edge_index: (2, num_edge)
        edge_attr: (num_edge, edge_dim)

        output: (num_node, output_dim)
        """
        for i, gnn in enumerate(self.stack):
            if i == 0:
                node_embeddings = gnn(x, edge_index, edge_attr=edge_attr)
                # (num_node, heads * output_dim)
            else:
                node_embeddings = gnn(
                    torch.cat([node_embeddings, x], dim=-1),
                    edge_index,
                    edge_attr=edge_attr,
                )
                # (num_node, heads * output_dim)
        return self.linear(node_embeddings)
        # (num_node, output_dim)


class ZeroPositionalEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Simply return zero encodings. Useful for ablation studies.

        positions: (*)

        output: (*, channels)
        """
        return torch.zeros(*positions.size(), self.channels, device=positions.device)


class DynamicGNN(nn.Module):
    def __init__(
        self,
        timestamp_enc_dim: int,
        event_embedding_dim: int,
        output_dim: int,
        transformer_conv_num_block: int,
        transformer_conv_num_heads: int,
        dropout: float = 0.3,
        zero_timestamp_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.timestamp_enc_dim = timestamp_enc_dim
        self.event_embedding_dim = event_embedding_dim
        self.output_dim = output_dim

        # timestamp encoder
        self.timestamp_encoder: nn.Module
        if zero_timestamp_encoder:
            self.timestamp_encoder = ZeroPositionalEncoder(timestamp_enc_dim // 2)
        else:
            self.timestamp_encoder = PositionalEncoder(timestamp_enc_dim // 2, 128)

        # TransformerConvStack for the final node embeddings
        self.gnn = TransformerConvStack(
            event_embedding_dim + timestamp_enc_dim,
            output_dim,
            transformer_conv_num_block,
            heads=transformer_conv_num_heads,
            edge_dim=event_embedding_dim + timestamp_enc_dim,
            dropout=dropout,
        )

    def forward(self, batched_graph: Batch) -> torch.Tensor:
        """
        Calculate the node embeddings for the given batched graph.

        batched_graph: diagonally stacked graph, Batch(
            batch: (prev_num_node)
            x: (prev_num_node, event_embedding_dim)
            node_last_update: (prev_num_node, 2)
            edge_index: (2, prev_num_edge)
            edge_attr: (prev_num_edge, event_embedding_dim)
            edge_last_update: (prev_num_edge, 2)
        )

        output: node_embeddings, (num_node, output_dim)
        """
        x = torch.cat(
            [
                batched_graph.x,
                self.timestamp_encoder(batched_graph.node_last_update).view(
                    batched_graph.num_nodes,
                    -1,
                ),
            ],
            dim=-1,
        )
        # (num_node, event_embedding_dim + time_enc_dim)
        if batched_graph.edge_index.size(1) > 0:
            edge_attr = torch.cat(
                [
                    batched_graph.edge_attr,
                    self.timestamp_encoder(batched_graph.edge_last_update).view(
                        batched_graph.num_edges, -1
                    ),
                ],
                dim=-1,
            )
            # (num_edge, time_enc_dim + event_embedding_dim)
        else:
            edge_attr = torch.empty(
                0, self.event_embedding_dim + self.timestamp_enc_dim, device=x.device
            )
        return self.gnn(x, batched_graph.edge_index, edge_attr=edge_attr)
        # (num_node, output_dim)
