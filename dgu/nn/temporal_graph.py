import torch
import torch.nn as nn

from typing import Tuple, Optional
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_scatter import scatter

from dgu.constants import EVENT_TYPE_ID_MAP


class TransformerConvStack(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int,
        num_block: int,
        heads: int = 1,
        edge_dim: Optional[int] = None,
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


class TemporalGraphNetwork(nn.Module):
    def __init__(
        self,
        event_type_emb_dim: int,
        memory_dim: int,
        time_enc_dim: int,
        event_embedding_dim: int,
        output_dim: int,
        transformer_conv_num_block: int,
        transformer_conv_num_heads: int,
    ) -> None:
        super().__init__()
        self.event_type_emb_dim = event_type_emb_dim
        self.memory_dim = memory_dim
        self.time_enc_dim = time_enc_dim
        self.event_embedding_dim = event_embedding_dim
        self.output_dim = output_dim
        self.message_dim = (
            event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        )

        # event type embedding
        self.event_type_emb = nn.Embedding(len(EVENT_TYPE_ID_MAP), event_type_emb_dim)

        # time encoder
        self.time_encoder = TimeEncoder(time_enc_dim)

        # RNN to update memories
        self.rnn = nn.GRUCell(
            event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim,
            memory_dim,
        )

        # TransformerConvStack for the final node embeddings
        self.gnn = TransformerConvStack(
            event_embedding_dim + memory_dim,
            output_dim,
            transformer_conv_num_block,
            heads=transformer_conv_num_heads,
            edge_dim=time_enc_dim + event_embedding_dim,
        )

    def forward(
        self,
        node_event_type_ids: torch.Tensor,
        node_event_node_ids: torch.Tensor,
        node_event_embeddings: torch.Tensor,
        node_event_timestamps: torch.Tensor,
        node_event_mask: torch.Tensor,
        edge_event_type_ids: torch.Tensor,
        edge_event_src_ids: torch.Tensor,
        edge_event_dst_ids: torch.Tensor,
        edge_event_edge_ids: torch.Tensor,
        edge_event_embeddings: torch.Tensor,
        edge_event_timestamps: torch.Tensor,
        edge_event_mask: torch.Tensor,
        memory: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_timestamps: torch.Tensor,
        edge_last_update: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the updated node embeddings based on the given events. Node
        embeddings are calculated for nodes specified by node_ids.

        node_event_type_ids: (num_node_event)
        node_event_node_ids: (num_node_event)
        node_event_embeddings: (num_node_event, event_embedding_dim)
        node_event_timestamps: (num_node_event)
        node_event_mask: (num_node_event)
        edge_event_type_ids: (num_edge_event)
        edge_event_src_ids: (num_edge_event)
        edge_event_dst_ids: (num_edge_event)
        edge_event_edge_ids: (num_edge_event)
        edge_event_embeddings: (num_edge_event, event_embedding_dim)
        edge_event_timestamps: (num_edge_event)
        edge_event_mask: (num_edge_event)
        memory: (num_node, memory_dim)
            Includes zeroed out memories for new nodes that were
            added by the given graph events.
        node_features: (num_node, event_embedding_dim)
            These are node features after the given graph events.
        edge_index: (2, num_edge)
            These are edge indices after the given graph events.
        edge_features: (num_edge, event_embedding_dim)
            These are edge features after the given graph events.
        edge_timestamps: (num_edge) or scalar
        edge_last_update: (num_edge)
            These are last edge update timestamps before the given graph events.

        output: (num_node, output_dim),
        """
        # calculate messages
        node_msgs = self.node_message(
            node_event_type_ids,
            node_event_node_ids,
            node_event_embeddings,
            node_event_timestamps,
            node_event_mask,
            memory,
        )
        # (batch, message_dim)
        src_msgs, dst_msgs = self.edge_message(
            edge_event_type_ids,
            edge_event_src_ids,
            edge_event_dst_ids,
            edge_event_edge_ids,
            edge_event_embeddings,
            edge_event_timestamps,
            edge_event_mask,
            memory,
            edge_last_update,
        )
        # src_msgs: (batch, message_dim)
        # dst_msgs: (batch, message_dim)

        # aggregate messages
        event_node_ids = torch.cat(
            [node_event_node_ids, edge_event_src_ids, edge_event_dst_ids]
        )
        agg_msgs = scatter(
            torch.cat([node_msgs, src_msgs, dst_msgs]), event_node_ids, dim=0
        )
        # (max_event_node_id, message_dim)

        # update the memories
        unique_event_node_ids = event_node_ids.unique()
        memory[unique_event_node_ids] = self.rnn(
            agg_msgs[unique_event_node_ids], memory[unique_event_node_ids]
        )

        # calculate the node embeddings
        if node_features.size(0) != 0:
            rel_t = edge_timestamps - edge_last_update
            # (num_edge)
            rel_t_embs = self.time_encoder(rel_t)
            # (num_edge, time_enc_dim)
            x = torch.cat([node_features, memory], dim=-1)
            # (num_node, event_embedding_dim + memory_dim)
            edge_attr = torch.cat([rel_t_embs, edge_features], dim=-1)
            # (num_node, time_enc_dim + event_embedding_dim)
            node_embeddings = self.gnn(x, edge_index, edge_attr=edge_attr)
            # (num_node, output_dim)
        else:
            # no nodes, so no node embeddings either
            node_embeddings = torch.zeros(
                0, self.output_dim, device=node_features.device
            )
            # (0, output_dim)

        return node_embeddings
        # (batch, num_node, output_dim)

    def node_message(
        self,
        event_type_ids: torch.Tensor,
        node_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
        event_mask: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate node event messages. We concatenate the event type,
        node memory, time embedding and event embedding. Note that node messages
        created here do contain placeholder zeros for the destination node in
        order to keep the same dimension as edge messages.

        Special events like pad, start, end are masked out to be zeros, which means
        these will go away when messages are aggregated.

        event_type_ids: (batch)
        node_ids: (batch)
        event_embeddings: (batch event_embedding_dim)
        event_timestamps: (batch)
        event_mask: (batch)
        memory: (num_node, memory_dim)

        output: (batch, message_dim)
        """
        event_type_embs = self.event_type_emb(event_type_ids)
        # (batch, event_type_emb_dim)

        node_memory = memory[node_ids]
        # (batch, memory_dim)

        timestamp_emb = self.time_encoder(event_timestamps)
        # (batch, time_enc_dim)

        return (
            torch.cat(
                [
                    event_type_embs,
                    node_memory,
                    torch.zeros_like(node_memory),
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=-1,
            )
            # mask out special events
            * event_mask.unsqueeze(-1)
        )
        # (batch, message_dim)

    def edge_message(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
        event_mask: torch.Tensor,
        memory: torch.Tensor,
        last_update: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate edge event messages. We concatenate the event type,
        source and destination memories, relative time embedding and event embedding.

        Special events like pad, start, end are masked out.

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        event_edge_ids: (batch)
        event_embeddings: (batch event_embedding_dim)
        event_timestamps: (batch)
        event_mask: (batch)
        memory: (num_node, memory_dim)
        last_update: (num_edge)

        output:
            src_messages: (batch, message_dim)
            dst_messages: (batch, message_dim)
        """
        event_type_embs = self.event_type_emb(event_type_ids)
        # (batch, event_type_emb_dim)

        src_memory = memory[src_ids]
        # (batch, memory_dim)
        dst_memory = memory[dst_ids]
        # (batch, memory_dim)

        # calculate relative timestamps for edge events
        edge_last_update = last_update[event_edge_ids]
        # (batch)

        rel_edge_timestamps = event_timestamps - edge_last_update
        # (batch)

        timestamp_emb = self.time_encoder(rel_edge_timestamps)
        # (batch, time_enc_dim)

        src_messages = (
            torch.cat(
                [
                    event_type_embs,
                    src_memory,
                    dst_memory,
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=-1,
            )
            # mask out special events
            * event_mask.unsqueeze(-1)
        )
        # (batch, message_dim)
        dst_messages = (
            torch.cat(
                [
                    event_type_embs,
                    dst_memory,
                    src_memory,
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=-1,
            )
            # mask out special events
            * event_mask.unsqueeze(-1)
        )
        # (batch, message_dim)
        return src_messages, dst_messages

    def update_node_features(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> None:
        """
        Update node features using node-add event embeddings.

        event_type_ids: (batch, event_seq_len)
        src_ids: (batch, event_seq_len)
        event_embeddings: (batch, event_seq_len, event_embedding_dim)
        """
        # update node features using node-add event embeddings
        is_node_add = event_type_ids.flatten() == EVENT_TYPE_ID_MAP["node-add"]
        # (batch * event_seq_len)
        node_add_src_ids = src_ids.flatten()[is_node_add]
        # (num_node_add)

        # Duplicates are not possible because we generate a new ID for
        # every added node.
        node_add_event_embeddings = event_embeddings.flatten(end_dim=1)[is_node_add]
        # (num_node_add, event_embedding_dim)
        self.node_features[node_add_src_ids] = node_add_event_embeddings  # type: ignore

    def update_edge_features(
        self,
        event_type_ids: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> None:
        """
        Update edge features using edge-add event embeddings.

        event_type_ids: (batch, event_seq_len)
        event_edge_ids: (batch, event_seq_len)
        event_embeddings: (batch, event_seq_len, event_embedding_dim)
        """
        # update edge features using edge-add event embeddings
        is_edge_add = event_type_ids.flatten() == EVENT_TYPE_ID_MAP["edge-add"]
        # (batch * event_seq_len)
        edge_add_edge_ids = event_edge_ids.flatten()[is_edge_add]
        # (num_edge_add)

        # Duplicates are not possible because we generate a new ID for
        # every added edge.
        edge_add_event_embeddings = event_embeddings.flatten(end_dim=1)[is_edge_add]
        # (num_edge_add, event_embedding_dim)
        self.edge_features[  # type: ignore
            edge_add_edge_ids
        ] = edge_add_event_embeddings

    def update_last_update(
        self,
        event_type_ids: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> None:
        """
        Update last update timestamps for edges.

        event_type_ids: (batch, event_seq_len)
        event_edge_ids: (batch, event_seq_len)
        event_timestamps: (batch, event_seq_len)
        """
        # update last update timestamps using edge events
        flat_event_type_ids = event_type_ids.flatten()
        is_edge_event = torch.logical_or(
            flat_event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
            flat_event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
        )
        # (batch * event_seq_len)

        # Duplicates are possible here but it's OK, b/c PyTorch
        # automatically assigns the latest last update value in the given batch.
        edge_ids = event_edge_ids.flatten()[is_edge_event]
        # (num_edge_event)
        edge_timestamps = event_timestamps.flatten()[is_edge_event]
        # (num_edge_event)
        self.last_update[edge_ids] = edge_timestamps  # type: ignore
