import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @staticmethod
    def update_features_helper(
        add_event_type_id: int,
        features: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_obj_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # update node features using node-add event embeddings
        is_add_event = event_type_ids == add_event_type_id
        # (num_event)
        add_event_obj_ids = event_obj_ids[is_add_event]
        # (num_add_event)

        if add_event_obj_ids.numel() == 0:
            # nothing has been added, so nothing to do
            return features.clone()

        # expand the given features if a node/edge with a bigger id has been added
        num_new_objs = int(add_event_obj_ids.max() + 1 - features.size(0))
        if num_new_objs > 0:
            new_features = F.pad(features, (0, 0, 0, num_new_objs))
        else:
            # no need to expand, just copy
            new_features = features.clone()

        # update features
        add_event_embeddings = event_embeddings[is_add_event]
        # (num_add_event, event_embedding_dim)
        new_features[add_event_obj_ids] = add_event_embeddings

        return new_features

    @staticmethod
    def update_node_features(
        node_features: torch.Tensor,
        node_event_type_ids: torch.Tensor,
        node_event_node_ids: torch.Tensor,
        node_event_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update node features using node-add event embeddings.

        node_features: (num_node, event_embedding_dim)
        node_event_type_ids: (num_node_event)
        node_event_node_ids: (num_node_event)
        node_event_embeddings: (num_node_event, event_embedding_dim)

        output: (new_num_node, event_embedding_dim)
        """
        return TemporalGraphNetwork.update_features_helper(
            EVENT_TYPE_ID_MAP["node-add"],
            node_features,
            node_event_type_ids,
            node_event_node_ids,
            node_event_embeddings,
        )

    @staticmethod
    def update_edge_features(
        edge_features: torch.Tensor,
        edge_event_type_ids: torch.Tensor,
        edge_event_edge_ids: torch.Tensor,
        edge_event_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update edge features using edge-add event embeddings.

        edge_features: (num_edge, event_embedding_dim)
        event_type_ids: (num_edge_event)
        event_edge_ids: (num_edge_event)
        event_embeddings: (num_edge_event, event_embedding_dim)

        output: (new_num_edge, event_embedding_dim)
        """
        return TemporalGraphNetwork.update_features_helper(
            EVENT_TYPE_ID_MAP["edge-add"],
            edge_features,
            edge_event_type_ids,
            edge_event_edge_ids,
            edge_event_embeddings,
        )

    @staticmethod
    def expand_last_update(
        edge_last_update: torch.Tensor,
        edge_event_type_ids: torch.Tensor,
        edge_event_edge_ids: torch.Tensor,
        edge_event_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand last update timestamps for edges with last updates for new edges added.
        The last updates for new edges are initialized to be the given event timestamps
        so that the relative timestamps would be 0.

        edge_last_update: (num_edge)
        edge_event_type_ids: (num_event)
        edge_event_edge_ids: (num_event)
        edge_event_timestamps: (num_event)

        output: (new_num_edge)
        """
        is_edge_add_event = edge_event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
        # (num_event)
        edge_add_event_edge_ids = edge_event_edge_ids[is_edge_add_event]
        # (num_edge_add_event)

        if edge_add_event_edge_ids.numel() == 0:
            # nothing has been added, so no need to expand
            expanded_last_update = edge_last_update.clone()
        else:
            num_new_edges = int(
                edge_add_event_edge_ids.max() + 1 - edge_last_update.size(0)
            )
            if num_new_edges > 0:
                expanded_last_update = F.pad(edge_last_update, (0, num_new_edges))
            else:
                # no need to expand
                expanded_last_update = edge_last_update.clone()
        # (new_num_edge)

        # set the last updates of the new edges as their event timestamps
        expanded_last_update[edge_add_event_edge_ids] = edge_event_timestamps[
            is_edge_add_event
        ]

        return expanded_last_update

    @staticmethod
    def update_last_update(
        edge_last_update: torch.Tensor,
        edge_event_type_ids: torch.Tensor,
        edge_event_edge_ids: torch.Tensor,
        edge_event_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update last update timestamps for edges with the given graph event.
        We assume that all the newly added edges already have up-to-date last updates
        from expand_last_update(), so we don't touch them.

        edge_last_update: (num_edge)
        edge_event_type_ids: (num_event)
        edge_event_edge_ids: (num_event)
        edge_event_timestamps: (num_event)

        output: (num_edge)
        """
        is_edge_delete_event = edge_event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
        # (num_edge_delete_event)
        edge_delete_event_edge_ids = edge_event_edge_ids[is_edge_delete_event]
        # (num_edge_add_event)
        updated_last_update = edge_last_update.clone()
        updated_last_update[edge_delete_event_edge_ids] = edge_event_timestamps[
            is_edge_delete_event
        ]

        return updated_last_update

    def expand_memory(
        self,
        memory: torch.Tensor,
        node_event_type_ids: torch.Tensor,
        node_event_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand memory for newly added nodes and initialize to be 0.

        memory: (num_node, memory_dim)
        node_event_type_ids: (num_node_event)
        node_event_node_ids: (num_node_event)

        output: (new_num_node, memory_dim)
        """
        is_node_add_event = node_event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (num_event)
        node_add_event_node_ids = node_event_node_ids[is_node_add_event]
        # (num_node_add_event)

        if node_add_event_node_ids.numel() == 0:
            # nothing has been added, so no need to expand
            expanded_memory = memory.clone()
        else:
            num_new_nodes = int(node_add_event_node_ids.max() + 1 - memory.size(0))
            if num_new_nodes > 0:
                expanded_memory = F.pad(memory, (0, 0, 0, num_new_nodes))
            else:
                # no need to expand
                expanded_memory = memory.clone()
        # (new_num_edge)

        # set the memory of new nodes to 0
        expanded_memory.index_fill_(0, node_add_event_node_ids, 0)

        return expanded_memory
