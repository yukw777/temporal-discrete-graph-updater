import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Dict
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.data import Batch
from torch_scatter import scatter

from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.nn.utils import index_edge_attr, get_edge_index_co_occurrence_matrix


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
        edge_event_type_ids: torch.Tensor,
        edge_event_src_ids: torch.Tensor,
        edge_event_dst_ids: torch.Tensor,
        edge_event_embeddings: torch.Tensor,
        edge_event_timestamps: torch.Tensor,
        memory: torch.Tensor,
        node_memory_update_index: torch.Tensor,
        node_memory_update_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_timestamps: torch.Tensor,
        edge_last_update: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the updated node embeddings based on the given events.

        edge_event_type_ids: (num_edge_event)
        edge_event_src_ids: (num_edge_event)
        edge_event_dst_ids: (num_edge_event)
        edge_event_embeddings: (num_edge_event, event_embedding_dim)
        edge_event_timestamps: (num_edge_event)
        memory: (prev_num_node, memory_dim)
            Node memories before the given graph events.
        node_memory_update_index: (prev_num_node)
            New indices for nodes in the previous memory.
        node_memory_update_mask: (prev_num_node)
            Mask for nodes that have not been deleted in the previous memory.
        node_features: (num_node, event_embedding_dim)
            Node features after the given graph events.
        edge_index: (2, num_edge)
            Edge indices after the given graph events.
        edge_features: (num_edge, event_embedding_dim)
            Edge features after the given graph events.
        edge_timestamps: (num_edge) or scalar
        edge_last_update: (num_edge)
            Last edge update timestamps before the given graph events.

        output: {
            "node_embeddings": (num_node, output_dim),
            "updated_memory": (num_node, memory_dim),
        }
        """
        src_msgs, dst_msgs = self.edge_message(
            edge_event_type_ids,
            edge_event_src_ids,
            edge_event_dst_ids,
            edge_event_embeddings,
            edge_event_timestamps,
            memory,
            edge_index,
            edge_last_update,
        )
        # src_msgs: (num_edge_event, message_dim)
        # dst_msgs: (num_edge_event, message_dim)

        # aggregate messages
        event_node_ids = torch.cat([edge_event_src_ids, edge_event_dst_ids])
        # (2 * num_edge_event)
        agg_msgs = scatter(torch.cat([src_msgs, dst_msgs]), event_node_ids, dim=0)
        # (max_event_node_id, message_dim)

        # update the memories
        updated_memory = self.update_memory(
            memory,
            node_memory_update_index,
            node_memory_update_mask,
            node_features,
            event_node_ids,
            agg_msgs,
        )
        # (num_node, memory_dim)

        # calculate the node embeddings
        if node_features.size(0) != 0:
            rel_t = edge_timestamps - edge_last_update
            # (num_edge)
            rel_t_embs = self.time_encoder(rel_t)
            # (num_edge, time_enc_dim)
            x = torch.cat([node_features, updated_memory], dim=-1)
            # (num_node, event_embedding_dim + memory_dim)
            edge_attr = torch.cat([rel_t_embs, edge_features], dim=-1)
            # (num_edge, time_enc_dim + event_embedding_dim)
            node_embeddings = self.gnn(x, edge_index, edge_attr=edge_attr)
            # (num_node, output_dim)
        else:
            # no nodes, so no node embeddings either
            node_embeddings = torch.zeros(
                0, self.output_dim, device=node_features.device
            )
            # (0, output_dim)

        return {
            "node_embeddings": node_embeddings,
            # (num_node, output_dim)
            "updated_memory": updated_memory
            # (num_node, memory_dim)
        }

    def update_memory(
        self,
        memory: torch.Tensor,
        delete_node_mask: torch.Tensor,
        sorted_node_indices: torch.Tensor,
        event_node_ids: torch.Tensor,
        agg_msgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the memory by resizing and setting new memories.

        memory: (prev_num_node, memory_dim)
            Node memories before the given graph events.
        num_added_node: number of added nodes
        delete_node_mask: mask for deleting nodes, (prev_num_node)
        sorted_node_indices: sorted indices for nodes in the updated memory, (num_node)
        event_node_ids: (2 * num_message), batched node IDs
        agg_msgs: (max_event_node_id, message_dim)

        output: (num_node, memory_dim)
        """
        # mask out the deleted nodes, add new memories on the bottom then apply the new
        # node ordering using the given indices.
        updated_memory = memory[delete_node_mask]
        # (prev_num_node - num_deleted_node)
        num_added_node = int(sorted_node_indices.size(0) - updated_memory.size(0))
        updated_memory = F.pad(updated_memory, (0, 0, 0, num_added_node))[
            sorted_node_indices
        ]
        # (num_node, memory_dim)
        # (prev_num_node - num_deleted_node + num_added_node, memory_dim)

        # update the memory with messages
        if event_node_ids.numel() > 0:
            unique_event_node_ids = event_node_ids.unique()
            updated_memory[unique_event_node_ids] = self.rnn(
                agg_msgs[unique_event_node_ids], updated_memory[unique_event_node_ids]
            )
        # (num_node, memory_dim)
        return updated_memory

    def edge_message(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        last_update: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate edge event messages. We concatenate the event type,
        source and destination memories, relative time embedding and event embedding.

        Special events like pad, start, end are masked out.

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        event_embeddings: (batch, event_embedding_dim)
        event_timestamps: (batch)
        memory: (num_node, memory_dim)
        edge_index: (2, num_edge)
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
        edge_last_update = index_edge_attr(
            edge_index, last_update, torch.stack([src_ids, dst_ids])
        )
        # (batch)

        rel_edge_timestamps = event_timestamps - edge_last_update
        # (batch)

        timestamp_emb = self.time_encoder(rel_edge_timestamps)
        # (batch, time_enc_dim)

        src_messages = torch.cat(
            [
                event_type_embs,
                src_memory,
                dst_memory,
                timestamp_emb,
                event_embeddings,
            ],
            dim=-1,
        )
        # (batch, message_dim)
        dst_messages = torch.cat(
            [
                event_type_embs,
                dst_memory,
                src_memory,
                timestamp_emb,
                event_embeddings,
            ],
            dim=-1,
        )
        # (batch, message_dim)
        return src_messages, dst_messages

    @staticmethod
    def calculate_node_id_offsets(batch_size: int, batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate the node id offsets for turning subgraph node IDs into a batched
        graph node IDs.

        batch_size: scalar
        batch: (num_node)

        output: (batch_size)
        """
        subgraph_size_cumsum = batch.bincount().cumsum(0)
        return F.pad(
            subgraph_size_cumsum, (1, batch_size - subgraph_size_cumsum.size(0) - 1)
        )

    @classmethod
    def update_batched_graph(
        cls,
        batched_graph: Batch,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
        """
        Update the given batch of graph events to the given batched graph.
        Also returns the deleted node mask and sorted batch indices to be used
        for updating the memory.
        All the events are assumed to be valid, and each event in the batch is assuemd
        to be applied only to the corresponding graph in the batched graph.

        batched_graph: diagonally stacked graph BEFORE the given graph events: Batch(
            batch: (num_node)
            x: (num_node, event_embedding_dim)
            edge_index: (2, num_edge)
            edge_attr: (num_edge, event_embedding_dim)
            edge_last_update: (num_edge)
        )
        event_type_ids: (batch)
        event_src_ids: (batch)
        event_dst_ids: (batch)
        event_embeddings: (batch, event_embedding_dim)
        event_timestamps: (batch)

        output: (
            updated batch of graphs,
            deleted_node_mask: (num_node),
            sorted_node_indices: (num_node-num_deleted_node+num_added_node),
        )
        """
        # translate src_ids and dst_ids to batched versions
        node_id_offsets = cls.calculate_node_id_offsets(
            event_type_ids.size(0), batched_graph.batch
        )
        # (batch)

        # take care of the edges first
        # first turn existing edge_index into subgraph node IDs so that it's easier
        # to manipulate
        edge_index_batch = batched_graph.batch[batched_graph.edge_index[0]]
        # (num_edge)
        subgraph_edge_index = batched_graph.edge_index - node_id_offsets[
            edge_index_batch
        ].unsqueeze(0).expand(2, -1)
        # (2, num_edge)

        # collect edge add events
        edge_add_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
        # (batch)
        added_edge_index_batch = edge_add_event_mask.nonzero().squeeze(-1)
        # (num_added_edge)
        added_edge_index = torch.stack(
            [event_src_ids[edge_add_event_mask], event_dst_ids[edge_add_event_mask]]
        )
        # (2, num_added_edge)
        added_edge_attr = event_embeddings[edge_add_event_mask]
        # (num_added_edge, event_embedding_dim)
        added_edge_last_update = event_timestamps[edge_add_event_mask]
        # (num_added_edge)

        # collect edge delete events
        edge_delete_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
        # (batch)
        deleted_edge_co_occur = get_edge_index_co_occurrence_matrix(
            subgraph_edge_index,
            torch.stack(
                [
                    event_src_ids[edge_delete_event_mask],
                    event_dst_ids[edge_delete_event_mask],
                ]
            ),
        )
        # (num_edge, num_deleted_edge)
        deleted_edge_mask = torch.ones(
            batched_graph.num_edges,
            dtype=torch.bool,
            device=deleted_edge_co_occur.device,
        ).masked_fill(deleted_edge_co_occur.any(1), False)
        # (num_edge)

        new_edge_index_batch = torch.cat(
            [edge_index_batch[deleted_edge_mask], added_edge_index_batch]
        )
        # (num_edge-num_deleted_edge+num_added_edge)
        new_subgraph_edge_index = torch.cat(
            [subgraph_edge_index[:, deleted_edge_mask], added_edge_index], dim=1
        )
        # (2, num_edge-num_deleted_edge+num_added_edge)
        new_edge_attr = torch.cat(
            [batched_graph.edge_attr[deleted_edge_mask], added_edge_attr]
        )
        # (num_edge-num_deleted_edge+num_added_edge, event_embedding_dim)
        new_edge_last_update = torch.cat(
            [batched_graph.edge_last_update[deleted_edge_mask], added_edge_last_update]
        )
        # (num_edge-num_deleted_edge+num_added_edge)

        # take care of the nodes now
        # collect node add events
        node_add_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (batch)
        added_x = event_embeddings[node_add_event_mask]
        # (num_added_node, event_embedding_dim)
        added_node_batch = node_add_event_mask.nonzero().squeeze(-1)
        # (num_added_node)

        # collect node delete events
        node_delete_event_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
        # (batch)
        batch_src_ids = event_src_ids + node_id_offsets
        # (batch)
        node_delete_node_ids = batch_src_ids[node_delete_event_mask]
        # (batch)
        delete_node_mask = torch.ones(
            batched_graph.num_nodes,
            dtype=torch.bool,
            device=node_delete_node_ids.device,
        ).index_fill(0, node_delete_node_ids, False)
        # (num_node)

        # take care of the nodes
        new_batch = torch.cat([batched_graph.batch[delete_node_mask], added_node_batch])
        # (num_node-num_deleted_node+num_added_node)
        new_x = torch.cat([batched_graph.x[delete_node_mask], added_x])
        # (num_node-num_deleted_node+num_added_node, event_embedding_dim)

        # sort the new batch in ascending order
        sorted_batch, sorted_node_indices = new_batch.sort()
        # sorted_batch: (num_node-num_deleted_node+num_added_node)
        # sorted_node_indices: (num_node-num_deleted_node+num_added_node)

        # update the new subgraph edge index to match the new sorted node indices
        new_node_id_offsets = cls.calculate_node_id_offsets(
            event_type_ids.size(0), sorted_batch
        )
        # (batch)
        new_edge_index = new_subgraph_edge_index + new_node_id_offsets[
            new_edge_index_batch
        ].unsqueeze(0).expand(2, -1)
        # (2, num_edge-num_deleted_edge+num_added_edge)

        return (
            Batch(
                batch=sorted_batch,
                x=new_x[sorted_node_indices],
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                edge_last_update=new_edge_last_update,
            ),
            delete_node_mask,
            sorted_node_indices,
        )
