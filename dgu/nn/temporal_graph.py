import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Dict
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.data import Batch

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
        # graph events
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
        # diagonally stacked graph BEFORE the given graph events
        batched_graph: Batch,
        # memory
        memory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Update the given graphs using the given events and calculate the node embeddings

        graph events: {
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_embeddings: (batch, event_embedding_dim)
            event_timestamps: (batch)
        }
        batched_graph: diagonally stacked graph BEFORE the given graph events: Batch(
            batch: (prev_num_node)
            x: (prev_num_node, event_embedding_dim)
            edge_index: (2, prev_num_edge)
            edge_attr: (prev_num_edge, event_embedding_dim)
            edge_last_update: (prev_num_edge)
        )
        memory: (prev_num_node, memory_dim)

        output: {
            "node_embeddings": (num_node, output_dim),
            "updated_batched_graph": Batch(
                batch: (num_node)
                x: (num_node, event_embedding_dim)
                edge_index: (2, num_edge)
                edge_attr: (num_edge, event_embedding_dim)
                edge_last_update: (num_edge)
            )
            "updated_memory": (num_node, memory_dim),
        }
        """
        # update the batched graph
        updated_batched_graph, updated_memory = self.update_batched_graph_memory(
            batched_graph,
            memory,
            event_type_ids,
            event_src_ids,
            event_dst_ids,
            event_embeddings,
            event_timestamps,
        )

        # calculate the node embeddings
        if updated_batched_graph.num_nodes != 0:
            edge_timestamps = self.get_edge_timestamps(
                event_timestamps,
                updated_batched_graph.batch,
                updated_batched_graph.edge_index,
            )
            # (num_edge)
            rel_t = edge_timestamps - updated_batched_graph.edge_last_update
            # (num_edge)
            rel_t_embs = self.time_encoder(rel_t)
            # (num_edge, time_enc_dim)
            x = torch.cat([updated_batched_graph.x, updated_memory], dim=-1)
            # (num_node, event_embedding_dim + memory_dim)
            edge_attr = torch.cat([rel_t_embs, updated_batched_graph.edge_attr], dim=-1)
            # (num_edge, time_enc_dim + event_embedding_dim)
            node_embeddings = self.gnn(
                x, updated_batched_graph.edge_index, edge_attr=edge_attr
            )
            # (num_node, output_dim)
        else:
            # no nodes, so no node embeddings either
            node_embeddings = torch.zeros(
                0, self.output_dim, device=event_type_ids.device
            )
            # (0, output_dim)

        return {
            "node_embeddings": node_embeddings,
            "updated_batched_graph": updated_batched_graph,
            "updated_memory": updated_memory,
        }

    @staticmethod
    def get_edge_timestamps(
        timestamps: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign an appropriate timestamp for each edge.

        timestamps: (batch)
        batch: (num_node)
        edge_index: (2, num_edge)

        output: (num_edge)
        """
        # figure out which batch element the source node belongs to
        # then get the timestamps
        return timestamps[batch[edge_index[0]]]

    def get_event_messages(
        self,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
        node_id_offsets: torch.Tensor,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        edge_last_update: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate event messages. We ignore node delete events, b/c the memories of
        deleted nodes are removed anyway. For node add events, we concatenate the event
        type embedding, zero source memory and zero destination memory, time embedding
        and event embedding. For edge events, we concatenate the event type, source
        and destination memories, relative time embedding and event embedding.

        Special events like pad, start, end are masked out.

        event_type_ids: (batch)
        event_src_ids: (batch)
        event_dst_ids: (batch)
        event_embeddings: (batch, event_embedding_dim)
        event_timestamps: (batch)
        node_id_offsets: (batch)
        memory: (num_node, memory_dim)
        edge_index: (2, num_edge)
        edge_last_update: (num_edge)

        output:
            # 2 * num_edge_msgs b/c an edge event produces a source message
            # and a destination message
            node_add_event_messages: (num_node_add_event, message_dim)
            edge_event_node_ids: (2 * num_edge_event), batched node IDs
            edge_event_batch: (2 * num_edge_event)
            edge_event_messages: (2 * num_edge_event, message_dim)
        """
        # generate node add event messages
        is_node_add_event = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (batch)
        node_add_event_type_embs = self.event_type_emb(
            event_type_ids[is_node_add_event]
        )
        # (num_node_add_event, event_type_emb_dim)
        node_add_event_time_embeddings = self.time_encoder(
            event_timestamps[is_node_add_event]
        )
        # (num_node_add_event, time_enc_dim)
        node_add_event_embeddings = event_embeddings[is_node_add_event]
        # (num_node_add_event, event_embedding_dim)
        node_add_event_msgs = torch.cat(
            [
                node_add_event_type_embs,
                torch.zeros(
                    node_add_event_type_embs.size(0),
                    self.memory_dim,
                    device=is_node_add_event.device,
                ),
                torch.zeros(
                    node_add_event_type_embs.size(0),
                    self.memory_dim,
                    device=is_node_add_event.device,
                ),
                node_add_event_time_embeddings,
                node_add_event_embeddings,
            ],
            dim=-1,
        )
        # (num_node_add_event, message_dim)

        # generate edge event messages
        is_edge_event = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
        )
        # (batch)
        edge_node_id_offsets = node_id_offsets[is_edge_event]
        # (num_edge_event)
        edge_event_src_ids = event_src_ids[is_edge_event]
        # (num_edge_event)
        batch_edge_event_src_ids = edge_event_src_ids + edge_node_id_offsets
        # (num_edge_event)
        edge_event_dst_ids = event_dst_ids[is_edge_event]
        # (num_edge_event)
        batch_edge_event_dst_ids = edge_event_dst_ids + edge_node_id_offsets
        # (num_edge_event)
        edge_event_type_embs = self.event_type_emb(event_type_ids[is_edge_event])
        # (num_edge_event, event_type_emb_dim)
        edge_event_src_memory = memory[batch_edge_event_src_ids]
        # (num_edge_event, memory_dim)
        edge_event_dst_memory = memory[batch_edge_event_dst_ids]
        # (num_edge_event, memory_dim)
        edge_event_last_update = index_edge_attr(
            edge_index,
            edge_last_update,
            torch.stack([batch_edge_event_src_ids, batch_edge_event_dst_ids]),
        )
        # (num_edge_event)
        rel_edge_timestamps = event_timestamps[is_edge_event] - edge_event_last_update
        # (num_edge_event)
        edge_event_timestamp_emb = self.time_encoder(rel_edge_timestamps)
        # (num_edge_event, time_enc_dim)
        edge_event_embeddings = event_embeddings[is_edge_event]
        # (num_edge_event, event_embedding_dim)
        edge_src_messages = torch.cat(
            [
                edge_event_type_embs,
                edge_event_src_memory,
                edge_event_dst_memory,
                edge_event_timestamp_emb,
                edge_event_embeddings,
            ],
            dim=-1,
        )
        # (num_edge_event, message_dim)
        edge_dst_messages = torch.cat(
            [
                edge_event_type_embs,
                edge_event_dst_memory,
                edge_event_src_memory,
                edge_event_timestamp_emb,
                edge_event_embeddings,
            ],
            dim=-1,
        )
        # (num_edge_event, message_dim)

        return (
            node_add_event_msgs,
            torch.cat([edge_event_src_ids, edge_event_dst_ids]),
            is_edge_event.nonzero().expand(-1, 2).t().flatten(),
            torch.cat([edge_src_messages, edge_dst_messages]),
        )

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

    def update_batched_graph_memory(
        self,
        batched_graph: Batch,
        memory: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> Tuple[Batch, torch.Tensor]:
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
        memory: (num_node, memory_dim)
        event_type_ids: (batch)
        event_src_ids: (batch)
        event_dst_ids: (batch)
        event_embeddings: (batch, event_embedding_dim)
        event_timestamps: (batch)

        output: (
            updated batch of graphs,
            updated_memory: (num_node-num_deleted_node+num_added_node, memory_dim)
        )
        """
        # translate src_ids and dst_ids to batched versions
        node_id_offsets = self.calculate_node_id_offsets(
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
        batch_src_ids = event_src_ids + node_id_offsets
        # (batch)
        batch_dst_ids = event_dst_ids + node_id_offsets
        # (batch)
        deleted_edge_co_occur = get_edge_index_co_occurrence_matrix(
            batched_graph.edge_index,
            torch.stack(
                [
                    batch_src_ids[edge_delete_event_mask],
                    batch_dst_ids[edge_delete_event_mask],
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
        new_node_id_offsets = self.calculate_node_id_offsets(
            event_type_ids.size(0), sorted_batch
        )
        # (batch)
        new_edge_index = new_subgraph_edge_index + new_node_id_offsets[
            new_edge_index_batch
        ].unsqueeze(0).expand(2, -1)
        # (2, num_edge-num_deleted_edge+num_added_edge)

        # take care of the memory
        # first get the messages
        (
            node_add_event_msgs,
            edge_event_node_ids,
            edge_event_batch,
            edge_event_msgs,
        ) = self.get_event_messages(
            event_type_ids,
            event_src_ids,
            event_dst_ids,
            event_embeddings,
            event_timestamps,
            node_id_offsets,
            memory,
            batched_graph.edge_index,
            batched_graph.edge_last_update,
        )
        num_node_add_msg = node_add_event_msgs.size(0)
        # update memory with messages
        if num_node_add_msg > 0 or edge_event_node_ids.size(0) > 0:
            updated_memory = self.rnn(
                torch.cat([node_add_event_msgs, edge_event_msgs]),
                torch.cat(
                    [
                        torch.zeros(
                            num_node_add_msg, self.memory_dim, device=memory.device
                        ),
                        memory[edge_event_node_ids + node_id_offsets[edge_event_batch]],
                    ]
                ),
            )
        else:
            updated_memory = torch.empty(0, self.memory_dim, device=memory.device)
        # (num_added_node + num_edge_event_msg)
        new_memory = torch.cat(
            [memory[delete_node_mask], updated_memory[:num_node_add_msg]]
        )[sorted_node_indices]
        # (num_node-num_deleted_node+num_added_node, memory_dim)
        new_memory[
            edge_event_node_ids + new_node_id_offsets[edge_event_batch]
        ] = updated_memory[num_node_add_msg:]
        # (num_node-num_deleted_node+num_added_node, memory_dim)

        return (
            Batch(
                batch=sorted_batch,
                x=new_x[sorted_node_indices],
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                edge_last_update=new_edge_last_update,
            ),
            new_memory,
        )
