import torch
import torch.nn as nn

from typing import Tuple
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_scatter import scatter

from dgu.constants import EVENT_TYPE_ID_MAP


class TemporalGraphNetwork(nn.Module):
    def __init__(self, max_num_nodes: int, max_num_edges: int, hidden_dim: int) -> None:
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.hidden_dim = hidden_dim

        # memory, not persistent as we shouldn't save memories from one game to another
        self.register_buffer(
            "memory", torch.zeros(max_num_nodes, hidden_dim), persistent=False
        )

        # last updated timestamp for edges, not persistent as we shouldn't save last
        # updated timestamps from one game to another
        self.register_buffer(
            "last_update", torch.zeros(max_num_edges), persistent=False
        )

        # node features, not persistent as we shouldn't save node features
        # from one game to another
        self.register_buffer(
            "node_features", torch.zeros(max_num_nodes, hidden_dim), persistent=False
        )

        # edge features, not persistent as we shouldn't save edge features
        # from one game to another
        self.register_buffer(
            "edge_features", torch.zeros(max_num_edges, hidden_dim), persistent=False
        )

        # time encoder
        self.time_encoder = TimeEncoder(hidden_dim)

        # RNN to update memories
        self.rnn = nn.GRUCell(5 * hidden_dim, hidden_dim)

        # GNN for the final node embeddings
        self.gnn = TransformerConv(
            hidden_dim, hidden_dim, edge_dim=hidden_dim + hidden_dim
        )

    def forward(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        dst_ids: torch.Tensor,
        dst_mask: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_mask: torch.Tensor,
        event_timestamps: torch.Tensor,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        edge_index: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the updated node embeddings based on the given events. Node
        embeddings are calculated for nodes specified by node_ids.

        event_type_ids: (event_seq_len)
        src_ids: (event_seq_len)
        src_mask: (event_seq_len)
        dst_ids: (event_seq_len)
        dst_mask: (event_seq_len)
        event_edge_ids: (event_seq_len)
        event_embeddings: (event_seq_len, hidden_dim)
        event_mask: (event_seq_len)
        event_timestamps: (event_seq_len)
        node_ids: (num_node)
        edge_ids: (num_edge)
        edge_index: (2, num_edge)
        time: scalar

        output: (num_node, hidden_dim)
        """
        # calculate messages
        src_msgs, dst_msgs = self.message(
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
        # src_msgs: (event_seq_len, 5 * hidden_dim)
        # dst_msgs: (event_seq_len, 5 * hidden_dim)

        # aggregate messages
        agg_src_msgs = self.agg_message(src_msgs, src_ids)
        # (max_src_id, 5 * hidden_dim)
        agg_dst_msgs = self.agg_message(dst_msgs, dst_ids)
        # (max_dst_id, 5 * hidden_dim)

        # get unique IDs
        uniq_src_ids = src_ids.unique()
        # (num_uniq_src_ids)
        uniq_dst_ids = dst_ids.unique()
        # (num_uniq_dst_ids)
        uniq_ids = torch.cat([uniq_src_ids, uniq_dst_ids])
        # (num_uniq_src_ids + num_uniq_dst_ids)

        # update the memories
        self.memory[uniq_ids] = self.rnn(  # type: ignore
            torch.cat([agg_src_msgs[uniq_src_ids], agg_dst_msgs[uniq_dst_ids]]),
            self.memory[uniq_ids],  # type: ignore
        )

        # calculate the node embeddings
        rel_t = time - self.last_update[edge_ids]  # type: ignore
        # (num_edge)
        rel_t_embs = self.time_encoder(rel_t)
        # (num_edge, hidden_dim)
        node_embeddings = self.gnn(
            self.node_features[node_ids] + self.memory[node_ids],  # type: ignore
            edge_index,
            torch.cat(
                [rel_t_embs, self.edge_features[edge_ids]], dim=-1  # type: ignore
            ),
        )
        # (num_node, hidden_dim)

        # update node features
        self.update_node_features(event_type_ids, src_ids, event_embeddings)

        # update edge features
        self.update_edge_features(event_type_ids, event_edge_ids, event_embeddings)

        # update last updated timestamps
        self.update_last_update(event_type_ids, event_edge_ids, event_timestamps)

        return node_embeddings

    def message(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        dst_ids: torch.Tensor,
        dst_mask: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_mask: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate graph event messages. We concatenate the event type,
        source and destination memories, time embedding and event embedding.

        Special events like pad, start, end are masked out. Node events are also
        masked out for destination node messages.

        event_type_ids: (event_seq_len)
        src_ids: (event_seq_len)
        src_mask: (event_seq_len)
        dst_ids: (event_seq_len)
        dst_mask: (event_seq_len)
        event_embeddings: (event_seq_len, hidden_dim)
        event_mask: (event_seq_len)
        event_edge_ids: (event_seq_len)
        event_timestamps: (event_seq_len)

        output:
            src_messages: (event_seq_len, 5 * hidden_dim)
            dst_messages: (event_seq_len, 5 * hidden_dim)
        """
        # repeat event type id for event type embeddings
        event_type_embs = event_type_ids.unsqueeze(-1).expand(-1, self.hidden_dim)
        # (event_seq_len, hidden_dim)

        # use the memory for node embeddings
        src_embs = self.memory[src_ids] * src_mask.unsqueeze(-1)  # type: ignore
        # (event_seq_len, hidden_dim)
        dst_embs = self.memory[dst_ids] * dst_mask.unsqueeze(-1)  # type: ignore
        # (event_seq_len, hidden_dim)

        # calculate relative timestamps for edge events
        edge_last_update = self.last_update[event_edge_ids]  # type: ignore
        # (event_seq_len)

        # multiply edge_last_update by dst_mask so that we
        # only subtract last update timestamps for edge events
        rel_edge_timestamps = event_timestamps - edge_last_update * dst_mask
        # (event_seq_len)

        # only select node events from event_timestamps
        # and only select edge events from rel_edge_timestamps (use dst_mask)
        # then add them and pass it through the time encoder to get the embeddings
        # finally, mask out timestamp embeddings for special events (use event_mask).
        is_node_event = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["node-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["node-delete"],
        )
        # (event_seq_len)
        timestamp_emb = self.time_encoder(
            event_timestamps * is_node_event + rel_edge_timestamps * dst_mask
        ) * event_mask.unsqueeze(-1)
        # (event_seq_len, hidden_dim)

        # mask out special events
        src_messages = (
            torch.cat(
                [
                    event_type_embs,
                    src_embs,
                    dst_embs,
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=1,
            )
            * event_mask.unsqueeze(-1)
        )
        # (event_seq_len, 5 * hidden_dim)
        dst_messages = (
            torch.cat(
                [
                    event_type_embs,
                    dst_embs,
                    src_embs,
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=1,
            )
            * dst_mask.unsqueeze(-1)
        )
        # (event_seq_len, 5 * hidden_dim)
        return src_messages, dst_messages

    def agg_message(self, messages: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages based on the given node IDs. For now we calculate the mean.

        messages: (event_seq_len, 5 * hidden_dim)
        ids: (event_seq_len)

        output: (num_uniq_ids, 5 * hidden_dim)
        """
        return scatter(messages, ids, dim=0, reduce="mean")

    def update_node_features(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> None:
        """
        Update node features using node-add event embeddings.

        event_type_ids: (event_seq_len)
        src_ids: (event_seq_len)
        event_embeddings: (event_seq_len, hidden_dim)
        """
        # update node features using node-add event embeddings
        is_node_add = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (num_node_add)
        if is_node_add.size(0) > 0:
            # there could technically be duplicates, but we ignore them.
            # PyTorch seems to assign the first of the duplicates.
            node_add_src_ids = src_ids[is_node_add]
            # (num_node_add)
            node_add_event_embeddings = event_embeddings[is_node_add]
            # (num_node_add, hidden_dim)
            self.node_features[
                node_add_src_ids
            ] = node_add_event_embeddings  # type: ignore

    def update_edge_features(
        self,
        event_type_ids: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> None:
        """
        Update edge features using edge-add event embeddings.

        event_type_ids: (event_seq_len)
        event_edge_ids: (event_seq_len)
        event_embeddings: (event_seq_len, hidden_dim)
        """
        # update edge features using edge-add event embeddings
        is_edge_add = event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
        # (num_edge_add)
        if is_edge_add.size(0) > 0:
            # there could technically be duplicates, but we ignore them.
            # PyTorch seems to assign the first of the duplicates.
            edge_add_edge_ids = event_edge_ids[is_edge_add]
            # (num_node_add)
            edge_add_event_embeddings = event_embeddings[is_edge_add]
            # (num_node_add, hidden_dim)
            self.edge_features[
                edge_add_edge_ids
            ] = edge_add_event_embeddings  # type: ignore

    def update_last_update(
        self,
        event_type_ids: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> None:
        """
        Update last update timestamps for edges.

        event_type_ids: (event_seq_len)
        event_edge_ids: (event_seq_len)
        event_timestamps: (event_seq_len)
        """
        # update last update timestamps using edge events
        is_edge_event = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
        )
        # (num_edge_event)
        if is_edge_event.size(0) > 0:
            # there could technically be duplicates, but we ignore them.
            # PyTorch seems to assign the first of the duplicates.
            edge_ids = event_edge_ids[is_edge_event]
            # (num_edge_event)
            edge_timestamps = event_timestamps[is_edge_event]
            # (num_edge_event)
            self.last_update[edge_ids] = edge_timestamps  # type: ignore
