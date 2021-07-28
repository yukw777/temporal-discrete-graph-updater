import torch
import torch.nn as nn

from typing import Tuple
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_scatter import scatter

from dgu.constants import EVENT_TYPE_ID_MAP


class TemporalGraphNetwork(nn.Module):
    def __init__(
        self,
        max_num_nodes: int,
        max_num_edges: int,
        event_type_emb_dim: int,
        memory_dim: int,
        time_enc_dim: int,
        event_embedding_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.event_type_emb_dim = event_type_emb_dim
        self.memory_dim = memory_dim
        self.time_enc_dim = time_enc_dim
        self.event_embedding_dim = event_embedding_dim
        self.output_dim = output_dim

        # event type embedding
        self.event_type_emb = nn.Embedding(len(EVENT_TYPE_ID_MAP), event_type_emb_dim)

        # memory, not persistent as we shouldn't save memories from one game to another
        self.register_buffer(
            "memory", torch.zeros(max_num_nodes, memory_dim), persistent=False
        )

        # last updated timestamp for edges, not persistent as we shouldn't save last
        # updated timestamps from one game to another
        self.register_buffer(
            "last_update", torch.zeros(max_num_edges), persistent=False
        )

        # node features, not persistent as we shouldn't save node features
        # from one game to another
        self.register_buffer(
            "node_features",
            torch.zeros(max_num_nodes, event_embedding_dim),
            persistent=False,
        )

        # edge features, not persistent as we shouldn't save edge features
        # from one game to another
        self.register_buffer(
            "edge_features",
            torch.zeros(max_num_edges, event_embedding_dim),
            persistent=False,
        )

        # time encoder
        self.time_encoder = TimeEncoder(time_enc_dim)

        # RNN to update memories
        self.rnn = nn.GRUCell(
            event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim,
            memory_dim,
        )

        # GNN for the final node embeddings
        self.gnn = TransformerConv(
            event_embedding_dim + memory_dim,
            output_dim,
            edge_dim=time_enc_dim + event_embedding_dim,
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
        event_timestamps: torch.Tensor,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the updated node embeddings based on the given events. Node
        embeddings are calculated for nodes specified by node_ids.

        event_type_ids: (batch, event_seq_len)
        src_ids: (batch, event_seq_len)
        src_mask: (batch, event_seq_len)
        dst_ids: (batch, event_seq_len)
        dst_mask: (batch, event_seq_len)
        event_edge_ids: (batch, event_seq_len)
        event_embeddings: (batch, event_seq_len, event_embedding_dim)
        event_mask: (batch, event_seq_len)
        event_timestamps: (batch, event_seq_len)
        node_ids: (batch, num_node)
        edge_ids: (batch, num_edge)
        edge_index: (batch, 2, num_edge)
        edge_timestamps: (batch, num_edge)

        output: (batch, num_node, hidden_dim)
        """
        # calculate messages
        src_msgs, dst_msgs = self.message(
            event_type_ids,
            src_ids,
            src_mask,
            dst_ids,
            dst_mask,
            event_embeddings,
            event_edge_ids,
            event_timestamps,
        )
        # src_msgs: (
        #     batch,
        #     event_seq_len,
        #     event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        # )
        # dst_msgs: (
        #     batch,
        #     event_seq_len,
        #     event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        # )

        # aggregate messages
        agg_src_msgs = self.agg_message(src_msgs, src_ids).flatten(end_dim=1)
        # (batch * max_src_id,
        #  event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim)
        agg_dst_msgs = self.agg_message(dst_msgs, dst_ids).flatten(end_dim=1)
        # (batch * max_dst_id,
        #  event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim)

        # get unique IDs
        uniq_src_ids = src_ids.unique()
        # (num_uniq_src_ids)
        uniq_dst_ids = dst_ids.unique()
        # (num_uniq_dst_ids)
        uniq_ids = torch.cat([uniq_src_ids, uniq_dst_ids])
        # (num_uniq_src_ids + num_uniq_dst_ids)

        # update the memories
        # note that source IDs and destination IDs never overlap as we don't have
        # self-loops in our graphs (except for the pad edge, but that doesn't matter).
        if uniq_ids.size(0) > 0:
            self.memory[uniq_ids] = self.rnn(  # type: ignore
                torch.cat([agg_src_msgs[uniq_src_ids], agg_dst_msgs[uniq_dst_ids]]),
                # (num_uniq_src_ids + num_uniq_dst_ids,
                #  event_type_emb_dim + 2 * memory_dim + time_enc_dim +
                #  event_embedding_dim)
                self.memory[uniq_ids],  # type: ignore
                # (num_uniq_src_ids + num_uniq_dst_ids, memory_dim)
            )

        # calculate the node embeddings
        if node_ids.size(1) != 0:
            rel_t = edge_timestamps - self.last_update[edge_ids]  # type: ignore
            # (batch, num_edge)
            rel_t_embs = self.time_encoder(rel_t).view(
                rel_t.size(0), -1, self.time_enc_dim
            )
            # (batch, num_edge, time_enc_dim)
            node_embeddings = self.gnn(
                torch.cat(
                    [
                        self.node_features[node_ids],  # type: ignore
                        # (batch, num_node)
                        self.memory[node_ids],  # type: ignore
                        # (batch, num_node)
                    ],
                    dim=-1,
                ).flatten(end_dim=1),
                # (batch * num_node, event_embedding_dim + memory_dim)
                self.localize_edge_index(edge_index, node_ids)
                .transpose(0, 1)
                .flatten(start_dim=1),
                # (2, batch * num_node)
                torch.cat(
                    [rel_t_embs, self.edge_features[edge_ids]], dim=-1  # type: ignore
                ).flatten(end_dim=1),
                # (batch * num_node, time_enc_dim + event_embedding_dim)
            )
            # (batch * num_node, output_dim)
        else:
            # no nodes, so no node embeddings either
            node_embeddings = torch.zeros(0, self.output_dim, device=node_ids.device)
            # (0, output_dim)

        # update node features
        self.update_node_features(event_type_ids, src_ids, event_embeddings)

        # update edge features
        self.update_edge_features(event_type_ids, event_edge_ids, event_embeddings)

        # update last updated timestamps
        self.update_last_update(event_type_ids, event_edge_ids, event_timestamps)

        return node_embeddings.view(node_ids.size(0), -1, self.output_dim)
        # (batch, num_node, output_dim)

    @staticmethod
    def localize_edge_index(
        edge_index: torch.Tensor, node_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Translate edge_index into indices based on the local node_ids.

        edge_index: (batch, 2, num_edge)
        node_ids: (batch, num_node)

        output: (batch, 2, num_edge)
        """
        flat_node_ids = node_ids.flatten()
        # (batch * num_node)
        flat_edge_index = edge_index.flatten()
        # (batch * 2 * num_edge)
        matches = flat_node_ids.unsqueeze(-1) == flat_edge_index
        # (batch * num_node, batch * 2 * num_edge)

        # now find the index for the first True (or nonzero) element from (flat) matches
        # by looking for an element that is nonzero and the cumulative sum of the
        # matches is 1.
        # this means that if there are overlaps in the elements in the batch
        # (e.g. node 0 for padding), the index of the first occurrence would be used.
        # this is OK as the only overlap should be the pdding node 0.
        nonzero = matches > 0
        # (batch * num_node, batch * 2 * num_edge)
        nonzero_and_cumsum_zero = (nonzero.cumsum(0) == 1) & nonzero
        # (batch * num_node, batch * 2 * num_edge)
        # find indices of True by taking the max
        _, indices = torch.max(nonzero_and_cumsum_zero, dim=0)
        # (batch * 2 * num_edge)

        return indices.view_as(edge_index)
        # (batch, 2, num_edge)

    def message(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        dst_ids: torch.Tensor,
        dst_mask: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_edge_ids: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate graph event messages. We concatenate the event type,
        source and destination memories, time embedding and event embedding.

        Special events like pad, start, end are masked out. Node events are also
        masked out for destination node messages.

        event_type_ids: (batch, event_seq_len)
        src_ids: (batch, event_seq_len)
        src_mask: (batch, event_seq_len)
        dst_ids: (batch, event_seq_len)
        dst_mask: (batch, event_seq_len)
        event_embeddings: (batch, event_seq_len, event_embedding_dim)
        event_edge_ids: (batch, event_seq_len)
        event_timestamps: (batch, event_seq_len)

        output:
            src_messages: (
                batch,
                event_seq_len,
                event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
            )
            dst_messages: (
                batch,
                event_seq_len,
                event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
            )
        """
        # repeat event type id for event type embeddings
        event_type_embs = self.event_type_emb(event_type_ids)
        # (batch, event_seq_len, event_type_emb_dim)

        # use the memory for node embeddings
        src_embs = self.memory[src_ids] * src_mask.unsqueeze(-1)  # type: ignore
        # (batch, event_seq_len, hidden_dim)
        dst_embs = self.memory[dst_ids] * dst_mask.unsqueeze(-1)  # type: ignore
        # (batch, event_seq_len, hidden_dim)

        # calculate relative timestamps for edge events
        edge_last_update = self.last_update[event_edge_ids]  # type: ignore
        # (batch, event_seq_len)

        # multiply edge_last_update by dst_mask so that we
        # only subtract last update timestamps for edge events
        rel_edge_timestamps = event_timestamps - edge_last_update * dst_mask
        # (batch, event_seq_len)

        # only select node events from event_timestamps
        # and only select edge events from rel_edge_timestamps (use dst_mask)
        # then add them and pass it through the time encoder to get the embeddings
        # finally, mask out timestamp embeddings for special events
        is_node_event = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["node-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["node-delete"],
        )
        is_not_special_event = torch.logical_not(
            torch.logical_or(
                torch.logical_or(
                    event_type_ids == EVENT_TYPE_ID_MAP["pad"],
                    event_type_ids == EVENT_TYPE_ID_MAP["start"],
                ),
                event_type_ids == EVENT_TYPE_ID_MAP["end"],
            )
        )
        # (batch, event_seq_len)
        timestamp_emb = self.time_encoder(
            event_timestamps * is_node_event + rel_edge_timestamps * dst_mask
        ).view(
            event_timestamps.size(0), -1, self.time_enc_dim
        ) * is_not_special_event.unsqueeze(
            -1
        )
        # (batch, event_seq_len, time_enc_dim)

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
                dim=-1,
            )
            * is_not_special_event.unsqueeze(-1)
        )
        # (
        #   batch,
        #   event_seq_len,
        #   event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        # )
        dst_messages = (
            torch.cat(
                [
                    event_type_embs,
                    dst_embs,
                    src_embs,
                    timestamp_emb,
                    event_embeddings,
                ],
                dim=-1,
            )
            * dst_mask.unsqueeze(-1)
        )
        # (
        #   batch,
        #   event_seq_len,
        #   event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        # )
        return src_messages, dst_messages

    def agg_message(self, messages: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages based on the given node IDs. For now we calculate the mean.

        messages: (
            batch,
            event_seq_len,
            event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        )
        ids: (batch, event_seq_len)

        output: (
            batch,
            max_src_id,
            event_type_emb_dim + 2 * memory_dim + time_enc_dim + event_embedding_dim
        )
        """
        return scatter(messages, ids, dim=1, reduce="mean")

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

    def reset(self) -> None:
        self.memory.data.fill_(0)  # type: ignore
        self.last_update.data.fill_(0)  # type: ignore
        self.node_features.data.fill_(0)  # type: ignore
        self.edge_features.data.fill_(0)  # type: ignore
