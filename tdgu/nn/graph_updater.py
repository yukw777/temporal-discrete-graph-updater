import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

if sys.version_info >= (3, 8):
    from typing import Optional, Dict, List, Tuple, Protocol
else:
    from typing import Optional, Dict, List, Tuple
    from typing_extensions import Protocol

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from tdgu.nn.rep_aggregator import ReprAggregator
from tdgu.nn.utils import (
    compute_masks_from_event_type_ids,
    masked_mean,
    calculate_node_id_offsets,
    masked_softmax,
    masked_gumbel_softmax,
    update_batched_graph,
)
from tdgu.nn.graph_event_decoder import (
    EventTypeHead,
    EventNodeHead,
    EventSequentialLabelHead,
)
from tdgu.nn.dynamic_gnn import DynamicGNN
from tdgu.constants import EVENT_TYPE_ID_MAP


class TextEncoderProto(Protocol):
    def __call__(self, word_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def get_input_embeddings(self) -> nn.Embedding:
        pass


class GraphEventDecoder(Protocol):
    @property
    def hidden_dim(self) -> int:
        pass

    def __call__(
        self,
        input_event_embedding: torch.Tensor,
        event_mask: torch.Tensor,
        aggr_obs_graph: torch.Tensor,
        obs_mask: torch.Tensor,
        aggr_prev_action_graph: torch.Tensor,
        prev_action_mask: torch.Tensor,
        aggr_graph_obs: torch.Tensor,
        aggr_graph_prev_action: torch.Tensor,
        node_mask: torch.Tensor,
        prev_input_event_emb_seq: Optional[torch.Tensor] = None,
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        pass


class TemporalDiscreteGraphUpdater(nn.Module):
    """
    TemporalDiscreteGraphUpdater is essentially a Seq2Seq model which encodes
    a sequence of game steps, each with an observation and a previous action, and
    decodes a sequence of graph events.
    """

    def __init__(
        self,
        text_encoder: TextEncoderProto,
        gnn_module: nn.Module,
        hidden_dim: int,
        dgnn_timestamp_enc_dim: int,
        dgnn_num_gnn_block: int,
        dgnn_num_gnn_head: int,
        dgnn_zero_timestamp_encoder: bool,
        graph_event_decoder: GraphEventDecoder,
        graph_event_decoder_event_type_emb_dim: int,
        graph_event_decoder_autoregressive_emb_dim: int,
        graph_event_decoder_key_query_dim: int,
        label_head_bos_token_id: int,
        label_head_eos_token_id: int,
        label_head_pad_token_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # text encoder
        self.text_encoder = text_encoder

        # dynamic graph neural network
        self.dynamic_gnn = DynamicGNN(
            gnn_module,
            dgnn_timestamp_enc_dim,
            hidden_dim,
            hidden_dim,
            dgnn_num_gnn_block,
            dgnn_num_gnn_head,
            dropout=dropout,
            zero_timestamp_encoder=dgnn_zero_timestamp_encoder,
        )

        # representation aggregator
        self.repr_aggr = ReprAggregator(self.hidden_dim)

        # graph event decoder
        self.event_type_embeddings = nn.Embedding(
            len(EVENT_TYPE_ID_MAP),
            graph_event_decoder_event_type_emb_dim,
            padding_idx=EVENT_TYPE_ID_MAP["pad"],
        )
        self.graph_event_decoder = graph_event_decoder

        # autoregressive heads
        self.event_type_head = EventTypeHead(
            self.graph_event_decoder.hidden_dim,
            hidden_dim,
            graph_event_decoder_autoregressive_emb_dim,
            dropout=dropout,
        )
        self.event_src_head = EventNodeHead(
            hidden_dim,
            graph_event_decoder_autoregressive_emb_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
        )
        self.event_dst_head = EventNodeHead(
            hidden_dim,
            graph_event_decoder_autoregressive_emb_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
        )
        self.event_label_head = EventSequentialLabelHead(
            graph_event_decoder_autoregressive_emb_dim,
            hidden_dim,
            self.text_encoder.get_input_embeddings(),
            label_head_bos_token_id,
            label_head_eos_token_id,
            label_head_pad_token_id,
        )

    def forward(  # type: ignore
        self,
        # graph events
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_word_ids: torch.Tensor,
        event_label_mask: torch.Tensor,
        # diagonally stacked graph BEFORE the given graph events
        prev_batched_graph: Batch,
        # game step
        obs_mask: torch.Tensor,
        prev_action_mask: torch.Tensor,
        timestamps: torch.Tensor,
        obs_word_ids: Optional[torch.Tensor] = None,
        prev_action_word_ids: Optional[torch.Tensor] = None,
        encoded_obs: Optional[torch.Tensor] = None,
        encoded_prev_action: Optional[torch.Tensor] = None,
        # previous input event embedding sequence
        prev_input_event_emb_seq: Optional[torch.Tensor] = None,
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None,
        # groundtruth events for decoder teacher-force training
        groundtruth_event: Optional[Dict[str, torch.Tensor]] = None,
        max_label_decode_len: int = 10,
        gumbel_greedy_decode_labels: bool = False,
        gumbel_tau: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        graph events: {
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_word_ids: (batch, event_label_len) or
                one-hot encoded (batch, event_label_len, num_word)
            event_label_mask: (batch, event_label_len)
        }
        prev_batched_graph: diagonally stacked graph BEFORE the given graph events:
            Batch(
                batch: (prev_num_node)
                x: node label word IDs, (prev_num_node, prev_node_label_len) or
                    one-hot encoded (prev_num_node, prev_node_label_len, num_word)
                node_label_mask: (prev_num_node, prev_node_label_len)
                node_last_update: (prev_num_node)
                edge_index: (2, prev_num_edge)
                edge_attr: edge label word IDs, (prev_num_edge, prev_edge_label_len) or
                    one-hot encoded (prev_num_edge, prev_edge_label_len, num_word)
                edge_label_mask: (prev_num_edge, prev_edge_label_len)
                edge_last_update: (prev_num_edge)
            )
        prev_input_event_emb_seq:
            (graph_event_decoder_num_dec_blocks, batch,
             prev_input_seq_len, graph_event_decoder_hidden_dim)
        prev_input_event_emb_seq_mask: (batch, prev_input_seq_len)


        game step {
            obs_mask: (batch, obs_len)
            prev_action_mask: (batch, prev_action_len)
            timestamps: (batch)
            obs_word_ids: (batch, obs_len)
            prev_action_word_ids: (batch, prev_action_len)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)
        }
        If encoded_obs and encoded_prev_action are given, they're used
        Otherwise, they are calculated from obs_word_ids, obs_mask,
        prev_action_word_ids and prev_action_mask.

        groundtruth event: {
            groundtruth_event_type_ids: (batch)
            groundtruth_event_mask: (batch)
            groundtruth_event_src_ids: (batch)
            groundtruth_event_src_mask: (batch)
            groundtruth_event_dst_ids: (batch)
            groundtruth_event_dst_mask: (batch)
            groundtruth_event_label_tgt_word_ids: (batch, groundtruth_event_label_len)
            groundtruth_event_label_tgt_mask: (batch, groundtruth_event_label_len)
        }
        Ground-truth graph events used for teacher forcing of the decoder

        max_label_decode_len: max length of labels to decode

        gumbel_greedy_decode_labels: whether to perform gumbel greedy decoding or not
        gumbel_tau: gumbel greedy decoding temperature

        output:
        {
            event_type_logits: (batch, num_event_type)
            event_src_logits: (batch, max_sub_graph_num_node)
            event_dst_logits: (batch, max_sub_graph_num_node)
            event_label_logits: (batch, groundtruth_event_label_len, num_word),
                returned only if groundtruth event is given
            decoded_event_label_word_ids: (batch, decoded_label_len) or
                one-hot encoded (batch, decoded_label_len, num_word)
                returned only if groundtruth event is not given
            decoded_event_label_mask: (batch, decoded_label_len)
                returned only if groundtruth event is not given
            updated_prev_input_event_emb_seq:
                (graph_event_decoder_num_dec_blocks, batch,
                 input_seq_len, graph_event_decoder_hidden_dim)
            updated_prev_input_event_emb_seq_mask: (batch, input_seq_len)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)
            updated_batched_graph: Batch(
                batch: (num_node)
                x: (num_node, node_label_len) or
                    one-hot encoded (num_node, node_label_len, num_word)
                node_label_mask: (num_node, node_label_len)
                node_last_update: (num_node, 2)
                edge_index: (2, num_edge)
                edge_attr: (num_edge, edge_label_len) or
                    one-hot encoded (num_edge, edge_label_len, num_word)
                edge_label_mask: (num_edge, edge_label_len)
                edge_last_update: (num_edge, 2)
            )
            batch_node_embeddings: (batch, max_sub_graph_num_node, hidden_dim)
            batch_node_mask: (batch, max_sub_graph_num_node)
            self_attn_weights:
                len([(batch, 1, input_seq_len), ...]) == num_decoder_block,
            obs_graph_attn_weights:
                len([(batch, 1, obs_len), ...]) == num_decoder_block,
            prev_action_graph_attn_weights:
                len([(batch, 1, prev_action_len), ...]) == num_decoder_block,
            graph_obs_attn_weights:
                len([(batch, 1, num_node), ...]) == num_decoder_block,
            graph_prev_action_attn_weights:
                len([(batch, 1, num_node), ...]) == num_decoder_block,
        }
        """
        if encoded_obs is None:
            assert obs_word_ids is not None
            encoded_obs = self.text_encoder(obs_word_ids, obs_mask)
            # (batch, obs_len, hidden_dim)
        if encoded_prev_action is None:
            assert prev_action_word_ids is not None
            encoded_prev_action = self.text_encoder(
                prev_action_word_ids, prev_action_mask
            )
            # (batch, prev_action_len, hidden_dim)

        # update the batched graph
        # each timestamp is a 2-dimensional vector (game step, graph event step)
        event_timestamps = torch.stack(
            [
                timestamps,
                torch.tensor(
                    prev_input_event_emb_seq.size(2)
                    if prev_input_event_emb_seq is not None
                    else 0,
                    device=timestamps.device,
                ).expand(timestamps.size(0)),
            ],
            dim=1,
        )
        # (batch, 2)
        updated_batched_graph = update_batched_graph(
            prev_batched_graph,
            event_type_ids,
            event_src_ids,
            event_dst_ids,
            event_label_word_ids,
            event_label_mask,
            event_timestamps,
        )
        if updated_batched_graph.num_nodes == 0:
            node_embeddings = torch.zeros(
                0, self.hidden_dim, device=event_type_ids.device
            )
            # (0, hidden_dim)
        else:
            node_embeddings = self.dynamic_gnn(
                Batch(
                    batch=updated_batched_graph.batch,
                    x=self.embed_label(
                        updated_batched_graph.x, updated_batched_graph.node_label_mask
                    ),
                    node_last_update=updated_batched_graph.node_last_update,
                    edge_index=updated_batched_graph.edge_index,
                    edge_attr=self.embed_label(
                        updated_batched_graph.edge_attr,
                        updated_batched_graph.edge_label_mask,
                    ),
                    edge_last_update=updated_batched_graph.edge_last_update,
                )
            )
            # (num_node, hidden_dim)

        # batchify node_embeddings
        batch_node_embeddings, batch_node_mask = to_dense_batch(
            node_embeddings,
            batch=updated_batched_graph.batch,
            batch_size=obs_mask.size(0),
        )
        # batch_node_embeddings: (batch, max_sub_graph_num_node, hidden_dim)
        # batch_node_mask: (batch, max_sub_graph_num_node)

        h_og, h_go = self.repr_aggr(
            encoded_obs,
            batch_node_embeddings,
            obs_mask,
            batch_node_mask,
        )
        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            encoded_prev_action,
            batch_node_embeddings,
            prev_action_mask,
            batch_node_mask,
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, num_node, hidden_dim)

        masks = compute_masks_from_event_type_ids(event_type_ids)
        decoder_output, decoder_attn_weights = self.graph_event_decoder(
            self.get_decoder_input(
                event_type_ids,
                event_src_ids,
                event_dst_ids,
                event_label_word_ids,
                event_label_mask,
                prev_batched_graph.batch,
                prev_batched_graph.x,
                prev_batched_graph.node_label_mask,
                masks,
            ),
            event_type_ids != EVENT_TYPE_ID_MAP["pad"],
            h_og,
            obs_mask,
            h_ag,
            prev_action_mask,
            h_go,
            h_ga,
            batch_node_mask,
            prev_input_event_emb_seq=prev_input_event_emb_seq,
            prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
        )

        event_type_logits = self.event_type_head(decoder_output["output"])
        # (batch, num_event_type)
        if groundtruth_event is not None:
            autoregressive_emb = self.event_type_head.get_autoregressive_embedding(
                decoder_output["output"],
                groundtruth_event["groundtruth_event_type_ids"],
                groundtruth_event["groundtruth_event_mask"],
            )
            # (batch, hidden_dim)
        else:
            autoregressive_emb = self.event_type_head.get_autoregressive_embedding(
                decoder_output["output"],
                F.gumbel_softmax(event_type_logits, hard=True, tau=gumbel_tau)
                if gumbel_greedy_decode_labels
                else event_type_logits.argmax(dim=1),
                masks["event_mask"],
            )
            # (batch, hidden_dim)
        event_src_logits, event_src_key = self.event_src_head(
            autoregressive_emb, batch_node_embeddings
        )
        # event_src_logits: (batch, max_sub_graph_num_node)
        # event_src_key:
        #   (batch, max_sub_graph_num_node, graph_event_decoder_key_query_dim)
        if groundtruth_event is not None:
            autoregressive_emb = self.event_src_head.update_autoregressive_embedding(
                autoregressive_emb,
                groundtruth_event["groundtruth_event_src_ids"],
                batch_node_embeddings,
                batch_node_mask,
                groundtruth_event["groundtruth_event_src_mask"],
                event_src_key,
            )
        elif event_src_logits.size(1) > 0:
            autoregressive_emb = self.event_src_head.update_autoregressive_embedding(
                autoregressive_emb,
                masked_gumbel_softmax(
                    event_src_logits, batch_node_mask, hard=True, tau=gumbel_tau
                )
                if gumbel_greedy_decode_labels
                else masked_softmax(event_src_logits, batch_node_mask, dim=1).argmax(
                    dim=1
                ),
                batch_node_embeddings,
                batch_node_mask,
                masks["src_mask"],
                event_src_key,
            )
        event_dst_logits, event_dst_key = self.event_dst_head(
            autoregressive_emb, batch_node_embeddings
        )
        # event_dst_logits: (batch, max_sub_graph_num_node)
        # event_dst_key:
        #   (batch, max_sub_graph_num_node, graph_event_decoder_key_query_dim)
        if groundtruth_event is not None:
            autoregressive_emb = self.event_dst_head.update_autoregressive_embedding(
                autoregressive_emb,
                groundtruth_event["groundtruth_event_dst_ids"],
                batch_node_embeddings,
                batch_node_mask,
                groundtruth_event["groundtruth_event_dst_mask"],
                event_dst_key,
            )
        elif event_dst_logits.size(1) > 0:
            autoregressive_emb = self.event_dst_head.update_autoregressive_embedding(
                autoregressive_emb,
                masked_gumbel_softmax(
                    event_dst_logits, batch_node_mask, hard=True, tau=gumbel_tau
                )
                if gumbel_greedy_decode_labels
                else masked_softmax(event_dst_logits, batch_node_mask, dim=1).argmax(
                    dim=1
                ),
                batch_node_embeddings,
                batch_node_mask,
                masks["dst_mask"],
                event_dst_key,
            )
        output = {
            "event_type_logits": event_type_logits,
            "event_src_logits": event_src_logits,
            "event_dst_logits": event_dst_logits,
            "updated_prev_input_event_emb_seq": decoder_output[
                "updated_prev_input_event_emb_seq"
            ],
            "updated_prev_input_event_emb_seq_mask": (
                decoder_output["updated_prev_input_event_emb_seq_mask"]
            ),
            "encoded_obs": encoded_obs,
            "encoded_prev_action": encoded_prev_action,
            "updated_batched_graph": updated_batched_graph,
            "batch_node_embeddings": batch_node_embeddings,
            "batch_node_mask": batch_node_mask,
            **decoder_attn_weights,
        }
        if groundtruth_event is not None:
            event_label_logits, _ = self.event_label_head(
                groundtruth_event["groundtruth_event_label_tgt_word_ids"],
                groundtruth_event["groundtruth_event_label_tgt_mask"],
                autoregressive_embedding=autoregressive_emb,
            )
            # (batch, groundtruth_event_label_len, num_word)
            output["event_label_logits"] = event_label_logits
        else:
            if gumbel_greedy_decode_labels:
                (
                    decoded_event_label_word_ids,
                    decoded_event_label_mask,
                ) = self.event_label_head.gumbel_greedy_decode(
                    autoregressive_emb,
                    max_decode_len=max_label_decode_len,
                    tau=gumbel_tau,
                )
                # decoded_event_label_word_ids: one-hot encoded
                # (batch, decoded_label_len, num_word)
                # decoded_event_label_mask: (batch, decoded_label_len)
            else:
                (
                    decoded_event_label_word_ids,
                    decoded_event_label_mask,
                ) = self.event_label_head.greedy_decode(
                    autoregressive_emb, max_decode_len=max_label_decode_len
                )
                # decoded_event_label_word_ids: (batch, decoded_label_len)
                # decoded_event_label_mask: (batch, decoded_label_len)
            output["decoded_event_label_word_ids"] = decoded_event_label_word_ids
            output["decoded_event_label_mask"] = decoded_event_label_mask
        return output

    def embed_label(
        self, label_word_ids: torch.Tensor, label_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        label_word_ids: (batch, label_len) or
            one-hot encoded (batch, label_len, num_word)
        label_mask: (batch, label_len)

        output: (batch, hidden_dim)
        """
        return masked_mean(self.text_encoder(label_word_ids, label_mask), label_mask)

    def get_decoder_input(
        self,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_word_ids: torch.Tensor,
        event_label_mask: torch.Tensor,
        batch: torch.Tensor,
        node_label_word_ids: torch.Tensor,
        node_label_mask: torch.Tensor,
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Turn the graph events into decoder inputs by concatenating their embeddings.

        input:
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_word_ids: (batch, event_label_len) or
                one-hot encoded (batch, event_label_len, num_word)
            event_label_mask: (batch, event_label_len)
            batch: (num_node)
            node_label_word_ids: (num_node, node_label_len) or
                one-hot encoded (num_node, node_label_len, num_word)
            node_label_mask: (num_node, node_label_len)
            masks: masks calculated by compute_masks_from_event_type_ids(event_type_ids)

        output: (batch, decoder_input_dim)
            where decoder_input_dim = event_type_embedding_dim + 3 * label_embedding_dim
        """
        batch_size = event_type_ids.size(0)

        # event type embeddings
        event_type_embs = self.event_type_embeddings(event_type_ids)
        # (batch, event_type_embedding_dim)

        # event label embeddings
        event_label_embs = torch.zeros(
            batch_size, self.hidden_dim, device=event_type_ids.device
        )
        # (batch, label_embedding_dim)
        event_label_embs[masks["label_mask"]] = self.embed_label(
            event_label_word_ids[masks["label_mask"]],
            event_label_mask[masks["label_mask"]],
        )
        # (batch, label_embedding_dim)

        # event source/destination node embeddings
        event_src_embs = torch.zeros(
            batch_size, self.hidden_dim, device=event_type_ids.device
        )
        # (batch, label_embedding_dim)
        event_dst_embs = torch.zeros(
            batch_size, self.hidden_dim, device=event_type_ids.device
        )
        # (batch, label_embedding_dim)
        node_id_offsets = calculate_node_id_offsets(batch_size, batch)
        # (batch)
        if masks["src_mask"].any():
            event_src_embs[masks["src_mask"]] = self.embed_label(
                node_label_word_ids[
                    (event_src_ids + node_id_offsets)[masks["src_mask"]]
                ],
                node_label_mask[(event_src_ids + node_id_offsets)[masks["src_mask"]]],
            )
            # (batch, label_embedding_dim)

        if masks["dst_mask"].any():
            event_dst_embs[masks["dst_mask"]] = self.embed_label(
                node_label_word_ids[
                    (event_dst_ids + node_id_offsets)[masks["dst_mask"]]
                ],
                node_label_mask[(event_dst_ids + node_id_offsets)[masks["dst_mask"]]],
            )
            # (batch, label_embedding_dim)

        return torch.cat(
            [event_type_embs, event_src_embs, event_dst_embs, event_label_embs], dim=1
        )
        # (batch, event_type_embedding_dim + 3 * label_embedding_dim)

    @staticmethod
    def filter_invalid_events(
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a mask for invalid events. False for valid events and
        True for invalid events.

        node-add: all are valid
        node-delete:
            - nodes should exist
            - nodes cannot have edges
        edge-add:
            - nodes should exist
        edge-delete:
            - nodes should exist

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        batch: (num_node)
        edge_index: (2, num_edge)

        output: invalid event mask (batch)
        """
        batch_size = event_type_ids.size(0)
        batch_bincount = batch.bincount()
        subgraph_num_node = F.pad(
            batch_bincount, (0, batch_size - batch_bincount.size(0))
        )
        # (batch)

        invalid_src_mask = src_ids >= subgraph_num_node
        # (batch)
        invalid_dst_mask = dst_ids >= subgraph_num_node
        # (batch)

        # node-delete
        node_id_offsets = calculate_node_id_offsets(event_type_ids.size(0), batch)
        # (batch)
        batch_src_ids = src_ids + node_id_offsets
        # (batch)
        nodes_with_edges = torch.any(
            batch_src_ids.unsqueeze(-1) == edge_index.flatten(), dim=1
        )
        # (batch)
        invalid_node_delete_event_mask = invalid_src_mask.logical_or(
            nodes_with_edges
        ).logical_and(event_type_ids == EVENT_TYPE_ID_MAP["node-delete"])
        # (batch)

        invalid_edge_mask = invalid_src_mask.logical_or(invalid_dst_mask)
        # (batch)
        invalid_edge_add_event_mask = invalid_edge_mask.logical_and(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
        )
        # (batch)

        # if the edge doesn't exist, we still output it as was done
        # in the original GATA paper
        invalid_edge_delete_event_mask = invalid_edge_mask.logical_and(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
        )
        # (batch)

        return invalid_node_delete_event_mask.logical_or(
            invalid_edge_add_event_mask
        ).logical_or(invalid_edge_delete_event_mask)
        # (batch)
