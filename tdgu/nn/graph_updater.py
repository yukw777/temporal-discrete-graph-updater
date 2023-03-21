from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from tdgu.constants import EVENT_TYPE_ID_MAP
from tdgu.data import TWCmdGenGraphEventStepInput
from tdgu.nn.dynamic_gnn import DynamicGNN
from tdgu.nn.graph_event_decoder import (
    EventNodeHead,
    EventSequentialLabelHead,
    EventTypeHead,
    TransformerGraphEventDecoder,
)
from tdgu.nn.rep_aggregator import ReprAggregator
from tdgu.nn.text import TextEncoder
from tdgu.nn.utils import (
    calculate_node_id_offsets,
    compute_masks_from_event_type_ids,
    masked_gumbel_softmax,
    masked_softmax,
    update_batched_graph,
)


class TemporalDiscreteGraphUpdater(pl.LightningModule):
    """TemporalDiscreteGraphUpdater is essentially a Seq2Seq model which
    encodes a sequence of game steps, each with an observation and a previous
    action, and decodes a sequence of graph events."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        dynamic_gnn: DynamicGNN,
        hidden_dim: int,
        graph_event_decoder_event_type_emb_dim: int,
        graph_event_decoder_autoregressive_emb_dim: int,
        graph_event_decoder_key_query_dim: int,
        graph_event_decoder_num_dec_blocks: int,
        graph_event_decoder_dec_block_num_heads: int,
        graph_event_decoder_hidden_dim: int,
        label_head_bos_token_id: int,
        label_head_eos_token_id: int,
        label_head_pad_token_id: int,
        vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # vocab size
        self.vocab_size = vocab_size

        # text encoder
        self.text_encoder = text_encoder

        # dynamic graph neural network
        self.dynamic_gnn = dynamic_gnn

        # representation aggregator
        self.repr_aggr = ReprAggregator(self.hidden_dim)

        # graph event decoder
        self.event_type_embeddings = nn.Embedding(
            len(EVENT_TYPE_ID_MAP),
            graph_event_decoder_event_type_emb_dim,
            padding_idx=EVENT_TYPE_ID_MAP["pad"],
        )
        self.graph_event_decoder = TransformerGraphEventDecoder(
            graph_event_decoder_event_type_emb_dim + 3 * hidden_dim,
            hidden_dim,
            graph_event_decoder_num_dec_blocks,
            graph_event_decoder_dec_block_num_heads,
            graph_event_decoder_hidden_dim,
            dropout=dropout,
        )

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
        obs_word_ids: torch.Tensor | None = None,
        prev_action_word_ids: torch.Tensor | None = None,
        encoded_obs: torch.Tensor | None = None,
        encoded_prev_action: torch.Tensor | None = None,
        # previous input event embedding sequence
        prev_input_event_emb_seq: torch.Tensor | None = None,
        prev_input_event_emb_seq_mask: torch.Tensor | None = None,
        # groundtruth events for decoder teacher-force training
        groundtruth_event: dict[str, torch.Tensor] | None = None,
        max_label_decode_len: int = 10,
        gumbel_greedy_decode: bool = False,
        gumbel_tau: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        graph events: {
            event_type_ids: (batch) or one-hot encoded (batch, num_event_type)
            event_src_ids: (batch) or
                one-hot encoded (batch, prev_max_sub_graph_num_node)
            event_dst_ids: (batch) or
                one-hot encoded (batch, prev_max_sub_graph_num_node)
            event_label_word_ids_or_one_hot: (batch, event_label_len) or
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

        gumbel_greedy_decode: whether to perform gumbel greedy decoding or not
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
            encoded_obs = self.text_encoder(obs_word_ids, obs_mask)["encoded"]
            # (batch, obs_len, hidden_dim)
        if encoded_prev_action is None:
            assert prev_action_word_ids is not None
            encoded_prev_action = self.text_encoder(
                prev_action_word_ids, prev_action_mask
            )["encoded"]
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
        event_type_ids_argmax = (
            event_type_ids.argmax(dim=-1) if gumbel_greedy_decode else event_type_ids
        )
        # (batch)
        if gumbel_greedy_decode:
            if event_src_ids.size(1) == 0:
                event_src_ids_argmax = torch.zeros(
                    event_src_ids.size(0), device=event_src_ids.device, dtype=torch.long
                )
            else:
                event_src_ids_argmax = event_src_ids.argmax(dim=-1)
        else:
            event_src_ids_argmax = event_src_ids
        # (batch)
        if gumbel_greedy_decode:
            if event_dst_ids.size(1) == 0:
                event_dst_ids_argmax = torch.zeros(
                    event_dst_ids.size(0), device=event_dst_ids.device, dtype=torch.long
                )
            else:
                event_dst_ids_argmax = event_dst_ids.argmax(dim=-1)
        else:
            event_dst_ids_argmax = event_dst_ids
        # (batch)
        updated_batched_graph = update_batched_graph(
            prev_batched_graph,
            event_type_ids_argmax,
            event_src_ids_argmax,
            event_dst_ids_argmax,
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
                    x=self.text_encoder(
                        updated_batched_graph.x,
                        updated_batched_graph.node_label_mask,
                        return_pooled_output=True,
                    )["pooled_output"],
                    node_last_update=updated_batched_graph.node_last_update,
                    edge_index=updated_batched_graph.edge_index,
                    edge_attr=self.text_encoder(
                        updated_batched_graph.edge_attr,
                        updated_batched_graph.edge_label_mask,
                        return_pooled_output=True,
                    )["pooled_output"],
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

        masks = compute_masks_from_event_type_ids(event_type_ids_argmax)
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
            event_type_ids_argmax != EVENT_TYPE_ID_MAP["pad"],
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
                if gumbel_greedy_decode
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
                if gumbel_greedy_decode
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
                if gumbel_greedy_decode
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
            if gumbel_greedy_decode:
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
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Turn the graph events into decoder inputs by concatenating their
        embeddings.

        input:
            event_type_ids_or_one_hot: (batch) or
                one-hot encoded (batch, num_event_type)
            event_src_ids: (batch) or
                one-hot encoded (batch, prev_max_sub_graph_num_node)
            event_dst_ids: (batch) or
                one-hot encoded (batch, prev_max_sub_graph_num_node)
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
        if event_type_ids.dim() == 1:
            event_type_embs = self.event_type_embeddings(event_type_ids)
        else:
            event_type_embs = event_type_ids.matmul(self.event_type_embeddings.weight)
        # (batch, event_type_embedding_dim)

        # event label embeddings
        event_label_embs = torch.zeros(
            batch_size, self.hidden_dim, device=event_type_ids.device
        )
        # (batch, label_embedding_dim)
        event_label_embs[masks["label_mask"]] = self.text_encoder(
            event_label_word_ids[masks["label_mask"]],
            event_label_mask[masks["label_mask"]],
            return_pooled_output=True,
        )["pooled_output"]
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
        if event_src_ids.dim() == 1:
            node_id_offsets = calculate_node_id_offsets(batch_size, batch)
            # (batch)
            if masks["src_mask"].any():
                event_src_embs[masks["src_mask"]] = self.text_encoder(
                    node_label_word_ids[
                        (event_src_ids + node_id_offsets)[masks["src_mask"]]
                    ],
                    node_label_mask[
                        (event_src_ids + node_id_offsets)[masks["src_mask"]]
                    ],
                    return_pooled_output=True,
                )["pooled_output"]
            if masks["dst_mask"].any():
                event_dst_embs[masks["dst_mask"]] = self.text_encoder(
                    node_label_word_ids[
                        (event_dst_ids + node_id_offsets)[masks["dst_mask"]]
                    ],
                    node_label_mask[
                        (event_dst_ids + node_id_offsets)[masks["dst_mask"]]
                    ],
                    return_pooled_output=True,
                )["pooled_output"]
        else:
            batch_node_label_word_ids, _ = to_dense_batch(
                node_label_word_ids, batch=batch, batch_size=batch_size
            )
            # batch_node_label_word_ids: one-hot encoded
            # (batch, prev_max_sub_graph_num_node, node_label_len, num_word)
            batch_node_label_mask, _ = to_dense_batch(
                node_label_mask, batch=batch, batch_size=batch_size
            )
            # batch_node_label_mask: one-hot encoded
            # (batch, prev_max_sub_graph_num_node, node_label_len)
            if masks["src_mask"].any():
                event_src_embs[masks["src_mask"]] = self.text_encoder(
                    torch.sum(
                        batch_node_label_word_ids
                        * event_src_ids.view(batch_size, -1, 1, 1),
                        dim=1,
                    )[masks["src_mask"]],
                    # (batch, node_label_len, num_word)
                    torch.sum(
                        batch_node_label_mask * event_src_ids.unsqueeze(-1), dim=1
                    ).bool()[masks["src_mask"]],
                    # (batch, node_label_len)
                    return_pooled_output=True,
                )["pooled_output"]
            if masks["dst_mask"].any():
                event_dst_embs[masks["dst_mask"]] = self.text_encoder(
                    torch.sum(
                        batch_node_label_word_ids
                        * event_dst_ids.view(batch_size, -1, 1, 1),
                        dim=1,
                    )[masks["dst_mask"]],
                    # (batch, node_label_len, num_word)
                    torch.sum(
                        batch_node_label_mask * event_dst_ids.unsqueeze(-1), dim=1
                    ).bool()[masks["dst_mask"]],
                    # (batch, node_label_len)
                    return_pooled_output=True,
                )["pooled_output"]

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
        """Return a mask for invalid events. False for valid events and True
        for invalid events.

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

    def greedy_decode(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        prev_batched_graph: Batch,
        max_event_decode_len: int = 100,
        max_label_decode_len: int = 10,
        gumbel_greedy_decode: bool = False,
        gumbel_tau: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        step_input: the current step input
        prev_batch_graph: diagonally stacked batch of current graphs
        max_event_decode_len: max length of decoded event sequence
        max_label_decode_len: max length of decoded labels
        gumbel: greedy decode using gumbel softmax
        gumbel_tau: gumbel temperature, only used if one_hot is True

        output:
        len([{
            decoded_event_type_ids: one-hot encoded
                (batch, num_event_type) if gumbel_greedy_decode is True
                (batch) otherwise
            decoded_event_src_ids: one-hot encoded
                (batch, max_sub_graph_num_node) if gumbel_greedy_decode is True
                (batch) otherwise
            decoded_event_dst_ids: one-hot encoded
                (batch, max_sub_graph_num_node) if gumbel_greedy_decode is True
                (batch) otherwise
            decoded_event_label_word_ids: one-hot encoded
                (batch, decoded_label_len, num_word) if gumbel_greedy_decode is True
                (batch, decoded_label_len) otherwise
            decoded_event_label_mask: (batch, decoded_label_len)
            updated_batched_graph: diagonally stacked batch of updated graphs.
                these are the graphs used to decode the graph events above
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
        }, ...]) == decode_len <= max_event_decode_len
        """
        # initialize the initial inputs
        batch_size = step_input.obs_word_ids.size(0)
        if gumbel_greedy_decode:
            decoded_event_type_ids = F.one_hot(
                torch.full(  # type: ignore
                    (batch_size,), EVENT_TYPE_ID_MAP["start"], device=self.device
                ),
                num_classes=len(EVENT_TYPE_ID_MAP),
            ).float()
            # (batch, num_event_type)
            prev_max_sub_graph_num_node = (
                prev_batched_graph.batch.bincount().max()
                if prev_batched_graph.num_nodes > 0
                else 0
            )
            decoded_src_ids = torch.zeros(  # type: ignore
                batch_size, prev_max_sub_graph_num_node, device=self.device
            )
            # (batch, prev_max_sub_graph_num_node)
            decoded_dst_ids = torch.zeros(  # type: ignore
                batch_size, prev_max_sub_graph_num_node, device=self.device
            )
            # (batch, prev_max_sub_graph_num_node)
            decoded_label_word_ids = torch.empty(  # type: ignore
                batch_size, 0, self.vocab_size, device=self.device
            )
            # (batch, 0, num_word)
        else:
            decoded_event_type_ids = torch.full(  # type: ignore
                (batch_size,), EVENT_TYPE_ID_MAP["start"], device=self.device
            )
            # (batch)
            decoded_src_ids = torch.zeros(  # type: ignore
                batch_size, device=self.device, dtype=torch.long
            )
            # (batch)
            decoded_dst_ids = torch.zeros(  # type: ignore
                batch_size, device=self.device, dtype=torch.long
            )
            # (batch)
            decoded_label_word_ids = torch.empty(  # type: ignore
                batch_size, 0, device=self.device, dtype=torch.long
            )
            # (batch, 0)
        decoded_label_mask = torch.empty(  # type: ignore
            batch_size, 0, device=self.device, dtype=torch.bool
        )
        # (batch, 0)

        end_event_mask = torch.full(
            (batch_size,),
            False,
            device=self.device,
        )  # type: ignore
        # (batch)

        prev_input_event_emb_seq: torch.Tensor | None = None
        prev_input_event_emb_seq_mask: torch.Tensor | None = None
        encoded_obs: torch.Tensor | None = None
        encoded_prev_action: torch.Tensor | None = None
        results_list: list[dict[str, Any]] = []
        for _ in range(max_event_decode_len):
            results = self(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                decoded_label_word_ids,
                decoded_label_mask,
                prev_batched_graph,
                step_input.obs_mask,
                step_input.prev_action_mask,
                step_input.timestamps,
                obs_word_ids=step_input.obs_word_ids,
                prev_action_word_ids=step_input.prev_action_word_ids,
                encoded_obs=encoded_obs,
                encoded_prev_action=encoded_prev_action,
                prev_input_event_emb_seq=prev_input_event_emb_seq,
                prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
                max_label_decode_len=max_label_decode_len,
                gumbel_greedy_decode=gumbel_greedy_decode,
                gumbel_tau=gumbel_tau,
            )

            # process the decoded result for the next iteration
            if gumbel_greedy_decode:
                # applying mask like this is OK, since the argmax of all zeros
                # is 0, which is the index of the pad event type.
                decoded_event_type_ids = F.gumbel_softmax(
                    results["event_type_logits"], hard=True, tau=gumbel_tau
                ) * end_event_mask.logical_not().unsqueeze(1)
                # (batch, num_event_type)
            else:
                decoded_event_type_ids = (
                    results["event_type_logits"]
                    .argmax(dim=1)
                    .masked_fill(end_event_mask, EVENT_TYPE_ID_MAP["pad"])
                )
                # (batch)

            if results["event_src_logits"].size(1) == 0:
                if gumbel_greedy_decode:
                    decoded_src_ids = torch.zeros(
                        batch_size,
                        0,
                        device=self.device,
                    )  # type: ignore
                    # (batch, 0)
                else:
                    decoded_src_ids = torch.zeros(  # type: ignore
                        batch_size, dtype=torch.long, device=self.device
                    )
                    # (batch)
            else:
                if gumbel_greedy_decode:
                    decoded_src_ids = masked_gumbel_softmax(
                        results["event_src_logits"],
                        results["batch_node_mask"],
                        hard=True,
                        tau=gumbel_tau,
                    )
                    # (batch, max_sub_graph_num_node)
                else:
                    decoded_src_ids = masked_softmax(
                        results["event_src_logits"], results["batch_node_mask"], dim=1
                    ).argmax(dim=1)
                    # (batch)
            if results["event_dst_logits"].size(1) == 0:
                if gumbel_greedy_decode:
                    decoded_dst_ids = torch.zeros(  # type: ignore
                        batch_size,
                        0,
                        device=self.device,
                    )
                    # (batch, 0)
                else:
                    decoded_dst_ids = torch.zeros(  # type: ignore
                        batch_size, dtype=torch.long, device=self.device
                    )
                    # (batch)
            else:
                if gumbel_greedy_decode:
                    decoded_dst_ids = masked_gumbel_softmax(
                        results["event_dst_logits"],
                        results["batch_node_mask"],
                        hard=True,
                        tau=gumbel_tau,
                    )
                    # (batch, max_sub_graph_num_node)
                else:
                    decoded_dst_ids = masked_softmax(
                        results["event_dst_logits"], results["batch_node_mask"], dim=1
                    ).argmax(dim=1)
                    # (batch)
            decoded_label_word_ids = results["decoded_event_label_word_ids"]
            # (batch, decoded_label_len, num_word) if one_hot is True
            # (batch, decoded_label_len) otherwise
            decoded_label_mask = results["decoded_event_label_mask"]
            # (batch, decoded_label_len)

            # filter out invalid decoded events
            if gumbel_greedy_decode:
                if decoded_src_ids.size(1) == 0:
                    decoded_src_ids_argmax = torch.zeros(
                        decoded_src_ids.size(0),
                        device=decoded_src_ids.device,
                        dtype=torch.long,
                    )
                else:
                    decoded_src_ids_argmax = decoded_src_ids.argmax(dim=-1)
            else:
                decoded_src_ids_argmax = decoded_src_ids
            # (batch)
            if gumbel_greedy_decode:
                if decoded_dst_ids.size(1) == 0:
                    decoded_dst_ids_argmax = torch.zeros(
                        decoded_dst_ids.size(0),
                        device=decoded_dst_ids.device,
                        dtype=torch.long,
                    )
                else:
                    decoded_dst_ids_argmax = decoded_dst_ids.argmax(dim=-1)
            else:
                decoded_dst_ids_argmax = decoded_dst_ids
            # (batch)
            invalid_event_mask = self.filter_invalid_events(
                decoded_event_type_ids.argmax(dim=-1)
                if gumbel_greedy_decode
                else decoded_event_type_ids,
                decoded_src_ids_argmax,
                decoded_dst_ids_argmax,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            # (batch)

            if gumbel_greedy_decode:
                decoded_event_type_ids = (
                    decoded_event_type_ids
                    * invalid_event_mask.logical_not().unsqueeze(1)
                )
            else:
                decoded_event_type_ids = decoded_event_type_ids.masked_fill(
                    invalid_event_mask, EVENT_TYPE_ID_MAP["pad"]
                )

            # collect the results
            results_list.append(
                {
                    "decoded_event_type_ids": decoded_event_type_ids,
                    "decoded_event_src_ids": decoded_src_ids,
                    "decoded_event_dst_ids": decoded_dst_ids,
                    "decoded_event_label_word_ids": decoded_label_word_ids,
                    "decoded_event_label_mask": decoded_label_mask,
                    "updated_batched_graph": results["updated_batched_graph"],
                    "batch_node_embeddings": results["batch_node_embeddings"],
                    "batch_node_mask": results["batch_node_mask"],
                    "self_attn_weights": results["self_attn_weights"],
                    "obs_graph_attn_weights": results["obs_graph_attn_weights"],
                    "prev_action_graph_attn_weights": results[
                        "prev_action_graph_attn_weights"
                    ],
                    "graph_obs_attn_weights": results["graph_obs_attn_weights"],
                    "graph_prev_action_attn_weights": results[
                        "graph_prev_action_attn_weights"
                    ],
                }
            )

            # update the batched graph
            prev_batched_graph = results["updated_batched_graph"]

            # update previous input event embedding sequence
            prev_input_event_emb_seq = results["updated_prev_input_event_emb_seq"]
            prev_input_event_emb_seq_mask = results[
                "updated_prev_input_event_emb_seq_mask"
            ]

            # save update the encoded observation and previous action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

            # update end_event_mask
            end_event_mask = end_event_mask.logical_or(
                (
                    decoded_event_type_ids.argmax(dim=-1)
                    if gumbel_greedy_decode
                    else decoded_event_type_ids
                )
                == EVENT_TYPE_ID_MAP["end"]
            )

            # if everything in the batch is done, break
            if end_event_mask.all():
                break

        return results_list
