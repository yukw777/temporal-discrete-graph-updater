import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Dict

from dgu.nn.text import TextEncoder
from dgu.nn.rep_aggregator import ReprAggregator
from dgu.nn.utils import masked_mean
from dgu.nn.graph_event_decoder import (
    StaticLabelGraphEventEncoder,
    StaticLabelGraphEventDecoder,
    RNNGraphEventSeq2Seq,
)
from dgu.nn.temporal_graph import TemporalGraphNetwork


class StaticLabelDiscreteGraphUpdater(pl.LightningModule):
    """
    StaticLabelDiscreteGraphUpdater is essentially a Seq2Seq model which encodes
    a sequence of game steps, each with an observation and a previous action, and
    decodes a sequence of graph events.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_num_nodes: int,
        max_num_edges: int,
        word_emb_dim: int,
        text_encoder_num_blocks: int,
        text_encoder_num_conv_layers: int,
        text_encoder_kernel_size: int,
        text_encoder_num_heads: int,
        graph_event_decoder_key_query_dim: int,
        max_decode_len: int,
        pretrained_word_embeddings: nn.Embedding,
        node_label_embeddings: torch.Tensor,
        edge_label_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        # constants
        self.hidden_dim = hidden_dim
        self.max_decode_len = max_decode_len

        # word embeddings
        assert word_emb_dim == pretrained_word_embeddings.embedding_dim
        self.word_embeddings = nn.Sequential(
            pretrained_word_embeddings, nn.Linear(word_emb_dim, hidden_dim)
        )

        # text encoder
        self.text_encoder = TextEncoder(
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            hidden_dim,
            text_encoder_num_heads,
        )

        # temporal graph network
        assert node_label_embeddings.size(1) == edge_label_embeddings.size(1)
        self.tgn = TemporalGraphNetwork(
            max_num_nodes, max_num_edges, hidden_dim, node_label_embeddings.size(1)
        )

        # representation aggregator
        self.repr_aggr = ReprAggregator(hidden_dim)

        # graph event seq2seq
        self.seq2seq = RNNGraphEventSeq2Seq(
            hidden_dim,
            max_decode_len,
            node_label_embeddings.size(1),
            StaticLabelGraphEventEncoder(),
            StaticLabelGraphEventDecoder(
                hidden_dim,
                graph_event_decoder_key_query_dim,
                node_label_embeddings,
                edge_label_embeddings,
            ),
        )

    def forward(  # type: ignore
        self,
        obs_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_word_ids: torch.Tensor,
        prev_action_mask: torch.Tensor,
        prev_node_embeddings: torch.Tensor,
        tgt_event_type_ids: Optional[torch.Tensor] = None,
        tgt_event_src_ids: Optional[torch.Tensor] = None,
        tgt_event_src_mask: Optional[torch.Tensor] = None,
        tgt_event_dst_ids: Optional[torch.Tensor] = None,
        tgt_event_dst_mask: Optional[torch.Tensor] = None,
        tgt_event_label_ids: Optional[torch.Tensor] = None,
        tgt_event_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input:
            obs_word_ids: (batch, obs_len)
            obs_mask: (batch, obs_len)
            prev_action_word_ids: (batch, prev_action_len)
            prev_action_mask: (batch, prev_action_len)
            prev_node_embeddings: (prev_num_node, hidden_dim)
            tgt_event_type_ids: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_src_ids: (graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_src_mask: (graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_dst_ids: (graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_dst_mask: (graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_label_ids: (graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_mask: (graph_event_seq_len)
                Used for teacher forcing.

        output:
        if training:
            {
                event_type_logits: (graph_event_seq_len, num_event_type)
                src_logits: (graph_event_seq_len, prev_num_node)
                dst_logits: (graph_event_seq_len, prev_num_node)
                label_logits: (graph_event_seq_len, num_label)
            }
        else:
            {
                decoded_event_type_ids: (decoded_len),
                decoded_src_ids: (decoded_len),
                decoded_dst_ids: (decoded_len),
                decoded_label_ids: (decoded_len),
            }
        """
        encoded_obs = self.encode_text(obs_word_ids, obs_mask)
        # (batch, obs_len, hidden_dim)
        encoded_prev_action = self.encode_text(prev_action_word_ids, prev_action_mask)
        # (batch, prev_action_len, hidden_dim)

        delta_g = self.f_delta(
            prev_node_embeddings,
            encoded_obs,
            obs_mask,
            encoded_prev_action,
            prev_action_mask,
        )
        # (batch, 4 * hidden_dim)

        # treat batches of delta_g as one sequence and pass it to the seq2seq model
        # this works b/c every batch is sequential (no shuffling).
        results = self.seq2seq(
            delta_g.unsqueeze(0),
            prev_node_embeddings,
            tgt_event_mask=None
            if tgt_event_mask is None
            else tgt_event_mask.unsqueeze(0),
            tgt_event_type_ids=None
            if tgt_event_type_ids is None
            else tgt_event_type_ids.unsqueeze(0),
            tgt_event_src_ids=None
            if tgt_event_src_ids is None
            else tgt_event_src_ids.unsqueeze(0),
            tgt_event_src_mask=None
            if tgt_event_src_mask is None
            else tgt_event_src_mask.unsqueeze(0),
            tgt_event_dst_ids=None
            if tgt_event_dst_ids is None
            else tgt_event_dst_ids.unsqueeze(0),
            tgt_event_dst_mask=None
            if tgt_event_dst_mask is None
            else tgt_event_dst_mask.unsqueeze(0),
            tgt_event_label_ids=None
            if tgt_event_label_ids is None
            else tgt_event_label_ids.unsqueeze(0),
        )
        # squeeze the batch dimension and return
        return {k: t.squeeze(0) for k, t in results.items()}

    def encode_text(self, word_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        word_ids: (batch, seq_len)
        mask: (batch, seq_len)
        output: (batch, seq_len, hidden_dim)
        """
        word_embs = self.word_embeddings(word_ids)
        # (batch, seq_len, hidden_dim)
        return self.text_encoder(word_embs, mask)
        # (batch, seq_len, hidden_dim)

    def f_delta(
        self,
        prev_node_embeddings: torch.Tensor,
        obs_embeddings: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_embeddings: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        prev_node_embeddings: (prev_num_node, hidden_dim)
        obs_embeddings: (batch, obs_len, hidden_dim)
        obs_mask: (batch, obs_len)
        prev_action_embeddings: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, 4 * hidden_dim)
        """
        batch_size = obs_embeddings.size(0)
        expanded_prev_node_embeddings = prev_node_embeddings.expand(batch_size, -1, -1)

        # no masks necessary for prev_node_embeddings, so just create a fake one
        prev_node_mask = torch.ones(
            batch_size, prev_node_embeddings.size(0), device=prev_node_embeddings.device
        )

        h_og, h_go = self.repr_aggr(
            obs_embeddings,
            expanded_prev_node_embeddings,
            obs_mask,
            prev_node_mask,
        )
        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, prev_num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            prev_action_embeddings,
            expanded_prev_node_embeddings,
            prev_action_mask,
            prev_node_mask,
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, prev_num_node, hidden_dim)

        mean_h_og = masked_mean(h_og, obs_mask)
        # (batch, hidden_dim)
        mean_h_go = masked_mean(h_go, prev_node_mask)
        # (batch, hidden_dim)
        mean_h_ag = masked_mean(h_ag, prev_action_mask)
        # (batch, hidden_dim)
        mean_h_ga = masked_mean(h_ga, prev_node_mask)
        # (batch, hidden_dim)

        return torch.cat([mean_h_og, mean_h_go, mean_h_ag, mean_h_ga], dim=1)
        # (batch, 4 * hidden_dim)
