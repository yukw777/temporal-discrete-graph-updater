import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Dict, List, Tuple
from torch.optim import Adam, Optimizer
from hydra.utils import to_absolute_path
from pathlib import Path

from dgu.nn.text import TextEncoder
from dgu.nn.rep_aggregator import ReprAggregator
from dgu.nn.utils import masked_mean, load_fasttext
from dgu.nn.graph_event_decoder import (
    StaticLabelGraphEventDecoder,
    RNNGraphEventDecoder,
)
from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS


class StaticLabelDiscreteGraphUpdater(pl.LightningModule):
    """
    StaticLabelDiscreteGraphUpdater is essentially a Seq2Seq model which encodes
    a sequence of game steps, each with an observation and a previous action, and
    decodes a sequence of graph events.
    """

    def __init__(
        self,
        hidden_dim: int = 8,
        max_num_nodes: int = 100,
        max_num_edges: int = 200,
        word_emb_dim: int = 300,
        tgn_event_type_emb_dim: int = 8,
        tgn_memory_dim: int = 8,
        tgn_time_enc_dim: int = 8,
        text_encoder_num_blocks: int = 1,
        text_encoder_num_conv_layers: int = 3,
        text_encoder_kernel_size: int = 5,
        text_encoder_num_heads: int = 1,
        graph_event_decoder_key_query_dim: int = 8,
        max_decode_len: int = 100,
        learning_rate: float = 5e-4,
        pretrained_word_embedding_path: Optional[str] = None,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "hidden_dim",
            "max_num_nodes",
            "max_num_edges",
            "word_emb_dim",
            "tgn_event_type_emb_dim",
            "tgn_memory_dim",
            "tgn_time_enc_dim",
            "text_encoder_num_blocks",
            "text_encoder_num_conv_layers",
            "text_encoder_kernel_size",
            "text_encoder_num_heads",
            "graph_event_decoder_key_query_dim",
            "max_decode_len",
            "learning_rate",
        )
        # preprocessor
        if word_vocab_path is not None:
            self.preprocessor = SpacyPreprocessor.load_from_file(
                to_absolute_path(word_vocab_path)
            )
        else:
            # just load with special tokens
            self.preprocessor = SpacyPreprocessor([PAD, UNK, BOS, EOS])

        # load pretrained word embedding and freeze it
        if pretrained_word_embedding_path is not None:
            abs_pretrained_word_embedding_path = Path(
                to_absolute_path(pretrained_word_embedding_path)
            )
            serialized_path = abs_pretrained_word_embedding_path.parent / (
                abs_pretrained_word_embedding_path.stem + ".pt"
            )
            pretrained_word_embeddings = load_fasttext(
                str(abs_pretrained_word_embedding_path),
                serialized_path,
                self.preprocessor,
            )
            assert word_emb_dim == pretrained_word_embeddings.embedding_dim
        else:
            pretrained_word_embeddings = nn.Embedding(
                len(self.preprocessor.word_to_id_dict), word_emb_dim
            )
        pretrained_word_embeddings.weight.requires_grad = False
        self.word_embeddings = nn.Sequential(
            pretrained_word_embeddings, nn.Linear(word_emb_dim, hidden_dim)
        )

        # calculate node/edge label embeddings
        if node_vocab_path is not None:
            with open(to_absolute_path(node_vocab_path), "r") as f:
                node_vocab = [node_name.strip() for node_name in f]
        else:
            # initialize with a single node
            node_vocab = ["node"]
        node_label_word_ids, node_label_mask = self.preprocessor.preprocess(node_vocab)
        node_label_embeddings = masked_mean(
            pretrained_word_embeddings(node_label_word_ids), node_label_mask
        )
        # load relation vocab
        if relation_vocab_path is not None:
            with open(to_absolute_path(relation_vocab_path), "r") as f:
                relation_vocab = [relation_name.strip() for relation_name in f]
        else:
            # initialize with a single relation
            relation_vocab = ["relation"]
        # add reverse relations
        relation_vocab += [rel + " reverse" for rel in relation_vocab]
        edge_label_word_ids, edge_label_mask = self.preprocessor.preprocess(
            relation_vocab
        )
        edge_label_embeddings = masked_mean(
            pretrained_word_embeddings(edge_label_word_ids), edge_label_mask
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
        self.tgn = TemporalGraphNetwork(
            max_num_nodes,
            max_num_edges,
            tgn_event_type_emb_dim,
            tgn_memory_dim,
            tgn_time_enc_dim,
            word_emb_dim,
            hidden_dim,
        )

        # representation aggregator
        self.repr_aggr = ReprAggregator(hidden_dim)

        # graph event seq2seq
        event_decoder = StaticLabelGraphEventDecoder(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
            node_label_embeddings,
            edge_label_embeddings,
        )
        self.decoder = RNNGraphEventDecoder(4 * hidden_dim, hidden_dim, event_decoder)
        self.label_embeddings = event_decoder.event_label_head.label_embeddings

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(  # type: ignore
        self,
        obs_mask: torch.Tensor,
        prev_action_mask: torch.Tensor,
        prev_event_type_ids: torch.Tensor,
        prev_event_src_ids: torch.Tensor,
        prev_event_src_mask: torch.Tensor,
        prev_event_dst_ids: torch.Tensor,
        prev_event_dst_mask: torch.Tensor,
        prev_event_edge_ids: torch.Tensor,
        prev_event_label_ids: torch.Tensor,
        prev_event_mask: torch.Tensor,
        prev_event_timestamps: torch.Tensor,
        prev_node_ids: torch.Tensor,
        prev_edge_ids: torch.Tensor,
        prev_edge_index: torch.Tensor,
        prev_edge_timestamps: torch.Tensor,
        decoder_hidden: Optional[torch.Tensor] = None,
        obs_word_ids: Optional[torch.Tensor] = None,
        prev_action_word_ids: Optional[torch.Tensor] = None,
        encoded_obs: Optional[torch.Tensor] = None,
        encoded_prev_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input:
            obs_mask: (batch, obs_len)
            prev_action_mask: (batch, prev_action_len)
            prev_event_type_ids: (batch, prev_event_seq_len)
            prev_event_src_ids: (batch, prev_event_seq_len)
            prev_event_src_mask: (batch, prev_event_seq_len)
            prev_event_dst_ids: (batch, prev_event_seq_len)
            prev_event_dst_mask: (batch, prev_event_seq_len)
            prev_event_edge_ids: (batch, prev_event_seq_len)
            prev_event_label_ids: (batch, prev_event_seq_len)
            prev_event_mask: (batch, prev_event_seq_len)
            prev_event_timestamps: (batch, prev_event_seq_len)
            prev_node_ids: (batch, prev_num_node)
            prev_edge_ids: (batch, prev_num_edge)
            prev_edge_index: (batch, 2, prev_num_edge)
            prev_edge_timestamps: (batch, prev_num_edge)
            decoder_hidden: (batch, hidden_dim)

            If encoded_obs and encoded_prev_action are given, they're used
            Otherwise, they are calculated from obs_word_ids, obs_mask,
            prev_action_word_ids and prev_action_mask.

            obs_word_ids: (batch, obs_len)
            prev_action_word_ids: (batch, prev_action_len)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)

        output:
        {
            event_type_logits: (batch, num_event_type)
            src_logits: (batch, num_node)
            dst_logits: (batch, num_node)
            label_logits: (batch, num_label)
            new_hidden: (batch, hidden_dim)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)
        }
        """
        if encoded_obs is None and encoded_prev_action is None:
            assert obs_word_ids is not None
            assert prev_action_word_ids is not None
            encoded_obs = self.encode_text(obs_word_ids, obs_mask)
            # (batch, obs_len, hidden_dim)
            encoded_prev_action = self.encode_text(
                prev_action_word_ids, prev_action_mask
            )
            # (batch, prev_action_len, hidden_dim)
        assert encoded_obs is not None
        assert encoded_prev_action is not None

        prev_node_embeddings = self.tgn(
            prev_event_type_ids,
            prev_event_src_ids,
            prev_event_src_mask,
            prev_event_dst_ids,
            prev_event_dst_mask,
            prev_event_edge_ids,
            self.label_embeddings[prev_event_label_ids],  # type: ignore
            prev_event_mask,
            prev_event_timestamps,
            prev_node_ids,
            prev_edge_ids,
            prev_edge_index,
            prev_edge_timestamps,
        )
        # (batch, prev_num_node, hidden_dim)

        delta_g = self.f_delta(
            prev_node_embeddings,
            encoded_obs,
            obs_mask,
            encoded_prev_action,
            prev_action_mask,
        )
        # (batch, 4 * hidden_dim)

        results = self.decoder(delta_g, prev_node_embeddings, hidden=decoder_hidden)
        results["encoded_obs"] = encoded_obs
        results["encoded_prev_action"] = encoded_prev_action

        return results

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
        node_embeddings: torch.Tensor,
        obs_embeddings: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_embeddings: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_embeddings: (batch, prev_num_node, hidden_dim)
        obs_embeddings: (batch, obs_len, hidden_dim)
        obs_mask: (batch, obs_len)
        prev_action_embeddings: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, 4 * hidden_dim)
        """
        # no masks necessary for prev_node_embeddings, so just create a fake one
        batch_size = obs_embeddings.size(0)
        prev_node_mask = torch.ones(
            batch_size, node_embeddings.size(1), device=node_embeddings.device
        )

        h_og, h_go = self.repr_aggr(
            obs_embeddings,
            node_embeddings,
            obs_mask,
            prev_node_mask,
        )
        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, prev_num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            prev_action_embeddings,
            node_embeddings,
            prev_action_mask,
            prev_node_mask,
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, prev_num_node, hidden_dim)

        mean_h_og = masked_mean(h_og, obs_mask)
        # (batch, hidden_dim)
        if h_go.size(1) == 0:
            # if there are no nodes, just use zeros
            mean_h_go = torch.zeros_like(mean_h_og)
        else:
            mean_h_go = masked_mean(h_go, prev_node_mask)
        # (batch, hidden_dim)
        mean_h_ag = masked_mean(h_ag, prev_action_mask)
        # (batch, hidden_dim)
        if h_ga.size(1) == 0:
            # if there are no nodes, just use zeros
            mean_h_ga = torch.zeros_like(mean_h_ag)
        else:
            mean_h_ga = masked_mean(h_ga, prev_node_mask)
        # (batch, hidden_dim)

        return torch.cat([mean_h_og, mean_h_go, mean_h_ag, mean_h_ga], dim=1)
        # (batch, 4 * hidden_dim)

    def training_step(  # type: ignore
        self,
        batch: List[Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]],
        batch_idx: int,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for textual_inputs, graph_event_inputs in batch:
            encoded_obs: Optional[torch.Tensor] = None
            encoded_prev_action: Optional[torch.Tensor] = None
            for graph_event in graph_event_inputs:
                hiddens: Optional[torch.Tensor] = None
                results = self(
                    textual_inputs["obs_mask"],
                    textual_inputs["prev_action_mask"],
                    graph_event["tgt_event_type_ids"],
                    graph_event["tgt_event_src_ids"],
                    graph_event["tgt_event_src_mask"],
                    graph_event["tgt_event_dst_ids"],
                    graph_event["tgt_event_dst_mask"],
                    graph_event["tgt_event_edge_ids"],
                    graph_event["tgt_event_label_ids"],
                    graph_event["tgt_event_mask"],
                    graph_event["tgt_event_timestamps"],
                    graph_event["node_ids"],
                    graph_event["edge_ids"],
                    graph_event["edge_index"],
                    graph_event["edge_timestamps"],
                    decoder_hidden=hiddens,
                    obs_word_ids=textual_inputs["obs_word_ids"],
                    prev_action_word_ids=textual_inputs["prev_action_word_ids"],
                    encoded_obs=encoded_obs,
                    encoded_prev_action=encoded_prev_action,
                )

                # calculate losses
                event_type_loss = torch.mean(
                    self.criterion(
                        results["event_type_logits"],
                        graph_event["groundtruth_event_type_ids"].flatten(),
                    )
                    * graph_event["groundtruth_event_mask"]
                )
                src_loss = torch.mean(
                    self.criterion(
                        results["src_logits"],
                        graph_event["groundtruth_event_src_ids"].flatten(),
                    )
                    * graph_event["groundtruth_event_src_mask"]
                )
                dst_loss = torch.mean(
                    self.criterion(
                        results["dst_logits"],
                        graph_event["groundtruth_event_dst_ids"].flatten(),
                    )
                    * graph_event["groundtruth_event_dst_mask"]
                )
                label_loss = torch.mean(
                    self.criterion(
                        results["label_logits"],
                        graph_event["groundtruth_event_label_ids"].flatten(),
                    )
                    * graph_event["groundtruth_event_mask"]
                )
                losses.append(event_type_loss + src_loss + dst_loss + label_loss)

                # update hiddens
                hiddens = results["new_hidden"].detach()

                # update encoded_obs and encoded_prev_action
                if encoded_obs is None:
                    encoded_obs = results["encoded_obs"].detach()
                if encoded_prev_action is None:
                    encoded_prev_action = results["encoded_prev_action"].detach()

                # detach memory
                self.tgn.memory.detach_()  # type: ignore

        loss = torch.stack(losses).mean()
        self.log("train_loss", loss, prog_bar=True)
        assert hiddens is not None
        return loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def on_epoch_start(self) -> None:
        self.tgn.reset()
