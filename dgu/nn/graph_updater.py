import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Dict
from torch.optim import Adam, Optimizer
from hydra.utils import to_absolute_path
from pathlib import Path

from dgu.nn.text import TextEncoder
from dgu.nn.rep_aggregator import ReprAggregator
from dgu.nn.utils import masked_mean, load_fasttext
from dgu.nn.graph_event_decoder import (
    StaticLabelGraphEventEncoder,
    StaticLabelGraphEventDecoder,
    RNNGraphEventSeq2Seq,
)
from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS
from dgu.constants import EVENT_TYPE_ID_MAP


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

        # raw message storage for training
        # initialize with an empty subgraph and [start, end] graph events.
        self.raw_msg = {
            "subgraph_node_ids": torch.zeros(0, dtype=torch.long),
            "subgraph_edge_ids": torch.zeros(0, dtype=torch.long),
            "subgraph_local_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "subgraph_edge_timestamps": torch.zeros(0),
            "tgt_event_timestamps": torch.zeros(2),
            "tgt_event_mask": torch.zeros(2),
            "tgt_event_type_ids": torch.tensor(
                [EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["end"]]
            ),
            "tgt_event_src_ids": torch.zeros(2, dtype=torch.long),
            "tgt_event_src_mask": torch.zeros(2),
            "tgt_event_dst_ids": torch.zeros(2, dtype=torch.long),
            "tgt_event_dst_mask": torch.zeros(2),
            "tgt_event_edge_ids": torch.zeros(2, dtype=torch.long),
            "tgt_event_label_ids": torch.zeros(2, dtype=torch.long),
        }

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(  # type: ignore
        self,
        obs_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_word_ids: torch.Tensor,
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
        subgraph_node_ids: Optional[torch.Tensor] = None,
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
            prev_event_type_ids: (prev_event_seq_len)
            prev_event_src_ids: (prev_event_seq_len)
            prev_event_src_mask: (prev_event_seq_len)
            prev_event_dst_ids: (prev_event_seq_len)
            prev_event_dst_mask: (prev_event_seq_len)
            prev_event_edge_ids: (prev_event_seq_len)
            prev_event_label_ids: (prev_event_seq_len)
            prev_event_mask: (prev_event_seq_len)
            prev_event_timestamps: (prev_event_seq_len)
            prev_node_ids: (prev_num_node)
            prev_edge_ids: (prev_num_edge)
            prev_edge_index: (2, prev_num_edge)
            prev_edge_timestamps: (prev_num_edge)
            subgraph_node_ids: (subgraph_num_node)
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
                src_logits: (graph_event_seq_len, subgraph_num_node)
                dst_logits: (graph_event_seq_len, subgraph_num_node)
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

        label_embeddings = (
            self.seq2seq.graph_event_decoder.event_label_head.label_embeddings
        )
        prev_label_embeds = label_embeddings[prev_event_label_ids]  # type: ignore
        # (prev_event_seq_len, label_embedding_dim)

        prev_node_embeddings = self.tgn(
            prev_event_type_ids,
            prev_event_src_ids,
            prev_event_src_mask,
            prev_event_dst_ids,
            prev_event_dst_mask,
            prev_event_edge_ids,
            prev_label_embeds,
            prev_event_mask,
            prev_event_timestamps,
            prev_node_ids,
            prev_edge_ids,
            prev_edge_index,
            prev_edge_timestamps,
        )
        # (prev_num_node, hidden_dim)

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
            self.tgn.memory,  # type: ignore
            subgraph_node_ids=subgraph_node_ids,
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
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        results = self(
            batch["obs_word_ids"],
            batch["obs_mask"],
            batch["prev_action_word_ids"],
            batch["prev_action_mask"],
            self.raw_msg["tgt_event_type_ids"].to(self.device),
            self.raw_msg["tgt_event_src_ids"].to(self.device),
            self.raw_msg["tgt_event_src_mask"].to(self.device),
            self.raw_msg["tgt_event_dst_ids"].to(self.device),
            self.raw_msg["tgt_event_dst_mask"].to(self.device),
            self.raw_msg["tgt_event_edge_ids"].to(self.device),
            self.raw_msg["tgt_event_label_ids"].to(self.device),
            self.raw_msg["tgt_event_mask"].to(self.device),
            self.raw_msg["tgt_event_timestamps"].to(self.device),
            self.raw_msg["subgraph_node_ids"].to(self.device),
            self.raw_msg["subgraph_edge_ids"].to(self.device),
            self.raw_msg["subgraph_local_edge_index"].to(self.device),
            self.raw_msg["subgraph_edge_timestamps"].to(self.device),
            batch["subgraph_node_ids"],
            tgt_event_type_ids=batch["tgt_event_type_ids"],
            tgt_event_src_ids=batch["tgt_event_src_ids"],
            tgt_event_src_mask=batch["tgt_event_src_mask"],
            tgt_event_dst_ids=batch["tgt_event_dst_ids"],
            tgt_event_dst_mask=batch["tgt_event_dst_mask"],
            tgt_event_label_ids=batch["tgt_event_label_ids"],
            tgt_event_mask=batch["tgt_event_mask"],
        )

        # update the raw message
        self.raw_msg["tgt_event_type_ids"] = batch["tgt_event_type_ids"]
        self.raw_msg["tgt_event_src_ids"] = batch["tgt_event_src_ids"]
        self.raw_msg["tgt_event_src_mask"] = batch["tgt_event_src_mask"]
        self.raw_msg["tgt_event_dst_ids"] = batch["tgt_event_dst_ids"]
        self.raw_msg["tgt_event_dst_mask"] = batch["tgt_event_dst_mask"]
        self.raw_msg["tgt_event_edge_ids"] = batch["tgt_event_edge_ids"]
        self.raw_msg["tgt_event_label_ids"] = batch["tgt_event_label_ids"]
        self.raw_msg["tgt_event_mask"] = batch["tgt_event_mask"]
        self.raw_msg["tgt_event_timestamps"] = batch["tgt_event_timestamps"]
        self.raw_msg["subgraph_node_ids"] = batch["subgraph_node_ids"]
        self.raw_msg["subgraph_edge_ids"] = batch["subgraph_edge_ids"]
        self.raw_msg["subgraph_local_edge_index"] = batch["subgraph_local_edge_index"]
        self.raw_msg["subgraph_edge_timestamps"] = batch["subgraph_edge_timestamps"]

        # calculate losses
        event_type_loss = self.criterion(
            results["event_type_logits"], batch["groundtruth_event_type_ids"]
        ).mean()
        src_loss = torch.mean(
            self.criterion(
                results["src_logits"], batch["groundtruth_event_subgraph_src_ids"]
            )
            * batch["groundtruth_event_src_mask"]
        )
        dst_loss = torch.mean(
            self.criterion(
                results["dst_logits"], batch["groundtruth_event_subgraph_dst_ids"]
            )
            * batch["groundtruth_event_dst_mask"]
        )
        label_loss = torch.mean(
            self.criterion(
                results["label_logits"], batch["groundtruth_event_label_ids"]
            )
            * batch["groundtruth_event_mask"]
        )
        return event_type_loss + src_loss + dst_loss + label_loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def on_train_batch_end(self, *unused) -> None:
        self.tgn.memory.detach_()  # type: ignore
