import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb
import dgu.metrics

from typing import Optional, Dict, List, Sequence, Tuple, Any
from torch.optim import AdamW, Optimizer
from hydra.utils import to_absolute_path
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.data import Batch

from dgu.nn.text import TextEncoder
from dgu.nn.rep_aggregator import ReprAggregator
from dgu.nn.utils import (
    masked_mean,
    load_fasttext,
    batchify_node_features,
    calculate_node_id_offsets,
    update_batched_graph,
)
from dgu.nn.graph_event_decoder import (
    StaticLabelGraphEventDecoder,
    RNNGraphEventDecoder,
)
from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS
from dgu.data import (
    TWCmdGenTemporalBatch,
    TWCmdGenTemporalGraphicalInput,
    TWCmdGenTemporalStepInput,
    read_label_vocab_files,
)
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
        word_emb_dim: int = 300,
        tgn_event_type_emb_dim: int = 8,
        tgn_time_enc_dim: int = 8,
        tgn_num_gnn_block: int = 1,
        tgn_num_gnn_head: int = 1,
        text_encoder_num_blocks: int = 1,
        text_encoder_num_conv_layers: int = 3,
        text_encoder_kernel_size: int = 5,
        text_encoder_num_heads: int = 1,
        graph_event_decoder_hidden_dim: int = 8,
        graph_event_decoder_key_query_dim: int = 8,
        max_decode_len: int = 100,
        learning_rate: float = 5e-4,
        truncated_bptt_steps: int = 0,
        pretrained_word_embedding_path: Optional[str] = None,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "hidden_dim",
            "word_emb_dim",
            "tgn_event_type_emb_dim",
            "tgn_time_enc_dim",
            "tgn_num_gnn_block",
            "tgn_num_gnn_head",
            "text_encoder_num_blocks",
            "text_encoder_num_conv_layers",
            "text_encoder_kernel_size",
            "text_encoder_num_heads",
            "graph_event_decoder_hidden_dim",
            "graph_event_decoder_key_query_dim",
            "max_decode_len",
            "learning_rate",
            "truncated_bptt_steps",
        )
        # preprocessor
        if word_vocab_path is None:
            # just load with special tokens
            self.preprocessor = SpacyPreprocessor([PAD, UNK, BOS, EOS])
        else:
            self.preprocessor = SpacyPreprocessor.load_from_file(
                to_absolute_path(word_vocab_path)
            )

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

        if node_vocab_path is not None and relation_vocab_path is not None:
            self.labels, self.label_id_map = read_label_vocab_files(
                to_absolute_path(node_vocab_path), to_absolute_path(relation_vocab_path)
            )
        else:
            self.labels = ["", "node", "relation"]
            self.label_id_map = {label: i for i, label in enumerate(self.labels)}

        # calculate node/edge label embeddings
        label_word_ids, label_mask = self.preprocessor.preprocess(self.labels)
        self.label_embeddings = nn.Embedding.from_pretrained(
            masked_mean(pretrained_word_embeddings(label_word_ids), label_mask)
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
            tgn_event_type_emb_dim,
            tgn_time_enc_dim,
            word_emb_dim,
            hidden_dim,
            tgn_num_gnn_block,
            tgn_num_gnn_head,
        )

        # representation aggregator
        self.repr_aggr = ReprAggregator(hidden_dim)

        # graph event seq2seq
        event_decoder = StaticLabelGraphEventDecoder(
            graph_event_decoder_hidden_dim,
            hidden_dim,
            word_emb_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
        )
        self.decoder = RNNGraphEventDecoder(
            4 * hidden_dim, graph_event_decoder_hidden_dim, event_decoder
        )

        self.criterion = nn.CrossEntropyLoss()

        self.event_type_f1 = torchmetrics.F1(ignore_index=EVENT_TYPE_ID_MAP["pad"])
        self.src_node_f1 = torchmetrics.F1()
        self.dst_node_f1 = torchmetrics.F1()
        self.label_f1 = torchmetrics.F1(ignore_index=0)

        self.graph_tf_exact_match = dgu.metrics.ExactMatch()
        self.token_tf_exact_match = dgu.metrics.ExactMatch()
        self.graph_tf_f1 = dgu.metrics.F1()
        self.token_tf_f1 = dgu.metrics.F1()

        self.graph_gd_exact_match = dgu.metrics.ExactMatch()
        self.token_gd_exact_match = dgu.metrics.ExactMatch()
        self.graph_gd_f1 = dgu.metrics.F1()
        self.token_gd_f1 = dgu.metrics.F1()

    def forward(  # type: ignore
        self,
        # graph events
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_ids: torch.Tensor,
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
        # decoder hidden
        decoder_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        graph events: {
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_ids: (batch)
        }
        prev_batched_graph: diagonally stacked graph BEFORE the given graph events:
            Batch(
                batch: (prev_num_node)
                x: node label IDs, (prev_num_node)
                node_last_update: (prev_num_node)
                edge_index: (2, prev_num_edge)
                edge_attr: edge label IDs, (prev_num_edge)
                edge_last_update: (prev_num_edge)
            )
        decoder_hidden: (batch, hidden_dim)

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

        output:
        {
            event_type_logits: (batch, num_event_type)
            event_src_logits: (batch, max_sub_graph_num_node)
            event_dst_logits: (batch, max_sub_graph_num_node)
            event_label_logits: (batch, num_label)
            new_decoder_hidden: (batch, hidden_dim)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)
            updated_batched_graph: Batch(
                batch: (num_node)
                x: (num_node)
                node_last_update: (num_node)
                edge_index: (2, num_edge)
                edge_attr: (num_edge)
                edge_last_update: (num_edge)
            )
        }
        """
        if encoded_obs is None:
            assert obs_word_ids is not None
            encoded_obs = self.encode_text(obs_word_ids, obs_mask)
            # (batch, obs_len, hidden_dim)
        if encoded_prev_action is None:
            assert prev_action_word_ids is not None
            encoded_prev_action = self.encode_text(
                prev_action_word_ids, prev_action_mask
            )
            # (batch, prev_action_len, hidden_dim)

        # update the batched graph
        updated_batched_graph = update_batched_graph(
            prev_batched_graph,
            event_type_ids,
            event_src_ids,
            event_dst_ids,
            event_label_ids,
            timestamps,
        )
        if updated_batched_graph.num_nodes == 0:
            node_embeddings = torch.zeros(
                0,
                self.hparams.hidden_dim,  # type: ignore
                device=self.device,
            )
            # (0, hidden_dim)
        else:
            node_embeddings = self.tgn(
                timestamps,
                Batch(
                    batch=updated_batched_graph.batch,
                    x=self.label_embeddings(updated_batched_graph.x),
                    node_last_update=updated_batched_graph.node_last_update,
                    edge_index=updated_batched_graph.edge_index,
                    edge_attr=self.label_embeddings(updated_batched_graph.edge_attr),
                    edge_last_update=updated_batched_graph.edge_last_update,
                ),
            )
            # (num_node, hidden_dim)

        # batchify node_embeddings
        batch_node_embeddings, batch_node_mask = batchify_node_features(
            node_embeddings, updated_batched_graph.batch, obs_mask.size(0)
        )
        # batch_node_embeddings: (batch, max_sub_graph_num_node, hidden_dim)
        # batch_node_mask: (batch, max_sub_graph_num_node)

        delta_g = self.f_delta(
            batch_node_embeddings,
            batch_node_mask,
            encoded_obs,
            obs_mask,
            encoded_prev_action,
            prev_action_mask,
        )
        # (batch, 4 * hidden_dim)

        decoder_results = self.decoder(
            delta_g,
            batch_node_embeddings,
            self.label_embeddings.weight,
            hidden=decoder_hidden,
        )
        return {
            "event_type_logits": decoder_results["event_type_logits"],
            "event_src_logits": decoder_results["src_logits"],
            "event_dst_logits": decoder_results["dst_logits"],
            "event_label_logits": decoder_results["label_logits"],
            "new_decoder_hidden": decoder_results["new_hidden"],
            "encoded_obs": encoded_obs,
            "encoded_prev_action": encoded_prev_action,
            "updated_batched_graph": updated_batched_graph,
        }

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
        node_mask: torch.Tensor,
        obs_embeddings: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_embeddings: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_embeddings: (batch, num_node, hidden_dim)
        node_mask: (batch, num_node)
        obs_embeddings: (batch, obs_len, hidden_dim)
        obs_mask: (batch, obs_len)
        prev_action_embeddings: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, 4 * hidden_dim)
        """
        h_og, h_go = self.repr_aggr(
            obs_embeddings,
            node_embeddings,
            obs_mask,
            node_mask,
        )
        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            prev_action_embeddings,
            node_embeddings,
            prev_action_mask,
            node_mask,
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, num_node, hidden_dim)

        mean_h_og = masked_mean(h_og, obs_mask)
        # (batch, hidden_dim)
        mean_h_go = masked_mean(h_go, node_mask)
        # (batch, hidden_dim)
        mean_h_ag = masked_mean(h_ag, prev_action_mask)
        # (batch, hidden_dim)
        mean_h_ga = masked_mean(h_ga, node_mask)
        # (batch, hidden_dim)

        return torch.cat([mean_h_og, mean_h_go, mean_h_ag, mean_h_ga], dim=1)
        # (batch, 4 * hidden_dim)

    def teacher_force(
        self,
        step_input: TWCmdGenTemporalStepInput,
        graphical_input_seq: Sequence[TWCmdGenTemporalGraphicalInput],
    ) -> List[Dict[str, Any]]:
        """
        step_input: the current step input
        graphical_input_seq: sequence of graphical inputs
        batch_graph: diagonally stacked batch of current graphs

        output:
        [{
            event_type_logits: (batch, num_event_type)
            event_src_logits: (batch, max_sub_graph_num_node)
            event_dst_logits: (batch, max_sub_graph_num_node)
            event_label_logits: (batch, num_label)
            new_decoder_hidden: (batch, hidden_dim)
            encoded_obs: (batch, obs_len, hidden_dim)
            encoded_prev_action: (batch, prev_action_len, hidden_dim)
            updated_batched_graph: diagonally stacked batch of updated graphs
        }, ...]
        """
        decoder_hidden: Optional[torch.Tensor] = None
        encoded_obs: Optional[torch.Tensor] = None
        encoded_prev_action: Optional[torch.Tensor] = None
        results_list: List[Dict[str, torch.Tensor]] = []
        for graphical_input in graphical_input_seq:
            results = self(
                graphical_input.tgt_event_type_ids,
                graphical_input.tgt_event_src_ids,
                graphical_input.tgt_event_dst_ids,
                graphical_input.tgt_event_label_ids,
                graphical_input.prev_batched_graph,
                step_input.obs_mask,
                step_input.prev_action_mask,
                step_input.timestamps,
                obs_word_ids=step_input.obs_word_ids,
                prev_action_word_ids=step_input.prev_action_word_ids,
                encoded_obs=encoded_obs,
                encoded_prev_action=encoded_prev_action,
                decoder_hidden=decoder_hidden,
            )

            # add results to the list
            results_list.append(results)

            # update decoder hidden
            decoder_hidden = results["new_decoder_hidden"]

            # save the encoded obs and prev action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

        return results_list

    def calculate_loss(
        self,
        event_type_logits: torch.Tensor,
        groundtruth_event_type_ids: torch.Tensor,
        event_src_logits: torch.Tensor,
        groundtruth_event_src_ids: torch.Tensor,
        event_dst_logits: torch.Tensor,
        groundtruth_event_dst_ids: torch.Tensor,
        event_label_logits: torch.Tensor,
        groundtruth_event_label_ids: torch.Tensor,
        groundtruth_event_mask: torch.Tensor,
        groundtruth_event_src_mask: torch.Tensor,
        groundtruth_event_dst_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss.

        event_type_logits: (batch, num_event_type)
        groundtruth_event_type_ids: (batch)
        event_src_logits: (batch, num_node)
        groundtruth_event_src_ids: (batch)
        event_dst_logits: (batch, num_node)
        groundtruth_event_dst_ids: (batch)
        event_label_logits: (batch, num_label)
        groundtruth_event_label_ids: (batch)
        groundtruth_event_mask: (batch)
        groundtruth_event_src_mask: (batch)
        groundtruth_event_dst_mask: (batch)

        output: (batch)
        """
        # event type loss
        loss = self.criterion(
            event_type_logits[groundtruth_event_mask],
            groundtruth_event_type_ids[groundtruth_event_mask],
        )
        if groundtruth_event_dst_mask.any():
            # source node loss
            loss += self.criterion(
                event_src_logits[groundtruth_event_src_mask],
                groundtruth_event_src_ids[groundtruth_event_src_mask],
            )
        if groundtruth_event_dst_mask.any():
            # destination node loss
            loss += self.criterion(
                event_dst_logits[groundtruth_event_dst_mask],
                groundtruth_event_dst_ids[groundtruth_event_dst_mask],
            )
        # label loss
        loss += self.criterion(
            event_label_logits[groundtruth_event_mask],
            groundtruth_event_label_ids[groundtruth_event_mask],
        )
        return loss

    def calculate_f1s(
        self,
        event_type_logits: torch.Tensor,
        groundtruth_event_type_ids: torch.Tensor,
        event_src_logits: torch.Tensor,
        groundtruth_event_src_ids: torch.Tensor,
        event_dst_logits: torch.Tensor,
        groundtruth_event_dst_ids: torch.Tensor,
        event_label_logits: torch.Tensor,
        groundtruth_event_label_ids: torch.Tensor,
        groundtruth_event_src_mask: torch.Tensor,
        groundtruth_event_dst_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate various F1 scores.

        event_type_logits: (batch, num_event_type)
        groundtruth_event_type_ids: (batch)
        event_src_logits: (batch, num_node)
        groundtruth_event_src_ids: (batch)
        event_dst_logits: (batch, num_node)
        groundtruth_event_dst_ids: (batch)
        event_label_logits: (batch, num_label)
        groundtruth_event_label_ids: (batch)
        groundtruth_event_src_mask: (batch)
        groundtruth_event_dst_mask: (batch)

        output: {
            'event_type_f1': scalar,
            'src_node_f1': optional, scalar,
            'dst_node_f1': optional, scalar,
            'label_f1': scalar,
        }
        """
        f1s: Dict[str, torch.Tensor] = {}
        f1s["event_type_f1"] = self.event_type_f1(
            event_type_logits.softmax(dim=1), groundtruth_event_type_ids
        )
        src_mask = groundtruth_event_src_mask.bool()
        if src_mask.any():
            f1s["src_node_f1"] = self.src_node_f1(
                event_src_logits[src_mask].softmax(dim=1),
                groundtruth_event_src_ids[src_mask],
            )
        dst_mask = groundtruth_event_dst_mask.bool()
        if dst_mask.any():
            f1s["dst_node_f1"] = self.dst_node_f1(
                event_dst_logits[dst_mask].softmax(dim=1),
                groundtruth_event_dst_ids[dst_mask],
            )
        f1s["label_f1"] = self.label_f1(
            event_label_logits.softmax(dim=1), groundtruth_event_label_ids
        )

        return f1s

    def generate_graph_triples(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        label_ids: torch.Tensor,
        batched_graph: Batch,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Generate graph triplets based on the given batch of graph events.

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        label_ids: (batch)
        batched_graph: batched graph before the given events

        output: (
            cmds: len([commands, ...]) = batch
            tokens: len([[token, ...], ...]) = batch
        )
        """
        # figure out the label IDs for the nodes in the batched graph
        # we compare the node features, which are label embeddings, with the
        # whole label embeddings and figure out the correct IDs.
        graph_node_label_ids = torch.all(
            batched_graph.x.unsqueeze(-1) == self.label_embeddings.weight.t(), dim=1
        ).nonzero()[:, 1]
        # (num_node)

        # we ignore last graphs without nodes, b/c they wouldn't produce valid
        # graph triples anyway
        batched_graph_node_label_ids = graph_node_label_ids.split(
            batched_graph.batch.bincount().tolist()
        )
        # (batch - num_last_empty_graphs)

        cmds: List[str] = []
        tokens: List[List[str]] = []
        for event_type_id, src_id, dst_id, label_id, node_label_ids in zip(
            event_type_ids.tolist(),
            src_ids,
            dst_ids,
            label_ids,
            batched_graph_node_label_ids,
        ):
            if event_type_id in {
                EVENT_TYPE_ID_MAP["edge-add"],
                EVENT_TYPE_ID_MAP["edge-delete"],
            }:
                cmd = (
                    "add"
                    if event_type_id == EVENT_TYPE_ID_MAP["edge-add"]
                    else "delete"
                )
                src_label = self.labels[node_label_ids[src_id]]
                dst_label = self.labels[node_label_ids[dst_id]]
                # in the original dataset, multi-word edge labels are joined by
                # an underscore
                edge_label = "_".join(self.labels[label_id].split())
                cmd_tokens = [cmd, src_label, dst_label, edge_label]
                cmds.append(" , ".join(cmd_tokens))
                tokens.append(cmd_tokens)
        return cmds, tokens

    def generate_batch_graph_triples_seq(
        self,
        event_type_id_seq: Sequence[torch.Tensor],
        src_id_seq: Sequence[torch.Tensor],
        dst_id_seq: Sequence[torch.Tensor],
        label_id_seq: Sequence[torch.Tensor],
        batched_graph_seq: Sequence[Batch],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        batch_size = event_type_id_seq[0].size(0)
        # (batch, event_seq_len, cmd_len)
        batch_cmds: List[List[str]] = [[] for _ in range(batch_size)]
        # (batch, event_seq_len, token_len)
        batch_tokens: List[List[str]] = [[] for _ in range(batch_size)]
        for step_id, (
            event_type_ids,
            src_ids,
            dst_ids,
            label_ids,
            batched_graph,
        ) in enumerate(
            zip(
                event_type_id_seq,
                src_id_seq,
                dst_id_seq,
                label_id_seq,
                batched_graph_seq,
            )
        ):
            for batch_id, (cmd, tokens) in enumerate(
                zip(
                    *self.generate_graph_triples(
                        event_type_ids, src_ids, dst_ids, label_ids, batched_graph
                    )
                )
            ):
                batch_cmds[batch_id].append(cmd)
                if step_id != 0:
                    batch_tokens[batch_id].append("<sep>")
                batch_tokens[batch_id].extend(tokens)
        return batch_cmds, batch_tokens

    @staticmethod
    def generate_batch_groundtruth_graph_triple_tokens(
        groundtruth_cmd_seq: Sequence[Sequence[str]],
    ) -> List[List[str]]:
        batch_groundtruth_tokens: List[List[str]] = []
        for groundtruth_cmds in groundtruth_cmd_seq:
            batch_groundtruth_tokens.append(
                " , <sep> , ".join(groundtruth_cmds).split(" , ")
            )
        return batch_groundtruth_tokens

    def training_step(  # type: ignore
        self, batch: TWCmdGenTemporalBatch, batch_idx: int
    ) -> torch.Tensor:
        """
        batch: the batch data
        batch_idx: the batch id, unused
        """
        results_list = self.teacher_force(batch.step_input, batch.graphical_input_seq)
        loss = torch.stack(
            [
                self.calculate_loss(
                    results["event_type_logits"],
                    graphical_input.groundtruth_event_type_ids,
                    results["event_src_logits"],
                    graphical_input.groundtruth_event_src_ids,
                    results["event_dst_logits"],
                    graphical_input.groundtruth_event_dst_ids,
                    results["event_label_logits"],
                    graphical_input.groundtruth_event_label_ids,
                    graphical_input.groundtruth_event_mask,
                    graphical_input.groundtruth_event_src_mask,
                    graphical_input.groundtruth_event_dst_mask,
                )
                for results, graphical_input in zip(
                    results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(
        self, batch: TWCmdGenTemporalBatch, log_prefix: str
    ) -> List[Tuple[str, ...]]:
        # [(id, groundtruth commands, teacher-force commands, greedy-decode commands)]
        # id = (game|walkthrough_step|random_step)
        table_data: List[Tuple[str, ...]] = []

        # loss from teacher forcing
        tf_results_list = self.teacher_force(
            batch.step_input, batch.graphical_input_seq
        )
        loss = torch.stack(
            [
                self.calculate_loss(
                    results["event_type_logits"],
                    graphical_input.groundtruth_event_type_ids,
                    results["event_src_logits"],
                    graphical_input.groundtruth_event_src_ids,
                    results["event_dst_logits"],
                    graphical_input.groundtruth_event_dst_ids,
                    results["event_label_logits"],
                    graphical_input.groundtruth_event_label_ids,
                    graphical_input.groundtruth_event_mask,
                    graphical_input.groundtruth_event_src_mask,
                    graphical_input.groundtruth_event_dst_mask,
                )
                for results, graphical_input in zip(
                    tf_results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        self.log(log_prefix + "_loss", loss, prog_bar=True)

        # log classification F1s from teacher forcing
        for results, graphical_input in zip(tf_results_list, batch.graphical_input_seq):
            f1s = self.calculate_f1s(
                results["event_type_logits"],
                graphical_input.groundtruth_event_type_ids,
                results["event_src_logits"],
                graphical_input.groundtruth_event_src_ids,
                results["event_dst_logits"],
                graphical_input.groundtruth_event_dst_ids,
                results["event_label_logits"],
                graphical_input.groundtruth_event_label_ids,
                graphical_input.groundtruth_event_src_mask,
                graphical_input.groundtruth_event_dst_mask,
            )
            self.log(log_prefix + "_event_type_f1", f1s["event_type_f1"])
            if "src_node_f1" in f1s:
                self.log(log_prefix + "_src_node_f1", f1s["src_node_f1"])
            if "dst_node_f1" in f1s:
                self.log(log_prefix + "_dst_node_f1", f1s["dst_node_f1"])
            self.log(log_prefix + "_label_f1", f1s["label_f1"])

        # calculate graph tuples from teacher forcing graph events
        # handle source/destination logits for empty graphs by setting them to zeros
        tf_src_id_seq = [
            results["event_src_logits"].argmax(dim=1)
            if results["event_src_logits"].size(1) > 0
            else torch.zeros(
                results["event_src_logits"].size(0),
                dtype=torch.long,
                device=self.device,
            )
            for results in tf_results_list
        ]
        tf_dst_id_seq = [
            results["event_dst_logits"].argmax(dim=1)
            if results["event_dst_logits"].size(1) > 0
            else torch.zeros(
                results["event_dst_logits"].size(0),
                dtype=torch.long,
                device=self.device,
            )
            for results in tf_results_list
        ]
        batch_tf_cmds, batch_tf_tokens = self.generate_batch_graph_triples_seq(
            [
                self.filter_invalid_events(
                    results["event_type_logits"].argmax(dim=1),
                    src_ids,
                    dst_ids,
                    results["updated_batched_graph"].batch,
                    results["updated_batched_graph"].edge_index,
                )
                for results, src_ids, dst_ids in zip(
                    tf_results_list, tf_src_id_seq, tf_dst_id_seq
                )
            ],
            tf_src_id_seq,
            tf_dst_id_seq,
            [
                results["event_label_logits"].argmax(dim=1)
                for results in tf_results_list
            ],
            [results["updated_batched_graph"] for results in tf_results_list],
        )

        # collect groundtruth command tokens
        batch_groundtruth_tokens = self.generate_batch_groundtruth_graph_triple_tokens(
            batch.graph_commands
        )

        # log teacher force graph based metrics
        self.log(
            log_prefix + "_graph_tf_em",
            self.graph_tf_exact_match(batch_tf_cmds, batch.graph_commands),
        )
        self.log(
            log_prefix + "_graph_tf_f1",
            self.graph_tf_f1(batch_tf_cmds, batch.graph_commands),
        )

        # log teacher force token based metrics
        self.log(
            log_prefix + "_token_tf_em",
            self.token_tf_exact_match(batch_tf_tokens, batch_groundtruth_tokens),
        )
        self.log(
            log_prefix + "_token_tf_f1",
            self.token_tf_f1(batch_tf_tokens, batch_groundtruth_tokens),
        )

        # greedy decoding
        gd_results_list = self.greedy_decode(
            batch.step_input, batch.initial_batched_graph
        )

        # calculate graph tuples from greedy decoded graph events
        batch_gd_cmds, batch_gd_tokens = self.generate_batch_graph_triples_seq(
            [results["decoded_event_type_ids"] for results in gd_results_list],
            [results["decoded_event_src_ids"] for results in gd_results_list],
            [results["decoded_event_dst_ids"] for results in gd_results_list],
            [results["decoded_event_label_ids"] for results in gd_results_list],
            [results["updated_batched_graph"] for results in gd_results_list],
        )

        # log greedy decode graph based metrics
        self.log(
            log_prefix + "_graph_gd_em",
            self.graph_gd_exact_match(batch_gd_cmds, batch.graph_commands),
        )
        self.log(
            log_prefix + "_graph_gd_f1",
            self.graph_gd_f1(batch_gd_cmds, batch.graph_commands),
        )

        # log greedy decode token based metrics
        self.log(
            log_prefix + "_token_gd_em",
            self.token_gd_exact_match(batch_gd_tokens, batch_groundtruth_tokens),
        )
        self.log(
            log_prefix + "_token_gd_f1",
            self.token_gd_f1(batch_gd_tokens, batch_groundtruth_tokens),
        )

        # collect graph triple table data
        table_data.extend(
            self.generate_predict_table_rows(
                batch.ids, batch.graph_commands, batch_tf_cmds, batch_gd_cmds
            )
        )

        return table_data

    def validation_step(  # type: ignore
        self, batch: TWCmdGenTemporalBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch, "val")

    def test_step(  # type: ignore
        self, batch: TWCmdGenTemporalBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch, "test")

    def wandb_log_gen_obs(
        self, outputs: List[List[Tuple[str, str, str, str]]], table_title: str
    ) -> None:
        eval_table_artifact = wandb.Artifact(
            table_title + f"_{self.logger.experiment.id}", "predictions"
        )
        eval_table = wandb.Table(
            columns=["id", "truth", "tf", "gd"],
            data=[item for sublist in outputs for item in sublist],
        )
        eval_table_artifact.add(eval_table, "predictions")
        self.logger.experiment.log_artifact(eval_table_artifact)

    def validation_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "val_gen_graph_triples")

    def test_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "test_gen_graph_triples")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def greedy_decode(
        self, step_input: TWCmdGenTemporalStepInput, prev_batched_graph: Batch
    ) -> List[Dict[str, Any]]:
        """
        step_input: the current step input
        prev_batch_graph: diagonally stacked batch of current graphs

        output:
        [{
            decoded_event_type_ids: (batch)
            decoded_event_src_ids: (batch)
            decoded_event_dst_ids: (batch)
            decoded_event_label_ids: (batch)
            updated_batched_graph: diagonally stacked batch of updated graphs.
                these are the graphs used to decode the graph events above
        }, ...]
        """
        # initialize the initial inputs
        batch_size = step_input.obs_word_ids.size(0)
        decoded_event_type_ids = torch.tensor(
            [EVENT_TYPE_ID_MAP["start"]] * batch_size, device=self.device
        )
        decoded_src_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        decoded_dst_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        decoded_label_ids = torch.tensor([0] * batch_size, device=self.device)

        end_event_mask = torch.tensor([False] * batch_size, device=self.device)
        # (batch)

        decoder_hidden: Optional[torch.Tensor] = None
        encoded_obs: Optional[torch.Tensor] = None
        encoded_prev_action: Optional[torch.Tensor] = None
        results_list: List[Dict[str, Any]] = []
        for _ in range(self.hparams.max_decode_len):  # type: ignore
            results = self(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                decoded_label_ids,
                prev_batched_graph,
                step_input.obs_mask,
                step_input.prev_action_mask,
                step_input.timestamps,
                obs_word_ids=step_input.obs_word_ids,
                prev_action_word_ids=step_input.prev_action_word_ids,
                encoded_obs=encoded_obs,
                encoded_prev_action=encoded_prev_action,
                decoder_hidden=decoder_hidden,
            )

            # process the decoded result for the next iteration
            decoded_event_type_ids = (
                results["event_type_logits"]
                .argmax(dim=1)
                .masked_fill(end_event_mask, EVENT_TYPE_ID_MAP["pad"])
            )
            # (batch)

            if results["event_src_logits"].size(1) == 0:
                decoded_src_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
            else:
                decoded_src_ids = (
                    results["event_src_logits"]
                    .argmax(dim=1)
                    .masked_fill(end_event_mask, 0)
                )
            # (batch)
            if results["event_dst_logits"].size(1) == 0:
                decoded_dst_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
            else:
                decoded_dst_ids = (
                    results["event_dst_logits"]
                    .argmax(dim=1)
                    .masked_fill(end_event_mask, 0)
                )
            # (batch)
            decoded_label_ids = (
                results["event_label_logits"]
                .argmax(dim=1)
                .masked_fill(end_event_mask, 0)
            )
            # (batch)

            # filter out invalid decoded events
            decoded_event_type_ids = self.filter_invalid_events(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            # (batch)

            # collect the results
            results_list.append(
                {
                    "decoded_event_type_ids": decoded_event_type_ids,
                    "decoded_event_src_ids": decoded_src_ids,
                    "decoded_event_dst_ids": decoded_dst_ids,
                    "decoded_event_label_ids": decoded_label_ids,
                    "updated_batched_graph": results["updated_batched_graph"],
                }
            )

            # update the batched graph
            prev_batched_graph = results["updated_batched_graph"]

            # update the decoder hidden
            decoder_hidden = results["new_decoder_hidden"]

            # save update the encoded observation and previous action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

            # update end_event_mask
            end_event_mask = end_event_mask.logical_or(
                decoded_event_type_ids == EVENT_TYPE_ID_MAP["end"]
            )

            # if everything in the batch is done, break
            if end_event_mask.all():
                break

        return results_list

    @staticmethod
    def filter_invalid_events(
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filter out invalid events by setting them to pad events.

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

        output: filtered event_type_ids (batch)
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

        invalid_event_mask = invalid_node_delete_event_mask.logical_or(
            invalid_edge_add_event_mask
        ).logical_or(invalid_edge_delete_event_mask)
        # (batch)

        return event_type_ids.masked_fill(invalid_event_mask, EVENT_TYPE_ID_MAP["pad"])

    @staticmethod
    def generate_predict_table_rows(
        ids: Sequence[Tuple[str, int, int]], *args: Sequence[Sequence[str]]
    ) -> List[Tuple[str, ...]]:
        """
        Generate rows for the prediction table.

        ids: len([(game, walkthrough_step, random_step), ...]) = batch
        args: various commands of shape (batch, event_seq_len)

        output: [
            ('game|walkthrough_step|random_step', groundtruth_cmd, tf_cmd, gd_cmd),
            ...
        ]
        """
        return list(
            zip(
                ["|".join(map(str, step_id)) for step_id in ids],
                *[[" | ".join(cmds) for cmds in batch_cmds] for batch_cmds in args],
            )
        )
