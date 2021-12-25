import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb
import dgu.metrics
import tqdm

from typing import Optional, Dict, List, Sequence, Tuple, Any
from torch.optim import AdamW, Optimizer
from hydra.utils import to_absolute_path
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from torch_geometric.nn import TransformerConv, GATv2Conv
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader

from dgu.nn.text import TextEncoder
from dgu.nn.rep_aggregator import ReprAggregator
from dgu.nn.utils import (
    compute_masks_from_event_type_ids,
    index_edge_attr,
    masked_mean,
    load_fasttext,
    batchify_node_features,
    calculate_node_id_offsets,
    masked_softmax,
    update_batched_graph,
)
from dgu.nn.graph_event_decoder import (
    TransformerGraphEventDecoder,
    EventTypeHead,
    EventNodeHead,
    EventStaticLabelHead,
)
from dgu.nn.dynamic_gnn import DynamicGNN
from dgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS
from dgu.data import (
    TWCmdGenGraphEventBatch,
    TWCmdGenGraphEventFreeRunDataset,
    TWCmdGenGraphEventGraphicalInput,
    TWCmdGenGraphEventStepInput,
    read_label_vocab_files,
)
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.graph import (
    batch_to_data_list,
    data_to_networkx,
    networkx_to_rdf,
    update_rdf_graph,
)


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
        dgnn_gnn: str = "TransformerConv",
        dgnn_timestamp_enc_dim: int = 8,
        dgnn_num_gnn_block: int = 1,
        dgnn_num_gnn_head: int = 1,
        dgnn_zero_timestamp_encoder: bool = False,
        text_encoder_num_blocks: int = 1,
        text_encoder_num_conv_layers: int = 3,
        text_encoder_kernel_size: int = 5,
        text_encoder_num_heads: int = 1,
        graph_event_decoder_event_type_emb_dim: int = 8,
        graph_event_decoder_hidden_dim: int = 8,
        graph_event_decoder_key_query_dim: int = 8,
        graph_event_decoder_num_dec_blocks: int = 1,
        graph_event_decoder_dec_block_num_heads: int = 1,
        max_decode_len: int = 100,
        learning_rate: float = 5e-4,
        dropout: float = 0.3,
        pretrained_word_embedding_path: Optional[str] = None,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "hidden_dim",
            "word_emb_dim",
            "dgnn_gnn",
            "dgnn_timestamp_enc_dim",
            "dgnn_num_gnn_block",
            "dgnn_num_gnn_head",
            "dgnn_zero_timestamp_encoder",
            "text_encoder_num_blocks",
            "text_encoder_num_conv_layers",
            "text_encoder_kernel_size",
            "text_encoder_num_heads",
            "graph_event_decoder_event_type_emb_dim",
            "graph_event_decoder_hidden_dim",
            "graph_event_decoder_key_query_dim",
            "graph_event_decoder_num_dec_blocks",
            "graph_event_decoder_dec_block_num_heads",
            "max_decode_len",
            "learning_rate",
            "dropout",
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

        # node/edge labels
        label_word_ids, label_mask = self.preprocessor.preprocess(self.labels)
        self.register_buffer("label_word_ids", label_word_ids)
        self.register_buffer("label_mask", label_mask)

        # text encoder
        self.text_encoder = TextEncoder(
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            hidden_dim,
            text_encoder_num_heads,
            dropout=dropout,
        )

        # temporal graph network
        gnn_module: nn.Module
        if dgnn_gnn == "TransformerConv":
            gnn_module = TransformerConv
        elif dgnn_gnn == "GATv2Conv":
            gnn_module = GATv2Conv
        else:
            raise ValueError(f"Unknown GNN: {dgnn_gnn}")
        self.dgnn = DynamicGNN(
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
        self.repr_aggr = ReprAggregator(hidden_dim)

        # graph event seq2seq
        self.event_type_embeddings = nn.Embedding(
            len(EVENT_TYPE_ID_MAP),
            graph_event_decoder_event_type_emb_dim,
            padding_idx=EVENT_TYPE_ID_MAP["pad"],
        )
        self.decoder = TransformerGraphEventDecoder(
            graph_event_decoder_event_type_emb_dim + 3 * hidden_dim,
            hidden_dim,
            graph_event_decoder_num_dec_blocks,
            graph_event_decoder_dec_block_num_heads,
            graph_event_decoder_hidden_dim,
            dropout=dropout,
        )
        self.event_type_head = EventTypeHead(
            graph_event_decoder_hidden_dim, hidden_dim, dropout=dropout
        )
        self.event_src_head = EventNodeHead(
            hidden_dim,
            graph_event_decoder_hidden_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
            dropout=dropout,
        )
        self.event_dst_head = EventNodeHead(
            hidden_dim,
            graph_event_decoder_hidden_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
            dropout=dropout,
        )
        self.event_label_head = EventStaticLabelHead(
            graph_event_decoder_hidden_dim,
            hidden_dim,
            hidden_dim,
            graph_event_decoder_key_query_dim,
            dropout=dropout,
        )

        self.criterion = UncertaintyWeightedLoss()

        self.event_type_f1 = torchmetrics.F1()
        self.src_node_f1 = torchmetrics.F1()
        self.dst_node_f1 = torchmetrics.F1()
        self.label_f1 = torchmetrics.F1()

        self.graph_tf_exact_match = dgu.metrics.ExactMatch()
        self.token_tf_exact_match = dgu.metrics.ExactMatch()
        self.graph_tf_f1 = dgu.metrics.F1()
        self.token_tf_f1 = dgu.metrics.F1()

        self.graph_gd_exact_match = dgu.metrics.ExactMatch()
        self.token_gd_exact_match = dgu.metrics.ExactMatch()
        self.graph_gd_f1 = dgu.metrics.F1()
        self.token_gd_f1 = dgu.metrics.F1()

        self.free_run_f1 = dgu.metrics.F1()
        self.free_run_em = dgu.metrics.ExactMatch()

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
        # previous input event embedding sequence
        prev_input_event_emb_seq: Optional[torch.Tensor] = None,
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None,
        # groundtruth events for decoder teacher-force training
        groundtruth_event: Optional[Dict[str, torch.Tensor]] = None,
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
        }
        Ground-truth graph events used for teacher forcing of the decoder

        output:
        {
            event_type_logits: (batch, num_event_type)
            event_src_logits: (batch, max_sub_graph_num_node)
            event_dst_logits: (batch, max_sub_graph_num_node)
            event_label_logits: (batch, num_label)
            updated_prev_input_event_emb_seq:
                (graph_event_decoder_num_dec_blocks, batch,
                 input_seq_len, graph_event_decoder_hidden_dim)
            updated_prev_input_event_emb_seq_mask: (batch, input_seq_len)
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
            encoded_obs = self.encode_text(obs_word_ids, obs_mask)
            # (batch, obs_len, hidden_dim)
        if encoded_prev_action is None:
            assert prev_action_word_ids is not None
            encoded_prev_action = self.encode_text(
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
            event_label_ids,
            event_timestamps,
        )
        if updated_batched_graph.num_nodes == 0:
            node_embeddings = torch.zeros(
                0,
                self.hparams.hidden_dim,  # type: ignore
                device=self.device,
            )
            # (0, hidden_dim)
        else:
            node_embeddings = self.dgnn(
                Batch(
                    batch=updated_batched_graph.batch,
                    x=self.embed_label(updated_batched_graph.x),
                    node_last_update=updated_batched_graph.node_last_update,
                    edge_index=updated_batched_graph.edge_index,
                    edge_attr=self.embed_label(updated_batched_graph.edge_attr),
                    edge_last_update=updated_batched_graph.edge_last_update,
                )
            )
            # (num_node, hidden_dim)

        # batchify node_embeddings
        batch_node_embeddings, batch_node_mask = batchify_node_features(
            node_embeddings, updated_batched_graph.batch, obs_mask.size(0)
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

        # (batch)
        decoder_output, decoder_attn_weights = self.decoder(
            self.get_decoder_input(
                event_type_ids,
                event_src_ids,
                event_dst_ids,
                event_label_ids,
                prev_batched_graph.batch,
                prev_batched_graph.x,
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
        # decoded_event_logits: (batch, hidden_dim)
        # updated_prev_input_event_emb_seq:
        #     (num_block, batch, input_seq_len, hidden_dim)
        # updated_prev_input_event_emb_seq_mask: (batch, input_seq_len)

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
            event_type_ids = event_type_logits.argmax(dim=1)
            masks = compute_masks_from_event_type_ids(event_type_ids)
            autoregressive_emb = self.event_type_head.get_autoregressive_embedding(
                decoder_output["output"], event_type_ids, masks["event_mask"]
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
                torch.argmax(
                    masked_softmax(event_src_logits, batch_node_mask, dim=1), dim=1
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
                torch.argmax(
                    masked_softmax(event_dst_logits, batch_node_mask, dim=1), dim=1
                ),
                batch_node_embeddings,
                batch_node_mask,
                masks["dst_mask"],
                event_dst_key,
            )
        label_logits = self.event_label_head(
            autoregressive_emb,
            self.embed_label(torch.arange(len(self.labels), device=self.device)),
        )
        # (batch, num_label)
        return {
            "event_type_logits": event_type_logits,
            "event_src_logits": event_src_logits,
            "event_dst_logits": event_dst_logits,
            "event_label_logits": label_logits,
            "updated_prev_input_event_emb_seq": decoder_output[
                "updated_prev_input_event_emb_seq"
            ],
            "updated_prev_input_event_emb_seq_mask": (
                decoder_output["updated_prev_input_event_emb_seq_mask"]
            ),
            "encoded_obs": encoded_obs,
            "encoded_prev_action": encoded_prev_action,
            "updated_batched_graph": updated_batched_graph,
            **decoder_attn_weights,
        }

    def embed_label(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        label_word_ids: (batch)
        output: (batch, hidden_dim)
        """
        return masked_mean(
            self.word_embeddings(self.label_word_ids[label_ids]),  # type: ignore
            self.label_mask[label_ids],  # type: ignore
        )

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

    def teacher_force(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        graphical_input_seq: Sequence[TWCmdGenGraphEventGraphicalInput],
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
        prev_input_event_emb_seq: Optional[torch.Tensor] = None
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None
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
                prev_input_event_emb_seq=prev_input_event_emb_seq,
                prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
                groundtruth_event={
                    "groundtruth_event_type_ids": (
                        graphical_input.groundtruth_event_type_ids
                    ),
                    "groundtruth_event_mask": graphical_input.groundtruth_event_mask,
                    "groundtruth_event_src_ids": (
                        graphical_input.groundtruth_event_src_ids
                    ),
                    "groundtruth_event_src_mask": (
                        graphical_input.groundtruth_event_src_mask
                    ),
                    "groundtruth_event_dst_ids": (
                        graphical_input.groundtruth_event_dst_ids
                    ),
                    "groundtruth_event_dst_mask": (
                        graphical_input.groundtruth_event_dst_mask
                    ),
                },
            )

            # add results to the list
            results_list.append(results)

            # update previous input event embedding sequence
            prev_input_event_emb_seq = results["updated_prev_input_event_emb_seq"]
            prev_input_event_emb_seq_mask = results[
                "updated_prev_input_event_emb_seq_mask"
            ]

            # save the encoded obs and prev action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

        return results_list

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
        groundtruth_event_mask: torch.Tensor,
        groundtruth_event_src_mask: torch.Tensor,
        groundtruth_event_dst_mask: torch.Tensor,
        groundtruth_event_label_mask: torch.Tensor,
    ) -> None:
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
        groundtruth_event_mask: (batch)
        groundtruth_event_src_mask: (batch)
        groundtruth_event_dst_mask: (batch)
        groundtruth_event_label_mask: (batch)
        """
        self.event_type_f1(
            event_type_logits[groundtruth_event_mask].softmax(dim=1),
            groundtruth_event_type_ids[groundtruth_event_mask],
        )
        if groundtruth_event_src_mask.any():
            self.src_node_f1(
                event_src_logits[groundtruth_event_src_mask].softmax(dim=1),
                groundtruth_event_src_ids[groundtruth_event_src_mask],
            )
        if groundtruth_event_dst_mask.any():
            self.dst_node_f1(
                event_dst_logits[groundtruth_event_dst_mask].softmax(dim=1),
                groundtruth_event_dst_ids[groundtruth_event_dst_mask],
            )
        if groundtruth_event_label_mask.any():
            self.label_f1(
                event_label_logits[groundtruth_event_label_mask].softmax(dim=1),
                groundtruth_event_label_ids[groundtruth_event_label_mask],
            )

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
        node_id_offsets = calculate_node_id_offsets(
            event_type_ids.size(0), batched_graph.batch
        )
        # (batch)
        batch_src_ids = src_ids + node_id_offsets
        # (batch)
        batch_dst_ids = dst_ids + node_id_offsets
        # (batch)

        cmds: List[str] = []
        tokens: List[List[str]] = []
        for event_type_id, batch_src_id, batch_dst_id, label_id in zip(
            event_type_ids.tolist(), batch_src_ids, batch_dst_ids, label_ids
        ):
            if event_type_id in {
                EVENT_TYPE_ID_MAP["edge-add"],
                EVENT_TYPE_ID_MAP["edge-delete"],
            }:
                src_label = self.labels[batched_graph.x[batch_src_id]]
                dst_label = self.labels[batched_graph.x[batch_dst_id]]
                if event_type_id == EVENT_TYPE_ID_MAP["edge-add"]:
                    cmd = "add"
                    edge_label = self.labels[label_id]
                else:
                    cmd = "delete"
                    edge_label = self.labels[
                        index_edge_attr(
                            batched_graph.edge_index,
                            batched_graph.edge_attr,
                            torch.stack(
                                [batch_src_id.unsqueeze(0), batch_dst_id.unsqueeze(0)]
                            ),
                        )
                    ]
                # in the original dataset, multi-word edge labels are joined by
                # an underscore
                cmd_tokens = [cmd, src_label, dst_label, "_".join(edge_label.split())]
                cmds.append(" , ".join(cmd_tokens))
                tokens.append(cmd_tokens)
            else:
                cmds.append("")
                tokens.append([])
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
        for event_type_ids, src_ids, dst_ids, label_ids, batched_graph in zip(
            event_type_id_seq,
            src_id_seq,
            dst_id_seq,
            label_id_seq,
            batched_graph_seq,
        ):
            for batch_id, (cmd, tokens) in enumerate(
                zip(
                    *self.generate_graph_triples(
                        event_type_ids, src_ids, dst_ids, label_ids, batched_graph
                    )
                )
            ):
                if cmd != "":
                    batch_cmds[batch_id].append(cmd)
                if len(tokens) != 0:
                    if len(batch_tokens[batch_id]) != 0:
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
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> torch.Tensor:
        """
        batch: the batch data
        batch_idx: the batch id, unused
        """
        results_list = self.teacher_force(batch.step_input, batch.graphical_input_seq)
        loss = torch.stack(
            [
                self.criterion(
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
                    graphical_input.groundtruth_event_label_mask,
                )
                for results, graphical_input in zip(
                    results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(
        self, batch: TWCmdGenGraphEventBatch, log_prefix: str
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
                self.criterion(
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
                    graphical_input.groundtruth_event_label_mask,
                )
                for results, graphical_input in zip(
                    tf_results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        self.log(log_prefix + "_loss", loss)

        # log classification F1s from teacher forcing
        for results, graphical_input in zip(tf_results_list, batch.graphical_input_seq):
            self.calculate_f1s(
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
                graphical_input.groundtruth_event_label_mask,
            )
        self.log(log_prefix + "_event_type_f1", self.event_type_f1)
        self.log(log_prefix + "_src_node_f1", self.src_node_f1)
        self.log(log_prefix + "_dst_node_f1", self.dst_node_f1)
        self.log(log_prefix + "_label_f1", self.label_f1)

        # calculate graph tuples from teacher forcing graph events
        tf_event_type_id_seq: List[torch.Tensor] = []
        tf_src_id_seq: List[torch.Tensor] = []
        tf_dst_id_seq: List[torch.Tensor] = []
        tf_label_id_seq: List[torch.Tensor] = []
        for results, graphical_input in zip(tf_results_list, batch.graphical_input_seq):
            # filter out pad events
            unfiltered_event_type_ids = (
                results["event_type_logits"].argmax(dim=1)
                * graphical_input.groundtruth_event_mask
            )
            # handle source/destination logits for empty graphs by setting them to zeros
            unfiltered_event_src_ids = (
                results["event_src_logits"].argmax(dim=1)
                if results["event_src_logits"].size(1) > 0
                else torch.zeros(
                    results["event_src_logits"].size(0),
                    dtype=torch.long,
                    device=self.device,
                )
            )
            unfiltered_event_dst_ids = (
                results["event_dst_logits"].argmax(dim=1)
                if results["event_dst_logits"].size(1) > 0
                else torch.zeros(
                    results["event_dst_logits"].size(0),
                    dtype=torch.long,
                    device=self.device,
                )
            )
            invalid_event_mask = self.filter_invalid_events(
                unfiltered_event_type_ids,
                unfiltered_event_src_ids,
                unfiltered_event_dst_ids,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            tf_event_type_id_seq.append(
                unfiltered_event_type_ids.masked_fill(
                    invalid_event_mask, EVENT_TYPE_ID_MAP["pad"]
                )
            )
            tf_src_id_seq.append(
                unfiltered_event_src_ids.masked_fill(invalid_event_mask, 0)
            )
            tf_dst_id_seq.append(
                unfiltered_event_dst_ids.masked_fill(invalid_event_mask, 0)
            )
            tf_label_id_seq.append(
                results["event_label_logits"]
                .argmax(dim=1)
                .masked_fill(invalid_event_mask, self.label_id_map[""])
            )

        batch_tf_cmds, batch_tf_tokens = self.generate_batch_graph_triples_seq(
            tf_event_type_id_seq,
            tf_src_id_seq,
            tf_dst_id_seq,
            tf_label_id_seq,
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

    def eval_free_run(
        self, dataset: TWCmdGenGraphEventFreeRunDataset, dataloader: DataLoader
    ) -> None:
        is_sanity_checking = (
            self.trainer.state.stage == RunningStage.SANITY_CHECKING  # type: ignore
        )
        if is_sanity_checking:
            total = self.trainer.num_sanity_val_steps  # type: ignore
        elif self.trainer.state.stage == RunningStage.VALIDATING:  # type: ignore
            if isinstance(self.trainer.limit_val_batches, float):  # type: ignore
                total = int(
                    self.trainer.limit_val_batches * len(dataset)  # type: ignore
                )
            elif isinstance(self.trainer.limit_val_batches, int):  # type: ignore
                total = self.trainer.limit_val_batches  # type: ignore
            else:
                total = len(dataset)
        elif self.trainer.state.stage == RunningStage.TESTING:  # type: ignore
            if isinstance(self.trainer.limit_test_batches, float):  # type: ignore
                total = int(
                    self.trainer.limit_test_batches * len(dataset)  # type: ignore
                )
            elif isinstance(self.trainer.limit_test_batches, int):  # type: ignore
                total = self.trainer.limit_test_batches  # type: ignore
            else:
                total = len(dataset)
        else:
            total = len(dataset)

        collator = self.trainer.datamodule.collator  # type: ignore
        game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], Data]] = {}
        with tqdm.tqdm(desc="Free Run", total=total) as pbar:
            for batch in dataloader:
                # finished games are the ones that were in game_id_to_graph, but are not
                # part of the new batch
                for finished_game_id in game_id_to_step_data_graph.keys() - {
                    game_id for game_id, _ in batch
                }:
                    step_data, graph = game_id_to_step_data_graph.pop(finished_game_id)
                    generated_rdfs = networkx_to_rdf(
                        data_to_networkx(graph, self.labels)
                    )
                    groundtruth_rdfs = update_rdf_graph(
                        set(step_data["previous_graph_seen"]),
                        step_data["target_commands"],
                    )
                    self.free_run_f1([generated_rdfs], [groundtruth_rdfs])
                    self.free_run_em([generated_rdfs], [groundtruth_rdfs])
                    pbar.update()
                    if pbar.n > total:
                        return

                # new games are the ones that were not in game_id_to_graph, but are now
                # part of the new batch.
                # due to Python's dictionary ordering (insertion order), new games are
                # added always to the end.
                for game_id, step_data in batch:
                    if game_id in game_id_to_step_data_graph:
                        _, graph = game_id_to_step_data_graph[game_id]
                        game_id_to_step_data_graph[game_id] = (step_data, graph)
                    else:
                        game_id_to_step_data_graph[game_id] = (
                            step_data,
                            Data(
                                x=torch.empty(0, dtype=torch.long),
                                node_last_update=torch.empty(0, 2, dtype=torch.long),
                                edge_index=torch.empty(2, 0, dtype=torch.long),
                                edge_attr=torch.empty(0, dtype=torch.long),
                                edge_last_update=torch.empty(0, 2, dtype=torch.long),
                            ).to(self.device),
                        )

                # sanity check
                assert [game_id for game_id, _ in batch] == [
                    game_id for game_id in game_id_to_step_data_graph
                ]

                # construct a batch
                batched_obs: List[str] = []
                batched_prev_actions: List[str] = []
                batched_timestamps: List[int] = []
                graph_list: List[Data] = []
                for game_id, (step_data, graph) in game_id_to_step_data_graph.items():
                    batched_obs.append(step_data["observation"])
                    batched_prev_actions.append(step_data["previous_action"])
                    batched_timestamps.append(step_data["timestamp"])
                    graph_list.append(graph)

                # greedy decode
                results_list = self.greedy_decode(
                    collator.collate_step_inputs(  # type: ignore
                        batched_obs, batched_prev_actions, batched_timestamps
                    ).to(self.device),
                    Batch.from_data_list(graph_list),
                )

                # update graphs in game_id_to_step_data_graph
                for (game_id, (step_data, _)), updated_graph in zip(
                    game_id_to_step_data_graph.items(),
                    batch_to_data_list(
                        results_list[-1]["updated_batched_graph"], len(batched_obs)
                    ),
                ):
                    game_id_to_step_data_graph[game_id] = (step_data, updated_graph)

    def validation_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch, "val")

    def test_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
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
        self.eval_free_run(
            self.trainer.datamodule.valid_free_run,  # type: ignore
            self.trainer.datamodule.val_free_run_dataloader(),  # type: ignore
        )
        self.log("val_free_run_f1", self.free_run_f1, prog_bar=True)
        self.log("val_free_run_em", self.free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "val_gen_graph_triples")

    def test_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        self.eval_free_run(
            self.trainer.datamodule.test_free_run,  # type: ignore
            self.trainer.datamodule.test_free_run_dataloader(),  # type: ignore
        )
        self.log("test_free_run_f1", self.free_run_f1)
        self.log("test_free_run_em", self.free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "test_gen_graph_triples")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def greedy_decode(
        self, step_input: TWCmdGenGraphEventStepInput, prev_batched_graph: Batch
    ) -> List[Dict[str, Any]]:
        """
        step_input: the current step input
        prev_batch_graph: diagonally stacked batch of current graphs

        output:
        len([{
            decoded_event_type_ids: (batch)
            decoded_event_src_ids: (batch)
            decoded_event_dst_ids: (batch)
            decoded_event_label_ids: (batch)
            updated_batched_graph: diagonally stacked batch of updated graphs.
                these are the graphs used to decode the graph events above
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
        }, ...]) == decode_len <= max_decode_len
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

        prev_input_event_emb_seq: Optional[torch.Tensor] = None
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None
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
                prev_input_event_emb_seq=prev_input_event_emb_seq,
                prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
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
                .masked_fill(end_event_mask, self.label_id_map[""])
            )
            # (batch)

            # filter out invalid decoded events
            invalid_event_mask = self.filter_invalid_events(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            # (batch)
            decoded_event_type_ids = decoded_event_type_ids.masked_fill(
                invalid_event_mask, EVENT_TYPE_ID_MAP["pad"]
            )
            decoded_src_ids = decoded_src_ids.masked_fill(invalid_event_mask, 0)
            decoded_dst_ids = decoded_dst_ids.masked_fill(invalid_event_mask, 0)
            decoded_label_ids = decoded_label_ids.masked_fill(
                invalid_event_mask, self.label_id_map[""]
            )

            # collect the results
            results_list.append(
                {
                    "decoded_event_type_ids": decoded_event_type_ids,
                    "decoded_event_src_ids": decoded_src_ids,
                    "decoded_event_dst_ids": decoded_dst_ids,
                    "decoded_event_label_ids": decoded_label_ids,
                    "updated_batched_graph": results["updated_batched_graph"],
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

    def get_decoder_input(
        self,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_ids: torch.Tensor,
        batch: torch.Tensor,
        node_label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Turn the graph events into decoder inputs by concatenating their embeddings.

        input:
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_ids: (batch)
            batch: (num_node)
            node_label_ids: (num_node)

        output: (batch, decoder_input_dim)
            where decoder_input_dim = event_type_embedding_dim + 3 * label_embedding_dim
        """
        batch_size = event_type_ids.size(0)

        # event type embeddings
        event_type_embs = self.event_type_embeddings(event_type_ids)
        # (batch, event_type_embedding_dim)

        # event label embeddings
        event_label_embs = torch.zeros(
            batch_size,
            self.hparams.hidden_dim,  # type: ignore
            device=event_type_ids.device,
        )
        # (batch, label_embedding_dim)
        node_add_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (batch)
        node_delete_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
        # (batch)
        edge_event_mask = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
        )
        # (batch)
        label_mask = node_add_mask.logical_or(node_delete_mask).logical_or(
            edge_event_mask
        )
        # (batch)
        event_label_embs[label_mask] = self.embed_label(event_label_ids[label_mask])
        # (batch, label_embedding_dim)

        # event source/destination node embeddings
        event_src_embs = torch.zeros(
            batch_size,
            self.hparams.hidden_dim,  # type: ignore
            device=event_type_ids.device,
        )
        # (batch, label_embedding_dim)
        event_dst_embs = torch.zeros(
            batch_size,
            self.hparams.hidden_dim,  # type: ignore
            device=event_type_ids.device,
        )
        # (batch, node_embedding_dim)
        src_node_mask = node_delete_mask.logical_or(edge_event_mask)
        # (batch)
        node_id_offsets = calculate_node_id_offsets(batch_size, batch)
        # (batch)
        if src_node_mask.any():
            event_src_embs[src_node_mask] = self.embed_label(
                node_label_ids[(event_src_ids + node_id_offsets)[src_node_mask]]
            )
            # (batch, label_embedding_dim)

        if edge_event_mask.any():
            event_dst_embs[edge_event_mask] = self.embed_label(
                node_label_ids[(event_dst_ids + node_id_offsets)[edge_event_mask]]
            )
            # (batch, node_embedding_dim)

        return torch.cat(
            [event_type_embs, event_src_embs, event_dst_embs, event_label_embs], dim=1
        )
        # (batch, event_type_embedding_dim + 3 * node_embedding_dim)


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # log variance for event type, source node, destination node and label
        # classification tasks
        self.log_var = nn.parameter.Parameter(torch.zeros(4))

    def forward(
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
        groundtruth_event_label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the total loss using the weighting strategy from Kendall, et al. 2018.
        with a small modification from Liebel, et al. 2018.

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
        groundtruth_event_label_mask: (batch)

        output: (batch)
        """
        # event type loss
        event_type_loss = (
            self.criterion(event_type_logits, groundtruth_event_type_ids)
            * groundtruth_event_mask
        )
        # (batch)
        src_node_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_src_mask.any():
            # source node loss
            src_node_loss += (
                self.criterion(event_src_logits, groundtruth_event_src_ids)
                * groundtruth_event_src_mask
            )
        dst_node_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_dst_mask.any():
            # destination node loss
            dst_node_loss += (
                self.criterion(event_dst_logits, groundtruth_event_dst_ids)
                * groundtruth_event_dst_mask
            )
        label_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_label_mask.any():
            # label loss
            label_loss += (
                self.criterion(event_label_logits, groundtruth_event_label_ids)
                * groundtruth_event_label_mask
            )

        # calculate the total loss
        precision = torch.exp(-self.log_var)
        # (4)
        stacked_losses = torch.stack(
            [event_type_loss, src_node_loss, dst_node_loss, label_loss]
        ).t()
        # (batch, 4)
        regularizer = torch.log1p(torch.exp(self.log_var))
        # (4)
        return torch.sum(stacked_losses * precision + regularizer, dim=1)
        # (batch)
