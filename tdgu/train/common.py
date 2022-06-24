import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Optional, List, Dict, Any
from torch_geometric.nn import TransformerConv, GATv2Conv
from torch_geometric.data import Batch
from hydra.utils import to_absolute_path
from pathlib import Path

from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.nn.text import QANetTextEncoder
from tdgu.nn.graph_event_decoder import TransformerGraphEventDecoder
from tdgu.data import TWCmdGenGraphEventStepInput
from tdgu.constants import EVENT_TYPE_ID_MAP
from tdgu.nn.utils import masked_softmax, load_fasttext, masked_gumbel_softmax
from tdgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS


class TDGULightningModule(  # type: ignore
    TemporalDiscreteGraphUpdater, pl.LightningModule
):
    """
    Base LightningModule class for TDGU.
    """

    def __init__(
        self,
        hidden_dim: int = 8,
        dgnn_gnn: str = "TransformerConv",
        dgnn_timestamp_enc_dim: int = 8,
        dgnn_num_gnn_block: int = 1,
        dgnn_num_gnn_head: int = 1,
        dgnn_zero_timestamp_encoder: bool = False,
        text_encoder_word_emb_dim: int = 300,
        text_encoder_num_blocks: int = 1,
        text_encoder_num_conv_layers: int = 3,
        text_encoder_kernel_size: int = 5,
        text_encoder_num_heads: int = 1,
        graph_event_decoder_event_type_emb_dim: int = 8,
        graph_event_decoder_hidden_dim: int = 8,
        graph_event_decoder_autoregressive_emb_dim: int = 8,
        graph_event_decoder_key_query_dim: int = 8,
        graph_event_decoder_num_dec_blocks: int = 1,
        graph_event_decoder_dec_block_num_heads: int = 1,
        max_event_decode_len: int = 100,
        max_label_decode_len: int = 10,
        learning_rate: float = 5e-4,
        dropout: float = 0.3,
        allow_objs_with_same_label: bool = False,
        word_vocab_path: Optional[str] = None,
        pretrained_word_embedding_path: Optional[str] = None,
    ) -> None:
        # preprocessor
        self.preprocessor = (
            SpacyPreprocessor([PAD, UNK, BOS, EOS])
            if word_vocab_path is None
            else SpacyPreprocessor.load_from_file(to_absolute_path(word_vocab_path))
        )

        # pretrained word embeddings
        pretrained_word_embeddings: Optional[nn.Embedding]
        if pretrained_word_embedding_path is None:
            pretrained_word_embeddings = nn.Embedding(
                self.preprocessor.vocab_size, text_encoder_word_emb_dim
            )
        else:
            abs_pretrained_word_embedding_path = Path(
                to_absolute_path(pretrained_word_embedding_path)
            )
            serialized_path = abs_pretrained_word_embedding_path.parent / (
                abs_pretrained_word_embedding_path.stem + ".pt"
            )
            pretrained_word_embeddings = load_fasttext(
                str(abs_pretrained_word_embedding_path),
                serialized_path,
                self.preprocessor.get_vocab(),
                self.preprocessor.pad_token_id,
            )
            pretrained_word_embeddings.requires_grad_(requires_grad=False)

        # text encoder
        text_encoder = QANetTextEncoder(
            pretrained_word_embeddings,
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
        super().__init__(
            text_encoder,
            gnn_module,
            hidden_dim,
            dgnn_timestamp_enc_dim,
            dgnn_num_gnn_block,
            dgnn_num_gnn_head,
            dgnn_zero_timestamp_encoder,
            TransformerGraphEventDecoder(
                graph_event_decoder_event_type_emb_dim + 3 * hidden_dim,
                hidden_dim,
                graph_event_decoder_num_dec_blocks,
                graph_event_decoder_dec_block_num_heads,
                graph_event_decoder_hidden_dim,
                dropout=dropout,
            ),
            graph_event_decoder_event_type_emb_dim,
            graph_event_decoder_autoregressive_emb_dim,
            graph_event_decoder_key_query_dim,
            self.preprocessor.bos_token_id,
            self.preprocessor.eos_token_id,
            self.preprocessor.pad_token_id,
            dropout,
        )
        self.save_hyperparameters(
            ignore=["word_vocab_path", "pretrained_word_embedding_path"]
        )

    def greedy_decode(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        prev_batched_graph: Batch,
        max_event_decode_len: int = 100,
        max_label_decode_len: int = 10,
        gumbel_greedy_decode: bool = False,
        gumbel_tau: float = 0.5,
    ) -> List[Dict[str, Any]]:
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
                torch.full(
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
            decoded_src_ids = torch.zeros(
                batch_size, prev_max_sub_graph_num_node, device=self.device
            )
            # (batch, prev_max_sub_graph_num_node)
            decoded_dst_ids = torch.zeros(
                batch_size, prev_max_sub_graph_num_node, device=self.device
            )
            # (batch, prev_max_sub_graph_num_node)
            decoded_label_word_ids = torch.empty(
                batch_size,
                0,
                self.preprocessor.vocab_size,
                device=self.device,
            )
            # (batch, 0, num_word)
        else:
            decoded_event_type_ids = torch.full(
                (batch_size,), EVENT_TYPE_ID_MAP["start"], device=self.device
            )
            # (batch)
            decoded_src_ids = torch.zeros(
                batch_size, device=self.device, dtype=torch.long
            )
            # (batch)
            decoded_dst_ids = torch.zeros(
                batch_size, device=self.device, dtype=torch.long
            )
            # (batch)
            decoded_label_word_ids = torch.empty(
                batch_size, 0, device=self.device, dtype=torch.long
            )
            # (batch, 0)
        decoded_label_mask = torch.empty(
            batch_size, 0, device=self.device, dtype=torch.bool
        )
        # (batch, 0)

        end_event_mask = torch.full((batch_size,), False, device=self.device)
        # (batch)

        prev_input_event_emb_seq: Optional[torch.Tensor] = None
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None
        encoded_obs: Optional[torch.Tensor] = None
        encoded_prev_action: Optional[torch.Tensor] = None
        results_list: List[Dict[str, Any]] = []
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
                    decoded_src_ids = torch.zeros(batch_size, 0, device=self.device)
                    # (batch, 0)
                else:
                    decoded_src_ids = torch.zeros(
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
                    decoded_dst_ids = torch.zeros(batch_size, 0, device=self.device)
                    # (batch, 0)
                else:
                    decoded_dst_ids = torch.zeros(
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
