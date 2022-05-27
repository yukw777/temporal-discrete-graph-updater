import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional
from torch_geometric.nn import TransformerConv, GATv2Conv

from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.nn.text import QANetTextEncoder
from tdgu.nn.graph_event_decoder import TransformerGraphEventDecoder


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
        text_encoder_vocab_size: int = 4,
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
        label_head_bos_token_id: int = 2,
        label_head_eos_token_id: int = 3,
        label_head_pad_token_id: int = 0,
        max_event_decode_len: int = 100,
        max_label_decode_len: int = 10,
        learning_rate: float = 5e-4,
        dropout: float = 0.3,
        allow_objs_with_same_label: bool = False,
        pretrained_word_embeddings: Optional[nn.Embedding] = None,
    ) -> None:
        if pretrained_word_embeddings is None:
            pretrained_word_embeddings = nn.Embedding(
                text_encoder_vocab_size, text_encoder_word_emb_dim
            )
        else:
            assert pretrained_word_embeddings.num_embeddings == text_encoder_vocab_size
            assert pretrained_word_embeddings.embedding_dim == text_encoder_word_emb_dim
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
            label_head_bos_token_id,
            label_head_eos_token_id,
            label_head_pad_token_id,
            dropout,
        )
        self.save_hyperparameters(ignore=["pretrained_word_embeddings"])
