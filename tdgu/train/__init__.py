import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional
from pathlib import Path
from hydra.utils import to_absolute_path
from torch_geometric.nn import TransformerConv, GATv2Conv

from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS
from tdgu.nn.text import QANetTextEncoder
from tdgu.nn.graph_event_decoder import TransformerGraphEventDecoder
from tdgu.nn.utils import load_fasttext


class TDGULightningModule(  # type: ignore
    TemporalDiscreteGraphUpdater, pl.LightningModule
):
    """
    Base LightningModule class for TDGU.
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
        graph_event_decoder_autoregressive_emb_dim: int = 8,
        graph_event_decoder_key_query_dim: int = 8,
        graph_event_decoder_num_dec_blocks: int = 1,
        graph_event_decoder_dec_block_num_heads: int = 1,
        max_event_decode_len: int = 100,
        max_label_decode_len: int = 10,
        learning_rate: float = 5e-4,
        dropout: float = 0.3,
        allow_objs_with_same_label: bool = False,
        pretrained_word_embedding_path: Optional[str] = None,
        word_vocab_path: Optional[str] = None,
    ) -> None:
        # preprocessor
        preprocessor = (
            SpacyPreprocessor([PAD, UNK, BOS, EOS])
            if word_vocab_path is None
            else SpacyPreprocessor.load_from_file(to_absolute_path(word_vocab_path))
        )

        # pretrained word embeddings
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
                preprocessor.get_vocab(),
                preprocessor.pad_token_id,
            )
            pretrained_word_embeddings.requires_grad_(requires_grad=False)
            assert word_emb_dim == pretrained_word_embeddings.embedding_dim
        else:
            pretrained_word_embeddings = nn.Embedding(
                preprocessor.vocab_size, word_emb_dim
            )
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
            preprocessor,
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
            dropout,
        )
        self.save_hyperparameters(
            ignore=["pretrained_word_embedding_path", "word_vocab_path"]
        )
