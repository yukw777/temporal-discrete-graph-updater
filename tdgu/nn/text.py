import torch
import torch.nn as nn

from typing import Optional

from hydra.utils import to_absolute_path
from pathlib import Path
from tdgu.nn.utils import PositionalEncoder, load_fasttext
from tdgu.preprocessor import SpacyPreprocessor


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise, separable 1d convolution to save computation in exchange for
    a bit of accuracy.
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pointwise_conv = torch.nn.Conv1d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, in_channels, seq_len_in)
        output: (batch, out_channels, seq_len_out)
        seq_len_out = (seq_len_in + 2 * (kernel_size // 2) - (kernel_size - 1) - 1) + 1
        """
        return self.relu(self.pointwise_conv(self.depthwise_conv(input)))


class TextEncoderConvBlock(nn.Module):
    """
    Convolutional blocks used in QANet.
    A layer norm followed by a depthwise, separable convolutional layer
    with a residual connection.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.3) -> None:
        super().__init__()
        assert (
            kernel_size % 2 == 1
        ), "kernel_size has to be odd in order to preserve the sequence length"
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv = DepthwiseSeparableConv1d(channels, channels, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, channels)
        output: (batch, seq_len, channels)
        """
        residual = input
        output = self.layer_norm(input)
        output = self.conv(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(output)
        return output + residual


class TextEncoderBlock(nn.Module):
    """
    Based on QANet (https://arxiv.org/abs/1804.09541)
    """

    def __init__(
        self,
        num_conv_layers: int,
        kernel_size: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim has to be even for positional encoding"
        self.pos_encoder = PositionalEncoder(hidden_dim, 512)
        self.conv_layers = nn.Sequential(
            *[
                TextEncoderConvBlock(hidden_dim, kernel_size)
                for _ in range(num_conv_layers)
            ]
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.self_attn_dropout = nn.Dropout(p=dropout)
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.linear_dropout = nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len)
        output: (batch, seq_len, hidden_dim)
        """
        # add the positional encodings using broadcast
        output = input + self.pos_encoder(
            torch.arange(input.size(1), device=input.device)
        )
        # (batch, seq_len, hidden_dim)

        # conv layers
        output = self.conv_layers(output)

        # self attention layer
        residual = output
        output = self.self_attn_layer_norm(output)
        output, _ = self.self_attn(output, output, output, key_padding_mask=mask == 0)
        output = self.self_attn_dropout(output)
        output += residual

        # linear layer
        residual = output
        output = self.linear_layer_norm(output)
        output = self.linear_layers(output)
        output = self.linear_dropout(output)
        output += residual

        return output


class TextEncoder(nn.Module):
    def __init__(
        self,
        preprocessor: SpacyPreprocessor,
        word_emb_dim: int,
        num_enc_blocks: int,
        enc_block_num_conv_layers: int,
        enc_block_kernel_size: int,
        enc_block_hidden_dim: int,
        enc_block_num_heads: int,
        dropout: float = 0.3,
        embedding_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.enc_block_hidden_dim = enc_block_hidden_dim
        if embedding_path is not None:
            abs_pretrained_word_embedding_path = Path(to_absolute_path(embedding_path))
            serialized_path = abs_pretrained_word_embedding_path.parent / (
                abs_pretrained_word_embedding_path.stem + ".pt"
            )
            self.pretrained_word_embeddings = load_fasttext(
                str(abs_pretrained_word_embedding_path),
                serialized_path,
                preprocessor,
            )
            assert word_emb_dim == self.pretrained_word_embeddings.embedding_dim
        else:
            self.pretrained_word_embeddings = nn.Embedding(
                preprocessor.vocab_size, word_emb_dim
            )

        self.pretrained_word_embeddings.weight.requires_grad = False
        self.word_embeddings = nn.Sequential(
            self.pretrained_word_embeddings,
            nn.Linear(word_emb_dim, enc_block_hidden_dim),
        )

        self.enc_blocks = nn.ModuleList(
            TextEncoderBlock(
                enc_block_num_conv_layers,
                enc_block_kernel_size,
                enc_block_hidden_dim,
                enc_block_num_heads,
                dropout=dropout,
            )
            for _ in range(num_enc_blocks)
        )

    def forward(self, word_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        word_ids: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        output:
            encoded: (batch_size, seq_len, enc_block_hidden_dim)
        """

        if word_ids.size(1) == 0:
            return torch.empty(
                word_ids.size(0), 0, self.enc_block_hidden_dim, device=word_ids.device
            )

        word_embs = self.word_embeddings(word_ids)
        # (batch_size, seq_len, enc_block_hidden_dim)

        for enc_block in self.enc_blocks:
            word_embs = enc_block(word_embs, mask)
        # (batch_size, seq_len, enc_block_hidden_dim)

        return word_embs

    @property
    def get_pretrained_embedding(self) -> nn.Embedding:
        return self.pretrained_word_embeddings
