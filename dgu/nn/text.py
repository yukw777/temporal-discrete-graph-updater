import torch
import torch.nn as nn
import math


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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, in_channels, seq_len_in)
        output: (batch, out_channels, seq_len_out)
        seq_len_out = (seq_len_in + 2 * (kernel_size // 2) - (kernel_size - 1) - 1) + 1
        """
        return self.pointwise_conv(self.depthwise_conv(input))


class TextEncoderConvBlock(nn.Module):
    """
    Convolutional blocks used in QANet.
    A layer norm followed by a depthwise, separable convolutional layer
    with a residual connection.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        assert (
            kernel_size % 2 == 1
        ), "kernel_size has to be odd in order to preserve the sequence length"
        self.layer_norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()
        self.conv = DepthwiseSeparableConv1d(channels, channels, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, channels)
        output: (batch, seq_len, channels)
        """
        residual = input
        output = self.layer_norm(input)
        output = self.relu(self.conv(output.transpose(1, 2))).transpose(1, 2)
        return output + residual


class PositionalEncoder(nn.Module):
    """
    The positional encoding from the original Transformer paper.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, d_model)
        output: (batch, seq_len, d_model)
        """
        # add positional encodings to the input using broadcast
        return input + self.pe[: input.size(1), :]  # type: ignore


class PositionalEncoderTensor2Tensor(nn.Module):
    """
    Add positional encodings to the given input. This is the tensor2tensor
    implementation of the positional encoding, which is slightly different
    from the one used by the original Transformer paper.
    Specifically, there are 2 key differences:
    1. Sine and cosine values are concatenated rather than interweaved.
    2. The divisor is calculated differently
        ((d_model (or channels) // 2) -1 vs. d_model)
    There are no material differences between positional encoding implementations.
    The important point is that you use the same implementation throughout. The
    original GATA code uses this version. I've cleaned up the implementation a bit,
    including a small optimization that caches all the positional encodings, which
    was shown in the PyTorch Transformer tutorial
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    def __init__(
        self,
        channels: int,
        max_len: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1e4,
    ) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        num_timescales = channels // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / (
            num_timescales - 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * -log_timescale_increment
        ).unsqueeze(0)
        scaled_time = position * inv_timescales
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).view(
            max_len, channels
        )
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, channels)
        output: (batch, seq_len, channels)
        """
        # add positional encodings to the input using broadcast
        return input + self.pe[: input.size(1)]  # type: ignore


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
    ) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim has to be even for positional encoding"
        self.pos_encoder = PositionalEncoderTensor2Tensor(hidden_dim, 512)
        self.conv_layers = nn.Sequential(
            *[
                TextEncoderConvBlock(hidden_dim, kernel_size)
                for _ in range(num_conv_layers)
            ]
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len)
        output: (batch, seq_len, hidden_dim)
        """
        # add the positional encodings
        output = self.pos_encoder(input)

        # conv layers
        output = self.conv_layers(output)

        # self attention layer
        residual = output
        # MultiheadAttention expects batch dim to be 1 for q, k, v
        # but 0 for key_padding_mask, so we need to transpose
        output = output.transpose(0, 1)
        output = self.self_attn_layer_norm(output)
        output, _ = self.self_attn(output, output, output, key_padding_mask=mask == 0)
        output = output.transpose(0, 1)
        output += residual

        # linear layer
        residual = output
        output = self.linear_layer_norm(output)
        output = self.linear_layers(output)
        output += residual

        return output


class TextEncoder(nn.Module):
    def __init__(
        self,
        num_enc_blocks: int,
        enc_block_num_conv_layers: int,
        enc_block_kernel_size: int,
        enc_block_hidden_dim: int,
        enc_block_num_heads: int,
    ) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            TextEncoderBlock(
                enc_block_num_conv_layers,
                enc_block_kernel_size,
                enc_block_hidden_dim,
                enc_block_num_heads,
            )
            for _ in range(num_enc_blocks)
        )

    def forward(
        self, input_word_embs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        input_word_embs: (batch_size, seq_len, enc_block_hidden_dim)
        mask: (batch_size, seq_len)
        output:
            encoded: (batch_size, seq_len, enc_block_hidden_dim)
        """
        output = input_word_embs
        # (batch_size, seq_len, enc_block_hidden_dim)
        for enc_block in self.enc_blocks:
            output = enc_block(output, mask)
        # (batch_size, seq_len, enc_block_hidden_dim)

        return output
