import torch
import pytest

from dgu.nn.text import (
    DepthwiseSeparableConv1d,
    TextEncoderConvBlock,
    PositionalEncoder,
    PositionalEncoderTensor2Tensor,
    TextEncoderBlock,
    TextEncoder,
)
from utils import increasing_mask


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,batch_size,seq_len_in,seq_len_out",
    [
        (10, 5, 3, 2, 5, 5),
        (15, 4, 2, 3, 10, 11),
    ],
)
def test_depthwise_separable_conv_1d(
    in_channels, out_channels, kernel_size, batch_size, seq_len_in, seq_len_out
):
    ds_conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size)
    assert ds_conv(torch.rand(batch_size, in_channels, seq_len_in)).size() == (
        batch_size,
        out_channels,
        seq_len_out,
    )


@pytest.mark.parametrize(
    "channels,kernel_size,batch_size,seq_len",
    [
        (10, 3, 2, 5),
        (15, 5, 3, 10),
        (15, 11, 5, 20),
    ],
)
def test_text_enc_conv_block(channels, kernel_size, batch_size, seq_len):
    conv = TextEncoderConvBlock(channels, kernel_size)
    assert conv(torch.rand(batch_size, seq_len, channels)).size() == (
        batch_size,
        seq_len,
        channels,
    )


@pytest.mark.parametrize(
    "d_model,max_len,batch_size,seq_len",
    [
        (4, 10, 5, 6),
        (10, 12, 3, 10),
    ],
)
def test_pos_encoder(d_model, max_len, batch_size, seq_len):
    pe = PositionalEncoder(d_model, max_len)
    encoded = pe(
        torch.zeros(
            batch_size,
            seq_len,
            d_model,
        )
    )
    assert encoded.size() == (batch_size, seq_len, d_model)
    # sanity check, make sure the values of the first dimension is sin(0, 1, 2, ...)
    # and the second dimension is cos(0, 1, 2, ...)
    for i in range(batch_size):
        assert encoded[i, :, 0].equal(torch.sin(torch.arange(seq_len).float()))
        assert encoded[i, :, 1].equal(torch.cos(torch.arange(seq_len).float()))


@pytest.mark.parametrize(
    "channels,max_len,batch_size,seq_len",
    [
        (4, 10, 5, 6),
        (10, 12, 3, 10),
    ],
)
def test_pos_encoder_tensor2tensor(channels, max_len, batch_size, seq_len):
    pe = PositionalEncoderTensor2Tensor(channels, max_len)
    encoded = pe(
        torch.zeros(
            batch_size,
            seq_len,
            channels,
        )
    )
    assert encoded.size() == (batch_size, seq_len, channels)
    # sanity check, make sure the values of the first dimension of both halves
    # of the channels is sin(0, 1, 2, ...) and cos(0, 1, 2, ...)
    for i in range(batch_size):
        assert encoded[i, :, 0].equal(torch.sin(torch.arange(seq_len).float()))
        assert encoded[i, :, channels // 2].equal(
            torch.cos(torch.arange(seq_len).float())
        )


@pytest.mark.parametrize(
    "num_conv_layers,kernel_size,hidden_dim,num_heads,batch_size,seq_len",
    [
        (1, 3, 10, 1, 3, 5),
        (3, 5, 12, 3, 3, 10),
    ],
)
def test_text_enc_block(
    num_conv_layers, kernel_size, hidden_dim, num_heads, batch_size, seq_len
):
    text_enc_block = TextEncoderBlock(
        num_conv_layers, kernel_size, hidden_dim, num_heads
    )
    # random tensors and increasing masks
    assert text_enc_block(
        torch.rand(batch_size, seq_len, hidden_dim),
        increasing_mask(batch_size, seq_len),
    ).size() == (
        batch_size,
        seq_len,
        hidden_dim,
    )


@pytest.mark.parametrize(
    "num_enc_blocks,enc_block_num_conv_layers,enc_block_kernel_size,"
    "enc_block_hidden_dim,enc_block_num_heads,batch_size,seq_len",
    [
        (1, 1, 3, 8, 1, 1, 1),
        (1, 1, 3, 8, 1, 2, 5),
        (3, 5, 5, 10, 5, 3, 7),
    ],
)
def test_text_encoder(
    num_enc_blocks,
    enc_block_num_conv_layers,
    enc_block_kernel_size,
    enc_block_hidden_dim,
    enc_block_num_heads,
    batch_size,
    seq_len,
):
    text_encoder = TextEncoder(
        num_enc_blocks,
        enc_block_num_conv_layers,
        enc_block_kernel_size,
        enc_block_hidden_dim,
        enc_block_num_heads,
    )
    # random word ids and increasing masks
    assert text_encoder(
        torch.rand(batch_size, seq_len, enc_block_hidden_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
        ),
    ).size() == (
        batch_size,
        seq_len,
        enc_block_hidden_dim,
    )