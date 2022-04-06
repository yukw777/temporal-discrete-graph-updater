import torch
import pytest

from tdgu.nn.text import (
    DepthwiseSeparableConv1d,
    TextEncoderConvBlock,
    TextEncoderBlock,
    TextEncoder,
)
from tdgu.preprocessor import SpacyPreprocessor
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
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_text_enc_conv_block(dropout, channels, kernel_size, batch_size, seq_len):
    if dropout is None:
        conv = TextEncoderConvBlock(channels, kernel_size)
    else:
        conv = TextEncoderConvBlock(channels, kernel_size, dropout=dropout)
    assert conv(torch.rand(batch_size, seq_len, channels)).size() == (
        batch_size,
        seq_len,
        channels,
    )


@pytest.mark.parametrize(
    "num_conv_layers,kernel_size,hidden_dim,num_heads,batch_size,seq_len",
    [
        (1, 3, 10, 1, 3, 5),
        (3, 5, 12, 3, 3, 10),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_text_enc_block(
    dropout, num_conv_layers, kernel_size, hidden_dim, num_heads, batch_size, seq_len
):
    if dropout is None:
        text_enc_block = TextEncoderBlock(
            num_conv_layers, kernel_size, hidden_dim, num_heads
        )
    else:
        text_enc_block = TextEncoderBlock(
            num_conv_layers, kernel_size, hidden_dim, num_heads, dropout=dropout
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
    "num_enc_blocks,word_emb_dim,enc_block_num_conv_layers,enc_block_kernel_size,"
    "enc_block_hidden_dim,enc_block_num_heads,batch_size,seq_len",
    [
        (1, 5, 1, 3, 8, 1, 1, 1),
        (1, 6, 1, 3, 8, 1, 2, 5),
        (3, 42, 5, 5, 10, 5, 3, 7),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_text_encoder(
    dropout,
    num_enc_blocks,
    word_emb_dim,
    enc_block_num_conv_layers,
    enc_block_kernel_size,
    enc_block_hidden_dim,
    enc_block_num_heads,
    batch_size,
    seq_len,
):
    preprocessor = SpacyPreprocessor.load_from_file("tests/data/test_word_vocab.txt")

    if dropout is None:
        text_encoder = TextEncoder(
            preprocessor,
            word_emb_dim,
            num_enc_blocks,
            enc_block_num_conv_layers,
            enc_block_kernel_size,
            enc_block_hidden_dim,
            enc_block_num_heads,
        )
    else:
        text_encoder = TextEncoder(
            preprocessor,
            word_emb_dim,
            num_enc_blocks,
            enc_block_num_conv_layers,
            enc_block_kernel_size,
            enc_block_hidden_dim,
            enc_block_num_heads,
            dropout=dropout,
        )
    # random word ids and increasing masks
    assert text_encoder(
        torch.randint(0, preprocessor.vocab_size, size=(batch_size, seq_len)),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
        ),
    ).size() == (
        batch_size,
        seq_len,
        enc_block_hidden_dim,
    )
