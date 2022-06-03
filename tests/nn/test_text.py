import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from tdgu.nn.text import (
    DepthwiseSeparableConv1d,
    TextEncoderConvBlock,
    TextEncoderBlock,
    QANetTextEncoder,
    TextDecoderBlock,
    TextDecoder,
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


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize(
    "num_enc_blocks,enc_block_num_conv_layers,enc_block_kernel_size,"
    "enc_block_hidden_dim,enc_block_num_heads,batch_size,seq_len",
    [
        (1, 1, 3, 8, 1, 1, 1),
        (1, 1, 3, 8, 1, 2, 5),
        (3, 5, 5, 10, 5, 3, 7),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_qanet_text_encoder(
    dropout,
    num_enc_blocks,
    enc_block_num_conv_layers,
    enc_block_kernel_size,
    enc_block_hidden_dim,
    enc_block_num_heads,
    batch_size,
    seq_len,
    one_hot,
):
    vocab_size = 10
    word_emb_dim = 12
    if dropout is None:
        text_encoder = QANetTextEncoder(
            nn.Embedding(vocab_size, word_emb_dim),
            num_enc_blocks,
            enc_block_num_conv_layers,
            enc_block_kernel_size,
            enc_block_hidden_dim,
            enc_block_num_heads,
        )
    else:
        text_encoder = QANetTextEncoder(
            nn.Embedding(vocab_size, word_emb_dim),
            num_enc_blocks,
            enc_block_num_conv_layers,
            enc_block_kernel_size,
            enc_block_hidden_dim,
            enc_block_num_heads,
            dropout=dropout,
        )
    # random word ids
    word_ids = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    if one_hot:
        word_ids = F.one_hot(word_ids, num_classes=vocab_size).float()
    # increasing masks
    mask = torch.tensor(
        [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
    )
    assert text_encoder(word_ids, mask).size() == (
        batch_size,
        seq_len,
        enc_block_hidden_dim,
    )


@pytest.mark.parametrize(
    "hidden_dim,num_heads,batch_size,input_seq_len,num_node,prev_action_len",
    [
        (10, 1, 1, 3, 5, 4),
        (20, 2, 3, 5, 10, 8),
    ],
)
def test_text_decoder_block(
    hidden_dim, num_heads, batch_size, input_seq_len, num_node, prev_action_len
):
    decoder_block = TextDecoderBlock(hidden_dim, num_heads)
    assert decoder_block(
        torch.rand(batch_size, input_seq_len, hidden_dim),
        torch.tensor(
            [
                [1.0] * (i + 1) + [0.0] * (input_seq_len - i - 1)
                for i in range(batch_size)
            ]
        ),
        torch.rand(batch_size, num_node, hidden_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (num_node - i - 1) for i in range(batch_size)]
        ),
        torch.rand(batch_size, prev_action_len, hidden_dim),
        torch.tensor(
            [
                [1.0] * (i + 1) + [0.0] * (prev_action_len - i - 1)
                for i in range(batch_size)
            ]
        ),
    ).size() == (batch_size, input_seq_len, hidden_dim)


@pytest.mark.parametrize(
    "num_embeddings,embedding_dim,num_dec_blocks,dec_block_num_heads,"
    "dec_block_hidden_dim,batch_size,input_seq_len,num_node,prev_action_len",
    [
        (2, 10, 1, 1, 10, 1, 3, 5, 4),
        (2, 20, 1, 2, 20, 3, 5, 10, 8),
        (8, 20, 3, 1, 10, 1, 3, 5, 4),
        (8, 40, 3, 2, 20, 3, 5, 10, 8),
    ],
)
def test_text_decoder(
    num_embeddings,
    embedding_dim,
    num_dec_blocks,
    dec_block_num_heads,
    dec_block_hidden_dim,
    batch_size,
    input_seq_len,
    num_node,
    prev_action_len,
):
    decoder = TextDecoder(
        nn.Embedding(num_embeddings, embedding_dim),
        num_dec_blocks,
        dec_block_num_heads,
        dec_block_hidden_dim,
    )
    assert decoder(
        torch.randint(num_embeddings, (batch_size, input_seq_len)),
        torch.tensor(
            [
                [1.0] * (i + 1) + [0.0] * (input_seq_len - i - 1)
                for i in range(batch_size)
            ]
        ),
        torch.rand(batch_size, num_node, dec_block_hidden_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (num_node - i - 1) for i in range(batch_size)]
        ),
        torch.randint(num_embeddings, (batch_size, prev_action_len)),
        torch.tensor(
            [
                [1.0] * (i + 1) + [0.0] * (prev_action_len - i - 1)
                for i in range(batch_size)
            ]
        ),
    ).size() == (batch_size, input_seq_len, num_embeddings)
