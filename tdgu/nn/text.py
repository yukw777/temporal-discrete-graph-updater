import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Dict
from transformers import PreTrainedModel

from tdgu.nn.utils import (
    PositionalEncoder,
    generate_square_subsequent_mask,
    masked_mean,
)
from tdgu.preprocessor import BertPreprocessor


class TextEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        return_pooled_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        word_ids: (batch_size, seq_len) or
            one-hot encoded (batch_size, seq_len, num_word)
        mask: (batch_size, seq_len)
        output:
            {
                "encoded": (batch_size, seq_len, encoder_hidden_dim),
                "pooled_output": (batch_size, pooled_hidden_dim)
                    if return_pooled_output is True
            }
        """

    @abstractmethod
    def get_input_embeddings(self) -> nn.Embedding:
        pass


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


class QANetTextEncoder(TextEncoder):
    def __init__(
        self,
        pretrained_word_embeddings: nn.Embedding,
        num_enc_blocks: int,
        enc_block_num_conv_layers: int,
        enc_block_kernel_size: int,
        enc_block_hidden_dim: int,
        enc_block_num_heads: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.enc_block_hidden_dim = enc_block_hidden_dim
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.hidden_dim = hidden_dim

        self.word_embedding_linear = nn.Linear(
            self.pretrained_word_embeddings.embedding_dim, enc_block_hidden_dim
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

        self.output_linear = nn.Linear(enc_block_hidden_dim, hidden_dim)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        return_pooled_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        word_ids: (batch_size, seq_len) or
            one-hot encoded (batch_size, seq_len, num_word)
        mask: (batch_size, seq_len)
        output:
                "encoded": (batch_size, seq_len, hidden_dim)
                "pooled_output": (batch_size, hidden_dim)
                    if return_pooled_output is True
        """
        if word_ids.size(1) == 0:
            batch_size = word_ids.size(0)
            encoded = torch.empty(
                batch_size, 0, self.enc_block_hidden_dim, device=word_ids.device
            )
        else:
            if word_ids.dim() == 3:
                # word_ids are one-hot encoded
                encoded = word_ids.matmul(self.pretrained_word_embeddings.weight)
            else:
                # word_ids are not one-hot encoded, so use word_embeddings directly
                encoded = self.pretrained_word_embeddings(word_ids)
            # (batch_size, seq_len, word_embedding_dim)
            encoded = self.word_embedding_linear(encoded)
            # (batch_size, seq_len, enc_block_hidden_dim)

            for enc_block in self.enc_blocks:
                encoded = enc_block(encoded, mask)
            # (batch_size, seq_len, enc_block_hidden_dim)
            encoded = self.output_linear(encoded)

        output = {"encoded": encoded}
        if return_pooled_output:
            output["pooled_output"] = masked_mean(encoded, mask)
        return output

    def get_input_embeddings(self) -> nn.Embedding:
        return self.pretrained_word_embeddings


class TextDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim has to be even for positional encoding"
        self.pos_encoder = PositionalEncoder(hidden_dim, 512)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.node_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.prev_action_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.combine_node_prev_action = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        prev_action_hidden: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input: (batch, input_seq_len, hidden_dim)
        input_mask: (batch, input_seq_len)
        node_embeddings: (batch, num_node, hidden_dim)
        node_mask: (batch, num_node)
        prev_action_hidden: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)
        output: (batch, input_seq_len, hidden_dim)
        """
        # calculate attention mask for decoding
        # this is the mask that prevents MultiheadAttention
        # from attending to future values
        input_seq_len = input.size(1)
        attn_mask = generate_square_subsequent_mask(input_seq_len, device=input.device)
        # (input_seq_len, input_seq_len)

        # add the positional encodings
        pos_encoded_input = input + self.pos_encoder(
            torch.arange(input_seq_len, device=input.device)
        )
        # (batch, input_seq_len, hidden_dim)

        # self attention layer
        input_residual = pos_encoded_input
        input_attn, _ = self.self_attn(
            pos_encoded_input,
            pos_encoded_input,
            pos_encoded_input,
            key_padding_mask=input_mask.logical_not(),
            attn_mask=attn_mask,
        )
        input_attn += input_residual
        # (batch, input_seq_len, hidden_dim)

        # apply layer norm to the input self attention output to calculate the query
        query = self.self_attn_layer_norm(input_attn)
        # (batch, input_seq_len, hidden_dim)

        # multihead attention for the nodes
        # If there are no nodes in the graph, MultiheadAttention returns nan due to
        # https://github.com/pytorch/pytorch/issues/41508
        # In order to get around it, set the mask of an empty graph to all True.
        # This is OK, b/c node embeddings for an empty graph are all 0.
        filled_node_mask = node_mask.masked_fill(
            node_mask.logical_not().all(dim=1).unsqueeze(1), True
        )
        node_attn, _ = self.node_attn(
            query,
            node_embeddings,
            node_embeddings,
            key_padding_mask=filled_node_mask.logical_not(),
        )
        # (batch, input_seq_len, hidden_dim)

        # multihead attention for the previous action
        prev_action_attn, _ = self.prev_action_attn(
            query,
            prev_action_hidden,
            prev_action_hidden,
            key_padding_mask=prev_action_mask.logical_not(),
        )
        # (batch, input_seq_len, hidden_dim)

        # combine self attention for the previous action and nodes with
        # input self attention
        combined_node_prev_attn = (
            self.combine_node_prev_action(
                torch.cat([prev_action_attn, node_attn], dim=-1)
            )
            + input_attn
        )
        # (batch, input_seq_len, hidden_dim)

        # linear layer
        output = self.linear_layer_norm(combined_node_prev_attn)
        output = self.linear_layers(output)
        output += combined_node_prev_attn
        # (batch, input_seq_len, hidden_dim)

        return output


class TextDecoder(nn.Module):
    def __init__(
        self,
        pretrained_word_embeddings: nn.Embedding,
        num_dec_blocks: int,
        dec_block_num_heads: int,
        dec_block_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.word_embedding_linear = nn.Linear(
            self.pretrained_word_embeddings.embedding_dim, dec_block_hidden_dim
        )
        self.dec_blocks = nn.ModuleList(
            TextDecoderBlock(dec_block_hidden_dim, dec_block_num_heads)
            for _ in range(num_dec_blocks)
        )
        self.output_linear = nn.Linear(
            dec_block_hidden_dim, self.pretrained_word_embeddings.num_embeddings
        )

    def forward(
        self,
        input_word_ids: torch.Tensor,
        input_mask: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        prev_action_word_ids: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_word_ids: (batch, input_seq_len)
        input_mask: (batch, input_seq_len)
        node_embeddings: (batch, num_node, hidden_dim)
        node_mask: (batch, num_node)
        prev_action_word_ids: (batch, prev_action_len)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, input_seq_len, vocab_size)
        """
        output = self.word_embedding_linear(
            self.pretrained_word_embeddings(input_word_ids)
        )
        # (batch, input_seq_len, hidden_dim)
        prev_action_hidden = self.word_embedding_linear(
            self.pretrained_word_embeddings(prev_action_word_ids)
        )
        # (batch, prev_action_len, hidden_dim)
        for dec_block in self.dec_blocks:
            output = dec_block(
                output,
                input_mask,
                node_embeddings,
                node_mask,
                prev_action_hidden,
                prev_action_mask,
            )
        # (batch_size, input_seq_len, hidden_dim)

        return self.output_linear(output)


class HuggingFaceTextEncoder(TextEncoder):
    model: PreTrainedModel

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()  # type: ignore


class BertTextEncoder(HuggingFaceTextEncoder):
    def __init__(self, bert_like_model: PreTrainedModel, hidden_dim: int) -> None:
        super().__init__()
        self.model = bert_like_model
        self.hidden_dim = hidden_dim

        self.output_linear = nn.Linear(self.model.config.hidden_size, hidden_dim)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        return_pooled_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        word_ids: (batch_size, seq_len) or
            one-hot encoded (batch_size, seq_len, num_word)
        mask: (batch_size, seq_len)
        output:
            {
                "encoded": (batch_size, seq_len, hidden_dim),
                "pooled_output": (batch_size, hidden_dim)
                    if return_pooled_output is True
            }
        """
        if word_ids.dim() == 2:
            model_output = self.model(input_ids=word_ids, attention_mask=mask)
        else:
            model_output = self.model(
                inputs_embeds=word_ids.matmul(
                    self.model.get_input_embeddings().weight  # type: ignore
                ),
                attention_mask=mask,
            )
        output = {"encoded": self.output_linear(model_output.last_hidden_state)}
        if return_pooled_output:
            output["pooled_output"] = self.output_linear(model_output.pooler_output)
        return output


class DistilBertTextEncoder(BertTextEncoder):
    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        return_pooled_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        word_ids: (batch_size, seq_len) or
            one-hot encoded (batch_size, seq_len, num_word)
        mask: (batch_size, seq_len)
        output:
            {
                "encoded": (batch_size, seq_len, hidden_dim),
                "pooled_output": (batch_size, hidden_dim)
                    if return_pooled_output is True
            }
        """
        if word_ids.dim() == 2:
            model_output = self.model(input_ids=word_ids, attention_mask=mask)
        else:
            model_output = self.model(
                inputs_embeds=word_ids.matmul(
                    self.model.get_input_embeddings().weight  # type: ignore
                ),
                attention_mask=mask,
            )
        encoded = self.output_linear(model_output.last_hidden_state)
        output = {"encoded": encoded}
        if return_pooled_output:
            output["pooled_output"] = encoded[:, 0]
        return output


HF_TEXT_ENCODER_INIT_MAP = {
    "bert": (BertPreprocessor, BertTextEncoder),
    "distil-bert": (BertPreprocessor, DistilBertTextEncoder),
}
