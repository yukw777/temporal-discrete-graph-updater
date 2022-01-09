import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List, Dict

from dgu.nn.utils import masked_mean, PositionalEncoder
from dgu.constants import EVENT_TYPES


class EventTypeHead(nn.Module):
    def __init__(
        self, graph_event_embedding_dim: int, hidden_dim: int, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(graph_event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, len(EVENT_TYPES)),
        )
        self.autoregressive_linear = nn.Sequential(
            nn.Linear(len(EVENT_TYPES), graph_event_embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, graph_event_embeddings: torch.Tensor) -> torch.Tensor:
        """
        graph_event_embeddings: (batch, graph_event_embedding_dim)

        output: event_type_logits: (batch, num_event_type)
        """
        # logits
        return self.linear(graph_event_embeddings)
        # (batch, num_event_type)

    def get_autoregressive_embedding(
        self,
        graph_event_embeddings: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        graph_event_embeddings: (batch, graph_event_embedding_dim)
        event_type_ids: (batch)
        event_mask: (batch)

        output: autoregressive_embedding, (batch, graph_event_embedding_dim)
        """
        # autoregressive embedding
        # get the one hot encoding of event
        one_hot_event_type = F.one_hot(
            event_type_ids, num_classes=len(EVENT_TYPES)
        ).float()
        # (batch, num_event_type)
        # pass it through a linear layer
        encoded_event_type = self.autoregressive_linear(one_hot_event_type)
        # (batch, graph_event_embedding_dim)
        # add it to graph_event_embeddings to calculate the autoregressive embedding
        return graph_event_embeddings + encoded_event_type * event_mask.unsqueeze(-1)
        # (batch, graph_event_embedding_dim)


class EventNodeHead(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        autoregressive_embedding_dim: int,
        hidden_dim: int,
        key_query_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.key_query_dim = key_query_dim
        self.key_linear = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(autoregressive_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.autoregressive_linear = nn.Linear(
            key_query_dim, autoregressive_embedding_dim
        )

    def forward(
        self, autoregressive_embedding: torch.Tensor, node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)
        node_embeddings: (batch, num_node, node_embedding_dim)

        output:
            node_logits: (batch, num_node)
            key: (batch, num_node, key_query_dim)
        """
        if node_embeddings.size(1) == 0:
            # if there are no nodes, just return
            batch = node_embeddings.size(0)
            return torch.empty(batch, 0, device=node_embeddings.device), torch.empty(
                batch, 0, self.key_query_dim
            )

        # calculate the key from node_embeddings
        key = self.key_linear(node_embeddings)
        # (batch, num_node, key_query_dim)

        # calculate the query from autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        return torch.matmul(key, query.unsqueeze(-1)).squeeze(-1), key
        # node_logits: (batch, num_node)
        # key: (batch, num_node, key_query_dim)

    def update_autoregressive_embedding(
        self,
        autoregressive_embedding: torch.Tensor,
        node_ids: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        event_node_mask: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)
        node_ids: (batch)
        node_embeddings: (batch, num_node, node_embedding_dim)
        node_mask: (batch, num_node)
        event_node_mask: (batch)
        key: (batch, num_node, key_query_dim)

        output: updated autoregressive embedding
            (batch, autoregressive_embedding_dim)
        """
        if node_embeddings.size(1) == 0:
            # if there are no nodes, just return without updating
            return autoregressive_embedding
        # get the one hot encoding of the selected nodes
        one_hot_selected_node = (
            F.one_hot(node_ids, num_classes=node_embeddings.size(1)).float() * node_mask
        )
        # (batch, num_node)
        # multiply by the key
        selected_node_embeddings = torch.bmm(
            one_hot_selected_node.unsqueeze(1), key
        ).squeeze(1)
        # (batch, key_query_dim)
        # pass it through a linear layer
        selected_node_embeddings = self.autoregressive_linear(
            selected_node_embeddings
        ) * node_mask.any(dim=1).unsqueeze(
            -1
        )  # mask out pad nodes
        # (batch, hidden_dim)
        # add it to the autoregressive embedding
        # NOTE: make sure not to do an in-place += here as it messes with gradients
        return (
            autoregressive_embedding
            + selected_node_embeddings
            * event_node_mask.unsqueeze(-1)  # mask out nodes that shouldn't be selected
        )
        # (batch, hidden_dim)


class EventStaticLabelHead(nn.Module):
    def __init__(
        self,
        autoregressive_embedding_dim: int,
        hidden_dim: int,
        num_label: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(autoregressive_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_label),
        )

    def forward(self, autoregressive_embedding: torch.Tensor) -> torch.Tensor:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)

        output:
            label_logits: (batch, num_label), logits for nodes first, then edges
        """
        return self.linear(autoregressive_embedding)


class RNNGraphEventDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        aggr_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim + 4 * aggr_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        input_event_embedding: torch.Tensor,
        aggr_obs_graph: torch.Tensor,
        obs_mask: torch.Tensor,
        aggr_prev_action_graph: torch.Tensor,
        prev_action_mask: torch.Tensor,
        aggr_graph_obs: torch.Tensor,
        aggr_graph_prev_action: torch.Tensor,
        node_mask: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input:
            input_event_embedding: (batch, input_dim)
            aggr_obs_graph: (batch, obs_len, aggr_dim)
            obs_mask: (batch, obs_len)
            aggr_prev_action_graph: (batch, prev_action_len, aggr_dim)
            prev_action_mask: (batch, prev_action_len)
            aggr_graph_obs: (batch, num_node, aggr_dim)
            aggr_graph_prev_action: (batch, num_node, aggr_dim)
            node_mask: (batch, num_node)
            hidden: (batch, hidden_dim)

        output: (batch, hidden_dim)
        """
        mean_aggr_obs_graph = masked_mean(aggr_obs_graph, obs_mask)
        # (batch, aggr_dim)
        mean_aggr_graph_obs = masked_mean(aggr_graph_obs, node_mask)
        # (batch, aggr_dim)
        mean_aggr_prev_action_graph = masked_mean(
            aggr_prev_action_graph, prev_action_mask
        )
        # (batch, aggr_dim)
        mean_aggr_graph_prev_action = masked_mean(aggr_graph_prev_action, node_mask)
        # (batch, aggr_dim)
        gru_input = self.linear(
            torch.cat(
                [
                    input_event_embedding,
                    mean_aggr_obs_graph,
                    mean_aggr_graph_obs,
                    mean_aggr_prev_action_graph,
                    mean_aggr_graph_prev_action,
                ],
                dim=1,
            )
        )
        # (batch, hidden_dim)
        return self.gru_cell(gru_input, hidden)
        # (batch, hidden_dim)


class TransformerGraphEventDecoderBlock(nn.Module):
    def __init__(
        self, aggr_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.self_attn_dropout = nn.Dropout(p=dropout)

        self.cross_multi_head_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.aggr_obs_graph_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            kdim=aggr_dim,
            vdim=aggr_dim,
            dropout=dropout,
        )
        self.aggr_graph_obs_attn_dropout = nn.Dropout(p=dropout)
        self.aggr_prev_action_graph_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            kdim=aggr_dim,
            vdim=aggr_dim,
            dropout=dropout,
        )
        self.aggr_prev_action_graph_attn_dropout = nn.Dropout(p=dropout)
        self.aggr_graph_obs_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            kdim=aggr_dim,
            vdim=aggr_dim,
            dropout=dropout,
        )
        self.aggr_graph_prev_action_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            kdim=aggr_dim,
            vdim=aggr_dim,
            dropout=dropout,
        )

        self.combine_attn = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.linear_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor,
        aggr_obs_graph: torch.Tensor,
        obs_mask: torch.Tensor,
        aggr_prev_action_graph: torch.Tensor,
        prev_action_mask: torch.Tensor,
        aggr_graph_obs: torch.Tensor,
        aggr_graph_prev_action: torch.Tensor,
        node_mask: torch.Tensor,
        prev_input_seq: torch.Tensor,
        prev_input_seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        input: (batch, hidden_dim)
        input_mask: (batch)
        aggr_obs_graph: (batch, obs_len, aggr_dim)
        obs_mask: (batch, obs_len)
        aggr_prev_action_graph: (batch, prev_action_len, aggr_dim)
        prev_action_mask: (batch, prev_action_len)
        aggr_graph_obs: (batch, num_node, aggr_dim)
        aggr_graph_prev_action: (batch, num_node, aggr_dim)
        node_mask: (batch, num_node)
        prev_input_seq: (batch, prev_input_seq_len, hidden_dim)
        prev_input_seq_mask: (batch, prev_input_seq_len)

        output: {
            "output": (batch, hidden_dim),
            "self_attn_weights": (batch, 1, input_seq_len),
            "obs_graph_attn_weights": (batch, 1, obs_len),
            "prev_action_graph_attn_weights": (batch, 1, prev_action_len),
            "graph_obs_attn_weights": (batch, 1, num_node),
            "graph_prev_action_attn_weights": (batch, 1, num_node),
        }
        """
        concat_input = torch.cat([prev_input_seq, input.unsqueeze(1)], dim=1)
        # (batch, input_seq_len, hidden_dim)
        concat_input_mask = torch.cat(
            [prev_input_seq_mask, input_mask.unsqueeze(1)], dim=1
        )
        # (batch, input_seq_len)

        # self attention
        normalized_concat_input = self.self_attn_layer_norm(concat_input)
        # (batch, input_seq_len, hidden_dim)
        self_attn, self_attn_weights = self.self_attn(
            # we only need to calculate the self attention for the last element
            # in the input sequence
            normalized_concat_input[:, -1:, :],
            normalized_concat_input,
            normalized_concat_input,
            key_padding_mask=~concat_input_mask,
        )
        # self_attn: (batch, 1, hidden_dim)
        # self_attn_weights: (batch, 1, input_seq_len)
        self_attn = self.self_attn_dropout(self_attn)
        # (batch, 1, hidden_dim)
        self_attn += concat_input[:, -1:, :]
        # (batch, 1, hidden_dim)

        # cross multi-head attention
        # apply layer norm to the self attention output.
        # this is the query for the cross multi-head attentions.
        query = self.cross_multi_head_attn_layer_norm(self_attn)
        # (batch, 1, hidden_dim)

        # calculate multi-headed attention using the aggregated observation and
        # graph representation
        obs_graph_attn, obs_graph_attn_weights = self.aggr_obs_graph_attn(
            query,
            aggr_obs_graph,
            aggr_obs_graph,
            key_padding_mask=~obs_mask,
        )
        # obs_graph_attn: (batch, 1, hidden_dim)
        # obs_graph_attn_weights: (batch, 1, obs_len)
        obs_graph_attn = self.aggr_graph_obs_attn_dropout(obs_graph_attn)
        # (batch, 1, hidden_dim)
        obs_graph_attn += self_attn
        # (batch, 1, hidden_dim)

        # calculate multi-headed attention using the aggregated previous action and
        # graph representation
        (
            prev_action_graph_attn,
            prev_action_graph_attn_weights,
        ) = self.aggr_prev_action_graph_attn(
            query,
            aggr_prev_action_graph,
            aggr_prev_action_graph,
            key_padding_mask=~prev_action_mask,
        )
        # prev_action_graph_attn: (batch, 1, hidden_dim)
        # prev_action_graph_attn_weights: (batch, 1, prev_action_len)
        prev_action_graph_attn = self.aggr_prev_action_graph_attn_dropout(
            prev_action_graph_attn
        )
        # (batch, 1, hidden_dim)
        prev_action_graph_attn += self_attn
        # (batch, 1, hidden_dim)

        # calculate multi-headed attention using the aggregated graph and
        # observation representation
        attn_node_mask = ~node_mask
        # (batch, num_node)

        # due to https://github.com/pytorch/pytorch/issues/41508
        # nn.MultiheadAttention returns nan's if a whole row is masked in the key,
        # i.e. the sub-graph doesn't have any nodes. So, we don't calculate
        # multihead attention for those sub-graphs using a mask.
        subgraphs_with_node_mask = node_mask.any(dim=1)
        # (batch)
        graph_obs_attn = torch.zeros_like(query)
        # (batch, 1, hidden_dim)
        graph_obs_attn_weights = torch.zeros(
            node_mask.size(0), 1, node_mask.size(1), device=node_mask.device
        )
        # (batch, 1, num_node)
        (
            subgraph_with_node_obs_attn,
            # (num_subgraph_with_node, 1, hidden_dim)
            subgraph_with_node_obs_attn_weights,
            # (num_subgraph_with_node, 1, num_node)
        ) = self.aggr_graph_obs_attn(
            query[subgraphs_with_node_mask],
            aggr_graph_obs[subgraphs_with_node_mask],
            aggr_graph_obs[subgraphs_with_node_mask],
            key_padding_mask=attn_node_mask[subgraphs_with_node_mask],
        )
        graph_obs_attn[subgraphs_with_node_mask] = subgraph_with_node_obs_attn
        # (batch, 1, hidden_dim)
        graph_obs_attn_weights[
            subgraphs_with_node_mask
        ] = subgraph_with_node_obs_attn_weights
        # (batch, 1, num_node)
        graph_obs_attn += self_attn
        # (batch, 1, hidden_dim)

        # calculate multi-headed attention using the aggregated graph and
        # previous action
        graph_prev_action_attn = torch.zeros_like(query)
        # (batch, 1, hidden_dim)
        graph_prev_action_attn_weights = torch.zeros(
            node_mask.size(0), 1, node_mask.size(1), device=node_mask.device
        )
        # (batch, 1, num_node)
        (
            subgraph_with_node_prev_action_attn,
            # (num_subgraph_with_node, 1, hidden_dim)
            subgraph_with_node_prev_action_attn_weights,
            # (num_subgraph_with_node, 1, num_node)
        ) = self.aggr_graph_prev_action_attn(
            query[subgraphs_with_node_mask],
            aggr_graph_prev_action[subgraphs_with_node_mask],
            aggr_graph_prev_action[subgraphs_with_node_mask],
            key_padding_mask=attn_node_mask[subgraphs_with_node_mask],
        )
        graph_prev_action_attn[
            subgraphs_with_node_mask
        ] = subgraph_with_node_prev_action_attn
        # (batch, 1, hidden_dim)
        graph_prev_action_attn_weights[
            subgraphs_with_node_mask
        ] = subgraph_with_node_prev_action_attn_weights
        # (batch, 1, num_node)
        graph_prev_action_attn += self_attn
        # (batch, 1, hidden_dim)

        # combine multi-headed attentions
        combined_attn = self.combine_attn(
            torch.cat(
                [
                    obs_graph_attn,
                    prev_action_graph_attn,
                    graph_obs_attn,
                    graph_prev_action_attn,
                ],
                dim=-1,
            )
        )
        # (batch, 1, hidden_dim)

        # linear layer
        output = self.linear_layer_norm(combined_attn)
        # (batch, 1, hidden_dim)
        output = self.linear_layers(output)
        # (batch, 1, hidden_dim)
        output = self.linear_dropout(output)
        # (batch, 1, hidden_dim)
        output += combined_attn
        # (batch, 1, hidden_dim)
        output = output.squeeze(1)
        # (batch, hidden_dim)
        return {
            "output": output,
            "self_attn_weights": self_attn_weights,
            "obs_graph_attn_weights": obs_graph_attn_weights,
            "prev_action_graph_attn_weights": prev_action_graph_attn_weights,
            "graph_obs_attn_weights": graph_obs_attn_weights,
            "graph_prev_action_attn_weights": graph_prev_action_attn_weights,
        }


class TransformerGraphEventDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        aggr_dim: int,
        num_dec_blocks: int,
        dec_block_num_heads: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoder(hidden_dim, 1024)
        self.dec_blocks = nn.ModuleList(
            TransformerGraphEventDecoderBlock(
                aggr_dim, hidden_dim, dec_block_num_heads, dropout=dropout
            )
            for _ in range(num_dec_blocks)
        )
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_event_embedding: torch.Tensor,
        event_mask: torch.Tensor,
        aggr_obs_graph: torch.Tensor,
        obs_mask: torch.Tensor,
        aggr_prev_action_graph: torch.Tensor,
        prev_action_mask: torch.Tensor,
        aggr_graph_obs: torch.Tensor,
        aggr_graph_prev_action: torch.Tensor,
        node_mask: torch.Tensor,
        prev_input_event_emb_seq: Optional[torch.Tensor] = None,
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        input_event_embedding: (batch, input_dim)
        event_mask: (batch)
        aggr_obs_graph: (batch, obs_len, aggr_dim)
        obs_mask: (batch, obs_len)
        aggr_prev_action_graph: (batch, prev_action_len, aggr_dim)
        prev_action_mask: (batch, prev_action_len)
        aggr_graph_obs: (batch, num_node, aggr_dim)
        aggr_graph_prev_action: (batch, num_node, aggr_dim)
        node_mask: (batch, num_node)
        prev_input_event_emb_seq: (num_block, batch, prev_input_seq_len, hidden_dim)
        prev_input_event_emb_seq_mask: (num_block, batch, prev_input_seq_len)

        output: (
        {
            "output": (batch, hidden_dim),
            "updated_prev_input_event_emb_seq":
                (num_block, batch, input_seq_len, hidden_dim),
            "updated_prev_input_event_emb_seq_mask": (batch, input_seq_len),
        },
        {
            "self_attn_weights": len([(batch, 1, input_seq_len), ...]) == num_block,
            "obs_graph_attn_weights": len([(batch, 1, obs_len), ...]) == num_block,
            "prev_action_graph_attn_weights":
                len([(batch, 1, prev_action_len), ...]) == num_block,
            "graph_obs_attn_weights": len([(batch, 1, num_node), ...]) == num_block,
            "graph_prev_action_attn_weights":
                len([(batch, 1, num_node), ...]) == num_block,
        },
        )
        """
        # linearly resize the input input event embedding and
        # add the positional encodings to it
        pos_encoded_input = self.input_linear(input_event_embedding) + self.pos_encoder(
            torch.tensor(
                prev_input_event_emb_seq.size(2), device=input_event_embedding.device
            )
            if prev_input_event_emb_seq is not None
            else torch.tensor(0, device=input_event_embedding.device),
        )
        # (batch, hidden_dim)

        batch = input_event_embedding.size(0)
        if prev_input_event_emb_seq is None:
            prev_input_event_emb_seq = torch.empty(
                len(self.dec_blocks),
                batch,
                0,
                self.hidden_dim,
                dtype=torch.bool,
                device=input_event_embedding.device,
            )
            # (num_block, batch, prev_input_seq_len, hidden_dim)
        if prev_input_event_emb_seq_mask is None:
            prev_input_event_emb_seq_mask = torch.empty(
                batch, 0, dtype=torch.bool, device=input_event_embedding.device
            )
            # (batch, prev_input_seq_len)

        # keep track of inputs for each block per step
        output = pos_encoded_input
        # (batch, hidden_dim)
        block_inputs: List[torch.Tensor] = []
        self_attn_weights: List[torch.Tensor] = []
        obs_graph_attn_weights: List[torch.Tensor] = []
        prev_action_graph_attn_weights: List[torch.Tensor] = []
        graph_obs_attn_weights: List[torch.Tensor] = []
        graph_prev_action_attn_weights: List[torch.Tensor] = []
        for i, dec_block in enumerate(self.dec_blocks):
            block_inputs.append(output)
            dec_block_results = dec_block(
                output,
                event_mask,
                aggr_obs_graph,
                obs_mask,
                aggr_prev_action_graph,
                prev_action_mask,
                aggr_graph_obs,
                aggr_graph_prev_action,
                node_mask,
                prev_input_seq=prev_input_event_emb_seq[i],
                prev_input_seq_mask=prev_input_event_emb_seq_mask,
            )
            output = dec_block_results["output"]
            # (batch, hidden_dim)
            self_attn_weights.append(dec_block_results["self_attn_weights"])
            obs_graph_attn_weights.append(dec_block_results["obs_graph_attn_weights"])
            prev_action_graph_attn_weights.append(
                dec_block_results["prev_action_graph_attn_weights"]
            )
            graph_obs_attn_weights.append(dec_block_results["graph_obs_attn_weights"])
            graph_prev_action_attn_weights.append(
                dec_block_results["graph_prev_action_attn_weights"]
            )
        output = self.final_layer_norm(output)
        # (batch, hidden_dim)
        stacked_block_inputs = torch.stack(block_inputs)
        # (num_block, batch, hidden_dim)
        updated_prev_input_event_emb_seq = torch.cat(
            [prev_input_event_emb_seq, stacked_block_inputs.unsqueeze(2)], dim=2
        )
        # (num_block, batch, input_seq_len, hidden_dim)
        updated_prev_input_event_emb_seq_mask = torch.cat(
            [prev_input_event_emb_seq_mask, event_mask.unsqueeze(-1)], dim=-1
        )
        # (batch, input_seq_len)

        return {
            "output": output,
            "updated_prev_input_event_emb_seq": updated_prev_input_event_emb_seq,
            "updated_prev_input_event_emb_seq_mask": (
                updated_prev_input_event_emb_seq_mask
            ),
        }, {
            "self_attn_weights": self_attn_weights,
            "obs_graph_attn_weights": obs_graph_attn_weights,
            "prev_action_graph_attn_weights": prev_action_graph_attn_weights,
            "graph_obs_attn_weights": graph_obs_attn_weights,
            "graph_prev_action_attn_weights": graph_prev_action_attn_weights,
        }
