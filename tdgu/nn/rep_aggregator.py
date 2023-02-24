import torch
import torch.nn as nn

from tdgu.nn.utils import masked_softmax


class ContextQueryAttention(nn.Module):
    """Based on Context-Query Attention Layer from QANet, which is in turn
    based on Attention Flow Layer from https://arxiv.org/abs/1611.01603."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        w_C = torch.empty(hidden_dim, 1)
        w_Q = torch.empty(hidden_dim, 1)
        w_CQ = torch.empty(hidden_dim, 1)
        torch.nn.init.xavier_uniform_(w_C)
        torch.nn.init.xavier_uniform_(w_Q)
        torch.nn.init.xavier_uniform_(w_CQ)
        self.w_C = torch.nn.Parameter(w_C)
        self.w_Q = torch.nn.Parameter(w_Q)
        self.w_CQ = torch.nn.Parameter(w_CQ)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(
        self,
        ctx: torch.Tensor,
        query: torch.Tensor,
        ctx_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        ctx: (batch, ctx_seq_len, hidden_dim)
        query: (batch, query_seq_len, hidden_dim)
        ctx_mask: (batch, ctx_seq_len)
        query_mask: (batch, query_seq_len)
        output: (batch, ctx_seq_len, 4 * hidden_dim)
        """
        ctx_seq_len = ctx.size(1)
        query_seq_len = query.size(1)

        # (batch, ctx_seq_len, query_seq_len)
        similarity = self.trilinear_for_attention(ctx, query)
        # (batch, ctx_seq_len, query_seq_len)
        s_ctx = masked_softmax(
            similarity, ctx_mask.unsqueeze(2).expand(-1, -1, query_seq_len), dim=1
        )
        # (batch, ctx_seq_len, query_seq_len)
        s_query = masked_softmax(
            similarity, query_mask.unsqueeze(1).expand(-1, ctx_seq_len, -1), dim=2
        )
        # (batch, ctx_seq_len, hidden_dim)
        P = torch.bmm(s_query, query)
        # (batch, ctx_seq_len, hidden_dim)
        Q = torch.bmm(torch.bmm(s_query, s_ctx.transpose(1, 2)), ctx)

        # (batch, ctx_seq_len, 4 * hidden_dim)
        return torch.cat([ctx, P, ctx * P, ctx * Q], dim=2)

    def trilinear_for_attention(
        self, ctx: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        ctx: (batch, ctx_seq_len, hidden_dim), context C
        query: (batch, query_seq_len, hidden_dim), query Q
        output: (batch, ctx_seq_len, query_seq_len), similarity matrix S
        This is an optimized implementation. The number of multiplications of the
        original equation S_ij = w^T[C_i; Q_j; C_i * Q_j] is
        O(ctx_seq_len * query_seq_len * 3 * hidden_dim)
        = O(ctx_seq_len * query_seq_len * hidden_dim)
        We can reduce this number by splitting the weight matrix w into three parts,
        one for each part of the concatenated vector [C_i; Q_j; C_i * Q_j].
        Specifically,
        S_ij = w^T[C_i; Q_j; C_i * Q_j]
        = w^1C^1_i + ... + w^dC^d_i + w^{d+1}Q^1_j + ... +w^{2d}Q^d_j
          + w^{2d+1}C^1_iQ^1_j + ... + w^{3d}C^d_iQ^d_j
        = w_CC_i + w_QQ_j + w_{C * Q}C_iQ_j
        where d = hidden_dim, and the superscript i denotes the i'th element of a
        vector. The number of multiplications of this formulation is
        O(hidden_dim + hidden_dim + hidden_dim + ctx_seq_len * query_seq_len)
        = O(hidden_dim + ctx_seq_len * query_seq_len)
        """
        ctx_seq_len = ctx.size(1)
        query_seq_len = query.size(1)

        # (batch, ctx_seq_len, query_seq_len)
        res_C = torch.matmul(ctx, self.w_C).expand(-1, -1, query_seq_len)
        # (batch, query_seq_len, ctx_seq_len)
        res_Q = torch.matmul(query, self.w_Q).expand(-1, -1, ctx_seq_len)
        # (batch, ctx_seq_len, query_seq_len)
        res_CQ = torch.matmul(self.w_CQ.squeeze() * ctx, query.transpose(1, 2))

        return res_C + res_Q.transpose(1, 2) + res_CQ + self.bias


class ReprAggregator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cqattn = ContextQueryAttention(hidden_dim)
        self.prj = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        repr1: torch.Tensor,
        repr2: torch.Tensor,
        repr1_mask: torch.Tensor,
        repr2_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        repr1: (batch, repr1_seq_len, hidden_dim)
        repr2: (batch, repr2_seq_len, hidden_dim)
        repr1_mask: (batch, repr1_seq_len)
        repr2_mask: (batch, repr2_seq_len)
        output: (batch, repr1_seq_len, hidden_dim), (batch, repr2_seq_len, hidden_dim)
        """
        return (
            self.prj(self.cqattn(repr1, repr2, repr1_mask, repr2_mask)),
            self.prj(self.cqattn(repr2, repr1, repr2_mask, repr1_mask)),
        )
