import pytest
import torch
import torch.nn.functional as F

from dgu.nn.repr_aggr import ContextQueryAttention, ReprAggregator


@pytest.mark.parametrize(
    "hidden_dim,batch_size,ctx_seq_len,query_seq_len",
    [
        (10, 1, 2, 4),
        (10, 3, 5, 10),
    ],
)
def test_cqattn_trilinear(hidden_dim, batch_size, ctx_seq_len, query_seq_len):
    ra = ContextQueryAttention(hidden_dim)
    batched_ctx = torch.rand(batch_size, ctx_seq_len, hidden_dim)
    batched_query = torch.rand(batch_size, query_seq_len, hidden_dim)
    batched_similarity = ra.trilinear_for_attention(batched_ctx, batched_query)

    # compare the result from the optimized version to the one from the naive version
    combined_w = torch.cat([ra.w_C, ra.w_Q, ra.w_CQ]).squeeze()
    for similarity, ctx, query in zip(batched_similarity, batched_ctx, batched_query):
        for i in range(ctx_seq_len):
            for j in range(query_seq_len):
                naive_s_ij = torch.matmul(
                    combined_w, torch.cat([ctx[i], query[j], ctx[i] * query[j]])
                )
                assert similarity[i, j].isclose(naive_s_ij, atol=1e-6)


@pytest.mark.parametrize(
    "hidden_dim,batch_size,ctx_seq_len,query_seq_len",
    [
        (10, 1, 3, 5),
        (10, 3, 5, 7),
    ],
)
def test_cqattn(hidden_dim, batch_size, ctx_seq_len, query_seq_len):
    # test against non masked version as masked softmax is the weak point
    ra = ContextQueryAttention(hidden_dim)
    ctx = torch.rand(batch_size, ctx_seq_len, hidden_dim)
    query = torch.rand(batch_size, query_seq_len, hidden_dim)
    output = ra(
        ctx,
        query,
        torch.ones(batch_size, ctx_seq_len),
        torch.ones(batch_size, query_seq_len),
    )
    assert output.size() == (batch_size, ctx_seq_len, 4 * hidden_dim)

    # (batch, ctx_seq_len, query_seq_len)
    similarity = ra.trilinear_for_attention(ctx, query)
    # (batch, ctx_seq_len, query_seq_len)
    s_ctx = F.softmax(similarity, dim=1)
    # (batch, ctx_seq_len, query_seq_len)
    s_query = F.softmax(similarity, dim=2)
    # (batch, ctx_seq_len, hidden_dim)
    P = torch.bmm(s_query, query)
    # (batch, ctx_seq_len, hidden_dim)
    Q = torch.bmm(torch.bmm(s_query, s_ctx.transpose(1, 2)), ctx)

    # (batch, ctx_seq_len, 4 * hidden_dim)
    no_mask_output = torch.cat([ctx, P, ctx * P, ctx * Q], dim=2)

    assert output.equal(no_mask_output)


@pytest.mark.parametrize(
    "hidden_dim,batch_size,repr1_seq_len,repr2_seq_len",
    [
        (10, 1, 3, 5),
        (10, 3, 5, 7),
    ],
)
def test_repr_aggr(hidden_dim, batch_size, repr1_seq_len, repr2_seq_len):
    ra = ReprAggregator(hidden_dim)
    repr1 = torch.rand(batch_size, repr1_seq_len, hidden_dim)
    repr2 = torch.rand(batch_size, repr2_seq_len, hidden_dim)
    repr12, repr21 = ra(
        repr1,
        repr2,
        torch.ones(batch_size, repr1_seq_len),
        torch.ones(batch_size, repr2_seq_len),
    )
    assert repr12.size() == (batch_size, repr1_seq_len, hidden_dim)
    assert repr21.size() == (batch_size, repr2_seq_len, hidden_dim)
