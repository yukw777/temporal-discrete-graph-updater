from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from tdgu.constants import EVENT_TYPE_ID_MAP
from tdgu.nn.utils import (
    PositionalEncoder,
    calculate_node_id_offsets,
    compute_masks_from_event_type_ids,
    find_indices,
    generate_square_subsequent_mask,
    get_edge_index_co_occurrence_matrix,
    index_edge_attr,
    load_fasttext,
    masked_gumbel_softmax,
    masked_log_softmax,
    masked_mean,
    masked_softmax,
    pad_batch_seq_of_seq,
    shift_tokens_right,
    update_batched_graph,
    update_edge_index,
    update_node_features,
)
from tdgu.preprocessor import PAD, UNK


@pytest.mark.parametrize(
    "batched_input,batched_mask,expected",
    [
        (torch.rand(3, 0, 8), torch.ones(3, 0), torch.zeros(3, 8)),
        (torch.rand(3, 0, 8), torch.zeros(3, 0).bool(), torch.zeros(3, 8)),
        (torch.rand(3, 1, 8), torch.zeros(3, 1), torch.zeros(3, 8)),
        (
            torch.tensor(
                [
                    [
                        [1, 2, 300],
                        [300, 100, 200],
                        [3, 4, 100],
                    ],
                    [
                        [300, 100, 200],
                        [6, 2, 300],
                        [10, 4, 100],
                    ],
                ]
            ).float(),
            torch.tensor(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ).float(),
            torch.tensor(
                [
                    [2, 3, 200],
                    [8, 3, 200],
                ]
            ).float(),
        ),
    ],
)
def test_masked_mean(batched_input, batched_mask, expected):
    assert masked_mean(batched_input, batched_mask).equal(expected)


@pytest.mark.parametrize(
    "batched_input,batched_mask",
    [
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float(),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor(
                [[True, True, False], [False, True, True], [True, True, True]]
            ),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3).bool(),
        ),
    ],
)
def test_masked_softmax(batched_input, batched_mask):
    batched_output = masked_softmax(batched_input, batched_mask, dim=1)
    # ensure that softmax outputs add up to 1 if at least one item is not maksed out
    # 0 if all the items are masked out.
    assert batched_output.sum(dim=1).equal(
        torch.any(batched_mask.float() == 1, dim=1).float()
    )
    # multiplying the output by the mask shouldn't change the values now.
    assert (batched_output * batched_mask).equal(batched_output)


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("tau", [0.1, 0.5, 1])
@pytest.mark.parametrize(
    "batched_input,batched_mask",
    [
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float(),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor(
                [[True, True, False], [False, True, True], [True, True, True]]
            ),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3).bool(),
        ),
    ],
)
def test_masked_gumbel_softmax(batched_input, batched_mask, tau, hard):
    batched_output = masked_gumbel_softmax(
        batched_input, batched_mask, tau=tau, hard=hard
    )
    # ensure that softmax outputs add up to 1 if at least one item is not maksed out
    # 0 if all the items are masked out.
    assert (
        batched_output.sum(dim=1)
        .isclose(torch.any(batched_mask.float() == 1, dim=1).float())
        .all()
    )
    # multiplying the output by the mask shouldn't change the values now.
    assert (batched_output * batched_mask).equal(batched_output)


@pytest.mark.parametrize(
    "batched_input,batched_mask",
    [
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float(),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.tensor(
                [[True, True, False], [False, True, True], [True, True, True]]
            ),
        ),
        (
            torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float(),
            torch.zeros(3, 3).bool(),
        ),
    ],
)
def test_masked_log_softmax(batched_input, batched_mask):
    batched_output = masked_log_softmax(batched_input, batched_mask, dim=1)
    # ensure that log softmax outputs add up to 0 if at least one item is not maksed out
    # 0 if all the items are masked out.
    assert (
        batched_output.exp()
        .sum(dim=1)
        .isclose(torch.any(batched_mask.float() == 1, dim=1).float())
        .all()
    )
    # multiplying the output by the mask shouldn't change the values now.
    assert (batched_output.exp() * batched_mask).isclose(batched_output.exp()).all()


@pytest.mark.parametrize(
    "event_type_ids,expected_event_mask,expected_src_mask,expected_dst_mask,"
    "expected_label_mask",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ]
                ]
            ),
            torch.tensor([[True, True, False]]),
            torch.zeros(1, 3).bool(),
            torch.zeros(1, 3).bool(),
            torch.zeros(1, 3).bool(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ]
                ]
            ),
            torch.ones(1, 2).bool(),
            torch.tensor([[False, True]]),
            torch.zeros(1, 2).bool(),
            torch.tensor([[True, False]]),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.ones(1, 2).bool(),
            torch.ones(1, 2).bool(),
            torch.ones(1, 2).bool(),
            torch.tensor([[True, False]]),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ]
                ]
            ),
            torch.ones(1, 5).bool(),
            torch.tensor([[False, False, True, True, True]]),
            torch.tensor([[False, False, True, True, False]]),
            torch.tensor([[False, True, True, False, False]]),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.ones(2, 5).bool(),
            torch.tensor(
                [[False, False, True, True, True], [False, True, True, True, False]]
            ),
            torch.tensor(
                [[False, False, True, True, False], [False, True, True, False, False]]
            ),
            torch.tensor(
                [[False, True, True, False, False], [True, True, False, False, False]]
            ),
        ),
    ],
)
def test_compute_masks_from_event_type_ids(
    event_type_ids,
    expected_event_mask,
    expected_src_mask,
    expected_dst_mask,
    expected_label_mask,
):
    masks = compute_masks_from_event_type_ids(event_type_ids)
    assert masks["event_mask"].equal(expected_event_mask)
    assert masks["src_mask"].equal(expected_src_mask)
    assert masks["dst_mask"].equal(expected_dst_mask)
    assert masks["label_mask"].equal(expected_label_mask)


def test_load_fasttext(tmpdir):
    serialized_path = Path(tmpdir / "word-emb.pt")
    assert not serialized_path.exists()
    vocab = {
        token: i for i, token in enumerate([PAD, UNK, "my", "name", "is", "peter"])
    }
    emb = load_fasttext(
        "tests/data/test-fasttext.vec", serialized_path, vocab, vocab[PAD]
    )
    word_ids = torch.tensor(
        [
            [
                vocab.get(w, vocab[UNK])
                for w in ["hi", "there", "what's", "your", "name"]
            ],
            [vocab.get(w, vocab[UNK]) for w in ["my", "name", "is", "peter", PAD]],
        ]
    )
    embedded = emb(word_ids)
    # OOVs
    assert embedded[0, :4].equal(
        emb(torch.tensor(vocab[UNK])).unsqueeze(0).expand(4, -1)
    )
    # name
    assert embedded[0, 4].equal(emb(torch.tensor(3)))
    # my name is peter
    assert embedded[1, :4].equal(emb(torch.tensor([2, 3, 4, 5])))
    # pad, should be zero
    assert embedded[1, 4].equal(torch.zeros(300))

    with open(serialized_path, "rb") as f:
        assert emb.weight.equal(torch.load(f).weight)


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        (
            torch.tensor([], dtype=torch.long),
            torch.tensor([1, 2]),
            torch.tensor([], dtype=torch.long),
        ),
        (
            torch.tensor([0, 1, 2, 5, 7], dtype=torch.long),
            torch.tensor([1, 7]),
            torch.tensor([1, 4], dtype=torch.long),
        ),
        (
            torch.tensor([3, 2, 5, 6, 8], dtype=torch.long),
            torch.tensor([6, 2]),
            torch.tensor([3, 1], dtype=torch.long),
        ),
    ],
)
def test_find_indices(haystack, needle, expected):
    assert find_indices(haystack, needle).equal(expected)


@pytest.mark.parametrize(
    "batch_seq_of_seq,max_len_outer,max_len_inner,outer_padding_value,"
    "inner_padding_value,expected",
    [
        (
            [[[0, 1], [1, 2]], [[3, 4, 5], [6, 7, 8, 9], [10, 11]]],
            3,
            4,
            0,
            0,
            torch.tensor(
                [
                    [[0, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]],
                    [[3, 4, 5, 0], [6, 7, 8, 9], [10, 11, 0, 0]],
                ]
            ),
        ),
        (
            [[[0, 1], [1, 2]], [[3, 4, 5], [6, 7, 8, 9], [10, 11]]],
            3,
            4,
            -1,
            -2,
            torch.tensor(
                [
                    [[0, 1, -2, -2], [1, 2, -2, -2], [-1, -1, -1, -1]],
                    [[3, 4, 5, -2], [6, 7, 8, 9], [10, 11, -2, -2]],
                ]
            ),
        ),
        (
            [
                [[(0, 1)], [(2, 3), (4, 5)]],
                [
                    [(6, 7), (8, 9)],
                    [(10, 11), (12, 13), (13, 14)],
                    [(15, 16), (0, 1), (0, 2), (0, 3)],
                ],
            ],
            3,
            4,
            (0, 0),
            0,
            torch.tensor(
                [
                    [
                        [[0, 1], [0, 0], [0, 0], [0, 0]],
                        [[2, 3], [4, 5], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0], [0, 0]],
                    ],
                    [
                        [[6, 7], [8, 9], [0, 0], [0, 0]],
                        [[10, 11], [12, 13], [13, 14], [0, 0]],
                        [[15, 16], [0, 1], [0, 2], [0, 3]],
                    ],
                ]
            ),
        ),
        (
            [
                [[(0, 1)], [(2, 3), (4, 5)]],
                [
                    [(6, 7), (8, 9)],
                    [(10, 11), (12, 13), (13, 14)],
                    [(15, 16), (0, 1), (0, 2), (0, 3)],
                ],
            ],
            3,
            4,
            (-1, -1),
            -2,
            torch.tensor(
                [
                    [
                        [[0, 1], [-2, -2], [-2, -2], [-2, -2]],
                        [[2, 3], [4, 5], [-2, -2], [-2, -2]],
                        [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                    ],
                    [
                        [[6, 7], [8, 9], [-2, -2], [-2, -2]],
                        [[10, 11], [12, 13], [13, 14], [-2, -2]],
                        [[15, 16], [0, 1], [0, 2], [0, 3]],
                    ],
                ]
            ),
        ),
    ],
)
def test_pad_batch_seq_of_seq(
    batch_seq_of_seq,
    max_len_outer,
    max_len_inner,
    outer_padding_value,
    inner_padding_value,
    expected,
):
    assert pad_batch_seq_of_seq(
        batch_seq_of_seq,
        max_len_outer,
        max_len_inner,
        outer_padding_value,
        inner_padding_value,
    ).equal(expected)


@pytest.mark.parametrize(
    "edge_index,edge_attr,indices,expected",
    [
        (
            torch.zeros(2, 0).long(),
            torch.rand(0),
            torch.zeros(2, 0).long(),
            torch.zeros(0),
        ),
        (
            torch.tensor([[0, 1, 3, 6], [0, 1, 3, 6]]),
            torch.tensor([[0] * 4, [1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([[0, 3, 6, 6], [0, 3, 6, 6]]),
            torch.tensor([[0] * 4, [2] * 4, [3] * 4, [3] * 4]),
        ),
        (
            torch.tensor([[0, 1, 3, 6], [2, 5, 4, 6]]),
            torch.tensor([[0] * 4, [1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([[3, 1, 3, 0, 0], [4, 5, 4, 2, 2]]),
            torch.tensor([[2] * 4, [1] * 4, [2] * 4, [0] * 4, [0] * 4]),
        ),
        (
            torch.tensor([[7, 0, 8], [9, 1, 10]]),
            torch.tensor([2, 1, 3]),
            torch.tensor([[0, 7], [1, 9]]),
            torch.tensor([1, 2]),
        ),
        (
            torch.zeros(2, 0).long(),
            torch.rand(0),
            torch.randint(5, (2, 4)).long(),
            torch.zeros(4),
        ),
        (
            torch.tensor([[0, 1, 3, 6], [0, 1, 3, 6]]),
            torch.tensor([[0] * 4, [1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([[6, 8], [6, 4]]),
            torch.tensor([[3] * 4, [0] * 4]),
        ),
        (
            torch.tensor([[0, 1, 3, 6], [2, 5, 4, 6]]),
            torch.tensor([[0] * 4, [1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([[8, 3], [4, 4]]),
            torch.tensor([[0] * 4, [2] * 4]),
        ),
    ],
)
def test_index_edge_attr(edge_index, edge_attr, indices, expected):
    assert index_edge_attr(edge_index, edge_attr, indices).equal(expected)


@pytest.mark.parametrize(
    "edge_index_a,edge_index_b,expected",
    [
        (torch.empty(2, 0).long(), torch.empty(2, 0).long(), torch.empty(0, 0).bool()),
        (
            torch.tensor([[0], [3]]),
            torch.tensor([[0], [3]]),
            torch.tensor([[True]]),
        ),
        (
            torch.tensor([[0], [3]]),
            torch.tensor([[0], [2]]),
            torch.tensor([[False]]),
        ),
        (
            torch.tensor([[0, 1, 3], [3, 2, 5]]),
            torch.tensor([[0, 3], [4, 5]]),
            torch.tensor([[False, False], [False, False], [False, True]]),
        ),
    ],
)
def test_get_edge_index_co_occurrence_matrix(edge_index_a, edge_index_b, expected):
    assert get_edge_index_co_occurrence_matrix(edge_index_a, edge_index_b).equal(
        expected
    )


@pytest.mark.parametrize(
    "batch_size,batch,expected",
    [
        (1, torch.empty(0).long(), torch.tensor([0])),
        (1, torch.tensor([0, 0, 0]), torch.tensor([0])),
        (3, torch.tensor([0, 1, 1, 2, 2, 2]), torch.tensor([0, 1, 3])),
        (5, torch.tensor([0, 2, 2, 3, 3, 3]), torch.tensor([0, 1, 1, 3, 6])),
    ],
)
def test_calculate_node_id_offsets(batch_size, batch, expected):
    node_id_offsets = calculate_node_id_offsets(batch_size, batch)
    assert node_id_offsets.equal(expected)


@pytest.mark.parametrize(
    "batched_graphs,event_type_ids,event_src_ids,event_dst_ids,event_label_word_ids,"
    "event_label_mask,event_timestamps,expected_updated_batched_graph",
    [
        (
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 0).long(),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.zeros(3).long(),
            torch.zeros(3).long(),
            torch.empty(3, 0).long(),
            torch.empty(3, 0).bool(),
            torch.zeros(3, 2).long(),
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 0).long(),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor([[6] * 4, [5] * 4, [4] * 4, [3] * 4, [2] * 4, [1] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [2, 0], [3, 1], [4, 3], [5, 2], [6, 4]]
                ),
                edge_index=torch.tensor([[2, 5], [0, 4]]),
                edge_attr=torch.tensor([[4] * 4, [5] * 4]),
                edge_label_mask=torch.tensor(
                    [[True, True, False, False], [True, True, True, True]]
                ),
                edge_last_update=torch.tensor([[2, 0], [3, 1]]),
            ),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["node-delete"], EVENT_TYPE_ID_MAP["node-delete"]]
            ),
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            torch.tensor([[3], [3]]),  # eos
            torch.tensor([[True], [True]]),
            torch.tensor([[4, 2], [5, 1]]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[6] * 3, [4] * 3, [2] * 3, [1] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [5, 2], [6, 4]]),
                edge_index=torch.tensor([[1, 3], [0, 2]]),
                edge_attr=torch.tensor([[4] * 4, [5] * 4]),
                edge_label_mask=torch.tensor(
                    [[True, True, False, False], [True, True, True, True]]
                ),
                edge_last_update=torch.tensor([[2, 0], [3, 1]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 3, [3] * 3, [2] * 3, [1] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, False, False],
                        [True, False, False],
                        [True, True, True],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [5, 2], [6, 4]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]),
            torch.tensor([[True, True, True, False], [True, True, True, True]]),
            torch.tensor([[3, 2], [5, 4]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1]),
                x=torch.tensor([[4] * 3, [3] * 3, [5] * 3, [2] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, False, False],
                        [True, True, True],
                        [True, False, False],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [3, 2], [5, 2]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 3, [3] * 3, [2] * 3, [1] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [5, 2], [6, 4]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]),
            torch.ones(2, 4).bool(),
            torch.tensor([[3, 2], [5, 4]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor(
                    [
                        [4] * 3 + [0],
                        [3] * 3 + [0],
                        [5] * 4,
                        [2] * 3 + [0],
                        [1] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [3, 2], [5, 2], [6, 4]]),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]),
                edge_label_mask=torch.ones(1, 4).bool(),
                edge_last_update=torch.tensor([[5, 4]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor(
                    [
                        [4] * 3 + [0],
                        [3] * 3 + [0],
                        [5] * 4,
                        [2] * 3 + [0],
                        [1] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [3, 2], [5, 2], [6, 4]]),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]),
                edge_label_mask=torch.ones(1, 4).bool(),
                edge_last_update=torch.tensor([[5, 4]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[3], [3]]),  # eos
            torch.tensor([[True], [True]]),
            torch.tensor([[2, 2], [3, 4]]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 3, [3] * 3, [2] * 3, [1] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [5, 2], [6, 4]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, False, False, False],
                    ]
                ),
                node_last_update=torch.tensor([[1, 0], [3, 1], [5, 2], [6, 4]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4]),
            torch.ones(3, 4).bool(),
            torch.tensor([[3, 2], [5, 1], [4, 7]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 1], [3, 2], [5, 2], [6, 4], [4, 7]]
                ),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]),
                edge_label_mask=torch.ones(1, 4).bool(),
                edge_last_update=torch.tensor([[5, 1]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 1], [3, 2], [5, 2], [6, 4], [4, 7]]
                ),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]),
                edge_label_mask=torch.ones(1, 4).bool(),
                edge_last_update=torch.tensor([[5, 1]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[3] * 3, [8] * 3, [9] * 3]),
            torch.tensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
            torch.tensor([[2, 2], [3, 1], [4, 9]]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 2], [6, 4], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 4]),
                edge_label_mask=torch.ones(1, 4).bool(),
                edge_last_update=torch.tensor([[5, 1]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 2], [6, 4], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 2]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[5, 1]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[10] * 3, [6] * 3, [11] * 3]),
            torch.tensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
            torch.tensor([[5, 9], [3, 1], [2, 4]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 3 + [0],
                        [2] * 4,
                        [1] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [6, 4], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 3]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[2, 4]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 3 + [0],
                        [2] * 4,
                        [1] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [6, 4], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 3]),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[2, 4]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[12] * 2, [1] * 2, [11] * 2]),
            torch.tensor([[True, True], [True, False], [True, False]]),
            torch.tensor([[3, 8], [1, 4], [6, 7]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 3 + [0],
                        [2] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 2]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[3, 8]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 3 + [0],
                        [2] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 2]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[3, 8]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([2, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([[12] * 4, [13] * 4, [14] * 4]),
            torch.tensor(
                [[True, False, False, False], [True, True, True, False], [True] * 4]
            ),
            torch.tensor([[2, 3], [8, 0], [7, 3]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 3 + [0],
                        [2] * 4,
                        [8] * 3 + [0],
                        [7] * 4,
                        [9] * 3 + [0],
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[3, 6], [4, 5]]),
                edge_attr=torch.tensor([[13] * 4, [14] * 4]),
                edge_label_mask=torch.tensor([[True, True, True, False], [True] * 4]),
                edge_last_update=torch.tensor([[8, 0], [7, 3]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor([[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [2, 3], [3, 1], [4, 6], [5, 9], [6, 1]]
                ),
                edge_index=torch.tensor([[2, 5], [0, 3]]),
                edge_attr=torch.tensor([[1] * 3, [2] * 3]),
                edge_label_mask=torch.tensor([[True, True, False], [True] * 3]),
                edge_last_update=torch.tensor([[3, 2], [2, 5]]),
            ),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([0, 2]),
            torch.tensor([1, 0]),
            torch.tensor([[7] * 4, [8] * 4]),
            torch.tensor([[True] * 4, [True, False, False, False]]),
            torch.tensor([[5, 2], [3, 0]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor([[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [2, 3], [3, 1], [4, 6], [5, 9], [6, 1]]
                ),
                edge_index=torch.tensor([[2, 0], [0, 1]]),
                edge_attr=torch.tensor([[1] * 3 + [0], [7] * 4]),
                edge_label_mask=torch.tensor([[True, True, False, False], [True] * 4]),
                edge_last_update=torch.tensor([[3, 2], [5, 2]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor([[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [2, 3], [3, 1], [4, 6], [5, 9], [6, 1]]
                ),
                edge_index=torch.tensor([[2, 5], [0, 3]]),
                edge_attr=torch.tensor([[1] * 3, [2] * 3]),
                edge_label_mask=torch.tensor([[True, True, False], [True] * 3]),
                edge_last_update=torch.tensor([[3, 2], [2, 5]]),
            ),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([0, 2]),
            torch.tensor([1, 0]),
            torch.tensor([[7] * 2, [8] * 2]),
            torch.tensor([[True] * 2, [True, False]]),
            torch.tensor([[5, 2], [3, 0]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor([[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [2, 3], [3, 1], [4, 6], [5, 9], [6, 1]]
                ),
                edge_index=torch.tensor([[2, 0], [0, 1]]),
                edge_attr=torch.tensor([[1] * 2, [7] * 2]),
                edge_label_mask=torch.ones(2, 2).bool(),
                edge_last_update=torch.tensor([[3, 2], [5, 2]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                x=torch.tensor(
                    [
                        [1] * 4,
                        [2] * 4,
                        [3] * 4,
                        [4] * 4,
                        [5] * 4,
                        [6] * 4,
                        [7] * 4,
                        [8] * 4,
                        [9] * 4,
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [
                        [0, 2],
                        [1, 0],
                        [2, 3],
                        [3, 1],
                        [4, 6],
                        [5, 9],
                        [6, 1],
                        [7, 3],
                        [8, 2],
                    ]
                ),
                edge_index=torch.tensor([[2, 5], [0, 3]]),
                edge_attr=torch.tensor([[1] * 4, [2] * 4]),
                edge_label_mask=torch.tensor(
                    [[True, True, False, False], [True, True, True, True]]
                ),
                edge_last_update=torch.tensor([[3, 2], [2, 2]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([2, 2, 2, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4, [10] * 4]),
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, False, False, False],
                    [True, True, False, False],
                ]
            ),
            torch.tensor([[5, 0], [3, 2], [2, 5], [1, 2]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]),
                x=torch.tensor(
                    [
                        [1] * 4,
                        [2] * 4,
                        [3] * 4,
                        [4] * 4,
                        [5] * 4,
                        [6] * 4,
                        [7] * 4,
                        [8] * 4,
                        [9] * 4,
                        [10] * 4,
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, False, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [
                        [0, 2],
                        [1, 0],
                        [2, 3],
                        [3, 1],
                        [4, 6],
                        [5, 9],
                        [6, 1],
                        [7, 3],
                        [8, 2],
                        [1, 2],
                    ]
                ),
                edge_index=torch.tensor([[2, 5, 8], [0, 3, 6]]),
                edge_attr=torch.tensor([[1] * 4, [2] * 4, [3] * 4]),
                edge_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                    ]
                ),
                edge_last_update=torch.tensor([[3, 2], [2, 2], [2, 5]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([2, 2, 3, 3]),
                x=torch.tensor([[3] * 3, [4] * 3, [5] * 3, [6] * 3]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True],
                    ]
                ),
                node_last_update=torch.tensor([[1, 2], [2, 0], [3, 7], [4, 1]]),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[1] * 2]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[4, 2]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 0, 1, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4, [4] * 4]),
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, False, False, False],
                    [True, True, True, False],
                ]
            ),
            torch.tensor([[3, 2], [5, 1], [2, 4], [3, 8]]),
            Batch(
                batch=torch.tensor([0, 1, 2, 2, 3, 3, 3]),
                x=torch.tensor(
                    [
                        [1] * 4,
                        [2] * 4,
                        [3] * 3 + [0],
                        [4] * 3 + [0],
                        [5] * 3 + [0],
                        [6] * 3 + [0],
                        [4] * 4,
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, True, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[3, 2], [5, 1], [1, 2], [2, 0], [3, 7], [4, 1], [3, 8]]
                ),
                edge_index=torch.tensor([[5, 3], [4, 2]]),
                edge_attr=torch.tensor([[1] * 2, [3] * 2]),
                edge_label_mask=torch.tensor([[True, True], [True, False]]),
                edge_last_update=torch.tensor([[4, 2], [2, 4]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 2, 2]),
                x=torch.tensor([[1] * 4, [2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4]),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, False, False, False],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 3], [2, 0], [3, 2], [4, 5], [5, 2], [6, 3]]
                ),
                edge_index=torch.tensor([[0, 2, 4], [1, 3, 5]]),
                edge_attr=torch.tensor([[3] * 3, [2] * 3, [1] * 3]),
                edge_label_mask=torch.tensor(
                    [[True, False, False], [True, True, True], [True, True, False]]
                ),
                edge_last_update=torch.tensor([[4, 1], [3, 2], [2, 0]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[7] * 4, [8] * 4, [9] * 4]),
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, True, False, False],
                ]
            ),
            torch.tensor([[3, 2], [5, 9], [2, 1]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                x=torch.tensor(
                    [
                        [1] * 4,
                        [2] * 4,
                        [7] * 4,
                        [3] * 4,
                        [4] * 4,
                        [8] * 4,
                        [5] * 4,
                        [6] * 4,
                        [9] * 4,
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, False],
                        [True, True, False, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [
                        [1, 3],
                        [2, 0],
                        [3, 2],
                        [3, 2],
                        [4, 5],
                        [5, 9],
                        [5, 2],
                        [6, 3],
                        [2, 1],
                    ]
                ),
                edge_index=torch.tensor([[0, 3, 6], [1, 4, 7]]),
                edge_attr=torch.tensor([[3] * 3, [2] * 3, [1] * 3]),
                edge_label_mask=torch.tensor(
                    [[True, False, False], [True, True, True], [True, True, False]]
                ),
                edge_last_update=torch.tensor([[4, 1], [3, 2], [2, 0]]),
            ),
        ),
        (
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 0, 13),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0, 13),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["node-add"]]
            ),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            F.one_hot(torch.tensor([[11] * 2, [12] * 2]), num_classes=13).float(),
            torch.tensor([[True, False], [True, True]]),
            torch.tensor([[6, 7], [8, 9]]),
            Batch(
                batch=torch.tensor([0, 1]),
                x=F.one_hot(torch.tensor([[11] * 2, [12] * 2]), num_classes=13).float(),
                node_label_mask=torch.tensor([[True, False], [True, True]]),
                node_last_update=torch.tensor([[6, 7], [8, 9]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0, 13),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=F.one_hot(
                    torch.tensor([[8] * 2, [9] * 2, [11] * 2, [12] * 2]), num_classes=13
                ).float(),
                node_label_mask=torch.tensor(
                    [[True, True], [True, False], [True, False], [True, True]]
                ),
                node_last_update=torch.tensor([[2, 3], [3, 4], [6, 7], [8, 9]]),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 0, 13),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-add"]]
            ),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            F.one_hot(torch.tensor([[11] * 2, [12] * 2]), num_classes=13).float(),
            torch.tensor([[True, False], [True, True]]),
            torch.tensor([[6, 7], [8, 9]]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=F.one_hot(
                    torch.tensor([[8] * 2, [9] * 2, [11] * 2, [12] * 2]), num_classes=13
                ).float(),
                node_label_mask=torch.tensor(
                    [[True, True], [True, False], [True, False], [True, True]]
                ),
                node_last_update=torch.tensor([[2, 3], [3, 4], [6, 7], [8, 9]]),
                edge_index=torch.tensor([[0, 3], [1, 2]]),
                edge_attr=F.one_hot(
                    torch.tensor([[11] * 2, [12] * 2]), num_classes=13
                ).float(),
                edge_label_mask=torch.tensor([[True, False], [True, True]]),
                edge_last_update=torch.tensor([[6, 7], [8, 9]]),
            ),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.cat(
                    [
                        F.one_hot(torch.tensor([[4] * 4, [5] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([10] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                        F.one_hot(torch.tensor([[2] * 4, [1] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([8] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                        F.one_hot(torch.tensor([[7] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([9] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                    ]
                ),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [6, 4], [3, 1], [4, 7], [4, 9]]
                ),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=F.one_hot(torch.tensor([[11] * 3]), num_classes=13).float(),
                edge_label_mask=torch.ones(1, 3).bool(),
                edge_last_update=torch.tensor([[2, 4]]),
            ),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([2, 1, 0, 0]),
            torch.tensor([0, 0, 1, 0]),
            F.one_hot(
                torch.tensor([[12] * 2, [1] * 2, [11] * 2, [12] * 2]), num_classes=13
            ).float(),
            torch.tensor([[True, True], [True, False], [True, False], [True, True]]),
            torch.tensor([[3, 8], [1, 4], [6, 7], [8, 9]]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2, 3]),
                x=torch.cat(
                    [
                        F.one_hot(torch.tensor([[4] * 4, [5] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([10] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                        F.one_hot(torch.tensor([[2] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([8] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                        F.one_hot(torch.tensor([[7] * 4]), num_classes=13),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([9] * 3), num_classes=13),
                                torch.zeros(1, 13),
                            ]
                        ).unsqueeze(0),
                        torch.cat(
                            [
                                F.one_hot(torch.tensor([12] * 2), num_classes=13),
                                torch.zeros(2, 13),
                            ]
                        ).unsqueeze(0),
                    ]
                ).float(),
                node_label_mask=torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, True, True, True],
                        [True, True, False, False],
                        [True, True, True, True],
                        [True, True, True, False],
                        [True, True, False, False],
                    ]
                ),
                node_last_update=torch.tensor(
                    [[1, 0], [3, 2], [5, 9], [5, 2], [3, 1], [4, 7], [4, 9], [8, 9]]
                ),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=F.one_hot(torch.tensor([[12] * 2]), num_classes=13).float(),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[3, 8]]),
            ),
        ),
    ],
)
def test_update_batched_graph(
    batched_graphs,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_label_word_ids,
    event_label_mask,
    event_timestamps,
    expected_updated_batched_graph,
):
    updated_batched_graph = update_batched_graph(
        batched_graphs,
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_label_word_ids,
        event_label_mask,
        event_timestamps,
    )
    assert updated_batched_graph.batch.equal(expected_updated_batched_graph.batch)
    assert updated_batched_graph.x.equal(expected_updated_batched_graph.x)
    assert updated_batched_graph.node_label_mask.equal(
        expected_updated_batched_graph.node_label_mask
    )
    assert updated_batched_graph.node_last_update.equal(
        expected_updated_batched_graph.node_last_update
    )
    assert updated_batched_graph.edge_index.equal(
        expected_updated_batched_graph.edge_index
    )
    assert updated_batched_graph.edge_attr.equal(
        expected_updated_batched_graph.edge_attr
    )
    assert updated_batched_graph.edge_label_mask.equal(
        expected_updated_batched_graph.edge_label_mask
    )
    assert updated_batched_graph.edge_last_update.equal(
        expected_updated_batched_graph.edge_last_update
    )


@pytest.mark.parametrize(
    "node_features,batch,delete_mask,added_features,added_batch,batch_size,expected",
    [
        (
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([True, False, True, True]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            1,
            torch.tensor([0, 0, 0, 0, 0]),
        ),
        (
            torch.tensor([1, 1, 2, 3, 3, 3, 5]),
            torch.tensor([1, 1, 2, 3, 3, 3, 5]),
            torch.tensor([False, True, True, True, False, True, True]),
            torch.tensor([0, 4, 4]),
            torch.tensor([0, 4, 4]),
            6,
            torch.tensor([0, 1, 2, 3, 3, 4, 4, 5]),
        ),
        (
            torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([True, False, True, True]),
            torch.tensor([[5] * 4, [6] * 4]),
            torch.tensor([0, 0]),
            1,
            torch.tensor([[4] * 4, [2] * 4, [1] * 4, [5] * 4, [6] * 4]),
        ),
        (
            torch.tensor(
                [[1] * 4, [1] * 4, [2] * 4, [3] * 4, [3] * 4, [3] * 4, [5] * 4]
            ),
            torch.tensor([1, 1, 2, 3, 3, 3, 5]),
            torch.tensor([False, True, True, True, False, True, True]),
            torch.tensor([[0] * 4, [4] * 4, [4] * 4]),
            torch.tensor([0, 4, 4]),
            6,
            torch.tensor(
                [[0] * 4, [1] * 4, [2] * 4, [3] * 4, [3] * 4, [4] * 4, [4] * 4, [5] * 4]
            ),
        ),
        (
            F.one_hot(
                torch.tensor(
                    [[1] * 4, [1] * 4, [2] * 4, [3] * 4, [3] * 4, [3] * 4, [5] * 4]
                ),
                num_classes=6,
            ),
            torch.tensor([1, 1, 2, 3, 3, 3, 5]),
            torch.tensor([False, True, True, True, False, True, True]),
            F.one_hot(torch.tensor([[0] * 4, [4] * 4, [4] * 4]), num_classes=6),
            torch.tensor([0, 4, 4]),
            6,
            F.one_hot(
                torch.tensor(
                    [
                        [0] * 4,
                        [1] * 4,
                        [2] * 4,
                        [3] * 4,
                        [3] * 4,
                        [4] * 4,
                        [4] * 4,
                        [5] * 4,
                    ]
                ),
                num_classes=6,
            ),
        ),
    ],
)
def test_update_node_features(
    node_features,
    batch,
    delete_mask,
    added_features,
    added_batch,
    batch_size,
    expected,
):
    assert update_node_features(
        node_features, batch, delete_mask, added_features, added_batch, batch_size
    ).equal(expected)


@pytest.mark.parametrize(
    "edge_index,batch,delete_node_mask,node_add_event_mask,expected_updated_edge_index",
    [
        (
            torch.tensor([[1], [0]]),
            torch.tensor([0, 0]),
            torch.tensor([True, True]),
            torch.tensor([True]),
            torch.tensor([[1], [0]]),
        ),
        (
            torch.tensor([[1], [0]]),
            torch.tensor([0, 0]),
            torch.tensor([True, True]),
            torch.tensor([False]),
            torch.tensor([[1], [0]]),
        ),
        (
            torch.tensor([[0, 2, 4], [1, 3, 5]]),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.tensor([True] * 6),
            torch.tensor([True] * 3),
            torch.tensor([[0, 3, 6], [1, 4, 7]]),
        ),
        (
            torch.tensor([[0, 2, 4], [1, 3, 5]]),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.tensor([True] * 6),
            torch.tensor([False, True, True]),
            torch.tensor([[0, 2, 5], [1, 3, 6]]),
        ),
        (
            torch.tensor([[0, 3, 6], [2, 5, 8]]),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            torch.tensor([True, False, True, True, False, True, True, False, True]),
            torch.tensor([False, False, False]),
            torch.tensor([[0, 2, 4], [1, 3, 5]]),
        ),
        (
            torch.tensor([[6, 8], [8, 7]]),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            torch.tensor([True, False, True, True, True, True, True, True, True]),
            torch.tensor([False, True, True]),
            torch.tensor([[6, 8], [8, 7]]),
        ),
    ],
)
def test_update_edge_index(
    edge_index,
    batch,
    delete_node_mask,
    node_add_event_mask,
    expected_updated_edge_index,
):
    assert update_edge_index(
        edge_index, batch, delete_node_mask, node_add_event_mask
    ).equal(expected_updated_edge_index)


@pytest.mark.parametrize(
    "channels,max_len,position_size",
    [
        (4, 10, ()),
        (4, 10, (5,)),
        (4, 10, (5, 6)),
        (10, 12, (3, 10)),
    ],
)
def test_pos_encoder(channels, max_len, position_size):
    pe = PositionalEncoder(channels, max_len)
    positions = torch.randint(max_len, position_size)
    encoding = pe(positions)
    assert encoding.size() == position_size + (channels,)
    # sanity check, make sure the values of the first dimension of both halves
    # of the channels is sin(...) and cos(...)
    flat_encoding = encoding.view(-1, channels)
    flat_positions = positions.flatten()
    assert flat_encoding[:, 0].equal(torch.sin(flat_positions.float()))
    assert flat_encoding[:, channels // 2].equal(torch.cos(flat_positions.float()))


@pytest.mark.parametrize("size", [1, 3, 5, 7])
def test_generate_subsequent_mask(size):
    mask = generate_square_subsequent_mask(size)
    # assert that the sum of tril and triu is the original mask
    assert mask.equal(torch.tril(mask) + torch.triu(mask, diagonal=1))


@pytest.mark.parametrize(
    "input_ids,decoder_start_token_id,expected",
    [
        (torch.tensor([[1, 2, 3]]), 5, torch.tensor([[5, 1, 2]])),
        (torch.tensor([[1, 2, 3], [2, 3, 0]]), 9, torch.tensor([[9, 1, 2], [9, 2, 3]])),
    ],
)
def test_shift_tokens_right(input_ids, decoder_start_token_id, expected):
    assert shift_tokens_right(input_ids, decoder_start_token_id).equal(expected)
