import pytest
import torch

from tdgu.preprocessor import HuggingFacePreprocessor, SpacyPreprocessor


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["My name is Peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[True] * 4]),
        ),
        (
            ["my name is peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[True] * 4]),
        ),
        (
            ["My name is Peter", "Is my name David?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[True, True, True, True, False], [True] * 5]),
        ),
    ],
)
def test_spacy_preprocessor_preprocess(batch, expected_preprocessed, expected_mask):
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    preprocessed, mask = sp.preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["My name is Peter"],
            torch.tensor([[101, 2026, 2171, 2003, 2848, 102]]),
            torch.tensor([[1, 1, 1, 1, 1, 1]]),
        ),
        (
            ["my name is peter"],
            torch.tensor([[101, 2026, 2171, 2003, 2848, 102]]),
            torch.tensor([[1, 1, 1, 1, 1, 1]]),
        ),
        (
            ["My name is Peter", "Is my name David?"],
            torch.tensor(
                [
                    [101, 2026, 2171, 2003, 2848, 102, 0],
                    [101, 2003, 2026, 2171, 2585, 1029, 102],
                ]
            ),
            torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]),
        ),
    ],
)
def test_BERT_preprocessor_preprocess(batch, expected_preprocessed, expected_mask):
    sp = HuggingFacePreprocessor("distilbert-base-uncased")
    preprocessed, mask = sp.preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)


def test_spacy_preprocessor_load_from_file():
    sp = SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    assert len(sp.word_to_id_dict) == 772
    assert sp.vocab_size == 772


def test_spacy_preprocessor_load():
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    assert sp.pad_token_id == 0
    assert sp.unk_token_id == 1


def test_bert_preprocessor_load():
    sp = HuggingFacePreprocessor("distilbert-base-uncased")
    assert sp.vocab_size != 0
    assert sp.pad_token_id == 0
    assert sp.unk_token_id == 100


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["$$$$$$$ My name is Peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[True] * 4]),
        ),
        (
            ["my   name     is  peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[True] * 4]),
        ),
        (
            ["My    name\n is Peter", "$$$$$$$Is   my name \n\nDavid?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[True, True, True, True, False], [True] * 5]),
        ),
    ],
)
def test_spacy_preprocessor_clean_preprocess(
    batch, expected_preprocessed, expected_mask
):
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    preprocessed, mask = sp.clean_and_preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["$$$$$$$ My name is Peter"],
            torch.tensor([[101, 2026, 2171, 2003, 2848, 102]]),
            torch.tensor([[1, 1, 1, 1, 1, 1]]),
        ),
        (
            ["my   name     is  peter"],
            torch.tensor([[101, 2026, 2171, 2003, 2848, 102]]),
            torch.tensor([[1, 1, 1, 1, 1, 1]]),
        ),
        (
            ["My    name\n is Peter", "$$$$$$$Is   my name \n\nDavid?"],
            torch.tensor(
                [
                    [101, 2026, 2171, 2003, 2848, 102, 0],
                    [101, 2003, 2026, 2171, 2585, 1029, 102],
                ]
            ),
            torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]),
        ),
    ],
)
def test_BERT_preprocessor_clean_preprocess(
    batch, expected_preprocessed, expected_mask
):
    sp = HuggingFacePreprocessor("distilbert-base-uncased")
    preprocessed, mask = sp.clean_and_preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)
