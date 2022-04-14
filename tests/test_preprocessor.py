import pytest
import torch

from tdgu.preprocessor import (
    clean,
    Preprocessor,
    HuggingFacePreprocessor,
    SpacyPreprocessor,
)


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
def test_hf_preprocessor_preprocess(batch, expected_preprocessed, expected_mask):
    sp = HuggingFacePreprocessor("distilbert-base-uncased")
    preprocessed, mask = sp.preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)


def test_spacy_preprocessor_load_from_file():
    sp = SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    assert sp.vocab_size == 772


def test_spacy_preprocessor_load():
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    assert sp.pad_token_id == 0
    assert sp.unk_token_id == 1


def test_hf_preprocessor_load():
    sp = HuggingFacePreprocessor("distilbert-base-uncased")
    assert sp.vocab_size == 30522
    assert sp.pad_token_id == 0
    assert sp.unk_token_id == 100


@pytest.mark.parametrize(
    "raw_str,cleaned",
    [
        (None, "nothing"),
        ("double  spaces!", "double spaces!"),
        ("many     spaces!", "many spaces!"),
        ("    ", "nothing"),
        (
            "\n\n\n"
            "                    ________  ________  __    __  ________        \n"
            "                   |        \\|        \\|  \\  |  \\|        \\       \n"
            "                    \\$$$$$$$$| $$$$$$$$| $$  | $$ \\$$$$$$$$       \n"
            "                      | $$   | $$__     \\$$\\/  $$   | $$          \n"
            "                      | $$   | $$  \\     >$$  $$    | $$          \n"
            "                      | $$   | $$$$$    /  $$$$\\    | $$          \n"
            "                      | $$   | $$_____ |  $$ \\$$\\   | $$          \n"
            "                      | $$   | $$     \\| $$  | $$   | $$          \n"
            "                       \\$$    \\$$$$$$$$ \\$$   \\$$    \\$$          \n"
            "              __       __   ______   _______   __        _______  \n"
            "             |  \\  _  |  \\ /      \\ |       \\ |  \\      |       \\ \n"
            "             | $$ / \\ | $$|  $$$$$$\\| $$$$$$$\\| $$      | $$$$$$$\\\n"
            "             | $$/  $\\| $$| $$  | $$| $$__| $$| $$      | $$  | $$\n"
            "             | $$  $$$\\ $$| $$  | $$| $$    $$| $$      | $$  | $$\n"
            "             | $$ $$\\$$\\$$| $$  | $$| $$$$$$$\\| $$      | $$  | $$\n"
            "             | $$$$  \\$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$\n"
            "             | $$$    \\$$$ \\$$    $$| $$  | $$| $$     \\| $$    $$\n"
            "              \\$$      \\$$  \\$$$$$$  \\$$   \\$$ \\$$$$$$$$ \\$$$$$$$"
            " \n\n"
            "You are hungry! Let's cook a delicious meal. "
            "Check the cookbook in the kitchen for the recipe. "
            "Once done, enjoy your meal!\n\n"
            "-= Kitchen =-\n"
            "If you're wondering why everything seems so normal all of a sudden, "
            "it's because you've just shown up in the kitchen.\n\n"
            "You can see a closed fridge, which looks conventional, "
            "right there by you. "
            "You see a closed oven right there by you. Oh, great. Here's a table. "
            "Unfortunately, there isn't a thing on it. Hm. "
            "Oh well You scan the room, seeing a counter. The counter is vast. "
            "On the counter you can make out a cookbook and a knife. "
            "You make out a stove. Looks like someone's already been here and "
            "taken everything off it, though. Sometimes, just sometimes, "
            "TextWorld can just be the worst.\n\n\n",
            "You are hungry! Let's cook a delicious meal. "
            "Check the cookbook in the kitchen for the recipe. "
            "Once done, enjoy your meal! -= Kitchen =- "
            "If you're wondering why everything seems so normal all of a sudden, "
            "it's because you've just shown up in the kitchen. "
            "You can see a closed fridge, which looks conventional, "
            "right there by you. You see a closed oven right there by you. "
            "Oh, great. Here's a table. Unfortunately, there isn't a thing on it. "
            "Hm. Oh well You scan the room, seeing a counter. The counter is vast. "
            "On the counter you can make out a cookbook and a knife. "
            "You make out a stove. "
            "Looks like someone's already been here and taken everything off it, "
            "though. Sometimes, just sometimes, TextWorld can just be the worst.",
        ),
    ],
)
def test_clean(raw_str, cleaned):
    assert clean(raw_str) == cleaned


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["$$$$$$$ My name is Peter"],
            torch.tensor([[99, 99, 99, 99]]),
            torch.tensor([[1, 1, 1, 1]]),
        ),
        (
            ["my   name     is       "],
            torch.tensor([[99, 99, 99]]),
            torch.tensor([[1, 1, 1]]),
        ),
        (
            ["My    name\n is Peter", "$$$$$$$Is   my name \n\nDavid?"],
            torch.tensor(
                [
                    [99, 99, 99, 99],
                    [99, 99, 99, 99],
                ]
            ),
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        ),
    ],
)
def test_mock_clean_and_preprocess(batch, expected_preprocessed, expected_mask):
    class TestPreprocessor(Preprocessor):
        def preprocess(self, batch, device=None):
            max_len = max(len(word_ids.strip().split()) for word_ids in batch)
            return (
                torch.tensor(
                    [[99] * (max_len) for _ in batch],
                    device=device,
                ),
                torch.tensor(
                    [[1] * (max_len) for _ in batch],
                    device=device,
                ),
            )

        def convert_ids_to_tokens(self, word_ids):
            return [-1]

        def convert_tokens_to_ids(self, word_ids):
            return [-1]

        @property
        def vocab_size(self) -> int:
            return -1

        def get_vocab(self):
            return {}

        @property
        def bos_token_id(self) -> int:
            return -1

        @property
        def eos_token_id(self) -> int:
            return -1

        @property
        def pad_token_id(self) -> int:
            return -1

        @property
        def unk_token_id(self) -> int:
            return -1

    sp = TestPreprocessor()
    preprocessed, mask = sp.clean_and_preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)
