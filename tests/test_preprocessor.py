import pytest
import torch

from dgu.preprocessor import SpacyPreprocessor


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["My name is Peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["my name is peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["My name is Peter", "Is my name David?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).float(),
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
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["my name is peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["My name is Peter", "Is my name David?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).float(),
        ),
    ],
)
def test_spacy_preprocessor_decode(batch, expected_preprocessed, expected_mask):
    sp = SpacyPreprocessor(
        ["<pad>", "<unk>", "my", "name", "is", "peter", "david", "?"]
    )
    preprocessed, _ = sp.preprocess(batch)
    assert sp.decode(preprocessed.tolist()) == [" ".join(sp.tokenize(s)) for s in batch]


def test_spacy_preprocessor_load_from_file():
    sp = SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    assert len(sp.word_to_id_dict) == 772


@pytest.mark.parametrize("batch_size", list(range(3)))
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
def test_spacy_preprocessor_clean(raw_str, cleaned, batch_size):
    sp = SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    assert sp.clean(raw_str) == cleaned
    assert sp.batch_clean([raw_str] * batch_size) == [cleaned] * batch_size


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["$$$$$$$ My name is Peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["my   name     is  peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["My    name\n is Peter", "$$$$$$$Is   my name \n\nDavid?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).float(),
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
