import torch

from typing import List, Tuple, Optional
from spacy.lang.en import English

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


class SpacyPreprocessor:
    def __init__(self, word_vocab: List[str]) -> None:
        super().__init__()
        self.tokenizer = English().tokenizer
        self.word_vocab = word_vocab
        self.word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}
        self.pad_id = self.word_to_id_dict[PAD]
        self.unk_id = self.word_to_id_dict[UNK]

    def id_to_word(self, word_id: int) -> str:
        return self.word_vocab[word_id]

    def ids_to_words(self, word_ids: List[int]) -> List[str]:
        return [self.id_to_word(word_id) for word_id in word_ids]

    def word_to_id(self, word: str) -> int:
        return self.word_to_id_dict.get(word, self.unk_id)

    def words_to_ids(self, words: List[str]) -> List[int]:
        return [self.word_to_id(word) for word in words]

    def tokenize(self, s: str) -> List[str]:
        return [t.text.lower() for t in self.tokenizer(s)]

    def pad(
        self, unpadded_batch: List[List[int]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return padded tensor and corresponding mask
        max_len = max(len(word_ids) for word_ids in unpadded_batch)
        return (
            torch.tensor(
                [
                    word_ids + [0] * (max_len - len(word_ids))
                    for word_ids in unpadded_batch
                ],
                device=device,
            ),
            torch.tensor(
                [
                    [1] * len(word_ids) + [0] * (max_len - len(word_ids))
                    for word_ids in unpadded_batch
                ],
                device=device,
                dtype=torch.float,
            ),
        )

    def preprocess_tokenized(
        self, tokenized_batch: List[List[str]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pad(
            [self.words_to_ids(tokenized) for tokenized in tokenized_batch],
            device=device,
        )

    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_tokenized(
            [self.tokenize(s) for s in batch], device=device
        )

    def clean_and_preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_tokenized(
            [self.tokenize(self.clean(s)) for s in batch], device=device
        )

    def batch_clean(self, batch_raw_str: List[str]) -> List[str]:
        return [self.clean(raw_str) for raw_str in batch_raw_str]

    def clean(self, raw_str: Optional[str]) -> str:
        """
        Copied from the original GATA code (preproc())
        """
        if raw_str is None:
            return "nothing"
        cleaned = raw_str.replace("\n", " ")
        if "$$$$$$$" in cleaned:
            cleaned = cleaned.split("$$$$$$$")[-1]
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        cleaned = cleaned.strip()
        if len(cleaned) == 0:
            return "nothing"
        return cleaned

    def decode(self, batch: List[List[int]]) -> List[str]:
        return [
            " ".join(
                self.ids_to_words(
                    [word_id for word_id in word_ids if word_id != self.pad_id]
                )
            )
            for word_ids in batch
        ]

    @classmethod
    def load_from_file(cls, word_vocab_path: str) -> "SpacyPreprocessor":
        with open(word_vocab_path, "r") as f:
            word_vocab = [word.strip() for word in f]
        return cls(word_vocab)
