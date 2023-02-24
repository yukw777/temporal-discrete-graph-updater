from abc import ABC, abstractmethod

import torch
from spacy.lang.en import English
from transformers import AutoTokenizer

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


def clean(raw_str: str | None) -> str:
    """Copied from the original GATA code (preproc())"""
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


class Preprocessor(ABC):
    def clean_and_preprocess(
        self, batch: list[str], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess([clean(s) for s in batch], device=device)

    @abstractmethod
    def preprocess(
        self, batch: list[str], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, word_ids: list[int]) -> list[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, word_ids: list[str]) -> list[int]:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        pass

    @abstractmethod
    def batch_decode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """
        token_ids: (batch, sent_len)
        mask: (batch, sent_len)
        skip_special_tokens: whether or not to skip special tokens.

        output: [sent, ...]
        """

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        pass


class SpacyPreprocessor(Preprocessor):
    def __init__(self, word_vocab: list[str]) -> None:
        self.tokenizer = English().tokenizer
        self.word_vocab = word_vocab
        self.word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}
        self.special_token_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        }

    def convert_ids_to_tokens(self, word_ids: list[int]) -> list[str]:
        return [self.word_vocab[word_id] for word_id in word_ids]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.word_to_id_dict.get(token, self.unk_token_id) for token in tokens]

    def tokenize(self, s: str) -> list[str]:
        return [BOS] + [t.text.lower() for t in self.tokenizer(s)] + [EOS]

    def pad(
        self, unpadded_batch: list[list[int]], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                dtype=torch.bool,
            ),
        )

    def preprocess(
        self, batch: list[str], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pad(
            [self.convert_tokens_to_ids(self.tokenize(s)) for s in batch], device=device
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """
        token_ids: (batch, sent_len)
        mask: (batch, sent_len)
        skip_special_tokens: whether or not to skip special tokens.

        output: [sent, ...]
        """
        decoded: list[str] = []
        for ids, size in zip(token_ids.tolist(), mask.sum(dim=1).tolist()):
            if skip_special_tokens:
                filtered_ids = [
                    token_id
                    for token_id in ids[:size]
                    if token_id not in self.special_token_ids
                ]
            else:
                filtered_ids = ids[:size]
            decoded.append(" ".join(self.convert_ids_to_tokens(filtered_ids)))
        return decoded

    @property
    def vocab_size(self) -> int:
        return len(self.word_vocab)

    def get_vocab(self) -> dict[str, int]:
        return self.word_to_id_dict

    @property
    def bos_token_id(self) -> int:
        return self.word_to_id_dict[BOS]

    @property
    def eos_token_id(self) -> int:
        return self.word_to_id_dict[EOS]

    @property
    def pad_token_id(self) -> int:
        return self.word_to_id_dict[PAD]

    @property
    def unk_token_id(self) -> int:
        return self.word_to_id_dict[UNK]

    @classmethod
    def load_from_file(cls, word_vocab_path: str) -> "SpacyPreprocessor":
        with open(word_vocab_path) as f:
            word_vocab = [word.strip() for word in f]
        return cls(word_vocab)


class HuggingFacePreprocessor(Preprocessor):
    def __init__(self, tokenizer_model: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def convert_ids_to_tokens(self, word_ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(word_ids)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def preprocess(
        self, batch: list[str], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: Assumes the batch has not been altered yet
        """

        processed = self.tokenizer(
            batch, return_tensors="pt", padding="longest", truncation="longest_first"
        )
        return (
            processed["input_ids"].to(device=device),
            processed["attention_mask"].bool().to(device=device),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """
        token_ids: (batch, sent_len)
        mask: (batch, sent_len)
        skip_special_tokens: whether or not to skip special tokens.

        output: [sent, ...]
        """
        sequences: list[list[str]] = []
        for ids, size in zip(token_ids.tolist(), mask.sum(dim=1).tolist()):
            sequences.append(ids[:size])
        return self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id


class BertPreprocessor(HuggingFacePreprocessor):
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.cls_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.sep_token_id
