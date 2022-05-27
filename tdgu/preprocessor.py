from abc import ABC, abstractmethod

import torch

from typing import List, Dict, Tuple, Optional
from spacy.lang.en import English

from transformers import AutoTokenizer

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


def clean(raw_str: Optional[str]) -> str:
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


class Preprocessor(ABC):
    def clean_and_preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess([clean(s) for s in batch], device=device)

    @abstractmethod
    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, word_ids: List[int]) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, word_ids: List[str]) -> List[int]:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        pass

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
    def __init__(self, word_vocab: List[str]) -> None:
        self.tokenizer = English().tokenizer
        self.word_vocab = word_vocab
        self.word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}

    def convert_ids_to_tokens(self, word_ids: List[int]) -> List[str]:
        return [self.word_vocab[word_id] for word_id in word_ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word_to_id_dict.get(token, self.unk_token_id) for token in tokens]

    def tokenize(self, s: str) -> List[str]:
        return [BOS] + [t.text.lower() for t in self.tokenizer(s)] + [EOS]

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
                dtype=torch.bool,
            ),
        )

    def preprocess_tokenized(
        self,
        tokenized_batch: List[List[str]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: Assumes the batch has already been tokenized by a preprocessor with the
        similar vocab mapping
        """

        return self.pad(
            [self.convert_tokens_to_ids(tokenized) for tokenized in tokenized_batch],
            device=device,
        )

    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: Assumes the batch has not been altered yet
        """

        return self.preprocess_tokenized(
            [self.tokenize(s) for s in batch], device=device
        )

    @property
    def vocab_size(self) -> int:
        return len(self.word_vocab)

    def get_vocab(self) -> Dict[str, int]:
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
        with open(word_vocab_path, "r") as f:
            word_vocab = [word.strip() for word in f]
        return cls(word_vocab)


class HuggingFacePreprocessor(Preprocessor):
    def __init__(self, tokenizer_model: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def convert_ids_to_tokens(self, word_ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(word_ids)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: Assumes the batch has not been altered yet
        """

        processed = self.tokenizer(
            batch, return_tensors="pt", padding="longest", truncation="longest_first"
        )
        return (
            processed["input_ids"].to(device=device),
            processed["attention_mask"].to(device=device),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

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
