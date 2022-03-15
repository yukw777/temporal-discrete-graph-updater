import torch

from typing import List, Tuple, Optional
from spacy.lang.en import English

from transformers import DistilBertTokenizer

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


class Preprocessor:
    def __init__(self, word_vocab: List[str]) -> None:
        pass

    def id_to_word(self, word_id: int) -> str:
        pass

    def ids_to_words(self, word_ids: List[int]) -> List[str]:
        pass

    def word_to_id(self, word: str) -> int:
        pass

    def words_to_ids(self, words: List[str]) -> List[int]:
        pass

    def tokenize(self, s: str) -> List[str]:
        pass

    def pad(
        self, unpadded_batch: List[List[int]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def preprocess_tokenized(
        self, tokenized_batch: List[List[str]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def clean_and_preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def batch_clean(self, batch_raw_str: List[str]) -> List[str]:
        pass

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
        pass

    @classmethod
    def load_from_file(cls, word_vocab_path: str) -> "Preprocessor":
        pass


class SpacyPreprocessor(Preprocessor):
    def __init__(self, word_vocab: List[str]) -> None:
        super().__init__(word_vocab)

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
                dtype=torch.bool,
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
        return super().clean(raw_str)

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


class BERTPreprocessor(Preprocessor):
    def __init__(self, word_vocab: List[str] = []) -> None:
        super().__init__(word_vocab)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", use_fast=True
        )

        # NOTE: Acceptable to have different tokens from Spacy
        # special_token_dict = {
        #     "pad_token" : PAD,
        #     "unk_token" : UNK,
        #     "bos_token" : BOS,
        #     "eos_token" : EOS,
        #     "sep_token" : SEP
        # }
        # self.tokenizer.add_special_tokens(special_token_dict)
        # self.pad_id = self.word_to_id_dict[PAD]
        # self.unk_id = self.word_to_id_dict[UNK]

        DistilBERT_vocab = self.tokenizer.get_vocab()
        self.word_vocab = list(DistilBERT_vocab.keys())
        self.word_to_id_dict = DistilBERT_vocab

    # NOTE: The BERT pretrained tokenizer is rather specific about
    # tokenization before mapping to ids
    def id_to_word(self, word_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(word_id)

    def ids_to_words(self, word_ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(word_ids)

    def word_to_id(self, word: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(word)

    def words_to_ids(self, words: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(words)

    def tokenize(self, s: str) -> List[str]:
        return self.tokenizer.tokenize(s)

    # NOTE: Default padding to max model length to maintain consistency
    # with tokenizer __call__ (in this case 512)
    def pad(
        self, unpadded_batch: List[List[int]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return padded tensor and corresponding mask
        max_len = self.tokenizer.max_model_input_sizes["distilbert-base-uncased"]
        max_len = min(max_len, max(len(word_ids) for word_ids in unpadded_batch))
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

    # NOTE: Assumes the input is already correctly formatted with CLS, etc. tokens
    # NOTE: Mask is in the form of bool(s) rather than (long) int(s) or float(s),
    # this distinction is kept to maintain consistency in data.py
    def preprocess_tokenized(
        self, tokenized_batch: List[List[str]], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pad(
            [self.words_to_ids(tokenized) for tokenized in tokenized_batch],
            device=device,
        )

    # NOTE: Assumes the input has no formatting at all (will add CLS, etc. tokens)
    # NOTE: Mask is in the form of (long) int(s) rather than bool(s),
    # this distinction is made so the transformer may accept it
    def preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        processed = self.tokenizer(
            batch, return_tensors="pt", padding="longest", truncation="longest_first"
        )
        return (
            processed["input_ids"].to(device=device),
            processed["attention_mask"].to(device=device),
        )

    # NOTE: preprocess_tokenized preceded by clean
    def clean_and_preprocess(
        self, batch: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_tokenized(
            [self.tokenize(self.clean(s)) for s in batch], device=device
        )

    def batch_clean(self, batch_raw_str: List[str]) -> List[str]:
        return [self.clean(raw_str) for raw_str in batch_raw_str]

    def clean(self, raw_str: Optional[str]) -> str:
        return super().clean(raw_str)

    def decode(self, batch: List[List[int]]) -> List[str]:
        return [
            self.tokenizer.decode(word_ids, skip_special_tokens=True)
            for word_ids in batch
        ]

    # NOTE: No need to load from file, simply downloads the vocabulary
    @classmethod
    def load_from_file(cls, word_vocab_path: str) -> "BERTPreprocessor":
        return cls()
