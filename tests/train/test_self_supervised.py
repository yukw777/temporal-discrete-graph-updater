import pytest
import torch
import shutil


from tdgu.train.self_supervised import ObsGenSelfSupervisedTDGU
from tdgu.data import TWCmdGenGraphEventStepInput, TWCmdGenObsGenBatch


@pytest.fixture
def obs_gen_self_supervised_tdgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return ObsGenSelfSupervisedTDGU(
        pretrained_word_embedding_path=f"{tmp_path}/test-fasttext.vec",
        word_vocab_path="tests/data/test_word_vocab.txt",
    )


@pytest.mark.parametrize(
    "batch,split_size,expected",
    [
        (
            TWCmdGenObsGenBatch(
                (TWCmdGenGraphEventStepInput(),), torch.tensor([[True]])
            ),
            1,
            [
                TWCmdGenObsGenBatch(
                    (TWCmdGenGraphEventStepInput(),), torch.tensor([[True]])
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                (TWCmdGenGraphEventStepInput(),), torch.tensor([[True]])
            ),
            2,
            [
                TWCmdGenObsGenBatch(
                    (TWCmdGenGraphEventStepInput(),), torch.tensor([[True]])
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                (
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                ),
                torch.tensor([[True] * 3, [True, False, True], [False, False, True]]),
            ),
            2,
            [
                TWCmdGenObsGenBatch(
                    (TWCmdGenGraphEventStepInput(), TWCmdGenGraphEventStepInput()),
                    torch.tensor([[True] * 3, [True, False, True]]),
                ),
                TWCmdGenObsGenBatch(
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[False, False, True]]),
                ),
            ],
        ),
    ],
)
def test_tbptt_split_batch(obs_gen_self_supervised_tdgu, batch, split_size, expected):
    assert obs_gen_self_supervised_tdgu.tbptt_split_batch(batch, split_size) == expected
