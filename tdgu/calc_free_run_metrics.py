from torch.utils.data import DataLoader

from tdgu.train.supervised import SupervisedTDGU
from tdgu.data import TWCmdGenGraphEventFreeRunDataset, TWCmdGenGraphEventDataCollator
from tdgu.preprocessor import SpacyPreprocessor
from tdgu.metrics.f1 import F1
from tdgu.metrics.exact_match import ExactMatch


def main(
    data_filename: str,
    ckpt_filename: str,
    word_vocab_path: str,
    batch_size: int,
    device: str,
) -> None:
    dataset = TWCmdGenGraphEventFreeRunDataset(data_filename, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    preprocessor = SpacyPreprocessor.load_from_file(word_vocab_path)
    collator = TWCmdGenGraphEventDataCollator(preprocessor)

    lm = SupervisedTDGU.load_from_checkpoint(
        ckpt_filename, word_vocab_path=word_vocab_path
    )
    lm.eval()
    lm = lm.to(device)
    graph_f1 = F1()
    graph_em = ExactMatch()

    generated_rdfs_list, groundtruth_rdfs_list = lm.eval_free_run(
        dataset, dataloader, collator, total=len(dataset)
    )
    graph_f1.update(generated_rdfs_list, groundtruth_rdfs_list)
    graph_em.update(generated_rdfs_list, groundtruth_rdfs_list)

    print(f"Free Run Graph F1: {graph_f1.compute()}")
    print(f"Free Run Graph EM: {graph_em.compute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--word-vocab-path", default="vocabs/word_vocab.txt")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.data_filename,
        args.ckpt_filename,
        args.word_vocab_path,
        args.batch_size,
        args.device,
    )
