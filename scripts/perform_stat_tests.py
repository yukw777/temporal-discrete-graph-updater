import torch
from scipy.stats import ttest_rel, wilcoxon


def main(scores_a_filename: str, scores_b_filename: str) -> None:
    scores_a_dict = torch.load(scores_a_filename)
    scores_b_dict = torch.load(scores_b_filename)
    assert scores_a_dict.keys() == scores_b_dict.keys()

    scores_a: list[float] = []
    scores_b: list[float] = []
    for key in scores_a_dict.keys():
        scores_a.append(scores_a_dict[key])
        scores_b.append(scores_b_dict[key])
    stat, pvalue = ttest_rel(scores_a, scores_b)
    print("Paired T-test")
    print(f"statistic: {stat}")
    print(f"p-value: {pvalue}")

    stat, pvalue = wilcoxon(scores_a, scores_b)
    print("Wilcoxon signed-rank test")
    print(f"statistic: {stat}")
    print(f"p-value: {pvalue}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scores_a_filename")
    parser.add_argument("scores_b_filename")
    args = parser.parse_args()
    main(args.scores_a_filename, args.scores_b_filename)
