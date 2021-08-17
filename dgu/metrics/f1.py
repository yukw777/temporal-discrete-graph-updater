import torch

from typing import List
from torchmetrics import Metric
from collections import Counter


class F1(Metric):
    """
    Measures accuracy or direct overlap between the predictions and ground truth
    """

    def __init__(self) -> None:
        super().__init__()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore
        self,
        batch_preds: List[List[str]],
        batch_targets: List[List[str]],
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets in zip(batch_preds, batch_targets):
            if preds == targets:
                self.score += 1  # type: ignore
            else:
                intersection = Counter(preds) & Counter(targets)
                matches = sum(intersection.values())
                if matches != 0:
                    # calculate the f1 score
                    precision = matches / len(preds)
                    recall = matches / len(targets)
                    self.score += (2 * precision * recall) / (  # type: ignore
                        precision + recall
                    )
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.score / self.total  # type: ignore
