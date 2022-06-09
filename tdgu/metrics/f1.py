import torch

from typing import List
from torchmetrics import Metric


class F1(Metric):
    """
    Measures accuracy or direct overlap between the predictions and ground truth
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore
        self, batch_preds: List[List[str]], batch_targets: List[List[str]]
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets in zip(batch_preds, batch_targets):
            preds_set = set(preds)
            targets_set = set(targets)
            if preds_set == targets_set:
                self.score += 1  # type: ignore
            else:
                matches = 0
                for pred in preds_set:
                    if pred in targets_set:
                        matches += 1
                if matches != 0:
                    # calculate the f1 score
                    precision = matches / len(preds_set)
                    recall = matches / len(targets_set)
                    self.score += (2 * precision * recall) / (  # type: ignore
                        precision + recall
                    )
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)  # type: ignore
        return self.score / self.total  # type: ignore
