import torch

from typing import List
from torchmetrics import Metric


class ExactMatch(Metric):
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
        batch_mask: List[bool],
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets, mask in zip(batch_preds, batch_targets, batch_mask):
            if not mask:
                continue
            # calculate the exact match score for each example in the batch
            if len(preds) == 0:
                if len(targets) == 0:
                    self.score += 1  # type: ignore
            else:
                matches = 0
                target_set = set(targets)
                for pred in preds:
                    if pred in target_set:
                        matches += 1
                self.score += matches / len(preds)  # type: ignore
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)  # type: ignore
        return self.score / self.total  # type: ignore
