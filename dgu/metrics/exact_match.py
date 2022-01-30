import torch

from typing import List
from torchmetrics import Metric


class ExactMatch(Metric):
    """
    Measures accuracy or direct overlap between the predictions and ground truth
    """

    def __init__(self) -> None:
        super().__init__()

        self.add_state("scores", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore
        self, batch_preds: List[List[str]], batch_targets: List[List[str]]
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets in zip(batch_preds, batch_targets):
            # calculate the exact match score for each example in the batch
            if len(preds) == 0:
                if len(targets) == 0:
                    self.scores = torch.cat(
                        [
                            self.scores,  # type: ignore
                            torch.tensor(1, device=self.device).unsqueeze(-1),
                        ]
                    )
            else:
                matches = 0
                pred_set = set(preds)
                target_set = set(targets)
                for pred in pred_set:
                    if pred in target_set:
                        matches += 1
                self.scores = torch.cat(
                    [
                        self.scores,
                        torch.tensor(
                            matches / len(pred_set), device=self.device
                        ).unsqueeze(-1),
                    ]
                )
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)  # type: ignore
        return self.scores.sum() / self.total  # type: ignore
