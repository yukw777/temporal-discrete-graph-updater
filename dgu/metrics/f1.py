import torch

from typing import List
from torchmetrics import Metric


class F1(Metric):
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
            preds_set = set(preds)
            targets_set = set(targets)
            if preds_set == targets_set:
                self.scores = torch.cat(
                    [
                        self.scores,  # type: ignore
                        torch.tensor(1, device=self.device).unsqueeze(-1),
                    ]
                )
            else:
                matches = 0
                for pred in preds_set:
                    if pred in targets_set:
                        matches += 1
                if matches != 0:
                    # calculate the f1 score
                    precision = matches / len(preds_set)
                    recall = matches / len(targets_set)
                    self.scores = torch.cat(
                        [
                            self.scores,
                            torch.tensor(
                                (2 * precision * recall) / (precision + recall),
                                device=self.device,
                            ).unsqueeze(-1),
                        ]
                    )
                else:
                    self.scores = torch.cat(
                        [
                            self.scores,  # type: ignore
                            torch.tensor(0, device=self.device).unsqueeze(-1),
                        ]
                    )
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)  # type: ignore
        return self.scores.sum() / self.total  # type: ignore
