import torch
from torchmetrics import Metric


class ExactMatch(Metric):
    """Measures accuracy or direct overlap between the predictions and ground
    truth."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore
        self, batch_preds: list[list[str]], batch_targets: list[list[str]]
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets in zip(batch_preds, batch_targets):
            # calculate the exact match score for each example in the batch
            if len(preds) == 0:
                if len(targets) == 0:
                    self.score += 1  # type: ignore
            else:
                matches = 0
                pred_set = set(preds)
                target_set = set(targets)
                for pred in pred_set:
                    if pred in target_set:
                        matches += 1
                self.score += matches / len(pred_set)  # type: ignore
            self.total += 1  # type: ignore

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)  # type: ignore
        return self.score / self.total  # type: ignore
