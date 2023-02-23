import torch

from typing import List
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_f1_score


class F1(Metric):
    """
    Measures accuracy or direct overlap between the predictions and ground truth
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    score: torch.Tensor
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, batch_preds: List[List[str]], batch_targets: List[List[str]]
    ) -> None:
        assert len(batch_preds) == len(batch_targets)
        for preds, targets in zip(batch_preds, batch_targets):
            preds_set = set(preds)
            targets_set = set(targets)
            if preds_set == targets_set:
                self.score += 1
            else:
                matches = 0
                for pred in preds_set:
                    if pred in targets_set:
                        matches += 1
                if matches != 0:
                    # calculate the f1 score
                    precision = matches / len(preds_set)
                    recall = matches / len(targets_set)
                    self.score += (2 * precision * recall) / (precision + recall)
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.total, dtype=torch.float)
        return self.score / self.total


class DynamicGraphNodeF1(Metric):
    """
    Measures the F1 score of dynamic graph nodes where the number of nodes varies.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    score: torch.Tensor

    def __init__(self) -> None:
        super().__init__()

        self.add_state("score", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        preds: float tensor of shape (batch, num_node) with values in [0, 1]
        target: integer tensor of shape (batch) with values in [0, num_node)
        """
        _, num_node = preds.size()
        if num_node > 1:
            # we only care about the F1 scores when we have more than one node
            scores = multiclass_f1_score(preds, target, num_node, average=None)
            # (num_node)
            self.score = torch.cat((self.score, scores))

    def compute(self) -> torch.Tensor:
        if self.score.size(0) == 0:
            return torch.tensor(0, device=self.device)
        return self.score.mean()
