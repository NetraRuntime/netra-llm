"""
Token-level NLL/PPL metrics for evaluation.

- NLLMetric: token-level mean negative log-likelihood (weighted mean over tokens).
- PPLMetric: exp(mean NLL) = perplexity.

Both use sync_on_compute=True so that compute() aggregates over all ranks.
"""

import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):
    """Token-level mean NLL. Weights should be the mask of predicted (e.g. masked) tokens."""

    def __init__(self, **kwargs):
        # Ensure cross-rank aggregation when compute() is called
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)

    def update(self, value, weight=1.0) -> None:
        # MeanMetric.update's _cast_and_nan_check_input evaluates two `.any()` results as
        # Python bools = 2 forced GPU->CPU syncs per call, and we update every micro-batch.
        # Accumulate the weighted sums directly: same math, no nan guard, no syncs.
        # (Do NOT use nan_strategy="disable" instead — torchmetrics 1.9 drops the weight there.)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.mean_value.device)
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32, device=value.device)
        weight = torch.broadcast_to(weight, value.shape)
        self.mean_value += (value.float() * weight.float()).sum()
        self.weight += weight.float().sum()


class PPLMetric(NLLMetric):
    """Token-level perplexity = exp(mean NLL)."""

    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)
