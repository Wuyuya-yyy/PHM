"""Transfer-learning interfaces for shared degradation representations."""

from __future__ import annotations

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer used by DANN-style domain adaptation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class DomainClassifier(nn.Module):
    """Classify latent vectors by domain for adversarial alignment."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32, num_domains: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.classifier(latent)


class DANNAdapter(nn.Module):
    """Prototype DANN interface with gradient reversal and a domain classifier."""

    def __init__(self, latent_dim: int, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.domain_classifier = DomainClassifier(latent_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        reversed_latent = GradientReversal.apply(latent, self.lambda_)
        return self.domain_classifier(reversed_latent)


class CORALLoss(nn.Module):
    """CORAL loss for second-order source-target alignment."""

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_cov = self._covariance(source)
        target_cov = self._covariance(target)
        dim = source.shape[1]
        return torch.mean((source_cov - target_cov) ** 2) / (4 * dim * dim)

    @staticmethod
    def _covariance(features: torch.Tensor) -> torch.Tensor:
        centered = features - features.mean(dim=0, keepdim=True)
        return centered.t().matmul(centered) / max(features.shape[0] - 1, 1)


class MMDLoss(nn.Module):
    """RBF-kernel MMD loss for distribution alignment."""

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5) -> None:
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernels = self._gaussian_kernel(source, target)
        n_source = source.shape[0]
        xx = kernels[:n_source, :n_source].mean()
        yy = kernels[n_source:, n_source:].mean()
        xy = kernels[:n_source, n_source:].mean()
        yx = kernels[n_source:, :n_source].mean()
        return xx + yy - xy - yx

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.shape[0], total.shape[0], total.shape[1])
        total1 = total.unsqueeze(1).expand(total.shape[0], total.shape[0], total.shape[1])
        distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(distance.detach()) / max(total.shape[0] ** 2 - total.shape[0], 1)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-distance / bw) for bw in bandwidths)
