from __future__ import annotations

import math
from typing import List, Tuple


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise ImportError(
            "PyTorch is required for Bayesian layers. "
            "Install with `pip install torch`."
        ) from exc
    return torch, nn, F


class BayesianLinear:
    """Factory that returns a torch.nn.Module when torch is available."""

    @staticmethod
    def build(in_features: int, out_features: int, prior_sigma: float = 1.0, bias: bool = True):
        torch, nn, F = _require_torch()

        class _BayesianLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.prior_sigma = prior_sigma
                self.prior_log_sigma = math.log(prior_sigma)

                self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
                self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
                if bias:
                    self.bias_mu = nn.Parameter(torch.empty(out_features))
                    self.bias_rho = nn.Parameter(torch.empty(out_features))
                else:
                    self.register_parameter("bias_mu", None)
                    self.register_parameter("bias_rho", None)

                self._reset_parameters()

            def _reset_parameters(self):
                nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
                # rho init -5 → softplus(-5) ≈ 0.0067 — near-deterministic start
                nn.init.constant_(self.weight_rho, -5.0)
                if self.bias_mu is not None:
                    fan_in = self.in_features
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(self.bias_mu, -bound, bound)
                    nn.init.constant_(self.bias_rho, -5.0)

            @property
            def weight_sigma(self):
                return F.softplus(self.weight_rho)

            @property
            def bias_sigma(self):
                if self.bias_rho is None:
                    return None
                return F.softplus(self.bias_rho)

            def forward(self, x):
                w_sigma = self.weight_sigma
                w = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
                if self.bias_mu is not None:
                    b_sigma = self.bias_sigma
                    b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
                else:
                    b = None
                return F.linear(x, w, b)

            def kl(self) -> torch.Tensor:
                """Closed-form KL(q || N(0, prior_sigma²)) summed over all params."""
                prior_s = self.prior_sigma
                prior_log_s = self.prior_log_sigma

                def _kl_gauss(mu, sigma):
                    # KL = sum [ log(prior/sigma) + (sigma² + mu²)/(2*prior²) - 0.5 ]
                    return (
                        prior_log_s
                        - sigma.log()
                        + (sigma.pow(2) + mu.pow(2)) / (2.0 * prior_s ** 2)
                        - 0.5
                    ).sum()

                kl_val = _kl_gauss(self.weight_mu, self.weight_sigma)
                if self.bias_mu is not None:
                    kl_val = kl_val + _kl_gauss(self.bias_mu, self.bias_sigma)
                return kl_val

        return _BayesianLinear()


class BayesianMLP:
    """Factory: heteroscedastic Bayesian MLP outputting (mean, log_var)."""

    @staticmethod
    def build(
        in_features: int,
        hidden_dims: List[int],
        prior_sigma: float = 1.0,
        dropout: float = 0.0,
    ):
        torch, nn, F = _require_torch()

        bayes_layers_modules = []
        dims = [in_features] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            bayes_layers_modules.append(BayesianLinear.build(d_in, d_out, prior_sigma=prior_sigma))

        # deterministic output heads (mean + log-variance)
        out_mu = nn.Linear(hidden_dims[-1], 1)
        out_logs = nn.Linear(hidden_dims[-1], 1)

        class _BayesianMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.bayes_layers = nn.ModuleList(bayes_layers_modules)
                self.out_mu = out_mu
                self.out_logs = out_logs
                self.dropout_p = dropout

            def forward(self, x) -> Tuple:
                for layer in self.bayes_layers:
                    x = F.relu(layer(x))
                    if self.dropout_p > 0 and self.training:
                        x = F.dropout(x, p=self.dropout_p)
                mu = self.out_mu(x).view(-1)
                log_var = self.out_logs(x).view(-1).clamp(-10.0, 10.0)
                return mu, log_var

            def kl(self) -> torch.Tensor:
                return sum(layer.kl() for layer in self.bayes_layers)

        return _BayesianMLP()


def elbo_loss(
    mu,
    log_var,
    y,
    kl: "torch.Tensor",
    n_batches: int,
    kl_weight: float = 1.0,
    aleatoric: bool = True,
) -> "torch.Tensor":
    """ELBO minibatch estimator (Blundell et al. 2015, uniform 1/M weighting).

    Per batch: NLL_batch_mean + beta * KL / M, M = n_batches per epoch.
    Summed over an epoch the KL term contributes exactly KL once.
    NLL is a per-example mean; KL/M is the per-batch share of the dataset KL.
    """
    torch, _, F = _require_torch()
    if aleatoric:
        nll = 0.5 * (torch.exp(-log_var) * (y - mu).pow(2) + log_var).mean()
    else:
        nll = F.mse_loss(mu, y)
    return nll + kl_weight * kl / max(n_batches, 1)
