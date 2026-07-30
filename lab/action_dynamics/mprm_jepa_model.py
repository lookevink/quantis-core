"""Frozen mean-preserving residual-mixture implementation for MPRM-JEPA."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from quantis_core.action_conditioned_dynamics import (
    ActionConditionedWindows,
    MixtureTrajectoryDistribution,
)
from quantis_core.mprm_jepa import (
    canonicalize_mixture_weights,
    mean_preserving_component_means,
)

from prototype_multi_hypothesis_jepa import (
    MultiHypothesisJepaPrototype,
    PrototypeConfig,
    _build_network,
)


class MeanPreservingResidualJepa(
    MultiHypothesisJepaPrototype  # type: ignore[misc]
):
    """Fit residual hypotheses whose transported mean equals a frozen anchor."""

    kind = "mean_preserving_residual_mixture_jepa_v1"

    def fit_anchored(
        self,
        windows: ActionConditionedWindows,
        anchor_mean: NDArray[Any],
    ) -> "MeanPreservingResidualJepa":
        """Fit the exact frozen residual objective on fit-role evidence."""

        import torch

        anchor = np.asarray(anchor_mean, dtype=np.float32)
        if anchor.shape != windows.future_states.shape:
            raise ValueError("MPRM-JEPA fit anchor shape differs")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        shape = (
            windows.histories.shape[2],
            windows.histories.shape[3],
            windows.future_states.shape[1],
            windows.future_controls.shape[2],
            windows.future_actions.shape[3],
        )
        self._shape = shape
        self._network = _build_network(torch, self.config, shape)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self._network.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed)
        metrics = []
        for epoch in range(self.config.epochs):
            order = generator.permutation(len(windows.histories))
            totals = {
                "total": 0.0,
                "negative_log_likelihood": 0.0,
                "latent": 0.0,
                "target_reconstruction": 0.0,
                "context_reconstruction": 0.0,
            }
            batches = 0
            self._network.train()
            for start in range(0, len(order), self.config.batch_size):
                selection = order[start : start + self.config.batch_size]
                histories = torch.as_tensor(
                    windows.histories[selection], dtype=torch.float32
                )
                controls = torch.as_tensor(
                    windows.future_controls[selection], dtype=torch.float32
                )
                actions = torch.as_tensor(
                    windows.future_actions[selection], dtype=torch.float32
                )
                future = torch.as_tensor(
                    windows.future_states[selection], dtype=torch.float32
                )
                batch_anchor = torch.as_tensor(anchor[selection])
                optimizer.zero_grad(set_to_none=True)
                output = dict(
                    self._network.predict(histories, controls, actions)
                )
                weight = output["weight"]
                residual_mean = output["component_mean"]
                weighted_residual = torch.sum(
                    weight[:, :, None, None, None] * residual_mean,
                    dim=1,
                    keepdim=True,
                )
                output["component_mean"] = (
                    batch_anchor[:, None]
                    + residual_mean
                    - weighted_residual
                )
                residual_target = future - batch_anchor
                with torch.no_grad():
                    output["target_latent"] = torch.nn.functional.gelu(
                        self._network.target_encoder(residual_target)
                    )
                output["target_reconstruction"] = self._network.decoder(
                    output["target_latent"]
                )
                output["context_reconstruction"] = self._network.decoder(
                    self._network.encode_states(histories)[:, -1]
                )
                losses = _anchored_losses(
                    torch,
                    output,
                    histories,
                    future,
                    residual_target,
                    self.config,
                )
                losses["total"].backward()
                optimizer.step()
                if self.config.objective == "jepa":
                    self._network.update_target(self.config.ema_decay)
                for name in totals:
                    totals[name] += float(losses[name].detach())
                batches += 1
            row = {
                name: value / float(batches)
                for name, value in totals.items()
            }
            row["epoch"] = float(epoch + 1)
            metrics.append(row)
        self.training_metrics = tuple(metrics)
        return self

    def rollout_anchored(
        self,
        histories: NDArray[Any],
        future_controls: NDArray[Any],
        future_actions: NDArray[Any],
        anchor_mean: NDArray[Any],
    ) -> MixtureTrajectoryDistribution:
        """Return the canonical float64 mean-preserving mixture."""

        import torch

        network, shape = self._fitted()
        values = np.asarray(histories, dtype=np.float32)
        controls = np.asarray(future_controls, dtype=np.float32)
        actions = np.asarray(future_actions, dtype=np.float32)
        anchor = np.asarray(anchor_mean, dtype=np.float64)
        if (
            values.ndim != 4
            or values.shape[2:] != shape[:2]
            or controls.shape != (len(values), shape[2], shape[3])
            or actions.shape
            != (len(values), shape[2], shape[0], shape[4])
            or anchor.shape
            != (len(values), shape[2], shape[0], shape[1])
        ):
            raise ValueError("MPRM-JEPA rollout inputs are invalid")
        residuals = []
        variances = []
        weights = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                end = start + self.config.batch_size
                output = network.predict(
                    torch.as_tensor(values[start:end]),
                    torch.as_tensor(controls[start:end]),
                    torch.as_tensor(actions[start:end]),
                )
                residuals.append(output["component_mean"].detach().numpy())
                variances.append(
                    output["component_variance"].detach().numpy()
                )
                weights.append(output["weight"].detach().numpy())
        canonical_weight = canonicalize_mixture_weights(
            np.concatenate(weights), floor=1e-9
        )
        component_mean = mean_preserving_component_means(
            anchor,
            np.concatenate(residuals).astype(np.float64, copy=False),
            canonical_weight,
        )
        return MixtureTrajectoryDistribution(
            component_mean=component_mean,
            component_variance=np.concatenate(variances).astype(
                np.float64, copy=False
            ),
            weight=canonical_weight,
        )


def _anchored_losses(
    torch: Any,
    output: Mapping[str, Any],
    histories: Any,
    future: Any,
    residual_target: Any,
    config: PrototypeConfig,
) -> Mapping[str, Any]:
    mean = output["component_mean"]
    variance = output["component_variance"]
    weight = output["weight"]
    terms = -0.5 * (
        torch.square(future[:, None] - mean) / variance
        + torch.log(variance)
        + math.log(2.0 * math.pi)
    )
    log_joint = torch.log(weight) + torch.sum(
        terms, dim=(2, 3, 4)
    )
    coordinate_count = int(np.prod(future.shape[1:]))
    nll = torch.mean(
        -torch.logsumexp(log_joint, dim=1) / coordinate_count
    )
    responsibilities = torch.softmax(log_joint, dim=1).detach()
    latent_error = torch.mean(
        torch.abs(
            output["component_latent"]
            - output["target_latent"][:, None]
        ),
        dim=(2, 3, 4),
    )
    latent = torch.mean(
        torch.sum(responsibilities * latent_error, dim=1)
    )
    target_reconstruction = torch.mean(
        torch.square(
            output["target_reconstruction"] - residual_target
        )
    )
    context_reconstruction = torch.mean(
        torch.square(
            output["context_reconstruction"] - histories[:, -1]
        )
    )
    total = nll
    if config.objective == "jepa":
        total = (
            total
            + config.latent_weight * latent
            + config.target_reconstruction_weight
            * target_reconstruction
            + config.context_reconstruction_weight
            * context_reconstruction
        )
    return {
        "total": total,
        "negative_log_likelihood": nll,
        "latent": latent,
        "target_reconstruction": target_reconstruction,
        "context_reconstruction": context_reconstruction,
    }
