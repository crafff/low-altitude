"""Shared-policy attention and PPO; collection/checkpoints live in the runner.

Paper-derived settings and explicit reconstruction choices are recorded in
configs/paper_ppo.json. Tests and training must be launched by tools/lab.py.
CPU with one intra-op thread remains the default. Explicit CUDA selection and
up to four intra-op threads support the controller's bounded speed benchmark.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import operator
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


DEFAULTS = {
    "device": "cpu", "torch_num_threads": 1,
    "own_dim": 7, "intruder_dim": 10, "action_dim": 60,
    "attention_dim": 64, "hidden_sizes": [128, 128],
    "learning_rate": 1e-5, "gamma": .99, "gae_lambda": .95,
    "clip_epsilon": .2, "value_coefficient": .5,
    "entropy_coefficient": 1e-4, "update_epochs": 1,
    "minibatch_size": 64, "normalize_advantages": True,
    "advantage_normalization_epsilon": 1e-8, "max_grad_norm": .5,
    "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0.,
}


def _device(value: str | torch.device) -> torch.device:
    if not isinstance(value, (str, torch.device)):
        raise ValueError("device must be 'cpu', 'cuda', or an indexed CUDA device")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be 'cpu', 'cuda', or an indexed CUDA device") from error
    if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
        raise ValueError("device must be 'cpu', 'cuda', or an indexed CUDA device")
    return device


def _config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result = {**DEFAULTS, **cfg}
    result["device"] = str(_device(result["device"]))
    threads = result["torch_num_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
        raise ValueError("torch_num_threads must be an integer in [1, 4]")
    for key, expected in (("own_dim", 7), ("intruder_dim", 10),
                          ("action_dim", 60)):
        if result[key] != expected:
            raise ValueError(f"{key} must be {expected!r}")
    for key in ("attention_dim", "update_epochs", "minibatch_size"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    hidden = result["hidden_sizes"]
    if len(hidden) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in hidden):
        raise ValueError("hidden_sizes must contain two positive integers")
    for key in ("learning_rate", "advantage_normalization_epsilon", "max_grad_norm", "adam_eps"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("value_coefficient", "entropy_coefficient", "weight_decay"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("gamma", "gae_lambda"):
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not math.isfinite(result["clip_epsilon"]) or not 0 < result["clip_epsilon"] < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not isinstance(result["normalize_advantages"], bool):
        raise ValueError("normalize_advantages must be boolean")
    betas = result["adam_betas"]
    if len(betas) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in betas):
        raise ValueError("adam_betas must contain two values in [0, 1)")
    return result


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("rollout inputs must be on CPU")
        value = value.detach().numpy()
    return np.asarray(value)


def collate(observations: Sequence[Mapping[str, Any]],
            device: str | torch.device = "cpu") -> tuple[torch.Tensor, ...]:
    """Validate CPU rollout data, then copy to the requested forward device.

    Neighbor padding has N >= 1 even for an entirely neighbor-free batch.
    The boolean action mask is required; it is never inferred or recomputed.
    An unavailable requested CUDA device raises; there is no CPU fallback.
    """
    device = _device(device)
    if not observations:
        raise ValueError("cannot collate an empty observation batch")
    own_rows, neighbor_rows, mask_rows = [], [], []
    for obs in observations:
        own = np.array(_numpy(obs["own"]), dtype=np.float32, copy=True)
        neighbors = np.array(_numpy(obs["intruders"]), dtype=np.float32, copy=True)
        mask = _numpy(obs["action_mask"])
        if own.shape != (7,) or neighbors.ndim != 2 or neighbors.shape[1] != 10:
            raise ValueError("observations require own[7] and intruders[N, 10]")
        if mask.shape != (60,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("action_mask must be bool[60] with at least one valid action")
        if not np.isfinite(own).all() or not np.isfinite(neighbors).all():
            raise ValueError("observation features must be finite")
        own_rows.append(own)
        neighbor_rows.append(neighbors)
        mask_rows.append(mask.copy())
    count = len(observations)
    width = max(1, max(len(row) for row in neighbor_rows))
    padded = np.zeros((count, width, 10), dtype=np.float32)
    present = np.zeros((count, width), dtype=np.bool_)
    for i, neighbors in enumerate(neighbor_rows):
        padded[i, :len(neighbors)] = neighbors
        present[i, :len(neighbors)] = True
    tensors = (torch.from_numpy(np.stack(own_rows)), torch.from_numpy(padded),
               torch.from_numpy(present), torch.from_numpy(np.stack(mask_rows)))
    return tuple(tensor.to(device=device) for tensor in tensors)


class SharedActorCritic(nn.Module):
    """One current-neighbor query and a shared actor/critic backbone.

    Initialization always consumes the CPU RNG before any device transfer, so
    identical CPU seeds give identical starting weights for CPU and CUDA.
    Construction sets the process-wide PyTorch intra-op thread count; callers
    comparing thread settings must construct and measure models sequentially.
    """

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.cfg = _config(cfg)
        torch.set_num_threads(self.cfg["torch_num_threads"])
        dim = self.cfg["attention_dim"]
        factory = {"device": torch.device("cpu"), "dtype": torch.float32}
        self.query = nn.Linear(7, dim, bias=False, **factory)
        self.key = nn.Linear(10, dim, bias=False, **factory)
        self.value = nn.Linear(10, dim, bias=False, **factory)
        first, second = self.cfg["hidden_sizes"]
        self.shared = nn.Sequential(nn.Linear(7 + dim, first, **factory), nn.Tanh(),
                                    nn.Linear(first, second, **factory), nn.Tanh())
        self.actor = nn.Linear(second, 60, **factory)
        self.critic = nn.Linear(second, 1, **factory)
        self._attention_scale = math.sqrt(dim)
        if self.cfg["device"] != "cpu":
            self.to(device=_device(self.cfg["device"]))

    def forward(self, own: torch.Tensor, intruders: torch.Tensor,
                intruder_mask: torch.Tensor, action_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = (own, intruders, intruder_mask, action_mask)
        device = self.query.weight.device
        if device.type not in ("cpu", "cuda") or any(p.device != device for p in self.parameters()):
            raise ValueError("SharedActorCritic parameters must share one CPU or CUDA device")
        if any(t.device != device for t in tensors):
            raise ValueError("forward tensors must match the model device")
        if own.dtype != torch.float32 or intruders.dtype != torch.float32:
            raise ValueError("own and intruders must be float32")
        if own.ndim != 2 or own.shape[1] != 7 or own.shape[0] < 1:
            raise ValueError("own must have shape [B, 7], B >= 1")
        batch = own.shape[0]
        if intruders.ndim != 3 or intruders.shape[0] != batch or intruders.shape[1] < 1 or intruders.shape[2] != 10:
            raise ValueError("intruders must have shape [B, N, 10], N >= 1")
        if intruder_mask.shape != intruders.shape[:2] or intruder_mask.dtype != torch.bool:
            raise ValueError("intruder_mask must have shape [B, N] and boolean dtype")
        if action_mask.shape != (batch, 60) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must have shape [B, 60] and boolean dtype")
        if not bool(action_mask.any(dim=-1).all()):
            raise ValueError("each row must have at least one valid action")
        _finite(own, "own")
        # Remove padding before the projections too: even NaN padding cannot leak.
        neighbors = intruders.masked_fill(~intruder_mask.unsqueeze(-1), 0.)
        _finite(neighbors, "unmasked intruders")
        keys, values = self.key(neighbors), self.value(neighbors)
        _finite(values, "attention values")
        scores = (keys * self.query(own).unsqueeze(1)).sum(dim=-1) / self._attention_scale
        _finite(scores, "attention scores")
        scores = scores.masked_fill(~intruder_mask, -torch.inf)
        # All-masked rows cannot be passed to softmax as all -inf.
        scores = torch.where(intruder_mask.any(dim=-1, keepdim=True), scores,
                             torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).masked_fill(~intruder_mask, 0.)
        context = (weights.unsqueeze(-1) * values).sum(dim=1)
        _finite(context, "attention context")
        hidden = self.shared(torch.cat((own, context), dim=-1))
        logits, prediction = self.actor(hidden), self.critic(hidden).squeeze(-1)
        _finite(logits, "unmasked policy logits")
        _finite(prediction, "value predictions")
        return logits.masked_fill(~action_mask, -torch.inf), prediction


def compute_gae(rewards: Any, values: Any, terminated: Any,
                bootstrap_value: Any, gamma: float = .99, gae_lambda: float = .95
                ) -> tuple[np.ndarray, np.ndarray]:
    """Detached GAE/returns for ONE contiguous aircraft trajectory, float32.

    terminated[t] describes the transition after rewards[t]. A terminal flag
    may occur only at the final transition: arrival/deadline failure forces
    zero bootstrap. For a live collector cutoff, pass terminated[-1]=False and
    the final next-observation value. Do not concatenate different aircraft.
    """
    reward = np.array(_numpy(rewards), dtype=np.float64, copy=True)
    value = np.array(_numpy(values), dtype=np.float64, copy=True)
    terminal = _numpy(terminated)
    bootstrap = _numpy(bootstrap_value)
    if reward.ndim != 1 or value.shape != reward.shape or terminal.shape != reward.shape:
        raise ValueError("GAE inputs must be matching one-dimensional trajectories")
    if terminal.dtype != np.bool_:
        raise ValueError("terminated must contain booleans")
    if terminal[:-1].any():
        raise ValueError("a contiguous aircraft trajectory cannot continue after termination")
    if bootstrap.ndim != 0 or not np.isfinite(bootstrap):
        raise ValueError("bootstrap_value must be a finite scalar")
    if not np.isfinite(reward).all() or not np.isfinite(value).all():
        raise ValueError("rewards and rollout values must be finite")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    advantage = np.empty_like(reward)
    following_value, following_advantage = float(bootstrap), 0.
    for t in range(len(reward) - 1, -1, -1):
        continuation = 0. if terminal[t] else 1.
        delta = reward[t] + gamma * following_value * continuation - value[t]
        following_advantage = delta + gamma * gae_lambda * continuation * following_advantage
        advantage[t] = following_advantage
        following_value = value[t]
    returns = (advantage + value).astype(np.float32)
    advantage = advantage.astype(np.float32)
    if not np.isfinite(advantage).all() or not np.isfinite(returns).all():
        raise ValueError("computed GAE/returns exceed finite float32 range")
    return advantage, returns


def _sample_tensor(samples: Sequence[Mapping[str, Any]], field: str,
                   device: torch.device) -> torch.Tensor:
    result = []
    for sample in samples:
        value = _numpy(sample[field])
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError(f"sample {field} must be a finite scalar")
        result.append(float(value))
    tensor = torch.tensor(result, dtype=torch.float32, device=device)
    _finite(tensor, field)
    return tensor


def update(model: SharedActorCritic, optimizer: torch.optim.Optimizer,
           samples: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any],
           generator: torch.Generator) -> dict[str, float | int]:
    """Perform actual minibatch PPO updates using detached, stored rollout data.

    Required sample fields: own, intruders, action_mask, action, old_log_prob,
    advantage, return. Diagnostics average each sample visit before its step;
    gradient_norm is the global norm before clipping, gradient_norm_after is
    after clipping. The caller constructs Adam from the same configuration.
    Stored rollout data and shuffle RNG remain on CPU; update tensors follow
    the configured model device. Finite checks and scalar diagnostics retain
    their synchronization costs on CUDA.
    """
    options = _config(cfg)
    if not samples:
        raise ValueError("PPO requires at least one sample")
    if generator.device.type != "cpu":
        raise ValueError("PPO shuffle generator must be on CPU")
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("paper-like PPO requires Adam")
    parameters = list(model.parameters())
    device, requested = parameters[0].device, _device(options["device"])
    if (device.type != requested.type
            or (requested.index is not None and device.index != requested.index)
            or any(p.device != device for p in parameters)):
        raise ValueError("PPO configuration device must match every model parameter")
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {id(p) for p in parameters}:
        raise ValueError("Adam must optimize the complete shared actor-critic exactly once")
    for group in optimizer.param_groups:
        expected = {"lr": options["learning_rate"], "eps": options["adam_eps"],
                    "weight_decay": options["weight_decay"]}
        if any(group[key] != value for key, value in expected.items()) or tuple(group["betas"]) != tuple(options["adam_betas"]):
            raise ValueError("Adam hyperparameters must match the PPO configuration")
    observations = collate(samples, device=device)
    chosen = []
    for sample in samples:
        raw = _numpy(sample["action"])
        if raw.ndim != 0 or raw.dtype.kind not in "iu":
            raise ValueError("sample action must be an integer index")
        chosen.append(operator.index(raw.item()))
    actions = torch.tensor(chosen, dtype=torch.long, device=device)
    if not bool(((actions >= 0) & (actions < 60)).all()):
        raise ValueError("sample action index is outside [0, 60)")
    if not bool(observations[3].gather(1, actions[:, None]).all()):
        raise ValueError("a sampled action is invalid under its stored action mask")
    old_log_prob = _sample_tensor(samples, "old_log_prob", device)
    if bool((old_log_prob > 1e-6).any()):
        raise ValueError("stored action log probabilities cannot be positive")
    advantage = _sample_tensor(samples, "advantage", device)
    returns = _sample_tensor(samples, "return", device)
    advantage_mean, advantage_std = advantage.mean(), advantage.std(unbiased=False)
    if options["normalize_advantages"]:
        advantage = (advantage - advantage_mean) / (advantage_std + options["advantage_normalization_epsilon"])
    metrics = {key: 0. for key in ("loss", "policy_loss", "value_loss", "entropy",
                                  "approx_kl", "clip_fraction", "gradient_norm", "gradient_norm_after")}
    visits, batches = 0, 0
    model.train()
    for _ in range(options["update_epochs"]):
        order = torch.randperm(len(samples), generator=generator, device="cpu")
        for indexes in order.split(options["minibatch_size"]):
            indexes = indexes.to(device=device)
            logits, predictions = model(*(field[indexes] for field in observations))
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(actions[indexes])
            log_ratio = log_prob - old_log_prob[indexes]
            ratio = log_ratio.exp()
            _finite(ratio, "PPO probability ratios")
            epsilon = options["clip_epsilon"]
            surrogate = torch.minimum(ratio * advantage[indexes],
                                      ratio.clamp(1. - epsilon, 1. + epsilon) * advantage[indexes])
            policy_loss = -surrogate.mean()
            value_loss = (predictions - returns[indexes]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + options["value_coefficient"] * value_loss - options["entropy_coefficient"] * entropy
            _finite(loss, "PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"], error_if_nonfinite=True)
            after = torch.stack([p.grad.detach().square().sum() for p in parameters if p.grad is not None]).sum().sqrt()
            with torch.no_grad():
                values_now = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                              "entropy": entropy, "approx_kl": ((ratio - 1.) - log_ratio).mean(),
                              "clip_fraction": ((ratio - 1.).abs() > epsilon).float().mean(),
                              "gradient_norm": norm, "gradient_norm_after": after}
                for key, value in values_now.items():
                    _finite(value, key)
                    metrics[key] += float(value) * len(indexes)
            optimizer.step()
            for parameter in parameters:
                _finite(parameter, "updated parameters")
            visits += len(indexes)
            batches += 1
    return {**{key: value / visits for key, value in metrics.items()},
            "samples": len(samples), "sample_visits": visits, "minibatches": batches,
            "epochs": options["update_epochs"], "advantage_mean": float(advantage_mean),
            "advantage_std": float(advantage_std)}
