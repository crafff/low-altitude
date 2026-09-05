"""Disposable CPU gradient/update diagnostic; checkpoint reads require lab isolation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.distributions import Categorical

from shared_ppo import SharedActorCritic, collate, update
from paper_train import (capture_rng, collect_episode, load_checkpoint, restore_rng,
                         _validate_train_config, _versions)


SCOPE = "execution-semantics PPO gradient diagnostic; no causal interference claim"


def identity(value):
    """Content identity for CPU tensors and primitive model/optimizer/RNG trees."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("diagnostic state must remain on CPU")
            digest.update(json.dumps(["tensor", str(item.dtype), list(item.shape)]).encode())
            digest.update(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def gradient_geometry(first, second):
    first = torch.as_tensor(first, dtype=torch.float64, device="cpu").detach().reshape(-1)
    second = torch.as_tensor(second, dtype=torch.float64, device="cpu").detach().reshape(-1)
    if first.shape != second.shape or not bool(torch.isfinite(first).all() & torch.isfinite(second).all()):
        raise ValueError("gradient vectors must have equal size and finite entries")
    a, b = float(torch.linalg.vector_norm(first)), float(torch.linalg.vector_norm(second))
    dot = float(torch.dot(first, second))
    if not all(math.isfinite(value) for value in (a, b, dot)):
        raise ValueError("gradient geometry exceeds finite precision")
    cosine = None if a == 0. or b == 0. else max(-1., min(1., dot/a/b))
    return {"first_norm": a, "second_norm": b, "dot_product": dot, "cosine": cosine,
            "angle_degrees": None if cosine is None else math.degrees(math.acos(cosine)),
            "status": "undefined_zero_gradient" if cosine is None else "defined"}


def categorical_kl(old_probs, new_probs, action_mask):
    """Full categorical KL(old||new), including explicit zero-support semantics."""
    old = torch.as_tensor(old_probs, dtype=torch.float64, device="cpu").detach()
    new = torch.as_tensor(new_probs, dtype=torch.float64, device="cpu").detach()
    mask = torch.as_tensor(action_mask, device="cpu")
    if old.ndim != 2 or old.shape != new.shape or old.shape != mask.shape or len(old) == 0 or mask.dtype != torch.bool:
        raise ValueError("KL requires matching [B, A] probabilities and boolean action masks")
    if not bool(mask.any(dim=-1).all()) or not bool(torch.isfinite(old).all() & torch.isfinite(new).all()):
        raise ValueError("KL requires finite probabilities and nonempty legal support")
    if bool((old < 0).any() | (new < 0).any()) or bool((old[~mask] != 0).any() | (new[~mask] != 0).any()):
        raise ValueError("probabilities must be nonnegative and exactly zero outside the stored mask")
    for probabilities in (old, new):
        if not torch.allclose(probabilities.sum(dim=-1), torch.ones(len(old), dtype=torch.float64), rtol=1e-7, atol=1e-9):
            raise ValueError("probability rows must sum to one")
    lost = ((old > 0) & (new == 0)).any(dim=-1)
    terms = torch.zeros_like(old)
    positive = (old > 0) & (new > 0)
    terms[positive] = old[positive]*(old[positive].log()-new[positive].log())
    per_row = terms.sum(dim=-1)
    finite_rows = per_row[~lost]
    if bool((finite_rows < -1e-10).any()):
        raise ValueError("negative KL exceeds floating-point tolerance")
    finite_rows = finite_rows.clamp_min(0.)
    infinite = int(lost.sum())
    return {"direction": "KL(old||new)", "weighting": "uniform over collected aircraft-decision samples",
            "rows": len(old), "infinite_rows": infinite,
            "mean": None if infinite else float(finite_rows.mean()),
            "max": None if infinite else float(finite_rows.max()),
            "finite_rows_mean": None if len(finite_rows) == 0 else float(finite_rows.mean()),
            "status": "infinite_support_loss" if infinite else "finite"}


def explained_variance(predictions, returns):
    predicted = torch.as_tensor(predictions, dtype=torch.float64, device="cpu").detach()
    targets = torch.as_tensor(returns, dtype=torch.float64, device="cpu").detach()
    if predicted.ndim != 1 or predicted.shape != targets.shape or len(targets) == 0:
        raise ValueError("explained variance requires matching nonempty vectors")
    if not bool(torch.isfinite(predicted).all() & torch.isfinite(targets).all()):
        raise ValueError("explained variance requires finite values")
    variance = float(targets.var(unbiased=False))
    return {"value": None if variance == 0. else 1.-float((targets-predicted).var(unbiased=False))/variance,
            "return_variance": variance, "status": "zero_return_variance" if variance == 0. else "defined"}


def _prepare(samples, cfg):
    observations = collate(samples)
    actions = torch.tensor([row["action"] for row in samples], dtype=torch.long, device="cpu")
    old = torch.tensor([row["old_log_prob"] for row in samples], dtype=torch.float32, device="cpu")
    raw_advantage = torch.tensor([row["advantage"] for row in samples], dtype=torch.float32, device="cpu")
    targets = torch.tensor([row["return"] for row in samples], dtype=torch.float32, device="cpu")
    mean, std = raw_advantage.mean(), raw_advantage.std(unbiased=False)
    advantage = ((raw_advantage-mean)/(std+cfg["advantage_normalization_epsilon"])
                 if cfg["normalize_advantages"] else raw_advantage)
    return observations, actions, old, advantage, targets, float(mean), float(std)


def measure_gradients(model, prepared, indexes, cfg, *, gradient_capture=None):
    """Static loss geometry on a disposable copy; no optimizer step or hook."""
    observations, actions, old, advantage, targets, _, _ = prepared
    parameters = list(model.named_parameters())
    shared_indexes = [i for i, (name, _) in enumerate(parameters)
                      if name.startswith(("query.", "key.", "value.", "shared."))]
    if not shared_indexes or any(not name.startswith(("query.", "key.", "value.", "shared.", "actor.", "critic.")) for name, _ in parameters):
        raise ValueError("unexpected shared-policy parameter partition")
    weights = [parameter for _, parameter in parameters]
    logits, prediction = model(*(field[indexes] for field in observations))
    distribution = Categorical(logits=logits)
    log_ratio = distribution.log_prob(actions[indexes])-old[indexes]
    ratio = log_ratio.exp()
    epsilon = cfg["clip_epsilon"]
    policy_loss = -torch.minimum(ratio*advantage[indexes],
                                ratio.clamp(1.-epsilon, 1.+epsilon)*advantage[indexes]).mean()
    value_loss = (prediction-targets[indexes]).square().mean()
    entropy = distribution.entropy().mean()
    terms = {"policy_surrogate": policy_loss, "entropy_bonus": -cfg["entropy_coefficient"]*entropy,
             "weighted_critic": cfg["value_coefficient"]*value_loss}
    combined = policy_loss+cfg["value_coefficient"]*value_loss-cfg["entropy_coefficient"]*entropy
    gradients = {}
    for name, loss in [*terms.items(), ("combined", combined)]:
        values = torch.autograd.grad(loss, weights, retain_graph=name!="combined", allow_unused=True)
        gradients[name] = [torch.zeros_like(weight) if value is None else value.detach()
                           for weight, value in zip(weights, values)]
    summed = [sum(gradients[name][i] for name in terms) for i in range(len(weights))]
    for component_sum, direct in zip(summed, gradients["combined"]):
        torch.testing.assert_close(component_sum, direct, rtol=1e-4, atol=1e-6)
    if gradient_capture is not None:
        gradient_capture.update({name: {parameter_name: gradient.clone()
                                       for (parameter_name, _), gradient in zip(parameters, values)}
                                 for name, values in gradients.items()})
    def shared_vector(values):
        return torch.cat([values[i].reshape(-1).double() for i in shared_indexes])
    vectors = {name: shared_vector(values) for name, values in gradients.items()}
    vectors["actor_total"] = vectors["policy_surrogate"]+vectors["entropy_bonus"]
    for parameter, gradient in zip(weights, gradients["combined"]):
        parameter.grad = gradient.clone()
    norm = torch.nn.utils.clip_grad_norm_(weights, cfg["max_grad_norm"], error_if_nonfinite=True)
    norm_after = torch.stack([p.grad.square().sum() for p in weights]).sum().sqrt()
    shared_after = shared_vector([p.grad for p in weights])
    report = {"sample_count": len(indexes), "sample_indexes_sha256": identity(indexes),
        "shared_parameter_names": [parameters[i][0] for i in shared_indexes],
        "shared_parameter_count": sum(weights[i].numel() for i in shared_indexes),
        "shared_norms": {name: float(torch.linalg.vector_norm(vector)) for name, vector in vectors.items()},
        "actor_total_vs_weighted_critic": gradient_geometry(vectors["actor_total"], vectors["weighted_critic"]),
        "surrogate_vs_weighted_critic": gradient_geometry(vectors["policy_surrogate"], vectors["weighted_critic"]),
        "entropy_vs_weighted_critic": gradient_geometry(vectors["entropy_bonus"], vectors["weighted_critic"]),
        "shared_combined_norm_after_global_clip": float(torch.linalg.vector_norm(shared_after)),
        "gradient_norm": float(norm), "gradient_norm_after": float(norm_after),
        "global_clip_applied": float(norm) > cfg["max_grad_norm"],
        "global_clip_scale": None if float(norm) == 0. else float(norm_after)/float(norm),
        "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
        "weighted_value_loss": float(terms["weighted_critic"].detach()), "entropy": float(entropy.detach()),
        "weighted_entropy_loss": float(terms["entropy_bonus"].detach()), "loss": float(combined.detach()),
        "approx_kl": float(((ratio-1.)-log_ratio).mean().detach()),
        "clip_fraction": float(((ratio-1.).abs()>epsilon).float().mean()),
        "component_gradient_sum_assertion": "passed"}
    return report


def _predict(model, observations, batch_size):
    probabilities, values = [], []
    with torch.no_grad():
        for start in range(0, len(observations[0]), batch_size):
            logits, prediction = model(*(field[start:start+batch_size] for field in observations))
            probabilities.append(torch.softmax(logits.double(), dim=-1))
            values.append(prediction)
    return torch.cat(probabilities), torch.cat(values)


def _optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
                            eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])


def parameter_displacement(before, after, gradients, shared_names):
    """Actual parameter motion and first-order loss changes, in float64.

    Gradients are the original, unclipped loss gradients. A negative dot product
    predicts a decrease of that minimized loss to first order; it is not a
    counterfactual optimizer comparison or evidence of causal interference.
    """
    names = list(before)
    if not names or set(after) != set(names) or not shared_names or not set(shared_names) <= set(names):
        raise ValueError("displacement requires matching parameters and a nonempty shared subset")
    if any(set(values) != set(names) for values in gradients.values()):
        raise ValueError("component gradients must cover all parameters")
    for name in names:
        values = [before[name], after[name], *(component[name] for component in gradients.values())]
        if any(value.device.type != "cpu" or value.shape != before[name].shape
               or not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("displacement tensors must have matching shapes and finite CPU entries")
    # Convert each endpoint before subtracting to retain the actual float32
    # parameter change without rounding the subtraction back to float32.
    delta = {name: after[name].detach().double()-before[name].detach().double() for name in names}
    report = {}
    for partition, selected in (("shared", [name for name in names if name in shared_names]), ("all", names)):
        motion = torch.cat([delta[name].reshape(-1) for name in selected])
        projections = {component: float(torch.dot(torch.cat([values[name].detach().double().reshape(-1)
                                                             for name in selected]), motion))
                       for component, values in gradients.items()}
        projections["actor_total"] = projections["policy_surrogate"]+projections["entropy_bonus"]
        norm = float(torch.linalg.vector_norm(motion))
        if not all(math.isfinite(value) for value in (norm, *projections.values())):
            raise ValueError("displacement geometry exceeds finite precision")
        report[partition] = {"parameter_count": motion.numel(), "delta_theta_norm": norm,
                             "original_gradient_dot_delta_theta": projections}
    return report


def _update_with_first_step_capture(model, optimizer, samples, cfg, shuffle):
    """Run the original full update with a temporary, passive step observer.

    Only endpoint tensor copies occur inside the observer. It never evaluates
    losses, alters gradients, skips a step, or draws from a random stream.
    """
    original_step = optimizer.step
    had_instance_step = "step" in vars(optimizer)
    previous_instance_step = vars(optimizer).get("step")
    captured = {"step_count": 0}
    def snapshot():
        return {name: value.detach().clone() for name, value in model.state_dict().items()}
    def observed_step(*args, **kwargs):
        first = captured["step_count"] == 0
        if first:
            captured["before"] = snapshot()
        result = original_step(*args, **kwargs)
        if first:
            captured["after"] = snapshot()
        captured["step_count"] += 1
        return result
    optimizer.step = observed_step
    try:
        metrics = update(model, optimizer, samples, cfg, shuffle)
    finally:
        if had_instance_step:
            optimizer.step = previous_instance_step
        else:
            del optimizer.step
    if not captured["step_count"] or captured["step_count"] != metrics["minibatches"]:
        raise AssertionError("first-step observer count differs from original update")
    return metrics, captured


def _fixed_minibatch_objectives(model, prepared, indexes, cfg):
    """Re-evaluate original losses on fixed indices with full-rollout advantages.

    This repeats the original update's loss arithmetic solely for measurement;
    its before-step values are asserted against measure_gradients below.
    """
    observations, actions, old, advantage, targets, _, _ = prepared
    with torch.no_grad():
        logits, prediction = model(*(field[indexes] for field in observations))
        distribution = Categorical(logits=logits)
        ratio = (distribution.log_prob(actions[indexes])-old[indexes]).exp()
        epsilon = cfg["clip_epsilon"]
        policy = -torch.minimum(ratio*advantage[indexes],
                                ratio.clamp(1.-epsilon, 1.+epsilon)*advantage[indexes]).mean()
        value = (prediction-targets[indexes]).square().mean()
        entropy = distribution.entropy().mean()
        weighted_value, weighted_entropy = cfg["value_coefficient"]*value, -cfg["entropy_coefficient"]*entropy
        total = policy+cfg["value_coefficient"]*value-cfg["entropy_coefficient"]*entropy
        report = {"policy_loss": float(policy), "actor_surrogate_objective": float(-policy),
                  "value_loss": float(value), "weighted_value_loss": float(weighted_value),
                  "entropy": float(entropy), "weighted_entropy_loss": float(weighted_entropy),
                  "actor_total_loss": float(policy+weighted_entropy), "loss": float(total)}
        if not all(math.isfinite(value) for value in report.values()):
            raise ValueError("first-step objective measurement must remain finite")
        return report, torch.softmax(logits.double(), dim=-1)


def diagnose_samples(model, optimizer, samples, cfg, shuffle, *, prediction_batch_size=256, deadline=None):
    """Measure then invoke the unchanged original update on a separate copy."""
    def boundary():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("gradient probe budget exhausted between phases")
    before_model, before_optimizer = identity(model.state_dict()), identity(optimizer.state_dict())
    before_shuffle = identity(shuffle.get_state())
    prepared = _prepare(samples, cfg)
    observations, actions, old_logs, _, targets, advantage_mean, advantage_std = prepared
    old_probs, old_values = _predict(model, observations, prediction_batch_size)
    selected = old_probs.gather(1, actions[:, None]).squeeze(-1).log()
    torch.testing.assert_close(selected, old_logs.double(), rtol=1e-5, atol=5e-6)
    boundary()
    full = measure_gradients(copy.deepcopy(model), prepared, torch.arange(len(samples)), cfg)
    boundary()
    inspection_shuffle = torch.Generator(device="cpu")
    inspection_shuffle.set_state(shuffle.get_state())
    first_indexes = torch.randperm(len(samples), generator=inspection_shuffle)[:cfg["minibatch_size"]]
    first_gradients = {}
    first = measure_gradients(copy.deepcopy(model), prepared, first_indexes, cfg, gradient_capture=first_gradients)
    boundary()
    updated = copy.deepcopy(model)
    updated_optimizer = _optimizer(updated, cfg)
    updated_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    private_shuffle = torch.Generator(device="cpu")
    private_shuffle.set_state(shuffle.get_state())
    updated_before = identity(updated.state_dict())
    metrics = update(updated, updated_optimizer, samples, cfg, private_shuffle)
    boundary()
    new_probs, new_values = _predict(updated, observations, prediction_batch_size)
    expected_loss = metrics["policy_loss"]+cfg["value_coefficient"]*metrics["value_loss"]-cfg["entropy_coefficient"]*metrics["entropy"]
    if not math.isclose(metrics["loss"], expected_loss, rel_tol=2e-5, abs_tol=2e-6):
        raise AssertionError("original update aggregate loss decomposition differs")
    if (metrics["samples"] != len(samples) or metrics["sample_visits"] != len(samples)*cfg["update_epochs"]
            or metrics["minibatches"] != math.ceil(len(samples)/cfg["minibatch_size"])*cfg["update_epochs"]):
        raise AssertionError("original update sample accounting differs")
    if metrics["advantage_mean"] != advantage_mean or metrics["advantage_std"] != advantage_std:
        raise AssertionError("original update advantage normalization statistics differ")
    if metrics["gradient_norm_after"] > cfg["max_grad_norm"]+1e-6:
        raise AssertionError("original update exceeds the configured gradient cap")
    exact_fields = []
    if metrics["minibatches"] == 1:
        for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm", "gradient_norm_after"):
            if not math.isclose(first[key], metrics[key], rel_tol=2e-4, abs_tol=2e-6):
                raise AssertionError(f"first-minibatch measurement disagrees with original update: {key}")
            exact_fields.append(key)
    boundary()
    # An additional original model/Adam/RNG copy observes step 1 while still
    # executing the complete original update. The unobserved path above remains
    # the reference for native metrics and the final full-rollout policy KL.
    observed = copy.deepcopy(model)
    observed_optimizer = _optimizer(observed, cfg)
    observed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    observed_shuffle = torch.Generator(device="cpu")
    observed_shuffle.set_state(shuffle.get_state())
    observed_metrics, captured = _update_with_first_step_capture(observed, observed_optimizer, samples, cfg, observed_shuffle)
    if (observed_metrics != metrics or identity(observed.state_dict()) != identity(updated.state_dict())
            or identity(observed_optimizer.state_dict()) != identity(updated_optimizer.state_dict())
            or identity(observed_shuffle.get_state()) != identity(private_shuffle.get_state())):
        raise AssertionError("passive first-step observation changed the original complete update")
    if identity(captured["before"]) != before_model:
        raise AssertionError("first Adam step did not start at the original checkpoint model")
    boundary()
    objectives_before, probs_before = _fixed_minibatch_objectives(model, prepared, first_indexes, cfg)
    matched_fields = ("policy_loss", "value_loss", "weighted_value_loss", "entropy", "weighted_entropy_loss", "loss")
    for key in matched_fields:
        if objectives_before[key] != first[key]:
            raise AssertionError(f"first-step initial objective differs from original gradient measurement: {key}")
    # Reuse the observer's disposable model only after its final identity has
    # been compared; no parameters or optimizer state are promoted or saved.
    observed_final_sha = identity(observed.state_dict())
    observed.load_state_dict(captured["after"])
    objectives_after, probs_after = _fixed_minibatch_objectives(observed, prepared, first_indexes, cfg)
    parameter_names = dict(model.named_parameters())
    motion = parameter_displacement({name: captured["before"][name] for name in parameter_names},
                                    {name: captured["after"][name] for name in parameter_names},
                                    first_gradients, first["shared_parameter_names"])
    first_step = {"captured_step_ordinal": 1, "captured_step_samples": len(first_indexes),
        "complete_update_steps": captured["step_count"], "sample_indexes_sha256": identity(first_indexes),
        "advantage_normalization": "unchanged full-rollout statistics, never recomputed on the first minibatch",
        "full_rollout_advantage_mean": advantage_mean, "full_rollout_advantage_std": advantage_std,
        "first_minibatch_used_advantage_mean": float(prepared[3][first_indexes].mean()),
        "parameter_displacement": motion,
        "gradient_definitions": {"policy_surrogate": "gradient of minimized negative clipped actor surrogate",
            "entropy_bonus": "gradient of negative entropy coefficient times entropy",
            "actor_total": "policy_surrogate plus entropy_bonus",
            "weighted_critic": "gradient of value coefficient times mean squared value error",
            "combined": "gradient of original total loss before clipping"},
        "projection_interpretation": "Original unclipped gradient dot actual Adam delta; negative predicts first-order decrease of the named loss. No causal interference claim.",
        "fixed_first_minibatch_before": objectives_before, "fixed_first_minibatch_after": objectives_after,
        "fixed_first_minibatch_kl_after_one_step": categorical_kl(probs_before, probs_after, observations[3][first_indexes]),
        "model_before_step_sha256": identity(captured["before"]),
        "model_after_one_step_sha256": identity(captured["after"]),
        "model_after_complete_update_sha256": observed_final_sha,
        "consistency_assertions": {"original_complete_update_metrics_exact": "passed",
            "original_final_model_adam_shuffle_exact": "passed", "initial_objective_fields_exact": list(matched_fields),
            "temporary_step_wrapper_restored": "passed"},
        "limitation": "Displacement and fixed-minibatch losses describe Adam step 1 only. Existing full-batch post-update KL describes all complete_update_steps."}
    boundary()
    if (identity(model.state_dict()) != before_model or identity(optimizer.state_dict()) != before_optimizer
            or identity(shuffle.get_state()) != before_shuffle):
        raise AssertionError("diagnostic changed reference model, Adam or shuffle state")
    return {"full_rollout_static_gradients": full, "first_original_minibatch_static_gradients": first,
            "first_adam_step_displacement": first_step,
            "rollout_tensor_sha256": identity(prepared[:5]), "first_original_minibatch_indexes": first_indexes.tolist(),
            "original_update_metrics": metrics, "full_batch_post_update_kl": categorical_kl(old_probs, new_probs, observations[3]),
            "explained_variance_before": explained_variance(old_values, targets),
            "explained_variance_after": explained_variance(new_values, targets),
            "old_legal_action_probabilities_sha256": identity(old_probs),
            "old_legal_action_probabilities_storage": "retained in memory until full-support post-update KL is computed",
            "old_selected_log_probability_consistency": "passed within float32 batching tolerance",
            "reference_model_before_sha256": before_model, "reference_model_after_sha256": identity(model.state_dict()),
            "reference_optimizer_before_sha256": before_optimizer, "reference_optimizer_after_sha256": identity(optimizer.state_dict()),
            "disposable_model_before_sha256": updated_before, "disposable_model_after_sha256": identity(updated.state_dict()),
            "disposable_optimizer_after_sha256": identity(updated_optimizer.state_dict()),
            "private_shuffle_after_sha256": identity(private_shuffle.get_state()),
            "consistency_assertions": {"aggregate_loss_decomposition": "passed", "advantage_statistics": "passed",
                "sample_accounting": "passed", "gradient_cap": "passed", "single_minibatch_exact_fields": exact_fields,
                "limitation": "Multi-minibatch returned fields are visit-weighted averages at changing weights; static gradients are not asserted equal to those averages."}}


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Read-only checkpoint declared with lab --input")
    args = parser.parse_args(argv)
    if os.environ.get("LAB_RUN_DIR") != "/output" or Path.cwd() != Path("/workspace"):
        raise ValueError("checkpoint diagnostics must run through the isolated lab launcher")
    cfg = json.loads(Path(args.config).read_text())
    if cfg["schema"] != "ppo-gradient-probe.v1" or cfg["scope"] != SCOPE:
        raise ValueError("unsupported gradient diagnostic configuration")
    if cfg["device"] != "cpu" or cfg["torch_num_threads"] != 1 or cfg["torch_num_interop_threads"] != 1:
        raise ValueError("gradient diagnostic requires one CPU thread")
    if not 0 < cfg["wall_seconds"] <= 110 or not 0 < cfg["save_grace_seconds"] < cfg["wall_seconds"]:
        raise ValueError("inner wall budget must leave margin under the 120-second outer bound")
    if not 1 <= cfg["maximum_rollout_samples"] <= 10000 or not 1 <= cfg["prediction_batch_size"] <= 512:
        raise ValueError("invalid bounded diagnostic batch size")
    checkpoint = Path(args.checkpoint or cfg["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(Path("/workspace")) or not checkpoint.is_file():
        raise ValueError("checkpoint must be declared as a read-only project input")
    if not os.statvfs(checkpoint).f_flag & os.ST_RDONLY:
        raise ValueError("checkpoint must be on the read-only source mount")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from paper_environment import PaperEnvironment, load_environment_config
    training = json.loads(Path(cfg["training_config"]).read_text())
    _validate_train_config(training, training["episodes"], training["wall_seconds"], cfg["eval_limit"])
    environment, parts = load_environment_config(training["environment_config"])
    ppo = json.loads(Path(training["ppo_config"]).read_text())
    configs = {"training": training, "environment": environment, "environment_parts": parts,
               "aircraft_types": json.loads(Path(environment["types_config"]).read_text()), "ppo": ppo,
               "effective_development_cases": copy.deepcopy(training["development_cases"][:cfg["eval_limit"]])}
    versions = _versions()
    model = SharedActorCritic(ppo)
    optimizer = _optimizer(model, ppo)
    sampling, shuffle = torch.Generator(device="cpu"), torch.Generator(device="cpu")
    payload = load_checkpoint(checkpoint, model, optimizer, sampling_generator=sampling,
                              shuffle_generator=shuffle, configs=configs, versions=versions)
    if payload["completed_episodes"] != cfg["expected_completed_episodes"]:
        raise ValueError("checkpoint episode count differs from declared diagnostic")
    seed = training["training_scenario_seed_start"]+payload["next_seed_index"]
    if seed != cfg["expected_next_training_seed"] or seed in {case["seed"] for case in training["development_cases"]}:
        raise ValueError("probe must use the explicitly declared next training seed")
    restored_rng = capture_rng(sampling, shuffle)
    if identity(restored_rng) != identity(payload["rng"]):
        raise AssertionError("checkpoint RNG restoration differs")
    env = PaperEnvironment(environment, parts)
    restore_rng(restored_rng, sampling, shuffle)
    deadline = started+cfg["wall_seconds"]-cfg["save_grace_seconds"]
    print(json.dumps({"phase": "collecting_next_training_episode", "seed": seed, "scope": SCOPE}), flush=True)
    samples, episode_summary = collect_episode(env, model, sampling, ppo,
                                               observations=env.reset(seed), deadline=deadline)
    if len(samples) > cfg["maximum_rollout_samples"]:
        raise ValueError("collected rollout exceeds declared diagnostic sample cap")
    after_collection_rng = capture_rng(sampling, shuffle)
    if identity(shuffle.get_state()) != identity(restored_rng["shuffle"]):
        raise AssertionError("collection advanced the shuffle stream")
    print(json.dumps({"phase": "measuring_and_disposable_update", "samples": len(samples), "scope": SCOPE}), flush=True)
    diagnostic = diagnose_samples(model, optimizer, samples, ppo, shuffle,
                                  prediction_batch_size=cfg["prediction_batch_size"], deadline=deadline)
    if identity(capture_rng(sampling, shuffle)) != identity(after_collection_rng):
        raise AssertionError("measurement/update advanced the reference RNG streams")
    checkpoint_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_after != checkpoint_sha:
        raise AssertionError("source checkpoint changed")
    result = {"schema": "ppo-gradient-probe-result.v1", "scope": SCOPE, "state": "completed",
              "training_scope": training["scope"], "probe_config": cfg, "strict_science_configs": configs,
              "strict_science_versions": versions, "checkpoint_path": str(checkpoint.relative_to("/workspace")),
              "checkpoint_sha256_before": checkpoint_sha, "checkpoint_sha256_after": checkpoint_after,
              "checkpoint_completed_episodes": payload["completed_episodes"], "training_scenario_seed": seed,
              "next_seed_index": payload["next_seed_index"], "episode_summary": episode_summary,
              "restored_rng_sha256": identity(restored_rng), "after_collection_rng_sha256": identity(after_collection_rng),
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "wall_seconds": time.perf_counter()-started,
              "updated_checkpoint_saved": False, "diagnostic": diagnostic,
              "interpretation": "One fixed rollout and disposable update diagnose scale/gradient geometry, not causal critic interference or a justification for tuning. No baseline efficacy or containment claim."}
    Path("/output/gradient_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"phase": "completed", "result": "gradient_probe.json", "scope": SCOPE}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
