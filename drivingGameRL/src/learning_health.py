"""Small, defensive health summary shared by driving learning runtimes.

The dashboard consumes many independently produced telemetry dictionaries.  This
module provides one stable, finite contract at that boundary so a bad diagnostic
value cannot crash rendering or quietly display ``nan`` as if training were healthy.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any


HEALTH_STATUSES = ("healthy", "warming_up", "warning", "critical")


def _finite_number(
    value: object,
    *,
    name: str,
    alerts: list[str],
    default: float = 0.0,
    minimum: float | None = None,
) -> tuple[float, bool]:
    if isinstance(value, bool) or not isinstance(value, Real):
        alerts.append(f"malformed:{name}")
        return default, False
    numeric = float(value)
    if not math.isfinite(numeric):
        alerts.append(f"non_finite:{name}")
        return default, False
    if minimum is not None and numeric < minimum:
        alerts.append(f"out_of_range:{name}")
        return default, False
    return numeric, True


def _mapping(value: object, *, name: str, alerts: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    alerts.append(f"malformed:{name}")
    return {}


def build_learning_health(
    *,
    learning: Mapping[str, Any] | object,
    replay: Mapping[str, Any] | object,
    safety: Mapping[str, Any] | object | None = None,
    environment: Mapping[str, Any] | object | None = None,
    throughput: Mapping[str, Any] | object | None = None,
    environment_decisions: int | float = 0,
    optimization_decisions: int | float | None = None,
    batch_size: int | float = 1,
    warmup_steps: int | float = 0,
    gradient_clip: int | float = 1.0,
    optimization_updates: int | float | None = None,
    gradient_clip_events: int | float | None = None,
    wall_contact_decisions: int | float | None = None,
    collision_loop_terminations: int | float | None = None,
    worker_failed: bool = False,
    worker_failure_type: str | None = None,
    replay_enabled: bool = True,
) -> dict[str, Any]:
    """Return a bounded, JSON-friendly health block with stable nested keys.

    Missing optional counters fall back to the corresponding current snapshot.
    Malformed and non-finite supplied values are replaced with safe zeros and
    reported through ``finite`` and ``alerts`` instead of leaking into the UI.
    """

    alerts: list[str] = []
    learning_map = _mapping(learning, name="learning", alerts=alerts)
    replay_map = _mapping(replay, name="replay", alerts=alerts)
    safety_map = (
        {} if safety is None else _mapping(safety, name="safety", alerts=alerts)
    )
    environment_map = (
        {}
        if environment is None
        else _mapping(environment, name="environment", alerts=alerts)
    )
    throughput_map = (
        {}
        if throughput is None
        else _mapping(throughput, name="throughput", alerts=alerts)
    )

    decisions, decisions_ok = _finite_number(
        environment_decisions,
        name="environment_decisions",
        alerts=alerts,
        minimum=0.0,
    )
    optimization_decision_source = (
        decisions if optimization_decisions is None else optimization_decisions
    )
    optimization_decision_count, optimization_decisions_ok = _finite_number(
        optimization_decision_source,
        name="optimization.decisions",
        alerts=alerts,
        minimum=0.0,
    )
    size, size_ok = _finite_number(
        replay_map.get("size", replay_map.get("transitions", 0)),
        name="replay.size",
        alerts=alerts,
        minimum=0.0,
    )
    capacity, capacity_ok = _finite_number(
        replay_map.get("capacity", 0),
        name="replay.capacity",
        alerts=alerts,
        minimum=0.0,
    )
    batch, batch_ok = _finite_number(
        batch_size,
        name="batch_size",
        alerts=alerts,
        default=1.0,
        minimum=1.0,
    )
    warmup, warmup_ok = _finite_number(
        warmup_steps,
        name="warmup_steps",
        alerts=alerts,
        minimum=0.0,
    )
    readiness_threshold = max(int(batch), int(warmup)) if replay_enabled else 0
    replay_ready = not replay_enabled or int(size) >= readiness_threshold
    replay_ready_ok = True
    if "ready" in replay_map:
        explicit_ready = replay_map["ready"]
        if isinstance(explicit_ready, bool):
            replay_ready = explicit_ready
        else:
            alerts.append("malformed:replay.ready")
            replay_ready_ok = False
    ready_members, ready_members_ok = _finite_number(
        replay_map.get("ready_members", int(replay_ready)),
        name="replay.ready_members",
        alerts=alerts,
        minimum=0.0,
    )
    replay_members, replay_members_ok = _finite_number(
        replay_map.get("member_count", 1),
        name="replay.member_count",
        alerts=alerts,
        default=1.0,
        minimum=1.0,
    )
    fill_ratio = min(1.0, size / capacity) if capacity > 0.0 else 0.0

    raw_updates = (
        learning_map.get("gradient_steps", 0)
        if optimization_updates is None
        else optimization_updates
    )
    updates, updates_ok = _finite_number(
        raw_updates,
        name="optimization.updates",
        alerts=alerts,
        minimum=0.0,
    )
    gradient_norm, gradient_ok = _finite_number(
        learning_map.get("gradient_norm", 0.0),
        name="optimization.gradient_norm",
        alerts=alerts,
        minimum=0.0,
    )
    clip_threshold, clip_ok = _finite_number(
        gradient_clip,
        name="optimization.clip_threshold",
        alerts=alerts,
        default=1.0,
        minimum=1e-12,
    )
    raw_clip_events = (
        learning_map.get("gradient_clip_events", 0)
        if gradient_clip_events is None
        else gradient_clip_events
    )
    clip_events, clip_events_ok = _finite_number(
        raw_clip_events,
        name="optimization.clip_events",
        alerts=alerts,
        minimum=0.0,
    )
    update_ratio = (
        updates / optimization_decision_count
        if optimization_decision_count > 0.0
        else 0.0
    )
    current_norm_ratio = gradient_norm / clip_threshold if clip_threshold > 0.0 else 0.0
    clip_ratio = clip_events / updates if updates > 0.0 else 0.0

    q_values = learning_map.get("q_values", ())
    q_abs_max = 0.0
    q_values_ok = True
    if not isinstance(q_values, (list, tuple)):
        alerts.append("malformed:values.q_values")
        q_values_ok = False
    else:
        for index, value in enumerate(q_values):
            numeric, valid = _finite_number(
                value,
                name=f"values.q_values[{index}]",
                alerts=alerts,
            )
            q_values_ok = q_values_ok and valid
            q_abs_max = max(q_abs_max, abs(numeric))
    td_error, td_ok = _finite_number(
        learning_map.get(
            "mean_absolute_td_error",
            learning_map.get("td_error", 0.0),
        ),
        name="values.td_error_abs_mean",
        alerts=alerts,
        minimum=0.0,
    )
    auxiliary_learning_ok = True
    for name in (
        "epsilon",
        "last_loss",
        "mean_predicted_q",
        "mean_target_q",
        "parameter_norm",
        "target_parameter_gap",
    ):
        if name not in learning_map:
            continue
        _, valid = _finite_number(
            learning_map[name],
            name=f"learning.{name}",
            alerts=alerts,
        )
        auxiliary_learning_ok = auxiliary_learning_ok and valid
    rejected_updates, rejected_updates_ok = _finite_number(
        learning_map.get("nonfinite_update_rejections", 0),
        name="optimization.nonfinite_update_rejections",
        alerts=alerts,
        minimum=0.0,
    )

    environment_ok = True
    for name in (
        "speed",
        "speed_ratio",
        "progress",
        "usable_clearance",
        "clearance_delta",
        "green_ray_fraction",
    ):
        if name not in environment_map:
            continue
        _, valid = _finite_number(
            environment_map[name],
            name=f"environment.{name}",
            alerts=alerts,
        )
        environment_ok = environment_ok and valid

    safety_decisions, safety_decisions_ok = _finite_number(
        safety_map.get("population_decisions", safety_map.get("decisions", 0)),
        name="safety.decisions",
        alerts=alerts,
        minimum=0.0,
    )
    interventions, interventions_ok = _finite_number(
        safety_map.get("population_interventions", safety_map.get("interventions", 0)),
        name="safety.interventions",
        alerts=alerts,
        minimum=0.0,
    )
    contacts_source = (
        int(bool(environment_map.get("wall_contact_active", False)))
        if wall_contact_decisions is None
        else wall_contact_decisions
    )
    contacts, contacts_ok = _finite_number(
        contacts_source,
        name="safety.wall_contact_decisions",
        alerts=alerts,
        minimum=0.0,
    )
    loops_source = (
        int(bool(environment_map.get("collision_looped", False)))
        if collision_loop_terminations is None
        else collision_loop_terminations
    )
    collision_loops, loops_ok = _finite_number(
        loops_source,
        name="safety.collision_loop_terminations",
        alerts=alerts,
        minimum=0.0,
    )
    intervention_rate = (
        interventions / safety_decisions if safety_decisions > 0.0 else 0.0
    )
    contact_rate = contacts / decisions if decisions > 0.0 else 0.0
    collision_loop_rate = collision_loops / decisions if decisions > 0.0 else 0.0

    decision_throughput, decision_throughput_ok = _finite_number(
        throughput_map.get(
            "decisions_per_second",
            throughput_map.get("decision_throughput", 0.0),
        ),
        name="throughput.decisions_per_second",
        alerts=alerts,
        minimum=0.0,
    )
    tick_throughput, tick_throughput_ok = _finite_number(
        throughput_map.get(
            "ticks_per_second", throughput_map.get("tick_throughput", 0.0)
        ),
        name="throughput.ticks_per_second",
        alerts=alerts,
        minimum=0.0,
    )
    last_batch_ms, last_batch_ok = _finite_number(
        throughput_map.get("last_batch_ms", 0.0),
        name="throughput.last_batch_ms",
        alerts=alerts,
        minimum=0.0,
    )
    workers, workers_ok = _finite_number(
        throughput_map.get("workers", throughput_map.get("parallel_workers", 1)),
        name="throughput.workers",
        alerts=alerts,
        default=1.0,
        minimum=1.0,
    )

    finite = all(
        (
            decisions_ok,
            optimization_decisions_ok,
            size_ok,
            capacity_ok,
            batch_ok,
            warmup_ok,
            replay_ready_ok,
            ready_members_ok,
            replay_members_ok,
            updates_ok,
            gradient_ok,
            clip_ok,
            clip_events_ok,
            q_values_ok,
            td_ok,
            auxiliary_learning_ok,
            rejected_updates_ok,
            environment_ok,
            safety_decisions_ok,
            interventions_ok,
            contacts_ok,
            loops_ok,
            decision_throughput_ok,
            tick_throughput_ok,
            last_batch_ok,
            workers_ok,
        )
    )

    if worker_failed:
        failure = str(worker_failure_type or "unknown")
        alerts.append(f"worker_failure:{failure}")
    if replay_ready is False:
        alerts.append("replay_warming_up")
    # One clipped batch is evidence, not a diagnosis. Wait for a useful window
    # before warning about persistent clipping; still expose every raw value.
    if updates >= 8.0 and (clip_ratio >= 0.50 or current_norm_ratio >= 5.0):
        alerts.append("gradient_clipping")
    if safety_decisions >= 32.0 and intervention_rate >= 0.50:
        alerts.append("high_safety_intervention_rate")
    if decisions >= 32.0 and contact_rate >= 0.05:
        alerts.append("high_wall_contact_rate")
    if collision_loops > 0.0:
        alerts.append("collision_loop_termination")
    if rejected_updates > 0.0:
        alerts.append("nonfinite_update_rejected")

    # Preserve insertion order while preventing repeated malformed-field alerts.
    alerts = list(dict.fromkeys(alerts))
    if not finite or worker_failed:
        status = "critical"
    elif any(
        alert
        in {
            "gradient_clipping",
            "high_safety_intervention_rate",
            "high_wall_contact_rate",
            "collision_loop_termination",
            "nonfinite_update_rejected",
        }
        for alert in alerts
    ):
        status = "warning"
    elif not replay_ready:
        status = "warming_up"
    else:
        status = "healthy"

    return {
        "status": status,
        "finite": finite,
        "alerts": alerts,
        "replay": {
            "applicable": bool(replay_enabled),
            "enabled": bool(replay_enabled),
            "size": int(size),
            "capacity": int(capacity),
            "fill_ratio": fill_ratio,
            "ready": replay_ready,
            "readiness_threshold": readiness_threshold,
            "ready_members": int(ready_members),
            "member_count": int(replay_members),
        },
        "optimization": {
            "applicable": bool(replay_enabled),
            "updates": int(updates),
            "decisions": int(optimization_decision_count),
            "update_to_decision_ratio": update_ratio,
            "gradient_norm": gradient_norm,
            "clip_threshold": clip_threshold,
            "clip_ratio": clip_ratio,
            "current_norm_ratio": current_norm_ratio,
            "clip_events": int(clip_events),
            "clip_event_rate": clip_ratio,
            "nonfinite_update_rejections": int(rejected_updates),
        },
        "values": {
            "q_applicable": True,
            "td_error_applicable": bool(replay_enabled),
            "q_abs_max": q_abs_max,
            "td_error_abs_mean": td_error,
        },
        "safety": {
            "decisions": int(safety_decisions),
            "interventions": int(interventions),
            "intervention_rate": intervention_rate,
            "wall_contact_decisions": int(contacts),
            "wall_contact_rate": contact_rate,
            "collision_loop_terminations": int(collision_loops),
            "collision_loop_rate": collision_loop_rate,
        },
        "throughput": {
            "decisions_per_second": decision_throughput,
            "ticks_per_second": tick_throughput,
            "last_batch_ms": last_batch_ms,
            "workers": int(workers),
            "worker_failed": bool(worker_failed),
            "worker_failure_type": (
                str(worker_failure_type) if worker_failure_type is not None else None
            ),
        },
    }


__all__ = ("HEALTH_STATUSES", "build_learning_health")
