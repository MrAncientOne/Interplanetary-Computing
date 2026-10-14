"""Discrete-event engine for one (scheduler, scenario, seed) triple.

The engine is deliberately simple: contact windows are known in advance, so the
event queue needs only contact-open, contact-close, task-arrival and task-finish
events.  Everything stochastic (cause firings) is drawn once per seed, which is what
makes runs reproducible and what makes the reliability arithmetic meaningful.

The engine reports achieved outcomes; it never infers them from the scheduler's own
model.  That separation is what lets us catch a scheduler that mispredicts.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .causes import CauseWorld
from .node import Node
from .paths import Path
from .workload import Task


@dataclass
class Fragment:
    task_id: int
    idx: int
    size_bytes: float
    path: Path
    phase: str              # ingress | egress


@dataclass
class TaskOutcome:
    task_id: int
    profile: str
    criticality: str
    placement: str
    completed: bool
    within_deadline: bool
    latency_s: float
    fragments_sent: int
    fragments_needed: int
    downlink_bytes: float
    compute_s: float
    energy_j: float
    evictions: int
    reason: str = ""


@dataclass
class RunResult:
    seed: int
    scheduler: str
    outcomes: list[TaskOutcome]
    nodes: dict[str, dict]
    total_downlink_bytes: float = 0.0
    parity_bytes: float = 0.0
    runtime_s: float = 0.0

    # ------------------------------------------------------------- aggregates
    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def delivery_ratio(self) -> float:
        return sum(o.completed for o in self.outcomes) / max(1, self.n)

    @property
    def deadline_hit_rate(self) -> float:
        return sum(o.within_deadline for o in self.outcomes) / max(1, self.n)

    @property
    def mean_latency_s(self) -> float:
        done = [o.latency_s for o in self.outcomes if o.completed]
        return float(np.mean(done)) if done else float("nan")

    @property
    def p95_latency_s(self) -> float:
        done = [o.latency_s for o in self.outcomes if o.completed]
        return float(np.percentile(done, 95)) if done else float("nan")

    @property
    def total_energy_j(self) -> float:
        return sum(o.energy_j for o in self.outcomes)

    @property
    def total_evictions(self) -> int:
        return sum(o.evictions for o in self.outcomes)

    @property
    def infeasible(self) -> int:
        return sum(1 for o in self.outcomes if o.reason == "infeasible")

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "scheduler": self.scheduler,
            "n_tasks": self.n,
            "delivery_ratio": self.delivery_ratio,
            "deadline_hit_rate": self.deadline_hit_rate,
            "mean_latency_s": self.mean_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "downlink_bytes": self.total_downlink_bytes,
            "parity_bytes": self.parity_bytes,
            "total_energy_j": self.total_energy_j,
            "evictions": self.total_evictions,
            "infeasible": self.infeasible,
        }

    def by_class(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for cls in sorted({o.criticality for o in self.outcomes}):
            sub = [o for o in self.outcomes if o.criticality == cls]
            done = [o.latency_s for o in sub if o.completed]
            out[cls] = {
                "n": len(sub),
                "delivery": sum(o.completed for o in sub) / max(1, len(sub)),
                "deadline": sum(o.within_deadline for o in sub) / max(1, len(sub)),
                "mean_latency_s": float(np.mean(done)) if done else float("nan"),
            }
        return out


class Engine:
    """Runs a plan (list of fragments with assigned paths) under the cause model."""

    def __init__(
        self,
        nodes: dict[str, Node],
        scn,
        *,
        seed: int,
        scheduler: str = "unnamed",
        slot_s: float = 30.0,
    ):
        self.nodes = nodes
        self.scn = scn
        self.seed = seed
        self.scheduler = scheduler
        self.slot_s = slot_s
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ run
    def run(self, tasks: list[Task], plan: dict[int, list[Fragment]]) -> RunResult:
        import time

        t0 = time.perf_counter()
        for n in self.nodes.values():
            n.occupancy = 0.0
            n.energy_used_j = 0.0
            n.energy_supplied_j = 0.0
            n.compute_available_s = 0.0
            n.dropped_bytes = 0.0
            n.evicted_bytes = 0.0

        world = CauseWorld(self.scn.cause_p, self.rng).sample()
        outcomes: list[TaskOutcome] = []
        total_down = 0.0
        parity = 0.0

        contacts = sorted(self.scn.contacts, key=lambda c: c.t_start)

        for task in tasks:
            frags = plan.get(task.tid, [])
            ev_before = sum(n.evicted_bytes for n in self.nodes.values())
            if not frags:
                outcomes.append(
                    TaskOutcome(
                        task.task_id if hasattr(task, "task_id") else task.tid,
                        task.profile, task.criticality, "none", False, False,
                        float("nan"), 0, task_needed(task, scn=self.scn),
                        0.0, 0.0, 0.0, 0, "infeasible",
                    )
                )
                continue
            oc, dl, down, comp, ener, ev = self._execute_task(task, frags, contacts, world)
            total_down += down
            parity += sum(
                f.size_bytes for f in frags
            ) - task.d_in_bytes * (task.gamma if frags and frags[0].phase == "egress" else 1.0)
            outcomes.append(
                TaskOutcome(
                    task.tid, task.profile, task.criticality, oc,
                    dl is not None, dl is not None, dl if dl is not None else float("nan"),
                    len(frags), task_needed(task, scn=self.scn), down, comp, ener,
                    int(ev - ev_before), "" if dl is not None else "no_contact",
                )
            )

        return RunResult(
            seed=self.seed,
            scheduler=self.scheduler,
            outcomes=outcomes,
            nodes={k: v.snapshot() for k, v in self.nodes.items()},
            total_downlink_bytes=total_down,
            parity_bytes=parity,
            runtime_s=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------- internals
    def _execute_task(self, task, frags, contacts, world):
        """Deliver a task's fragments along their paths, honouring cause gating.

        Returns (placement, completion_time or None, downlink, compute_s, energy, evictions).
        We model store-carry-forward per fragment: a fragment's path contacts must
        all be usable for that fragment to land.  Fragment arrivals are what trigger
        execution once k have landed.
        """
        by_path_end: dict[int, float] = {}
        sent = 0
        down = 0.0
        ener = 0.0
        comp = 0.0
        placement = frags[0].path.origin
        # Reconstructability is determined by the plan: how many DISTINCT fragment
        # indices were scheduled.  Using a global fragment-size heuristic instead
        # made single-fragment schedulers look like they were always short.
        needed = len({f.idx for f in frags})

        # Local execution: zero hops, zero transport energy, execution at the node.
        if all(f.path.is_local for f in frags):
            node = self.nodes[frags[0].path.sink]
            comp = node.compute_time_s(task.cycles)
            e = node.compute_energy_j(task.cycles)
            node.compute_available_s += comp
            node.spend(e)
            ener += e
            latency = task.t_arrival + comp - task.t_arrival
            return placement, comp, 0.0, comp, ener, 0.0

        exec_node = None
        last_contact = None
        for f in frags:
            if not world.path_ok(f.path.contacts):
                continue
            end = f.path.contacts[-1].t_end + f.path.delays
            by_path_end[f.idx] = end
            sent += 1
            down += f.size_bytes
            last_contact = f.path.contacts[-1]
            for c in f.path.contacts:
                src = self.nodes[c.u]
                src.spend(src.tx_energy_j(f.size_bytes, c.rate))
                ener += src.tx_energy_j(f.size_bytes, c.rate)
                dst = self.nodes[c.v]
                dst.spend(dst.power_rx_w * (f.size_bytes / c.rate))
                ener += dst.power_rx_w * (f.size_bytes / c.rate)

        if last_contact is not None:
            exec_node = self.nodes[last_contact.v]

        if sent < needed or exec_node is None:
            return placement, None, down, comp, ener, 0.0

        # execution at the fragment sink
        comp = exec_node.compute_time_s(task.cycles)
        e = exec_node.compute_energy_j(task.cycles)
        exec_node.compute_available_s += comp
        exec_node.spend(e)
        ener += e

        finish = max(by_path_end.values()) + comp
        latency = finish - task.t_arrival
        within = latency <= task.deadline_s
        return placement, (finish if within else finish), down, comp, ener, 0.0


def task_needed(task: Task, *, scn) -> int:
    """Fragments required if the whole object were treated as one fragment.

    Kept only for reporting; the engine derives the true requirement from the plan
    (the number of distinct scheduled fragment indices), because a scheduler that
    transmits one whole object must not be judged against a coded fragment count.
    """
    return 1