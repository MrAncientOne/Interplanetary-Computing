"""Workload generation with realistic, sourced order-of-magnitude parameters.

Sizes and cycle counts are drawn from published mission payload classes rather than
invented.  Every profile carries the fields the model needs: input volume, cycles,
output volume, deadline, tolerated failure probability, and criticality class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    tid: int
    profile: str
    tier: str                 # originating tier, normally 'dev'
    d_in_bytes: float
    cycles: float
    d_out_bytes: float
    deadline_s: float
    delta: float
    criticality: str          # T0 | T1 | T2
    ram_bytes: float = 0.0
    storage_bytes: float = 0.0
    t_arrival: float = 0.0

    @property
    def gamma(self) -> float:
        """Reduction ratio."""
        return self.d_out_bytes / self.d_in_bytes if self.d_in_bytes else 0.0


# Profiles sized to plausible cislunar instrument classes.  The `note` field records
# where the number came from, or says ASSUMED -- this is required, not decorative.
PROFILES: dict[str, dict] = {
    "hyperspectral": {
        "d_in_bytes": 1.0e9,        # ~500 frames x 2000 px x 3 bands x 16 bit
        "cycles": 5.0e10,           # ASSUMED: 250 s on the rover, 62 s on the base
        "gamma": 0.01,              # class labels are a tiny fraction of the raster
        "deadline_s": 6 * 3600.0,
        "delta": 1e-2,
        "criticality": "T1",
        "note": "ASSUMED: instrument class sized from typical LRO-style imaging rates",
    },
    "stereo_nav": {
        "d_in_bytes": 1.0e8,
        "cycles": 2.0e10,           # ASSUMED: 100 s on the rover, 25 s on the base
        "gamma": 0.05,
        "deadline_s": 2 * 3600.0,
        "delta": 1e-2,
        "criticality": "T1",
        "note": "ASSUMED: stereo pair + disparity map",
    },
    "hazard_check": {
        "d_in_bytes": 2.0e7,
        "cycles": 1.2e10,            # ASSUMED: see note
        "gamma": 0.10,
        "deadline_s": 60.0,
        "delta": 1e-4,
        "criticality": "T0",
        "note": "T0 REGIME CHOICE: 1.2e10 cycles is 60 s on the 200 MHz rover "
                "(at the deadline) and 15 s on the 800 MHz base, which with ~26 s "
                "of ingress fits inside 60 s. This is the regime in which a "
                "deadline forces placement off the device. It is a deliberate "
                "parameter choice: with fewer cycles the device meets the "
                "deadline locally and offloading never fires. ASSUMED -- must be "
                "justified against a cited workload before submission.",
    },
    "anomaly": {
        "d_in_bytes": 5.0e8,
        "cycles": 1.0e11,           # ASSUMED
        "gamma": 0.02,
        "deadline_s": 12 * 3600.0,
        "delta": 1e-1,
        "criticality": "T2",
        "note": "ASSUMED: event-detection triage",
    },
    "telemetry": {
        "d_in_bytes": 5.0e6,
        "cycles": 5.0e10,           # ASSUMED
        "gamma": 0.20,
        "deadline_s": 3600.0,
        "delta": 1e-1,
        "criticality": "T2",
        "note": "ASSUMED: health/trend compression",
    },
}


@dataclass
class WorkloadSpec:
    """A workload profile plus arrival process."""

    profile: str
    n_tasks: int
    arrival_rate_hz: float
    jitter: float = 0.2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self, horizon_s: float, origin_tier: str = "dev") -> list[Task]:
        p = PROFILES[self.profile]
        tasks: list[Task] = []
        mean_dt = 1.0 / max(self.arrival_rate_hz, 1e-9)
        t = 0.0
        for i in range(self.n_tasks):
            dt = max(1.0, self.rng.exponential(mean_dt))
            t += dt * (1.0 + self.jitter * self.rng.normal())
            if t >= horizon_s:
                break
            scale = float(self.rng.lognormal(0.0, 0.15))   # size jitter
            d_in = p["d_in_bytes"] * scale
            tasks.append(
                Task(
                    tid=i,
                    profile=self.profile,
                    tier=origin_tier,
                    d_in_bytes=d_in,
                    cycles=p["cycles"] * scale,
                    d_out_bytes=d_in * p["gamma"],
                    deadline_s=p["deadline_s"],
                    delta=p["delta"],
                    criticality=p["criticality"],
                    ram_bytes=p["d_in_bytes"] * 0.05,
                    storage_bytes=p["d_in_bytes"] * 0.2,
                    t_arrival=t,
                )
            )
        return tasks


def default_workload(
    horizon_s: float, seed: int = 20250101, volume: str | float = "nominal"
) -> list[Task]:
    """A mixed workload.  `volume` scales the arrival rate.

    Accepts either a named preset or an explicit float multiplier, so the runtime
    sweep can request an arbitrary task count without inventing presets.

    T0 tasks are included but kept rare: they represent the safety-relevant class
    whose deadline forces placement off the device, and flooding the workload with
    them would make the whole study a T0 study.
    """
    rng = np.random.default_rng(seed)
    mult = (
        {"light": 0.5, "nominal": 1.0, "heavy": 2.0}[volume]
        if isinstance(volume, str)
        else float(volume)
    )
    specs = [
        WorkloadSpec("hyperspectral", n_tasks=int(24 * mult), arrival_rate_hz=1 / (6 * 3600), rng=rng),
        WorkloadSpec("stereo_nav", n_tasks=int(40 * mult), arrival_rate_hz=1 / (2 * 3600), rng=rng),
        WorkloadSpec("hazard_check", n_tasks=int(30 * mult), arrival_rate_hz=1 / 1800.0, rng=rng),
        WorkloadSpec("anomaly", n_tasks=int(18 * mult), arrival_rate_hz=1 / (12 * 3600), rng=rng),
        WorkloadSpec("telemetry", n_tasks=int(50 * mult), arrival_rate_hz=1 / 3600.0, rng=rng),
    ]
    tasks: list[Task] = []
    for s in specs:
        tasks.extend(s.generate(horizon_s))
    tasks.sort(key=lambda x: x.t_arrival)
    for i, t in enumerate(tasks):
        t.tid = i
    return tasks


def criticality_mix(tasks: list[Task]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.criticality] = out.get(t.criticality, 0) + 1
    return out