"""Node state: buffer, energy with time-varying supply, and compute capacity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """One compute/relay node.

    Parameters are physical.  Compute energy is computed as
    ``power_draw_w * (cycles / f_hz)``, i.e. power times time, which is the only
    dimensionally consistent form.  The paper's eq. (ecmp) writes ``kappa*W*f``
    and calls it an energy; that product is a *power*, not an energy, so we correct
    it here and note the fix.

    supply_w is a callable t -> watts.  Making it time-varying is what couples the
    energy model to the contact model during lunar night.
    """

    nid: str
    tier: str
    f_hz: float
    buffer_bytes: float
    energy_budget_wh: float
    supply_w: object                       # callable t -> watts
    mtd_h: float = 72.0
    power_draw_w: float = 10.0             # compute power draw while busy
    power_tx_w: float = 2.0
    power_rx_w: float = 0.3
    power_idle_w: float = 0.1

    # runtime state
    occupancy: float = 0.0
    energy_used_j: float = 0.0
    energy_supplied_j: float = 0.0
    compute_available_s: float = 0.0
    dropped_bytes: float = 0.0
    evicted_bytes: float = 0.0

    # ---------------------------------------------------------------- buffer
    def admit(self, nbytes: float) -> bool:
        """Place nbytes in the buffer, honouring the hard capacity constraint (C3).

        Eviction is explicit and counted separately from failure, so a delivery
        ratio can never be flattered by silently discarding hard work.
        """
        if nbytes > self.buffer_bytes:
            self.dropped_bytes += nbytes
            return False
        if self.occupancy + nbytes <= self.buffer_bytes:
            self.occupancy += nbytes
            return True
        free = self.buffer_bytes - self.occupancy
        self.occupancy += max(0.0, free)
        self.evicted_bytes += nbytes - max(0.0, free)
        self.dropped_bytes += nbytes - max(0.0, free)
        return False

    def release(self, nbytes: float) -> None:
        self.occupancy = max(0.0, self.occupancy - nbytes)

    def free_space(self) -> float:
        return max(0.0, self.buffer_bytes - self.occupancy)

    # ---------------------------------------------------------------- energy
    def supplied_j(self, t0: float, t1: float, steps: int = 33) -> float:
        """Energy harvested in [t0, t1], by numeric integration of supply_w."""
        if t1 <= t0:
            return 0.0
        ts = np.linspace(t0, t1, steps)
        p = np.array([float(self.supply_w(float(t))) for t in ts])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(trapezoid(p, ts))

    def power_ok(self, t0: float, t1: float, cycles: float, margin: float = 1.0) -> bool:
        """Rate feasibility: is supply >= draw at *every* point of the window?

        A battery can hold joules it cannot deliver at the required rate, so this
        check is not implied by the stock check.  During lunar night supply falls to
        zero and this is the binding constraint.
        """
        draw = margin * self.power_draw_w
        ts = np.linspace(t0, t1, 33)
        return bool(np.all([float(self.supply_w(float(t))) >= draw for t in ts]))

    def energy_ok(
        self, t0: float, t1: float, need_j: float, reserve_j: float = 0.0
    ) -> bool:
        """Stock feasibility against the *running* ledger, not the horizon total.

        Energy harvested in hour 30 cannot pay for work in hour 2, which is why we
        keep a ledger rather than a scalar budget.
        """
        return (self.energy_supplied_j - self.energy_used_j - reserve_j) >= need_j

    def spend(self, joules: float) -> None:
        self.energy_used_j += joules

    def harvest(self, t0: float, t1: float) -> None:
        self.energy_supplied_j += self.supplied_j(t0, t1)

    @property
    def energy_remaining_j(self) -> float:
        return self.energy_budget_wh * 3600.0 - self.energy_used_j

    # ---------------------------------------------------------------- compute
    def compute_time_s(self, cycles: float) -> float:
        return cycles / self.f_hz

    def compute_energy_j(self, cycles: float) -> float:
        """Energy for `cycles` = draw power x compute time.  Dimensionally correct."""
        return self.power_draw_w * self.compute_time_s(cycles)

    def tx_energy_j(self, nbytes: float, rate: float) -> float:
        """E = (P_tx + P_idle) * S / R, in joules (eq. etx)."""
        return (self.power_tx_w + self.power_idle_w) * (nbytes / rate)

    def snapshot(self) -> dict:
        return {
            "nid": self.nid,
            "tier": self.tier,
            "occupancy": self.occupancy,
            "buffer_bytes": self.buffer_bytes,
            "buffer_fill": self.occupancy / self.buffer_bytes if self.buffer_bytes else 0.0,
            "energy_used_j": self.energy_used_j,
            "energy_supplied_j": self.energy_supplied_j,
            "compute_used_s": self.compute_available_s,
            "dropped_bytes": self.dropped_bytes,
            "evicted_bytes": self.evicted_bytes,
        }


# ---------------------------------------------------------------- supplies

def solar_supply(peak_w: float, period_s: float, phase: float = 0.0):
    """Lunar illumination: peak_w during the synodic day, ~0 during the night.

    The synodic month is 29.53 days, ~14.75 of them night, so the duty cycle is
    ~0.5.  A clipped sine rather than a step, so terminator ramps are finite and
    scheduling near dawn is not numerically brittle.
    """

    def supply(t: float) -> float:
        arg = 2.0 * np.pi * ((t + phase) % period_s) / period_s
        return peak_w * max(0.0, np.sin(arg))

    return supply


def constant_supply(power_w: float):
    def supply(t: float) -> float:
        return power_w

    return supply


def shadowed_supply(trickle_w: float = 0.02):
    """Permanently shadowed region: solar flux is effectively zero year-round.

    Real PSRs receive tens of milliwatts per square metre of scattered light,
    orders of magnitude below any compute load, so a shadowed rover is battery-only.
    The trickle is kept nonzero so division and normalisation behave.
    """

    def supply(t: float) -> float:
        return trickle_w

    return supply


def battery_supply(capacity_wh: float, drain_w: float, t_start: float = 0.0):
    """A finite battery discharging under a constant housekeeping load.

    Once depleted the supply is zero, which is how a rover's MTD limit manifests:
    after this horizon the node cannot execute anything.
    """
    state = {"t": t_start}

    def supply(t: float) -> float:
        if t < t_start:
            return 0.0
        elapsed = (t - t_start) / 3600.0
        remaining_wh = capacity_wh - drain_w * elapsed
        return max(0.0, remaining_wh * 3600.0 / max(1.0, 3600.0))

    return supply