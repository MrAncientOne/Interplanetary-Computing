"""Shared scenario construction so every experiment sees the same world."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcloud.node import (
    Node,
    constant_supply,
    shadowed_supply,
    solar_supply,
)
from ipcloud.scenario import lunar_topo

SYNODIC = 86400.0 * 29.53

# Start the horizon at local noon so the first synodic day is illuminated.  Starting
# at t=0 put every early task in lunar night, which made the power screen reject all
# remote tiers for a scheduling artefact rather than a physical reason.
EPOCH_OFFSET = SYNODIC / 4.0


def build_world(
    *,
    n_relays: int = 3,
    horizon_days: float = 2.0,
    relay_p: float = 0.90,
    surface_p: float = 0.995,
    earth_p: float = 0.999,
    local_links: int = 6,
    rover_draw_w: float = 3.0,
    base_draw_w: float = 25.0,
    relay_draw_w: float = 40.0,
    rover_battery_wh: float | None = 20.0,
):
    """Four-tier lunar south-pole world with realistic draw powers.

    The rover sits in a permanently shadowed region, so it is battery-only: its
    supply is a finite reservoir that depletes under housekeeping load.  This is
    what makes its MTD finite and what the Local Sufficiency Invariant is about.
    """
    horizon = horizon_days * 86400.0
    scn = lunar_topo(
        n_relays=n_relays,
        local_links=local_links,
        relay_p=relay_p,
        surface_p=surface_p,
        earth_p=earth_p,
        horizon=horizon,
    )

    nodes: dict[str, Node] = {}
    nodes["rover"] = Node(
        "rover", "dev", 200e6, 4e9, rover_battery_wh or 1e9,
        shadowed_supply(), mtd_h=6.0, power_draw_w=rover_draw_w,
        power_tx_w=1.5, power_rx_w=0.2,
    )
    nodes["base"] = Node(
        "base", "edge", 800e6, 64e9, 500.0,
        solar_supply(40.0, SYNODIC, phase=EPOCH_OFFSET), mtd_h=72.0,
        power_draw_w=base_draw_w, power_tx_w=5.0, power_rx_w=0.5,
    )
    for r in range(n_relays):
        nodes[f"relay{r}"] = Node(
            f"relay{r}", "relay", 1500e6, 16e9, 50.0,
            solar_supply(60.0, SYNODIC, phase=EPOCH_OFFSET + 3600.0 * r),
            mtd_h=0.5,
            power_draw_w=relay_draw_w, power_tx_w=8.0, power_rx_w=0.5,
        )
    nodes["earth"] = Node(
        "earth", "earth", 3e9, 1e12, 1e6, constant_supply(1e5),
        mtd_h=float("inf"), power_draw_w=1e6,
    )

    # Pre-harvest the horizon.  Real systems track a running ledger; pre-harvesting
    # is a modelling convenience for a batch experiment and is documented as such.
    for n in nodes.values():
        n.harvest(0.0, horizon)

    return scn, nodes


SCHEDULERS = {}


def scheduler_registry():
    from ipcloud.schedulers import (
        Ctx, chior, dtn_earth, earth_only, replication, top_r_reference,
    )

    return {
        "Earth-only": earth_only,
        "DTN+Earth": dtn_earth,
        "Replication-2": replication(2),
        "Top-r-by-q": top_r_reference,
        "CHIOR": chior,
    }
