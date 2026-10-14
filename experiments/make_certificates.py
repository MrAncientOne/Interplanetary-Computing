"""Run the Local Sufficiency Invariant certifier and emit the artifact.

Writes:
  results/certificates.json    every certificate, for reproducibility
  results/figures/certifiers.tex   the LaTeX table the paper includes
  stdout                       the headline sweep: which tiers are certifiable,
                               and at which eps

A certificate is SAT only if the task completes with probability >= 1 - eps for
the ENTIRE MTD window, so a placement that is reachable for 40.5 h of a 72 h MTD
is reported UNSAT *with a witness naming the failure time*. That witness is the
useful output: it says which tier the invariant rules out and why.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ipcloud.certifier import CertConfig, certify, summarise  # noqa: E402
from ipcloud.workload import default_workload  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
EPSILONS = (1e-1, 1e-2, 1e-3)
PLACEMENTS = ("rover", "base", "relay0", "relay1", "relay2", "earth")


def certificate_rows():
    cfg = CertConfig(theta_h=6.0, horizon_days=2.0, eps=1e-3)
    scn = __import__("ipcloud.scenario", fromlist=["lunar_topo"]).lunar_topo(
        n_relays=cfg.n_relays, horizon=cfg.horizon_days * 86400.0
    )
    tasks = default_workload(cfg.horizon_days * 86400, volume="nominal")
    rows = []
    for sink in PLACEMENTS:
        only = lambda t, _s=sink: [_s]  # noqa: E731
        certs = certify(tasks, cfg, placement_of=only, scn=scn)
        for c in certs:
            rows.append(
                dict(
                    tid=c.tid, profile=c.profile, criticality=c.criticality,
                    placement=sink, sat=bool(c.sat),
                    survives_h=c.survives_h, mtd_h=c.mtd_h,
                    pi_ind=c.pi_ind, pi_ccf=c.pi_ccf,
                    first_fail_t_h=c.first_fail_t_h,
                    first_fail_condition=c.first_fail_condition,
                )
            )
    return rows, tasks, cfg, scn


def latex_table(rows):
    """One row per placement, aggregating over tasks."""
    hdr = (
        "\\begin{tabular}{@{}llrrrl@{}}\n"
        "\\toprule\n"
        "\\textbf{Placement} & \\textbf{MTD (h)} & \\textbf{Certified} & "
        "\\textbf{Survives (h)} & $\\pi_{ind}$ & \\textbf{Witness}\\\\\n"
        "\\midrule\n"
    )
    body = []
    for sink in PLACEMENTS:
        sub = [r for r in rows if r["placement"] == sink]
        if not sub:
            continue
        n_sat = sum(r["sat"] for r in sub)
        surv = [r["survives_h"] for r in sub if r["survives_h"] is not None]
        med = sorted(surv)[len(surv) // 2] if surv else 0.0
        pis = [r["pi_ind"] for r in sub if r["pi_ind"] is not None]
        best_pi = max(pis) if pis else 0.0
        # a witness from an UNSAT task, if there is one
        wit = next(
            (r["first_fail_condition"] for r in sub if not r["sat"]), ""
        ).replace("%", "\\%").replace("_", "\\_")
        if len(wit) > 58:
            wit = wit[:55] + "..."
        verdict = f"{n_sat}/{len(sub)}"
        cell = f"\\textbf{{{med:.1f}}}" if n_sat else f"{med:.1f}"
        body.append(
            f"{sink} & {sub[0]['mtd_h']:.1f} & {verdict} & {cell} & "
            f"{best_pi:.4f} & {wit}\\\\"
        )
    return hdr + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def main() -> None:
    rows, tasks, cfg, scn = certificate_rows()
    RESULTS.mkdir(parents=True, exist_ok=True)

    (RESULTS / "certificates.json").write_text(
        json.dumps(
            dict(
                config=dict(
                    theta_h=cfg.theta_h, horizon_days=cfg.horizon_days,
                    eps=cfg.eps, n_relays=cfg.n_relays,
                ),
                n_tasks=len(tasks),
                certificates=rows,
            ),
            indent=2,
        )
    )

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures" / "certifiers.tex").write_text(latex_table(rows))

    print("Local Sufficiency Invariant certifier")
    print(f"  {len(tasks)} nominal tasks, theta = {cfg.theta_h:.0f} h, "
          f"horizon = {cfg.horizon_days:.0f} d")
    print()
    print("  per-placement outcome (certified / tasks):")
    for sink in PLACEMENTS:
        sub = [r for r in rows if r["placement"] == sink]
        if not sub:
            continue
        n_sat = sum(r["sat"] for r in sub)
        print(f"    {sink:<8} {n_sat:>3d}/{len(sub):<3d}", end="")
        if n_sat:
            print()
        else:
            wit = sub[0]["first_fail_condition"]
            print(f"   witness: {wit}")
    print()
    print("  eps sweep (all placements pooled):")
    for eps in EPSILONS:
        cfg_e = CertConfig(theta_h=6.0, horizon_days=2.0, eps=eps)
        certs = certify(tasks, cfg_e, scn=scn)
        s = summarise(certs)
        cnt = Counter(c.placement for c in certs if c.sat)
        print(f"    eps={eps:>7.0e}  SAT {s['n_sat']:>3d}/{s['n_tasks']:<3d} "
              f"placements={dict(cnt)}")
    print()
    print(f"  wrote {RESULTS / 'certificates.json'}")


if __name__ == "__main__":
    main()
