"""Figures for the paper. Every panel is drawn from results/experiments.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        # IEEE two-column: figures are included at ~3.5in (single) or ~7in
        # (double). Use serif to match IEEEtran + sizes that stay legible
        # after downscaling. No in-axes titles (captions live in LaTeX).
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "legend.frameon": False,
    }
)


def _save(fig, name: str):
    """Save vector (PDF) + high-res raster (PNG) so LaTeX can use PDF."""
    fig.tight_layout(pad=0.6)
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)

D = json.loads((RESULTS / "experiments.json").read_text())


def fig_delivery():
    s = D["E4_summary"]
    names = list(s)
    means = [s[n]["delivery_ratio"]["mean"] for n in names]
    ci = [s[n]["delivery_ratio"]["ci95"] for n in names]
    med = [s[n]["delivery_ratio"]["median"] for n in names]
    p05 = [s[n]["delivery_ratio"]["p05"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.errorbar(x, means, yerr=ci, fmt="o", capsize=3, label="mean ± 95% CI")
    ax.plot(x, med, "s", color="k", ms=4, label="median")
    ax.plot(x, p05, "_", color="crimson", ms=12, mew=2, label="5th pct")
    ax.set_xticks(x)
    # FIX: rotation=20 clipped in two-column layout; use short labels,
    # horizontal, no overlap. Full names go in the LaTeX caption.
    short = [n.replace("Replication-2", "Repl-2").replace("Top-r-by-q", "Top-r")
             for n in names]
    ax.set_xticklabels(short, rotation=0, ha="center")
    ax.set_ylabel("delivery ratio")
    ax.set_ylim(-0.03, 1.05)
    # The legend used to sit at loc="lower left" INSIDE the axes, which put it
    # exactly on top of the two 5th-percentile marks the caption quotes:
    # Replication-2 at 0.000 and Top-r-by-q at 0.010, both inside the legend
    # band.  Move it below the axes so it cannot occlude data at any y-limit.
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handletextpad=0.4, columnspacing=1.0)
    _save(fig, "delivery")


def fig_downlink():
    s = D["E4_summary"]
    names = list(s)
    means = np.array([s[n]["downlink_bytes"]["mean"] for n in names]) / 1e9
    ci = np.array([s[n]["downlink_bytes"]["ci95"] for n in names]) / 1e9
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(np.arange(len(names)), means, yerr=ci, capsize=3,
           color=["#999999", "#bbbbbb", "#7799bb", "#7799bb", "#cc4444"])
    ax.set_xticks(np.arange(len(names)))
    short = [n.replace("Replication-2", "Repl-2").replace("Top-r-by-q", "Top-r")
             for n in names]
    ax.set_xticklabels(short, rotation=0, ha="center")
    ax.set_ylabel("downlink volume (GB)")
    _save(fig, "downlink")


def fig_sign_flip():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    sh = D["E2_shared"]
    k = [r["k"] for r in sh]
    ax = axes[0]
    ax.plot(k, [r["ind"] - r["ccf"] for r in sh], "o-", color="crimson")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ks = D["E2_crossover_k"]
    if ks:
        ax.axvline(ks, color="crimson", ls=":", lw=1)
        # FIX: offset annotation inside axes so it cannot collide with ylabel
        ax.annotate(f"$k^\\star={ks}$", xy=(ks, 0), xytext=(6, 8),
                    textcoords="offset points", fontsize=8,
                    bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlabel("fragments required $k$")
    ax.set_ylabel("$F_{ind}-F_{ccf}$")
    # no in-figure title: explained in caption (Fig. 3)

    ex = D["E2_exchangeable"]
    ax = axes[1]
    for rho in sorted({r["rho"] for r in ex}):
        rs = [r for r in ex if r["rho"] == rho]
        ax.plot([r["k"] for r in rs], [r["gap"] for r in rs], "o-", ms=3, label=f"$\\rho$={rho}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("fragments required $k$")
    ax.legend(fontsize=7, loc="lower left")
    _save(fig, "sign_flip")


def fig_relay_disjointness():
    """fig:relaydisjoint — the G sweep from E1_relay_diversity.

    This figure previously plotted E1_contact_vs_cause, which only covers
    G in {1..4} for a fixed fragment count and is flat in G, so it could not
    support the caption's claim about behaviour as cause diversity increases.
    E1_relay_diversity is the experiment that sweeps G with n=6, k=3 and shows
    both the closure of the gap and its non-monotonicity at G=3.
    """
    rows = D["E1_relay_diversity"]
    rows = sorted(rows, key=lambda r: r["G"])
    G = [r["G"] for r in rows]
    ach = [r["achieved"] for r in rows]
    gap = [r["predicted"] - r["achieved"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(G, gap, "s-", color="#4477aa", ms=5, label="predicted $-$ achieved")
    ax.axhline(0, color="k", lw=0.8, ls="--")

    # Mark the non-monotone step so the caption's qualification is visible.
    dips = [i for i in range(1, len(gap)) if gap[i] >= gap[i - 1] - 1e-12]
    for i in dips:
        ax.annotate("$G{=}3$: gap reopens",
                    xy=(G[i], gap[i]), xytext=(G[i] + 0.55, gap[i] + 0.012),
                    fontsize=7, color="#cc4444",
                    arrowprops=dict(arrowstyle="->", color="#cc4444", lw=0.8))

    ax.set_xticks(G)
    ax.set_xlabel("distinct relay causes $G$")
    ax.set_ylabel("predicted $-$ achieved reliability")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(-0.012, max(gap) * 1.18)
    _save(fig, "relay_disjointness")


def fig_crossover():
    grid = D["E7_crossover"]["grid"]
    # FIX: old width 3.0*len(grid)=9in overflowed a 7in text block and
    # tiny 8pt panel titles were unreadable after downscaling.
    fig, axes = plt.subplots(1, len(grid), figsize=(7.0, 2.6), sharey=True)
    if len(grid) == 1:
        axes = [axes]
    for ax, g in zip(axes, grid):
        pts = [p for p in g["points"] if p["critical_rate"]]
        ax.plot([p["delay_s"] for p in pts], [p["critical_rate"] for p in pts],
                "k-", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("one-way delay (s)")
        # panel label, not a title that collides with caption
        ax.text(0.05, 0.92, g["profile"], transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(fc="white", ec="none", pad=1))
    axes[0].set_ylabel("critical bandwidth $\\bar{R}^{crit}$ (B/s)")
    _save(fig, "crossover")


def fig_runtime():
    r = D["E9_runtime"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    rows = r["vs_tasks"]
    axes[0].plot([x["n_tasks"] for x in rows], [x["runtime_s"] * 1e3 for x in rows],
                 "o-")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("tasks $|\\mathcal{J}|$")
    axes[0].set_ylabel("runtime (ms)")
    # FIX: exponent belongs on the axes, not as a floating title
    axes[0].text(0.05, 0.92, f"fit exponent {r['exponent_tasks']:.2f}",
                 transform=axes[0].transAxes, fontsize=8, va="top",
                 bbox=dict(fc="white", ec="none", pad=1))
    rows = r["vs_relays"]
    axes[1].plot([x["n_relays"] for x in rows], [x["runtime_s"] * 1e3 for x in rows],
                 "o-")
    axes[1].set_xlabel("relay count")
    axes[1].set_ylabel("runtime (ms)")
    axes[1].text(0.05, 0.92, f"fit exponent {r['exponent_relays']:.2f}",
                 transform=axes[1].transAxes, fontsize=8, va="top",
                 bbox=dict(fc="white", ec="none", pad=1))
    _save(fig, "runtime")


def fig_ablations():
    s = D["E5_summary"]
    names = list(s)
    # FIX: full "CHIOR -stage2" labels at 6pt/35deg were unreadable and
    # clipped by bbox=tight. Use compact tags; expand in caption.
    short = [n.replace("CHIOR", "").strip().replace("-", "−") or "full"
             for n in names]
    short = ["full" if t == "" else t for t in short]
    m = [s[n]["delivery_ratio"]["mean"] for n in names]
    c = [s[n]["delivery_ratio"]["ci95"] for n in names]
    d = [s[n]["downlink_bytes"]["mean"] / 1e9 for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(names))
    # Seven variants per panel, and the wider tags ("-stage2", "-buffer",
    # "-energy") still touch when set horizontal -- they ran together as
    # "-stage2-buffer-energy".  Rotate instead of shortening: the tags are
    # already near-minimal and abbreviating them further makes the figure
    # depend on the caption to be readable.  ha="right" anchors each label to
    # its own tick so rotation cannot shift one onto its neighbour.
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=25, ha="right", rotation_mode="anchor")
    axes[0].errorbar(x, m, yerr=c, fmt="o", capsize=3)
    axes[0].set_ylabel("delivery ratio")
    axes[0].set_ylim(0.4, 1.05)
    axes[1].bar(x, d, color="#7799bb")
    axes[1].set_ylabel("downlink (GB)")
    _save(fig, "ablations")


def fig_night():
    n = D["E8_night"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = ["day-start", "night-start"]
    vals = [n[l].get("base", 0) for l in labels]
    tot = [sum(n[l][k] for k in n[l] if k in ("base", "earth", "relay0", "rover"))
           for l in labels]
    frac = [v / t if t else 0 for v, t in zip(vals, tot, strict=True)]
    bars = ax.bar(labels, frac, color=["#cc8844", "#444488"])
    ax.set_ylabel("fraction placed off-device")
    ax.set_ylim(0, 1.35)
    # FIX: old ax.text at frac+0.03 overlapped the bar top and the
    # axes title ("supply..." clipped). Use bar_label offset + legend-style
    # note below, never inside the axes title area.
    for i, (l, s0, s6) in enumerate(
        [(l, n[l + "_supply_t0_w"], n[l + "_t_base"]) for l in labels]
    ):
        ax.text(i, frac[i] + 0.05, f"{s0:.1f} W → {s6:.1f} W",
                ha="center", fontsize=8,
                bbox=dict(fc="white", ec="none", pad=1))
    _save(fig, "night")


def fig_overestimate():
    v = D["E10b_relay_routed"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    names = list(v)
    g = [v[n]["mean_predicted_minus_true"] or 0.0 for n in names]
    gm = [v[n]["max_predicted_minus_true"] or 0.0 for n in names]
    x = np.arange(len(names))
    # FIX: two 0.36-wide bars on 2 slots overflowed, so the widths were widened
    # to 0.4 at offsets +-0.2 -- which made the two bars in a group touch with
    # no gap and read as one stacked block.  Narrow them slightly and pull the
    # offsets in so each group has a visible seam, and add headroom so the "max"
    # bar (0.0999) does not sit flush against the axis ceiling.
    ax.bar(x - 0.19, g, 0.34, label="mean", color="#cc4444")
    ax.bar(x + 0.19, gm, 0.34, label="max", color="#aa2222", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n).replace("-disjoint", "\n-disjoint") for n in names],
                       rotation=0, ha="center")
    ax.set_ylabel("predicted $-$ achieved reliability")
    ax.set_ylim(0, max(max(gm), 1e-9) * 1.22)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, "overestimate")


def main() -> None:
    for fn in [
        fig_delivery, fig_downlink, fig_sign_flip, fig_relay_disjointness,
        fig_crossover, fig_runtime, fig_ablations, fig_night, fig_overestimate,
    ]:
        try:
            fn()
            print(f"  {fn.__name__} ok")
        except Exception as e:  # noqa
            print(f"  {fn.__name__} FAILED: {e}")
    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()