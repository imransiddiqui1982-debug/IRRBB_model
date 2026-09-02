"""
plots.py  —  IRRBB visualisations (19-bucket BCBS 368 version)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calculator import ScenarioResult
from .scenarios import SCENARIOS
from .time_buckets import BUCKET_LABELS
from .yield_curve import YieldCurve, BASE_CURVE

BG = "#0a0c0f"
BG2 = "#0f1217"
BORDER = "#1e2530"
TEXT = "#c8d0dc"
DIM = "#5a6478"
AMBER = "#f0a500"
GREEN = "#00c87a"
RED = "#e03c3c"
BLUE = "#4a9eff"
ORANGE = "#ff7a30"
PURPLE = "#9b7fff"
CYAN = "#00c8c8"
YELLOW = "#ffe066"
SCENARIO_COLORS = [RED, BLUE, PURPLE, ORANGE, YELLOW, CYAN]

DISPLAY_IDX = [0, 2, 4, 6, 8, 10, 12, 15, 18]
DISPLAY_LABELS = [BUCKET_LABELS[i] for i in DISPLAY_IDX]


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG2,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "text.color": TEXT,
        "grid.color": BORDER,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "figure.dpi": 150,
    })


def _color(r):
    return RED if r.is_outlier else (AMBER if r.is_watch else GREEN)


def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, facecolor=BG, bbox_inches="tight")
        print(f"  Saved: {save_path}")


def plot_shock_curves(scenarios, save_path=None):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(BUCKET_LABELS)))
    for s, color in zip(scenarios, SCENARIO_COLORS):
        ax.plot(
            x, s.shocks_bp,
            color=color, lw=1.8, marker="o", markersize=3,
            label=s.name, zorder=3,
        )
    ax.axhline(0, color=DIM, lw=0.8)
    ax.set_xticks(DISPLAY_IDX)
    ax.set_xticklabels(DISPLAY_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Rate Shock (bp)", fontsize=9)
    ax.set_title(
        "BCBS 368 — Six Prescribed Shock Scenarios (19 Buckets, Interpolated)",
        fontsize=10, color=AMBER, pad=10,
    )
    ax.grid(zorder=0)
    ax.legend(
        fontsize=8, facecolor=BG2, edgecolor=BORDER,
        labelcolor=TEXT, ncol=2,
    )
    ax.text(
        0.01, 0.02,
        "Interpolated from reference tenors: O/N, 1Y, 2Y, 5Y, 10Y, 20Y"
        "  —  Source: BCBS 368 Annex 2",
        transform=ax.transAxes, fontsize=6.5, color=DIM,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_yield_curve(curve: YieldCurve = None, scenarios=None, save_path=None):
    _style()
    curve = curve or BASE_CURVE
    scenarios = scenarios or SCENARIOS
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(BUCKET_LABELS)))

    ax.plot(
        x, curve.base_rates * 100,
        color=AMBER, lw=2.5, marker="o", markersize=4,
        label="Base curve", zorder=5,
    )
    for s, color in zip(scenarios, SCENARIO_COLORS):
        shocked = curve.shocked_rates(s.shocks_bp) * 100
        ax.plot(
            x, shocked,
            color=color, lw=1.0, linestyle="--",
            alpha=0.7, label=f"Shocked: {s.name}", zorder=3,
        )
    ax.set_xticks(DISPLAY_IDX)
    ax.set_xticklabels(DISPLAY_LABELS, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Rate (%)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_title(
        "Yield Curve — Base and Shocked (All BCBS 368 Scenarios)",
        fontsize=10, color=AMBER, pad=10,
    )
    ax.grid(zorder=0)
    ax.legend(
        fontsize=7, facecolor=BG2, edgecolor=BORDER,
        labelcolor=TEXT, ncol=3,
    )
    ax.text(
        0.01, 0.02,
        "Stylised USD curve (late 2024) — not sourced from live market data",
        transform=ax.transAxes, fontsize=6.5, color=DIM,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_nii(results: list[ScenarioResult], save_path=None):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r.scenario.name for r in results]
    values = [r.delta_nii for r in results]
    bars = ax.bar(
        names, values,
        color=[GREEN if v >= 0 else RED for v in values],
        edgecolor=BORDER, lw=0.8, zorder=3,
    )
    ax.axhline(0, color=DIM, lw=0.8)
    ax.set_ylabel("Δ NII (USD millions)", fontsize=9)
    ax.set_title(
        "Net Interest Income Sensitivity — BCBS 368 Scenarios",
        fontsize=10, color=AMBER, pad=10,
    )
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(axis="y", zorder=0)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (1 if val >= 0 else -2.5),
            f"${val:+.1f}M",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8, color=TEXT,
        )
    ax.text(
        0.99, 0.97, "1-year horizon  ·  floating-rate repricing only",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, color=DIM,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_eve(results: list[ScenarioResult], save_path=None):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Economic Value of Equity Sensitivity — BCBS 368 (Cash Flow Discounting)",
        fontsize=10, color=AMBER, y=1.01,
    )
    names = [r.scenario.name for r in results]

    ax1 = axes[0]
    values = [r.delta_eve for r in results]
    bars = ax1.bar(
        names, values,
        color=[_color(r) for r in results],
        edgecolor=BORDER, lw=0.8, zorder=3,
    )
    ax1.axhline(0, color=DIM, lw=0.8)
    ax1.set_ylabel("Δ EVE (USD millions)", fontsize=9)
    ax1.set_title("Δ EVE by Scenario", fontsize=9)
    ax1.tick_params(axis="x", rotation=20, labelsize=7.5)
    ax1.grid(axis="y", zorder=0)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + (1.5 if val >= 0 else -3),
            f"${val:+.1f}M",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7.5, color=TEXT,
        )

    ax2 = axes[1]
    pcts = [r.delta_eve_pct for r in results]
    ax2.bar(
        names, pcts,
        color=[_color(r) for r in results],
        edgecolor=BORDER, lw=0.8, zorder=3,
    )
    ax2.axhline(15, color=RED, lw=1.5, linestyle="--",
                label="15% Outlier (BCBS 368 §99)")
    ax2.axhline(10, color=AMBER, lw=1.0, linestyle=":",
                label="10% Watch")
    ax2.set_ylabel("|ΔEVE| / Tier 1 (%)", fontsize=9)
    ax2.set_title("|ΔEVE| as % of Tier 1 Capital", fontsize=9)
    ax2.tick_params(axis="x", rotation=20, labelsize=7.5)
    ax2.grid(axis="y", zorder=0)
    ax2.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
    for bar, pct, r in zip(ax2.patches, pcts, results):
        suffix = " ⚠" if r.is_outlier else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 0.3,
            f"{pct:.1f}%{suffix}",
            ha="center", va="bottom", fontsize=7.5, color=TEXT,
        )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_repricing_gap(gap_df: pd.DataFrame, save_path=None):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Repricing Gap — 19 BCBS 368 Time Buckets",
        fontsize=10, color=AMBER,
    )
    x = np.arange(len(gap_df))
    w = 0.4

    ax1 = axes[0]
    ax1.bar(x - w / 2, gap_df["assets"], w,
            label="Assets", color=BLUE, alpha=0.85, edgecolor=BORDER)
    ax1.bar(x + w / 2, gap_df["liabilities"], w,
            label="Liabilities", color=ORANGE, alpha=0.85, edgecolor=BORDER)
    ax1.set_xticks(DISPLAY_IDX)
    ax1.set_xticklabels(DISPLAY_LABELS, rotation=35, ha="right", fontsize=7)
    ax1.set_ylabel("Notional (USD millions)", fontsize=9)
    ax1.set_title("Asset vs Liability by Repricing Bucket", fontsize=9)
    ax1.legend(fontsize=8, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
    ax1.grid(axis="y", zorder=0)

    ax2 = axes[1]
    net = gap_df["net_gap"].values
    ax2.bar(
        range(len(net)), net,
        color=[GREEN if v >= 0 else RED for v in net],
        edgecolor=BORDER, lw=0.8, zorder=3,
    )
    ax2.axhline(0, color=DIM, lw=0.8)
    ax2.set_xticks(DISPLAY_IDX)
    ax2.set_xticklabels(DISPLAY_LABELS, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("Net Gap (USD millions)", fontsize=9)
    ax2.set_title("Net Repricing Gap (Assets − Liabilities)", fontsize=9)
    ax2.grid(axis="y", zorder=0)
    ax2.text(
        0.99, 0.03,
        "Green = asset-sensitive  |  Red = liability-sensitive",
        transform=ax2.transAxes, ha="right", fontsize=6.5, color=DIM,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_nii_decomposition(results: list[ScenarioResult], save_path=None):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r.scenario.name for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, [r.nii_asset for r in results], w,
           label="Asset repricing income", color=BLUE, alpha=0.85, edgecolor=BORDER)
    ax.bar(x + w / 2, [r.nii_liability for r in results], w,
           label="Liability repricing saving", color=ORANGE, alpha=0.85, edgecolor=BORDER)
    ax.axhline(0, color=DIM, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Δ NII contribution (USD millions)", fontsize=9)
    ax.set_title(
        "NII Decomposition — Asset vs Liability Repricing",
        fontsize=10, color=AMBER, pad=10,
    )
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_instrument_eve_waterfall(
    detail_df: pd.DataFrame,
    scenario_name: str,
    save_path=None,
):
    """Horizontal waterfall showing per-instrument ΔEVE contribution."""
    _style()
    df = detail_df.sort_values("delta_eve")
    colors = [GREEN if v >= 0 else RED for v in df["delta_eve"]]
    labels = [
        f"{row['side'][:1]} · {row['instrument'][:32]}"
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.42)))
    bars = ax.barh(
        range(len(df)), df["delta_eve"],
        color=colors, edgecolor=BORDER, lw=0.5, height=0.7,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color=DIM, lw=0.8)
    ax.set_xlabel("Δ EVE contribution (USD millions)", fontsize=9)
    ax.set_title(
        f"Instrument-Level EVE Attribution — {scenario_name}",
        fontsize=10, color=AMBER, pad=10,
    )
    ax.grid(axis="x", zorder=0)
    for bar, val in zip(bars, df["delta_eve"]):
        ax.text(
            val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"${val:+.1f}M",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7, color=TEXT,
        )
    ax.legend(
        handles=[
            mpatches.Patch(color=GREEN, label="Positive EVE contribution"),
            mpatches.Patch(color=RED, label="Negative EVE contribution"),
        ],
        fontsize=8, facecolor=BG2, edgecolor=BORDER,
        labelcolor=TEXT, loc="lower right",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig
