"""Sensitivity of instability_score rankings to weight / threshold choices."""
from __future__ import annotations

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.join(
    os.path.dirname(__file__),
    "dual_gp",
    "dual GP- FA-BDA-AMP-20260602T193858Z-3-001",
)
OUT = os.path.dirname(__file__)


def load_data():
    df = pd.read_csv(os.path.join(BASE, "updated datasets", "combined_peak_data_full.csv"))
    mp = np.load(
        os.path.join(BASE, "updated datasets", "multiple_peaks_updated_20251106174553.npy"),
        allow_pickle=True,
    )
    compositions_per_iteration = 96
    df = df.copy()
    df["iteration"] = df.index // compositions_per_iteration
    df["composition_number"] = df.index % compositions_per_iteration + 1
    return df, mp


def component_terms(df, compositions_with_multiple_peaks, target_wavelength=780, wavelength_tolerance=30):
    """Unweighted continuous components + multiphase flag (pre-weight)."""
    multiple_peaks_set = set(tuple(row) for row in compositions_with_multiple_peaks)
    intensity_raw = []
    position_raw = []
    multi_flag = []
    for _, row in df.iterrows():
        current_comp = (row["composition_number"], row["iteration"])
        pi = row["initial_peak_positions"]
        pf = row["final_peak_positions"]
        ii = row["initial_peak_intensities"]
        fi = row["final_peak_intensities"]

        if ii == 0 and fi == 0:
            intensity_raw.append(np.nan)
            position_raw.append(np.nan)
            multi_flag.append(1.0 if current_comp in multiple_peaks_set else 0.0)
            continue

        intensity_change = np.abs(ii - fi) / max(ii + fi, 1e-10)
        intensity_raw.append(min(intensity_change, 1.0))

        init_dev = max(abs(pi - target_wavelength) - wavelength_tolerance, 0)
        fin_dev = max(abs(pf - target_wavelength) - wavelength_tolerance, 0)
        position_raw.append(min(init_dev, fin_dev) / target_wavelength)

        multi_flag.append(1.0 if current_comp in multiple_peaks_set else 0.0)

    return (
        np.asarray(intensity_raw, dtype=float),
        np.asarray(position_raw, dtype=float),
        np.asarray(multi_flag, dtype=float),
    )


def instability_score(
    df,
    compositions_with_multiple_peaks,
    target_wavelength=780,
    multiple_peak_penalty=0.5,
    wavelength_tolerance=30,
    degradation_weight=0.4,
    position_weight=0.6,
):
    max_score = 3
    stb_scores = []
    multiple_peaks_set = set(tuple(row) for row in compositions_with_multiple_peaks)

    for _, row in df.iterrows():
        current_comp = (row["composition_number"], row["iteration"])
        peak_positions_int = row["initial_peak_positions"]
        peak_positions_fin = row["final_peak_positions"]
        peak_intensities_int = row["initial_peak_intensities"]
        peak_intensities_fin = row["final_peak_intensities"]

        if peak_intensities_int == 0 and peak_intensities_fin == 0:
            stb_scores.append(max_score)
            continue

        intensity_change = np.abs(peak_intensities_int - peak_intensities_fin) / max(
            peak_intensities_int + peak_intensities_fin, 1e-10
        )
        intensity_score = min(intensity_change, 1) * degradation_weight

        initial_position_deviation = max(
            abs(peak_positions_int - target_wavelength) - wavelength_tolerance, 0
        )
        final_position_deviation = max(
            abs(peak_positions_fin - target_wavelength) - wavelength_tolerance, 0
        )
        position_score = (
            min(initial_position_deviation, final_position_deviation) / target_wavelength
        ) * position_weight

        multiple_peaks_score = multiple_peak_penalty if current_comp in multiple_peaks_set else 0
        total_score = min(intensity_score + position_score + multiple_peaks_score, max_score)
        stb_scores.append(total_score)

    return np.asarray(stb_scores, dtype=float)


def normalize(y):
    y = np.asarray(y, dtype=float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def topk_overlap(a, b, k):
    """Jaccard / fraction overlap of lowest-k (most stable) indices."""
    ia = set(np.argsort(a)[:k])
    ib = set(np.argsort(b)[:k])
    return len(ia & ib) / k


def main():
    df, mp = load_data()
    print(f"Loaded peak_data n={len(df)}, multiple_peaks n={len(mp)}")
    print(df.describe(include="all").to_string())

    # --- Component magnitude diagnostics (justifies weighting) ---
    i_raw, p_raw, m_flag = component_terms(df, mp)
    valid = np.isfinite(i_raw) & np.isfinite(p_raw)
    print("\n=== Unweighted component magnitudes (valid rows) ===")
    print(f"n_valid={valid.sum()} / {len(df)}; multiphase fraction={m_flag.mean():.3f}")
    print(
        "intensity_raw: mean={:.4f} median={:.4f} p90={:.4f} max={:.4f}".format(
            np.nanmean(i_raw), np.nanmedian(i_raw), np.nanpercentile(i_raw, 90), np.nanmax(i_raw)
        )
    )
    print(
        "position_raw:  mean={:.4f} median={:.4f} p90={:.4f} max={:.4f}".format(
            np.nanmean(p_raw), np.nanmedian(p_raw), np.nanpercentile(p_raw, 90), np.nanmax(p_raw)
        )
    )
    # With default weights, contribution sizes
    for dw, pw in [(0.4, 0.6), (0.5, 0.5), (0.3, 0.7), (0.6, 0.4)]:
        i_c = i_raw[valid] * dw
        p_c = p_raw[valid] * pw
        print(
            f"weights deg={dw:.1f}/pos={pw:.1f}: mean contrib intensity={i_c.mean():.4f}, "
            f"position={p_c.mean():.4f}, ratio I/P={i_c.mean() / max(p_c.mean(), 1e-12):.2f}"
        )

    baseline = instability_score(df, mp, degradation_weight=0.4, position_weight=0.6)
    baseline_n = normalize(baseline)

    # --- Weight split sensitivity (deg + pos = 1) ---
    splits = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    rows = []
    scores_by_split = {}
    for dw, pw in splits:
        y = instability_score(df, mp, degradation_weight=dw, position_weight=pw)
        yn = normalize(y)
        scores_by_split[(dw, pw)] = y
        rho, pval = spearmanr(baseline, y)
        rows.append(
            {
                "degradation_weight": dw,
                "position_weight": pw,
                "spearman_vs_0.4_0.6": rho,
                "spearman_p": pval,
                "topk10_overlap": topk_overlap(baseline, y, 10),
                "topk48_overlap": topk_overlap(baseline, y, 48),  # half plate
                "topk96_overlap": topk_overlap(baseline, y, 96),  # one plate
                "mean_abs_norm_diff": np.mean(np.abs(baseline_n - yn)),
                "score_mean": y.mean(),
                "score_std": y.std(),
            }
        )
    split_df = pd.DataFrame(rows)
    split_path = os.path.join(OUT, "weight_split_sensitivity.csv")
    split_df.to_csv(split_path, index=False)
    print("\n=== Weight split sensitivity vs defaults (0.4 / 0.6) ===")
    print(split_df.to_string(index=False))

    # --- Broader hyperparameter sweep ---
    tol_vals = [10, 20, 30, 40, 50]
    pen_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    weight_vals = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4)]
    broad = []
    for (dw, pw), tol, pen in product(weight_vals, tol_vals, pen_vals):
        y = instability_score(
            df,
            mp,
            degradation_weight=dw,
            position_weight=pw,
            wavelength_tolerance=tol,
            multiple_peak_penalty=pen,
        )
        rho, _ = spearmanr(baseline, y)
        broad.append(
            {
                "degradation_weight": dw,
                "position_weight": pw,
                "wavelength_tolerance": tol,
                "multiple_peak_penalty": pen,
                "spearman_vs_defaults": rho,
                "topk48_overlap": topk_overlap(baseline, y, 48),
                "topk96_overlap": topk_overlap(baseline, y, 96),
            }
        )
    broad_df = pd.DataFrame(broad)
    broad_path = os.path.join(OUT, "hyperparameter_sensitivity.csv")
    broad_df.to_csv(broad_path, index=False)
    print("\n=== Broad sweep summary vs defaults ===")
    print(
        "Spearman: min={:.3f} median={:.3f} mean={:.3f} max={:.3f}".format(
            broad_df["spearman_vs_defaults"].min(),
            broad_df["spearman_vs_defaults"].median(),
            broad_df["spearman_vs_defaults"].mean(),
            broad_df["spearman_vs_defaults"].max(),
        )
    )
    print(
        "Top-48 overlap: min={:.3f} median={:.3f} mean={:.3f}".format(
            broad_df["topk48_overlap"].min(),
            broad_df["topk48_overlap"].median(),
            broad_df["topk48_overlap"].mean(),
        )
    )
    print(
        "Top-96 overlap: min={:.3f} median={:.3f} mean={:.3f}".format(
            broad_df["topk96_overlap"].min(),
            broad_df["topk96_overlap"].median(),
            broad_df["topk96_overlap"].mean(),
        )
    )
    worst = broad_df.sort_values("spearman_vs_defaults").head(5)
    print("\nWorst 5 settings by Spearman:")
    print(worst.to_string(index=False))

    # Fraction of compositions that remain in "stable half" across weight splits
    stable_half = {}
    for key, y in scores_by_split.items():
        thr = np.median(y)
        stable_half[key] = set(np.where(y <= thr)[0])
    ref = stable_half[(0.4, 0.6)]
    print("\n=== Stable-half (score <= median) overlap with defaults ===")
    for key, s in stable_half.items():
        ov = len(ref & s) / len(ref)
        print(f"{key}: overlap={ov:.3f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].plot(split_df["degradation_weight"], split_df["spearman_vs_0.4_0.6"], "o-", color="#1f4e79")
    axes[0].axvline(0.4, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("degradation_weight (position = 1 - deg)")
    axes[0].set_ylabel("Spearman vs (0.4, 0.6)")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_title("Rank correlation")

    axes[1].plot(split_df["degradation_weight"], split_df["topk48_overlap"], "o-", label="top 48")
    axes[1].plot(split_df["degradation_weight"], split_df["topk96_overlap"], "s-", label="top 96")
    axes[1].axvline(0.4, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("degradation_weight")
    axes[1].set_ylabel("Overlap with defaults")
    axes[1].set_ylim(0.5, 1.02)
    axes[1].legend(frameon=False)
    axes[1].set_title("Most-stable set overlap")

    # Scatter: extreme splits vs baseline
    y_low = scores_by_split[(0.2, 0.8)]
    y_high = scores_by_split[(0.8, 0.2)]
    axes[2].scatter(baseline, y_low, s=8, alpha=0.5, label="0.2/0.8")
    axes[2].scatter(baseline, y_high, s=8, alpha=0.5, label="0.8/0.2")
    lims = [0, max(baseline.max(), y_low.max(), y_high.max()) * 1.05]
    axes[2].plot(lims, lims, "k--", lw=1)
    axes[2].set_xlabel("Score (0.4 / 0.6)")
    axes[2].set_ylabel("Score (altered split)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Score concordance")

    fig.tight_layout()
    fig_path = os.path.join(OUT, "weight_split_sensitivity.png")
    fig.savefig(fig_path, dpi=200)
    print(f"\nWrote:\n  {split_path}\n  {broad_path}\n  {fig_path}")


if __name__ == "__main__":
    main()
