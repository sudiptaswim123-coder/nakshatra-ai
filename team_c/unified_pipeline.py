"""Unified exoplanet pipeline that combines preprocessing, Member B classification,
and a lightweight Member C-style candidate payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from team_b.member_b_classifier import classify

from team_a.preprocessing import (
    detrend,
    make_global_view,
    make_local_view,
    sigma_clip_outliers,
)

def _synthetic_star(period: float = 4.05, t0: float = 2.0, depth: float = 0.009025,
                    duration_hours: float = 3.2, variability_amp: float = 0.0008,
                    n_points: int = 15000, baseline_days: float = 27.0,
                    noise_std: float = 0.0015, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, baseline_days, n_points)
    variability = (
        variability_amp * np.sin(2 * np.pi * time / 6.3)
        + 0.5 * variability_amp * np.sin(2 * np.pi * time / 2.1 + 1.0)
    )
    flux = 1.0 + variability + rng.normal(0, noise_std, n_points)
    duration_days = duration_hours / 24.0
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) < (duration_days / 2)
    flux[in_transit] -= depth
    outlier_idx = rng.choice(n_points, 20, replace=False)
    flux[outlier_idx] += rng.choice([-1, 1], 20) * rng.uniform(0.01, 0.03, 20)
    return time, flux


def _estimate_period_and_depth(time: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    try:
        from astropy.timeseries import BoxLeastSquares
    except Exception:
        # Lightweight fallback for environments without astropy.
        depth = float(max(1.0 - np.nanmin(flux), 1e-4))
        period = 4.0
        return period, depth

    model = BoxLeastSquares(time, flux)
    period_grid = np.linspace(0.5, 20.0, 1000)
    duration_grid = np.linspace(0.01, 0.2, 50)
    results = model.power(period_grid, duration_grid)
    best_index = int(np.argmax(results.power))
    best_period = float(results.period[best_index])
    best_depth = float(results.depth[best_index])
    return best_period, best_depth


def calculate_physics(period_days: float, depth: float, star_mass: float, star_radius: float) -> tuple[float, float]:
    try:
        import astropy.constants as const
        import astropy.units as u
    except Exception:
        return float(np.sqrt(depth)), 0.0

    r_star = star_radius * u.R_sun
    m_star = star_mass * u.M_sun
    rp_rs_ratio = float(np.sqrt(depth))
    period = period_days * u.day
    a_cubed = (const.G * m_star * period**2) / (4 * np.pi**2)
    a_m = np.cbrt(a_cubed.to_value(u.m**3)) * u.m
    a_au = float(a_m.to_value(u.AU))
    return rp_rs_ratio, a_au


def run_lightcurve_analysis(
    time: np.ndarray | list[float],
    flux: np.ndarray | list[float],
    star_mass: float = 1.0,
    star_radius: float = 1.0,
    *,
    period_guess: float | None = None,
    t0_guess: float | None = None,
    duration_hours: float = 3.2,
    checkpoint_path: str | Path = "models/member_b_classifier.pt",
    metadata: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    clipped_flux = sigma_clip_outliers(flux)
    flattened_flux, _trend = detrend(clipped_flux, time, window_length=0.5)
    period, depth = _estimate_period_and_depth(time, flattened_flux)
    if period_guess is not None:
        period = float(period_guess)
    if t0_guess is None:
        t0 = float(time[int(np.argmin(flattened_flux))])
    else:
        t0 = float(t0_guess)

    global_view = make_global_view(time, flattened_flux, period, t0, n_bins=2000)
    local_view = make_local_view(time, flattened_flux, period, t0, duration_hours, n_bins=200)

    if metadata is None:
        metadata_array = np.array([5800.0, float(star_radius), 4.4, float(star_mass)], dtype=np.float32)
    else:
        metadata_array = np.asarray(metadata, dtype=np.float32).reshape(4)

    classification = classify(global_view, local_view, metadata_array, None, checkpoint_path=checkpoint_path)
    rp_rs, a_au = calculate_physics(period, depth, star_mass, star_radius)

    candidate = {
        "period": round(period, 4),
        "depth": round(depth, 6),
        "duration_hours": round(duration_hours, 4),
        "t0": round(t0, 4),
        "Rp/R*": round(rp_rs, 4),
        "semi_major_axis": round(a_au, 4),
        "global_view": global_view.tolist(),
        "local_view": local_view.tolist(),
    }
    return {"status": "ok", "candidate": candidate, "classification": classification}


def run_demo_pipeline(seed: int = 7) -> dict[str, Any]:
    time, flux = _synthetic_star(seed=seed)
    result = run_lightcurve_analysis(time, flux, star_mass=1.05, star_radius=1.06, duration_hours=3.2)
    return result


def run_batch_manifest(manifest_csv: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_csv)
    rows = []
    with manifest_path.open(newline="") as handle:
        rows.extend(__import__("csv").DictReader(handle))

    outputs = []
    for row in rows:
        star_path = manifest_path.parent / row["file"]
        data = np.load(star_path)
        metadata = np.array([float(data["teff"]), float(data["radius"]), float(data["logg"]), float(data["mass"])], dtype=np.float32)
        classification = classify(
            np.asarray(data["global_view"], dtype=np.float32),
            np.asarray(data["local_view"], dtype=np.float32),
            metadata,
            None,
            checkpoint_path="models/member_b_classifier.pt",
        )
        outputs.append({
            "tic_id": row["tic_id"],
            "label": str(data["label"]),
            "classification": classification,
            "file": row["file"],
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified exoplanet pipeline")
    parser.add_argument("--demo", action="store_true", help="Run the synthetic demo workflow")
    parser.add_argument("--manifest", default=None, help="Optional manifest CSV to classify")
    args = parser.parse_args()

    if args.manifest:
        results = run_batch_manifest(args.manifest)
        print(json.dumps(results, indent=2))
        return

    if not args.demo:
        parser.error("Use --demo or --manifest")

    result = run_demo_pipeline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
