import json
from pathlib import Path
from typing import Any

import numpy as np

from .unified_pipeline import run_lightcurve_analysis


def calculate_physics(period_days: float, depth: float, star_mass: float, star_radius: float) -> tuple[float, float]:
    from .unified_pipeline import calculate_physics as shared_calculate_physics
    return shared_calculate_physics(period_days, depth, star_mass, star_radius)


def estimate_period_depth(time: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    from .unified_pipeline import _estimate_period_and_depth
    return _estimate_period_and_depth(time, flux)


def fit(time, flux, star_mass: float = 1.0, star_radius: float = 1.0,
        duration_hours: float = 3.2, checkpoint_path: str | Path = "models/member_b_classifier.pt") -> dict[str, Any]:
    analysis = run_lightcurve_analysis(
        time,
        flux,
        star_mass=star_mass,
        star_radius=star_radius,
        duration_hours=duration_hours,
        checkpoint_path=checkpoint_path,
    )
    payload = {
        "period": analysis["candidate"]["period"],
        "depth": analysis["candidate"]["depth"],
        "Rp/R*": analysis["candidate"]["Rp/R*"],
        "semi_major_axis": analysis["candidate"]["semi_major_axis"],
        "classification": analysis["classification"],
        "parameters": {
            "period": analysis["candidate"]["period"],
            "depth": analysis["candidate"]["depth"],
            "Rp/R*": analysis["candidate"]["Rp/R*"],
            "semi_major_axis": analysis["candidate"]["semi_major_axis"],
        },
    }
    return payload


if __name__ == "__main__":
    time_subset = np.linspace(0, 20, 500)
    flux_subset = np.random.normal(1, 0.005, 500)
    result = fit(time_subset, flux_subset, star_mass=1.05, star_radius=1.06)
    print(json.dumps(result, indent=2))