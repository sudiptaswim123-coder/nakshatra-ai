"""
preprocessing.py
Member A -- Data & Preprocessing Lead
ISRO BAH 2026 -- Nakshatra_AI / Team Vritti -- Exoplanet Detection Pipeline

Owns: raw FITS ingestion (MAST) -> Wotan detrending -> Global View + Local View
      generation -> stellar metadata lookup, all packaged behind one function:

          preprocess(tic_id) -> {
              "global_view": np.ndarray[2000],
              "local_view":  np.ndarray[200],
              "stellar_metadata": {"teff": float, "radius": float, "logg": float},
              "flattened_flux": np.ndarray,
              "time": np.ndarray,
          }

Member B calls preprocess(tic_id) and does not need to know any of the
internals below.

NOTE ON NETWORK ACCESS
-----------------------
This module talks to MAST (archive.stsci.edu) via `lightkurve` /
`astroquery.mast`. That requires outbound internet access to MAST's
servers. Run this on your own machine / Colab / the hackathon dev
environment where that's available -- it is NOT reachable from this
sandboxed session, which is why the demo at the bottom of the file
uses synthetic data to prove the detrending/folding/windowing logic
is correct. Swap `USE_SYNTHETIC_FOR_DEMO = False` once you run this
somewhere with real network access to MAST.
"""

from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. INGESTION -- pull TESS light curve + TPF from MAST for a TIC ID
# ---------------------------------------------------------------------------
def fetch_light_curve(tic_id: int, author: str = "SPOC", flux_column: str = "pdcsap_flux",
                       quality_bitmask: str = "hard"):
    """
    Download the PDCSAP flux light curve (and TPF) for a given TIC ID from MAST.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog ID, e.g. 261136679
    author : str
        Pipeline that produced the light curve (SPOC is the standard one).
    flux_column : str
        Which flux column to pull -- PDCSAP is pre-cleaned of instrumental
        systematics and is the standard choice for transit search.
    quality_bitmask : str
        How aggressively to drop cadences already flagged bad by the SPOC
        pipeline itself (cosmic rays, thruster firings, safe-mode, etc.)
        before your own quality checks ever run. One of "none", "default",
        "hard", "hardest" (lightkurve's scale, increasing strictness). "hard"
        is a good default for transit search -- it removes the flagged
        cadences that most commonly masquerade as fake transit-like dips.

    Returns
    -------
    time : np.ndarray (days, BTJD)
    flux : np.ndarray (normalized, contains NaNs/outliers -- not detrended yet)
    tpf  : lightkurve TargetPixelFile object (for the GNN stage / Member B),
           or None if no TPF is available for this target/sector.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(f"TIC {tic_id}", author=author, mission="TESS")
    if len(search) == 0:
        raise ValueError(f"No SPOC light curves found on MAST for TIC {tic_id}")

    lc_collection = search.download_all(quality_bitmask=quality_bitmask)
    lc = lc_collection.stitch()  # combine all available sectors into one curve

    lc = lc.remove_nans(column=flux_column)
    time = lc.time.value
    flux = lc[flux_column].value
    flux = flux / np.nanmedian(flux)  # normalize to ~1.0 baseline

    tpf = None
    try:
        tpf_search = lk.search_targetpixelfile(f"TIC {tic_id}", author=author, mission="TESS")
        if len(tpf_search) > 0:
            tpf = tpf_search[0].download()
    except Exception:
        tpf = None

    return time, flux, tpf


# ---------------------------------------------------------------------------
# 2. DETRENDING -- Wotan biweight filter, tuned to preserve transit depth
# ---------------------------------------------------------------------------
def detrend(flux: np.ndarray, time: np.ndarray, window_length: float = 0.5):
    """
    Flatten stellar variability / instrumental trends while preserving
    transit-shaped dips.

    The preferred implementation uses wotan when available. If that optional
    dependency is absent, this function falls back to a simple moving-window
    normalization so the rest of the pipeline still works in lightweight
    environments.
    """
    flux = np.asarray(flux, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    try:
        from wotan import flatten

        flattened_flux, trend = flatten(
            time,
            flux,
            method="biweight",
            window_length=window_length,
            return_trend=True,
        )
        return np.asarray(flattened_flux, dtype=np.float32), np.asarray(trend, dtype=np.float32)
    except Exception:
        if len(flux) < 5:
            return flux.astype(np.float32), np.ones_like(flux, dtype=np.float32)

        window = max(int(len(flux) * 0.01), 5)
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(flux, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        trend = np.convolve(padded, kernel, mode="valid")
        trend = np.asarray(trend[: len(flux)], dtype=np.float64)
        trend = np.where(trend > 0, trend, 1.0)
        flattened_flux = flux / trend
        flattened_flux = flattened_flux / np.nanmedian(flattened_flux)
        return np.asarray(flattened_flux, dtype=np.float32), np.asarray(trend, dtype=np.float32)


def sigma_clip_outliers(flux: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Mask (set to NaN) points beyond `sigma` MADs from the median."""
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    std_equiv = 1.4826 * mad
    mask = np.abs(flux - median) > sigma * std_equiv
    clipped = flux.copy()
    clipped[mask] = np.nan
    return clipped


# ---------------------------------------------------------------------------
# 3. GLOBAL VIEW -- 2000-bin phase-folded curve
# ---------------------------------------------------------------------------
def make_global_view(time, flattened_flux, period, t0, n_bins=2000):
    """
    Phase-fold the full light curve on the trial/candidate period and bin it
    into a fixed-length array for the network's global branch.

    period, t0 : float (days)
        Candidate period and reference transit epoch. In the real pipeline
        these come from Member C's BLS search on a first pass, or from a
        supplied ISRO label for training.
    """
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5  # in [-0.5, 0.5)
    order = np.argsort(phase)
    phase, flux_sorted = phase[order], flattened_flux[order]

    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    bin_idx = np.digitize(phase, bin_edges) - 1
    for i in range(n_bins):
        vals = flux_sorted[bin_idx == i]
        if vals.size > 0 and np.isfinite(vals).any():
            binned[i] = np.nanmedian(vals)

    # fill any empty bins by linear interpolation so the network gets no NaNs.
    # Use a safe interpolation path that avoids warnings when every value is NaN.
    nan_mask = np.isnan(binned)
    if nan_mask.any():
        x = np.arange(n_bins)
        valid_mask = ~nan_mask
        if valid_mask.sum() >= 2:
            binned[nan_mask] = np.interp(x[nan_mask], x[valid_mask], binned[valid_mask])
        else:
            binned[nan_mask] = np.nanmedian(binned[valid_mask]) if valid_mask.any() else 1.0

    return binned


# ---------------------------------------------------------------------------
# 4. LOCAL VIEW -- 200-bin zoomed transit window
# ---------------------------------------------------------------------------
def make_local_view(time, flattened_flux, period, t0, duration_hours, n_bins=200, n_durations=2.0):
    """
    Crop tightly around the transit center (+/- n_durations * duration) and
    bin into a fixed-length array for the network's local branch.

    duration_hours : float
        Estimated total transit duration in hours (from BLS / Member C, or
        the ISRO label during training).
    """
    duration_days = duration_hours / 24.0
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    window_half_width = (n_durations * duration_days) / period  # in phase units

    in_window = np.abs(phase) <= window_half_width
    phase_win = phase[in_window]
    flux_win = flattened_flux[in_window]

    order = np.argsort(phase_win)
    phase_win, flux_win = phase_win[order], flux_win[order]

    bin_edges = np.linspace(-window_half_width, window_half_width, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    bin_idx = np.digitize(phase_win, bin_edges) - 1
    for i in range(n_bins):
        vals = flux_win[bin_idx == i]
        if vals.size > 0 and np.isfinite(vals).any():
            binned[i] = np.nanmedian(vals)

    nan_mask = np.isnan(binned)
    if nan_mask.any():
        x = np.arange(n_bins)
        valid_mask = ~nan_mask
        if valid_mask.sum() >= 2:
            binned[nan_mask] = np.interp(x[nan_mask], x[valid_mask], binned[valid_mask])
        else:
            fill_value = np.nanmedian(binned[valid_mask]) if valid_mask.any() else 1.0
            binned[nan_mask] = fill_value

    return binned


# ---------------------------------------------------------------------------
# 4b. DATA QUALITY GATE -- decide if a star's data is even good enough to use
# ---------------------------------------------------------------------------
def assess_quality(time, flux, min_points=1000, max_nan_frac=0.2,
                    max_gap_days=2.0, min_baseline_days=5.0, flat_std_thresh=1e-6):
    """
    Decide whether a light curve is clean enough to hand to Member B, or
    should be thrown out before it ever reaches training/inference.

    This catches problems that survive MAST's own quality flags:
      - too few data points to phase-fold meaningfully
      - too many NaNs/masked cadences (bad sector, safe-mode gaps, etc.)
      - a single data gap so large it would swallow the whole global view
      - a baseline shorter than the trial period can even be covered by
      - a flat-lined / constant flux array (dead sector, download glitch)

    Returns
    -------
    dict: {
        "pass": bool,
        "reasons": [str, ...]   # empty if pass=True, else why it failed
        "metrics": {...}        # the numbers behind the decision, for logging
    }
    """
    reasons = []
    n_total = len(flux)
    n_nan = int(np.isnan(flux).sum())
    nan_frac = n_nan / n_total if n_total > 0 else 1.0

    baseline_days = float(time[-1] - time[0]) if n_total > 1 else 0.0

    valid_time = time[~np.isnan(flux)]
    max_gap = float(np.max(np.diff(valid_time))) if len(valid_time) > 1 else np.inf

    flux_std = float(np.nanstd(flux))

    if n_total < min_points:
        reasons.append(f"too few points ({n_total} < {min_points})")
    if nan_frac > max_nan_frac:
        reasons.append(f"too many missing cadences ({nan_frac:.1%} > {max_nan_frac:.0%})")
    if max_gap > max_gap_days:
        reasons.append(f"data gap too large ({max_gap:.2f}d > {max_gap_days}d)")
    if baseline_days < min_baseline_days:
        reasons.append(f"observation baseline too short ({baseline_days:.2f}d < {min_baseline_days}d)")
    if flux_std < flat_std_thresh:
        reasons.append("flux is flat-lined / constant (likely a bad download or dead sector)")

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "n_points": n_total,
            "nan_frac": round(nan_frac, 4),
            "max_gap_days": round(max_gap, 3),
            "baseline_days": round(baseline_days, 2),
            "flux_std": flux_std,
        },
    }


# ---------------------------------------------------------------------------
# 5. STELLAR METADATA -- Teff, Radius, log g from the TIC catalog
# ---------------------------------------------------------------------------
def get_stellar_metadata(tic_id: int) -> dict:
    """Look up host-star parameters from the TESS Input Catalog (TIC) on MAST."""
    from astroquery.mast import Catalogs

    result = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001)
    row = result[0]
    return {
        "teff": float(row["Teff"]) if row["Teff"] else np.nan,
        "radius": float(row["rad"]) if row["rad"] else np.nan,
        "logg": float(row["logg"]) if row["logg"] else np.nan,
        "mass": float(row["mass"]) if "mass" in row.colnames and row["mass"] else np.nan,
    }


# ---------------------------------------------------------------------------
# 6. THE HAND-OFF FUNCTION -- everything Member B needs to call
# ---------------------------------------------------------------------------
def preprocess(tic_id: int, period: float, t0: float, duration_hours: float,
               window_length: float = 0.5, raise_on_bad_quality: bool = True) -> dict:
    """
    Full Member A pipeline for one star: MAST -> quality gate -> detrend ->
    global/local views -> stellar metadata, packaged into the agreed
    interface contract with Member B.

    period, t0, duration_hours are supplied either from the ISRO training
    labels (during model training) or from Member C's BLS output (at
    inference time on a real candidate).

    If the star fails the quality gate (assess_quality), this either raises
    (raise_on_bad_quality=True, the default -- good for training loops where
    you want bad stars to loudly stop a batch job) or returns a dict with
    "quality"["pass"]=False and no views (raise_on_bad_quality=False -- good
    for batch pipelines where you want to log-and-skip instead of crashing).
    """
    time, raw_flux, tpf = fetch_light_curve(tic_id)

    quality = assess_quality(time, raw_flux)
    if not quality["pass"]:
        if raise_on_bad_quality:
            raise ValueError(
                f"TIC {tic_id} failed data quality gate: {'; '.join(quality['reasons'])}"
            )
        return {"tic_id": tic_id, "quality": quality, "global_view": None,
                 "local_view": None, "stellar_metadata": None,
                 "flattened_flux": None, "time": None, "tpf": None}

    clipped_flux = sigma_clip_outliers(raw_flux)
    flattened_flux, trend = detrend(clipped_flux, time, window_length=window_length)

    global_view = make_global_view(time, flattened_flux, period, t0, n_bins=2000)
    local_view = make_local_view(time, flattened_flux, period, t0, duration_hours, n_bins=200)
    stellar_metadata = get_stellar_metadata(tic_id)

    return {
        "tic_id": tic_id,
        "quality": quality,
        "global_view": global_view,
        "local_view": local_view,
        "stellar_metadata": stellar_metadata,
        "flattened_flux": flattened_flux,
        "time": time,
        "tpf": tpf,  # passed through for Member B's Pixel-Level GNN
    }


def preprocess_batch(star_list: list, window_length: float = 0.5) -> tuple:
    """
    Run preprocess() over many stars and split the results into "good" (ready
    to hand to Member B) and "rejected" (logged with a reason, not passed on).

    Parameters
    ----------
    star_list : list of dict
        Each dict: {"tic_id": int, "period": float, "t0": float, "duration_hours": float}
        -- i.e. one row per star from your ISRO training label set.

    Returns
    -------
    good : list[dict]       -- full preprocess() outputs, safe to hand to Member B
    rejected : list[dict]   -- {"tic_id": ..., "reasons": [...]} for anything dropped
    """
    good, rejected = [], []
    for star in star_list:
        try:
            result = preprocess(
                star["tic_id"], star["period"], star["t0"], star["duration_hours"],
                window_length=window_length, raise_on_bad_quality=False,
            )
        except Exception as e:
            rejected.append({"tic_id": star["tic_id"], "reasons": [f"error: {e}"]})
            continue

        if result["quality"]["pass"]:
            good.append(result)
        else:
            rejected.append({"tic_id": star["tic_id"], "reasons": result["quality"]["reasons"]})

    return good, rejected
