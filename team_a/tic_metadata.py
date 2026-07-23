from astroquery.mast import Catalogs
import numpy as np
def safe_value(value, default):
    try:
        if value is None:
            return default

        if np.ma.is_masked(value):
            return default

        if np.isnan(value):
            return default

        return float(value)

    except Exception:
        return default


def get_tic_metadata(tic_id):

    catalog = Catalogs.query_criteria(
        catalog="TIC",
        ID=int(tic_id)
    )

    if len(catalog) == 0:

        return np.array(
            [5800,1.0,4.4,1.0],
            dtype=np.float32
        )

    row = catalog[0]

    teff = (
        float(row["Teff"])
        if row["Teff"] is not None and not np.isnan(row["Teff"])
        else 5800
    )

    radius = (
        float(row["rad"])
        if row["rad"] is not None and not np.isnan(row["rad"])
        else 1.0
    )

    mass = (
        float(row["mass"])
        if row["mass"] is not None and not np.isnan(row["mass"])
        else 1.0
    )

    logg = (
        float(row["logg"])
        if row["logg"] is not None and not np.isnan(row["logg"])
        else 4.4
    )

    return np.array(
        [
            float(teff),
            float(radius),
            float(logg),
            float(mass)
        ],
        dtype=np.float32
    )