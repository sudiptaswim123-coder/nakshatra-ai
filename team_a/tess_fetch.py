import lightkurve as lk
import numpy as np

def fetch_tess_lightcurve(tic_id):

    tic_id = str(tic_id).replace("TIC", "").strip()

    target = f"TIC {tic_id}"

    search_result = lk.search_lightcurve(
        target,
        mission="TESS"
    )

    print(search_result)

    if len(search_result) == 0:
        raise ValueError(
            f"No TESS data found for {target}"
        )

    lc = search_result[0].download()

    lc = lc.remove_nans()

    time = np.array(
        lc.time.value,
        dtype=float
    )

    flux = np.array(
        lc.flux.value,
        dtype=float
    )

    flux = flux / np.nanmedian(flux)

    return time, flux