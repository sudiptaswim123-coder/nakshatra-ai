
from team_a.tess_fetch import fetch_tess_lightcurve
from team_a.tic_metadata import get_tic_metadata

from team_c.unified_pipeline import run_lightcurve_analysis


def run_pipeline(tic_id):

    time, flux = fetch_tess_lightcurve(tic_id)

    metadata = get_tic_metadata(tic_id)

    result = run_lightcurve_analysis(

        time=time,

        flux=flux,

        star_mass=float(metadata[3]),

        star_radius=float(metadata[1]),

        metadata=metadata
    )

    return {

        "prediction":
        result["classification"],

        "planet":{

            "period":
            result["candidate"]["period"],

            "rp_rs":
            result["candidate"]["Rp/R*"],

            "semi_major_axis":
            result["candidate"]["semi_major_axis"]

        },

        "time":time,

        "flux":flux,

        "metadata":metadata
    }
