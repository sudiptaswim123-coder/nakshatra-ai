print("PIPELINE LOADED")
from team_a.preprocessing import preprocess
from team_b.member_b_classifier import classify
from team_c.unified_pipeline import run_lightcurve_analysis

def run_pipeline(tic_id):
    print("RUN PIPELINE CALLED")

    # এখন dummy data
    # পরে Team A এর TESS data fetching এখানে বসবে

    import numpy as np

    time = np.linspace(0, 30, 500)
    flux = 1 + np.random.normal(0, 0.002, 500)

    result = run_lightcurve_analysis(
        time=time,
        flux=flux,
        star_mass=1.0,
        star_radius=1.0
    )

    prediction = result["classification"]
    candidate = result["candidate"]

    return {
        "prediction": prediction,
        "planet": {
            "period": candidate["period"],
            "rp_rs": candidate["Rp/R*"],
            "semi_major_axis": candidate["semi_major_axis"]
        }
    }