# pipeline.py

def preprocess(tic_id):
    return {
        "global_view": [],
        "local_view": [],
        "stellar_metadata": {
            "teff": 5778,
            "radius": 1.0,
            "logg": 4.4
        },
        "flattened_flux": []
    }


def classify(global_view, local_view, stellar_metadata):
    return {
        "class_label": "Transit",
        "confidence_scores": {
            "Transit": 0.91,
            "Eclipse": 0.03,
            "Blend": 0.04,
            "Noise": 0.02
        }
    }


def fit(flattened_flux):
    return {
        "period": 4.05,
        "rp_rs": 0.095,
        "semi_major_axis": 0.049
    }


def run_pipeline(tic_id):

    try:

        preprocessed = preprocess(tic_id)

        prediction = classify(
            preprocessed["global_view"],
            preprocessed["local_view"],
            preprocessed["stellar_metadata"]
        )

        result = {
            "status": "success",
            "prediction": prediction
        }

        if prediction["class_label"] == "Transit":

            result["planet"] = fit(
                preprocessed["flattened_flux"]
            )

        return result

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }