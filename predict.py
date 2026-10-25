import os
import pickle
import pandas as pd

from model.customer_syestem import CustomerSystem


MODEL_VERSION = "1.0.0"


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "customer_purchase_recommendation.pkl"
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as f:

    system = pickle.load(f)


# ============================================================
# CHECK LOADED SYSTEM
# ============================================================

if not isinstance(
    system,
    CustomerSystem
):

    raise TypeError(
        "Loaded pickle is not a valid CustomerSystem object."
    )


# ============================================================
# HELPER: build a single model's result block
# ============================================================

def _build_model_result(df, model_key):

    prediction = system.predict_purchase(
        df,
        model_name=model_key
    )[0]

    probability = system.purchase_probability(
        df,
        model_name=model_key
    )

    confidence = system.confidence(
        df,
        model_name=model_key
    )

    metrics = system.model_metrics.get(
        model_key,
        {}
    )

    return {

        "model":
            model_key,

        "prediction":
            bool(prediction),

        "purchase_probability":
            probability,

        "confidence":
            confidence,

        "accuracy":
            metrics.get("accuracy"),

        "precision":
            metrics.get("precision"),

        "recall":
            metrics.get("recall"),

        "f1_score":
            metrics.get("f1_score")

    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_output(
    user_input: dict,
    model_name="Best Model"
):

    df = pd.DataFrame(
        [user_input]
    )


    # ========================================================
    # ALL MODELS -> run every trained pipeline
    # ========================================================

    if model_name == "All Models":

        model_results = [

            _build_model_result(df, key)

            for key in system.purchase_pipelines.keys()

        ]

        best_result = next(
            r for r in model_results
            if r["model"] == system.best_model
        )

        prediction = best_result["prediction"]
        purchase_probability = best_result["purchase_probability"]
        confidence = best_result["confidence"]
        selected_model = "All Models"


    # ========================================================
    # SINGLE MODEL -> "Best Model" or a specific named model
    # ========================================================

    else:

        single_result = _build_model_result(
            df,
            model_name
        )

        model_results = [single_result]

        prediction = single_result["prediction"]
        purchase_probability = single_result["purchase_probability"]
        confidence = single_result["confidence"]

        selected_model = (
            system.best_model
            if model_name == "Best Model"
            else model_name
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "prediction":
            prediction,

        "purchase_probability":
            purchase_probability,

        "confidence":
            confidence,

        "recommendations":
            [],

        "selected_model":
            selected_model,

        "model_results":
            model_results

    }
