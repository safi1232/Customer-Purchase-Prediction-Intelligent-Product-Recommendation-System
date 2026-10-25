from fastapi import FastAPI, HTTPException
import pandas as pd

from model.predict import (
    predict_output,
    system,
    MODEL_VERSION
)

from schemas.prediction_response import PredictionResponse
from schemas.user_input import UserInput
from schemas.recommendation_input import RecommendationInput
import json
from pathlib import Path

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Purchase & Recommendation API",
    description="API for customer purchase prediction and product recommendation",
    version=MODEL_VERSION
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Customer Purchase & Recommendation API",
        "status": "ok",
        "version": MODEL_VERSION
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": MODEL_VERSION,
        "model_loaded": system is not None
    }


# ============================================================
# PURCHASE PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(user_input: UserInput):

    # --------------------------------------------------------
    # Convert request to dictionary
    # Exclude model_name because it is not a model feature.
    # --------------------------------------------------------

    if hasattr(user_input, "model_dump"):
        user_data = user_input.model_dump(
            exclude={"model_name"}
        )
    else:
        user_data = user_input.dict(
            exclude={"model_name"}
        )

    # Read selected model directly from the request.
    # Do NOT use user_data.pop("model_name").
    model_name = user_input.model_name

    print("\n===================================")
    print("PURCHASE PREDICTION REQUEST")
    print("===================================")
    print("Selected Model:", model_name)
    print("User Data:", user_data)

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:
        result = predict_output(
            user_data,
            model_name=model_name
        )

    except Exception as e:
        print("\nPREDICTION ERROR:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    print("\n===================================")
    print("PREDICTION RESULT")
    print("===================================")
    print(result)

    try:
        return PredictionResponse(**result)

    except Exception as e:
        print("\nRESPONSE VALIDATION ERROR:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Invalid prediction response: {str(e)}"
        )


# ============================================================
# PRODUCT RECOMMENDATION
# ============================================================

@app.post("/recommend")
def recommend(
    recommendation_input: RecommendationInput
):

    # --------------------------------------------------------
    # Check recommendation system
    # --------------------------------------------------------

    if system is None:
        raise HTTPException(
            status_code=500,
            detail="Recommendation system is not loaded."
        )

    # --------------------------------------------------------
    # Read request
    # Expected schema fields: product_name and n
    # --------------------------------------------------------

    if hasattr(recommendation_input, "model_dump"):
        data = recommendation_input.model_dump()
    else:
        data = recommendation_input.dict()

    product_name = data["product_name"].strip()
    n = data["n"]

    if not product_name:
        raise HTTPException(
            status_code=422,
            detail="Product name cannot be empty."
        )

    print("\n===================================")
    print("PRODUCT RECOMMENDATION REQUEST")
    print("===================================")
    print("Product Name:", product_name)
    print("Number of Recommendations:", n)

    # --------------------------------------------------------
    # Validate product data
    # --------------------------------------------------------

    products = system.products

    if products is None or products.empty:
        raise HTTPException(
            status_code=500,
            detail="Product data is empty or unavailable."
        )

    required_columns = {"Description", "StockCode"}

    if not required_columns.issubset(products.columns):
        raise HTTPException(
            status_code=500,
            detail=(
                "Product data must contain 'Description' "
                "and 'StockCode' columns."
            )
        )

    # --------------------------------------------------------
    # Find product by name (case-insensitive exact match)
    # --------------------------------------------------------

    descriptions = (
        products["Description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    matches = products[
        descriptions.str.casefold() == product_name.casefold()
    ]

    if matches.empty:
        # Return a few matching names to help the user
        similar_names = (
            products.loc[
                descriptions.str.contains(
                    product_name,
                    case=False,
                    regex=False
                ),
                "Description"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Product '{product_name}' was not found.",
                "suggestions": similar_names
            }
        )

    # --------------------------------------------------------
    # Collect every stock code for this description.
    # A description can occur under several stock codes, and the
    # recommender may expect a specific type (str vs int).
    # We try each code in its original type and as a clean string.
    # --------------------------------------------------------

    original_codes = matches["StockCode"].dropna().unique().tolist()

    candidate_codes = []

    for c in original_codes:
        for variant in (c, str(c).strip()):
            if variant not in candidate_codes:
                candidate_codes.append(variant)

    if not candidate_codes:
        raise HTTPException(
            status_code=404,
            detail="No stock code is available for this product."
        )

    # --------------------------------------------------------
    # TEMPORARY DEBUG OUTPUT (remove once the issue is found)
    # --------------------------------------------------------

    print("StockCode dtype in system.products:", products["StockCode"].dtype)
    print("Candidate codes:", [(repr(c), type(c).__name__) for c in candidate_codes])
    print(
        "Recommender attributes:",
        [a for a in dir(system) if not a.startswith("_")]
    )

    # --------------------------------------------------------
    # Generate recommendations (try each candidate code)
    # --------------------------------------------------------

    recommendations_df = None
    stock_code = None
    last_error = None

    for code in candidate_codes:

        try:
            out = system.recommend(code, n=n)

        except Exception as e:
            last_error = e
            print(f"RECOMMEND ERROR for {code!r} ({type(code).__name__}): {e}")
            continue

        print(
            f"recommend({code!r}) -> {type(out).__name__}, "
            f"empty={getattr(out, 'empty', 'n/a')}"
        )

        if out is not None and not out.empty:
            recommendations_df = out
            stock_code = str(code)
            break

    # --------------------------------------------------------
    # Handle no recommendations
    # --------------------------------------------------------

    if recommendations_df is None:

        detail = f"No recommendations were found for '{product_name}'."

        if last_error is not None:
            detail += f" Last error: {last_error}"

        raise HTTPException(
            status_code=404,
            detail=detail
        )

    # --------------------------------------------------------
    # Convert DataFrame to JSON-safe records
    # Replace missing values with None (JSON null).
    # --------------------------------------------------------

    recommendations_df = recommendations_df.astype(object).where(
        pd.notna(recommendations_df),
        None
    )

    recommendations = recommendations_df.to_dict(
        orient="records"
    )

    # --------------------------------------------------------
    # Return recommendations
    # --------------------------------------------------------

    response = {
        "product_name": product_name,
        "stock_code": stock_code,
        "number_of_recommendations": len(recommendations),
        "recommendations": recommendations
    }

    print("\n===================================")
    print("RECOMMENDATION RESULT")
    print("===================================")
    print(response)

    return response


# ============================================================
# PRODUCT LIST (for the frontend dropdown)
# ============================================================

@app.get("/products")
def list_products():

    if system is None or system.products is None or system.products.empty:
        raise HTTPException(
            status_code=500,
            detail="Product data is empty or unavailable."
        )

    names = (
        system.products["Description"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    product_names = sorted(n for n in names.unique() if n)

    return {"products": product_names}
METRICS_PATH = Path("model/model_metrics.json")


@app.get("/models")
def list_models():

    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found at '{METRICS_PATH}'. Generate it at training time."
        )

    try:
        with open(METRICS_PATH, "r") as f:
            saved = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read metrics file: {e}")

    models = saved.get("models", [])
    best_model = saved.get("best_model")

    for m in models:
        m["is_best"] = (m.get("model") == best_model)

    return {
        "best_model": best_model,
        "total_models": len(models),
        "models": models
    }
