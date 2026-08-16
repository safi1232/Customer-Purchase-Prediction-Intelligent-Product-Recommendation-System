from fastapi import FastAPI, HTTPException

from model.predict import (
    predict_output,
    system,
    MODEL_VERSION
)

from schemas.prediction_response import PredictionResponse
from schemas.user_input import PredictionRequest
from schemas.recommendation_input import RecommendationInput


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
def predict(user_input: PredictionRequest):

    # --------------------------------------------------------
    # Convert Pydantic object to dictionary
    # --------------------------------------------------------

    if hasattr(user_input, "model_dump"):

        user_data = user_input.model_dump()

    else:

        user_data = user_input.dict()


    print("\n===================================")
    print("USER DATA")
    print("===================================")

    print(user_data)


    # --------------------------------------------------------
    # Get selected model
    # --------------------------------------------------------

    model_name = user_data.pop(
        "model_name"
    )


    print("\nSelected Model:")
    print(model_name)


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
    # Print result
    # --------------------------------------------------------

    print("\n===================================")
    print("RESULT")
    print("===================================")

    print(result)


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return PredictionResponse(
        **result
    )


# ============================================================
# PRODUCT RECOMMENDATION
# ============================================================

@app.post(
    "/recommend"
)
def recommend(
    recommendation_input: RecommendationInput
):

    # --------------------------------------------------------
    # Check system
    # --------------------------------------------------------

    if system is None:

        raise HTTPException(
            status_code=500,
            detail="Recommendation system is not loaded."
        )


    # --------------------------------------------------------
    # Get input
    # --------------------------------------------------------

    if hasattr(
        recommendation_input,
        "model_dump"
    ):

        data = recommendation_input.model_dump()

    else:

        data = recommendation_input.dict()


    stock_code = data["stock_code"]

    n = data["n"]


    print("\n===================================")
    print("RECOMMENDATION REQUEST")
    print("===================================")

    print("Stock Code:", stock_code)
    print("Number:", n)


    # --------------------------------------------------------
    # Get recommendations
    # --------------------------------------------------------

    try:

        recommendations = system.recommend(
            stock_code,
            n=n
        )

    except Exception as e:

        print("\nRECOMMENDATION ERROR:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Product not found
    # --------------------------------------------------------

    if recommendations.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Product '{stock_code}' "
                "was not found in the recommendation system."
            )
        )


    # --------------------------------------------------------
    # Convert DataFrame to JSON
    # --------------------------------------------------------

    recommendations = (
        recommendations
        .where(
            recommendations.notna(),
            None
        )
        .to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # Return recommendations
    # --------------------------------------------------------

    return {

        "stock_code": stock_code,

        "number_of_recommendations": len(
            recommendations
        ),

        "recommendations": recommendations

    }