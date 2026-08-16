from pydantic import BaseModel, Field
from typing import List, Optional, Any


class PredictionResponse(BaseModel):

    model: str = Field(
        ...,
        description="Name of the model this result belongs to",
        examples=["Random Forest"]
    )

    prediction: bool = Field(
        ...,
        description="Predicted purchase outcome",
        examples=[True]
    )

    purchase_probability: Optional[float] = Field(
        default=None,
        description="Probability the customer will purchase",
        examples=[0.82]
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Model's confidence in its prediction",
        examples=[0.91]
    )

    accuracy: Optional[float] = Field(
        default=None,
        description="Model accuracy on the held-out test set",
        examples=[0.89]
    )

    precision: Optional[float] = Field(
        default=None,
        description="Model precision on the held-out test set",
        examples=[0.75]
    )

    recall: Optional[float] = Field(
        default=None,
        description="Model recall on the held-out test set",
        examples=[0.68]
    )

    f1_score: Optional[float] = Field(
        default=None,
        description="Model F1 score on the held-out test set",
        examples=[0.71]
    )


class PredictionResponse(BaseModel):

    prediction: bool = Field(
        ...,
        description="Predicted purchase outcome",
        examples=[True]
    )

    purchase_probability: Optional[float] = Field(
        default=None,
        description="Probability the customer will purchase",
        examples=[0.82]
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Model's confidence in its prediction",
        examples=[0.91]
    )

    recommendations: List[Any] = Field(
        default_factory=list,
        description="Product recommendations (currently always empty from /predict)"
    )

    selected_model: str = Field(
        ...,
        description="Name of the model actually used for this prediction",
        examples=["Random Forest"]
    )

    model_results: List[PredictionResponse] = Field(
        ...,
        description="Per-model breakdown of the prediction"
    )