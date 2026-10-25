from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    """Result returned by an individual machine learning model."""

    model: str
    prediction: bool

    purchase_probability: Optional[float] = None
    confidence: Optional[float] = None

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class PredictionResponse(BaseModel):
    """Overall response returned by the /predict endpoint."""

    prediction: bool

    purchase_probability: Optional[float] = None
    confidence: Optional[float] = None

    recommendations: List[Any] = Field(default_factory=list)

    selected_model: str

    model_results: List[ModelResult] = Field(default_factory=list)
