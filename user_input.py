from pydantic import BaseModel, Field
from typing import Annotated, Literal


class UserInput(BaseModel):

    Administrative: Annotated[
        int,
        Field(
            ...,
            description="Number of administrative pages visited",
            examples=[2]
        )
    ]

    Administrative_Duration: Annotated[
        float,
        Field(
            ...,
            description="Time spent on administrative pages",
            examples=[50.5]
        )
    ]

    Informational: Annotated[
        int,
        Field(
            ...,
            description="Number of informational pages visited",
            examples=[1]
        )
    ]

    Informational_Duration: Annotated[
        float,
        Field(
            ...,
            description="Time spent on informational pages",
            examples=[20.0]
        )
    ]

    ProductRelated: Annotated[
        int,
        Field(
            ...,
            description="Number of product-related pages visited",
            examples=[30]
        )
    ]

    ProductRelated_Duration: Annotated[
        float,
        Field(
            ...,
            description="Time spent on product-related pages",
            examples=[800.5]
        )
    ]

    BounceRates: Annotated[
        float,
        Field(
            ...,
            description="Bounce rate",
            examples=[0.02]
        )
    ]

    ExitRates: Annotated[
        float,
        Field(
            ...,
            description="Exit rate",
            examples=[0.04]
        )
    ]

    PageValues: Annotated[
        float,
        Field(
            ...,
            description="Page value",
            examples=[5.5]
        )
    ]

    SpecialDay: Annotated[
        float,
        Field(
            ...,
            description="Special day closeness",
            examples=[0.0]
        )
    ]

    Month: Annotated[
        str,
        Field(
            ...,
            description="Month",
            examples=["May"]
        )
    ]

    OperatingSystems: Annotated[
        int,
        Field(
            ...,
            examples=[2]
        )
    ]

    Browser: Annotated[
        int,
        Field(
            ...,
            examples=[1]
        )
    ]

    Region: Annotated[
        int,
        Field(
            ...,
            examples=[1]
        )
    ]

    TrafficType: Annotated[
        int,
        Field(
            ...,
            examples=[2]
        )
    ]

    VisitorType: Annotated[
        str,
        Field(
            ...,
            examples=["Returning_Visitor"]
        )
    ]

    Weekend: Annotated[
        bool,
        Field(
            ...,
            examples=[False]
        )
    ]


# ============================================================
# PREDICTION REQUEST
# ============================================================

class PredictionRequest(UserInput):

    model_name: Literal[
        "Best Model",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "All Models"
    ] = Field(
        default="Best Model",
        description="Machine learning model to use"
    )