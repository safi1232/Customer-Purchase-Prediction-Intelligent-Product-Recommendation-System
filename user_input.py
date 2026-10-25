from pydantic import BaseModel, Field
from typing import Annotated, Literal


class UserInput(BaseModel):
    Administrative: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of administrative pages visited",
            examples=[2],
        ),
    ]

    Administrative_Duration: Annotated[
        float,
        Field(
            ...,
            ge=0,
            description="Time spent on administrative pages",
            examples=[50.5],
        ),
    ]

    Informational: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of informational pages visited",
            examples=[1],
        ),
    ]

    Informational_Duration: Annotated[
        float,
        Field(
            ...,
            ge=0,
            description="Time spent on informational pages",
            examples=[20.0],
        ),
    ]

    ProductRelated: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of product-related pages visited",
            examples=[30],
        ),
    ]

    ProductRelated_Duration: Annotated[
        float,
        Field(
            ...,
            ge=0,
            description="Time spent on product-related pages",
            examples=[800.5],
        ),
    ]

    BounceRates: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=1,
            description="Bounce rate (between 0 and 1)",
            examples=[0.02],
        ),
    ]

    ExitRates: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=1,
            description="Exit rate (between 0 and 1)",
            examples=[0.04],
        ),
    ]

    PageValues: Annotated[
        float,
        Field(
            ...,
            ge=0,
            description="Page value",
            examples=[5.5],
        ),
    ]

    SpecialDay: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=1,
            description="Special day closeness (between 0 and 1)",
            examples=[0.0],
        ),
    ]

    Month: Annotated[
        str,
        Field(
            ...,
            description="Month",
            examples=["May"],
        ),
    ]

    OperatingSystems: Annotated[
        int,
        Field(
            ...,
            ge=1,
            description="Operating system category",
            examples=[2],
        ),
    ]

    Browser: Annotated[
        int,
        Field(
            ...,
            ge=1,
            description="Browser category",
            examples=[1],
        ),
    ]

    Region: Annotated[
        int,
        Field(
            ...,
            ge=1,
            description="Region category",
            examples=[1],
        ),
    ]

    TrafficType: Annotated[
        int,
        Field(
            ...,
            ge=1,
            description="Traffic source category",
            examples=[2],
        ),
    ]

    VisitorType: Annotated[
        str,
        Field(
            ...,
            description="Visitor type",
            examples=["Returning_Visitor"],
        ),
    ]

    Weekend: Annotated[
        bool,
        Field(
            ...,
            description="Whether the visit occurred on a weekend",
            examples=[False],
        ),
    ]
    model_name: Literal[
        "Best Model",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "All Models",
    ] = "Best Model"

