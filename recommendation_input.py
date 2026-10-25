from pydantic import BaseModel, Field


class RecommendationInput(BaseModel):
    product_name: str = Field(
        ...,
        min_length=1,
        description="Exact product description"
    )

    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations"
    )
