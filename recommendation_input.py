from pydantic import BaseModel, Field


class RecommendationInput(BaseModel):

    stock_code: str = Field(
        ...,
        description="Product StockCode used to generate similar product recommendations",
        examples=["85123A"]
    )

    n: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of products to recommend",
        examples=[10]
    )