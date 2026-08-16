import pandas as pd


class CustomerSystem:

    def __init__(self, purchase_pipelines, best_model, similarity_df, products, model_metrics):
        # purchase_pipelines: dict mapping model name -> fitted pipeline
        # e.g. {"Logistic Regression": pipe, "XGBoost": pipe, ...}
        self.purchase_pipelines = purchase_pipelines
        self.best_model = best_model
        self.similarity_df = similarity_df
        self.products = products
        self.model_metrics = model_metrics

    def _resolve_pipeline(self, model_name):
        # "Best Model" is an alias for whichever model actually won
        if model_name == "Best Model":
            model_name = self.best_model

        if model_name not in self.purchase_pipelines:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {list(self.purchase_pipelines.keys())}"
            )

        return self.purchase_pipelines[model_name]

    def predict_purchase(self, X, model_name="Best Model"):
        pipeline = self._resolve_pipeline(model_name)
        return pipeline.predict(X)

    def purchase_probability(self, X, model_name="Best Model"):
        pipeline = self._resolve_pipeline(model_name)
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X)[0]
            return float(probabilities[1])
        return None

    def confidence(self, X, model_name="Best Model"):
        pipeline = self._resolve_pipeline(model_name)
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X)[0]
            return float(max(probabilities))
        return None

    def recommend(self, stock_code, n=10):
        if stock_code not in self.similarity_df.index:
            return pd.DataFrame()

        similarity = self.similarity_df[stock_code]
        similar = similarity.sort_values(ascending=False)
        similar = similar[similar.index != stock_code]
        similar = similar.head(n)

        recommendations = self.products[
            self.products["StockCode"].isin(similar.index)
        ].copy()

        recommendations = recommendations.merge(
            similar.rename("Similarity"),
            left_on="StockCode",
            right_index=True,
            how="left"
        )

        recommendations = recommendations.sort_values(by="Similarity", ascending=False)

        return recommendations