# AI-Powered E-Commerce Customer Purchase Prediction & Product Recommendation System

An end-to-end machine learning system that combines **customer purchase prediction** and **product recommendation** into a single FastAPI application.

The project is designed for e-commerce scenarios where businesses want to predict whether a customer is likely to purchase and recommend similar products based on historical purchasing behavior.

---

## 📌 Project Overview

This project consists of two major machine learning components:

### 1. Customer Purchase Prediction

The purchase prediction component uses the **Online Shoppers Intention** dataset to predict whether an online visitor will generate revenue.

Several classification algorithms are trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The model with the best F1 Score is selected as the final purchase prediction model.

The system can also return:

* Purchase prediction
* Purchase probability
* Prediction confidence
* Selected model
* Model performance metrics

---

### 2. Product Recommendation

The recommendation component uses the **Online Retail** dataset.

The system creates a customer-product interaction matrix based on product quantities purchased by customers.

The workflow is:

```text
Online Retail Data
        ↓
Data Cleaning
        ↓
Customer-Product Matrix
        ↓
Product-Customer Matrix
        ↓
Cosine Similarity
        ↓
Similar Products
        ↓
Top-N Recommendations
```

When a product's StockCode is provided, the system finds products with the highest similarity and returns their:

* StockCode
* Description
* Similarity Score

---

# 🚀 Features

* Customer purchase prediction
* Multiple machine learning algorithms
* Automatic best-model selection
* Purchase probability
* Prediction confidence
* Product recommendation
* Cosine similarity
* Customer-product interaction matrix
* Model performance metrics
* FastAPI REST API
* API health monitoring
* Swagger API documentation
* Single serialized machine-learning system

---

# 🧠 Machine Learning Models

The following classification models are compared:

| Model               | Purpose        |
| ------------------- | -------------- |
| Logistic Regression | Classification |
| Decision Tree       | Classification |
| Random Forest       | Classification |
| Gradient Boosting   | Classification |
| XGBoost             | Classification |

The final model is selected according to the highest **F1 Score**.

---

# 📊 Datasets

## Online Shoppers Intention

The Online Shoppers Intention dataset contains information about users' browsing behavior on an e-commerce website.

Examples of features include:

* Administrative
* Administrative Duration
* Informational
* Informational Duration
* Product Related
* Product Related Duration
* Bounce Rates
* Exit Rates
* Page Values
* Special Day
* Month
* Operating Systems
* Browser
* Region
* Traffic Type
* Visitor Type
* Weekend

### Target

```text
Revenue
```

The target indicates whether the visitor completed a purchase.

---

## Online Retail

The Online Retail dataset contains transaction information from an online retailer.

Important columns include:

* InvoiceNo
* StockCode
* Description
* Quantity
* InvoiceDate
* UnitPrice
* CustomerID
* Country

The dataset is cleaned before creating the recommendation system.

---

# ⚙️ Data Processing

## Purchase Prediction

The purchase prediction pipeline performs:

```text
Raw Data
   ↓
Train/Test Split
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Feature Selection
   ↓
Machine Learning Model
   ↓
Prediction
```

Categorical features are encoded using `OneHotEncoder`.

Numerical features are scaled using `MinMaxScaler`.

Feature selection is performed using:

```text
SelectKBest + Chi-Square
```

---

## Product Recommendation

The recommendation system performs:

```text
Raw Transactions
        ↓
Remove Missing Customer IDs
        ↓
Remove Invalid Quantities
        ↓
Remove Invalid Prices
        ↓
Remove Cancelled Orders
        ↓
Create Customer-Product Matrix
        ↓
Transpose Matrix
        ↓
Calculate Cosine Similarity
        ↓
Generate Top-N Recommendations
```

---

# 🏗️ Project Structure

```text
projects prot/
│
├── model/
│   ├── __init__.py
│   ├── customer_system.py
│   ├── predict.py
│   └── customer_purchase_recommendation.pkl
│
├── schemas/
│   ├── prediction_response.py
│   ├── recommendation_input.py
│   └── user_input.py
│
├── app.py
├── fix_pickle.py
├── check_models.py
├── customer_purchase_recommendation.pkl
├── Online Retail.xlsx
├── top_10_recommendations.csv
├── requirements.txt
└── README.md
```

---

# 🔧 Installation

Clone or download the project and open a terminal inside the project directory.

Create a virtual environment:

```bash
python -m venv venv50
```

Activate it on Windows PowerShell:

```powershell
.\venv50\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the API

After activating the virtual environment, start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically provides interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger UI to test the prediction and recommendation endpoints directly.

---

# 🔗 API Endpoints

## Home

```http
GET /
```

Returns basic information about the API.

Example response:

```json
{
    "message": "Customer Purchase & Recommendation API",
    "status": "ok",
    "version": "1.0.0"
}
```

---

## Health Check

```http
GET /health
```

Checks whether the recommendation and prediction system has been loaded successfully.

Example:

```json
{
    "status": "ok",
    "version": "1.0.0",
    "model_loaded": true
}
```

---

## Purchase Prediction

```http
POST /predict
```

Accepts customer browsing information and returns the purchase prediction.

The response can contain:

```json
{
    "prediction": true,
    "purchase_probability": 0.82,
    "confidence": 0.82,
    "selected_model": "Random Forest"
}
```

The exact request fields depend on the Pydantic schema defined in:

```text
schemas/user_input.py
```

---

## Product Recommendation

```http
POST /recommend
```

Accepts a product StockCode and the desired number of recommendations.

Example request:

```json
{
    "stock_code": "85123A",
    "n": 5
}
```

Example response:

```json
{
    "stock_code": "85123A",
    "number_of_recommendations": 5,
    "recommendations": [
        {
            "StockCode": "22423",
            "Description": "REGENCY CAKESTAND 3 TIER",
            "Similarity": 0.91
        }
    ]
}
```

---

# 💾 Model Serialization

The trained prediction pipeline and recommendation system are stored together in:

```text
customer_purchase_recommendation.pkl
```

The saved object contains:

* Purchase prediction pipeline
* Product similarity matrix
* Product information
* Model performance metrics
* Best model name

This allows the FastAPI application to load the trained system without retraining the models every time the API starts.

---

# 🔐 Model Loading

The `CustomerSystem` class is defined separately in:

```text
model/customer_system.py
```

This is important because the class is serialized inside the pickle file.

The API loads the saved system through:

```text
model/predict.py
```

---

# 🧪 Model Evaluation

The classification models are evaluated using:

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many predicted purchasers were actually purchasers.

### Recall

Measures how many actual purchasers were correctly identified.

### F1 Score

Provides a balance between Precision and Recall.

The model with the highest F1 Score is selected as the final model.

---

# 🛠️ Technologies Used

| Technology        | Usage                   |
| ----------------- | ----------------------- |
| Python            | Programming language    |
| Pandas            | Data processing         |
| NumPy             | Numerical operations    |
| Scikit-learn      | Machine learning        |
| XGBoost           | Gradient boosting model |
| FastAPI           | REST API                |
| Pydantic          | API validation          |
| Uvicorn           | API server              |
| Pickle            | Model serialization     |
| Cosine Similarity | Product recommendation  |

---

# 🎯 Use Cases

This system can be used in e-commerce applications for:

* Customer purchase prediction
* Customer behavior analysis
* Product recommendations
* Personalized shopping experiences
* Cross-selling
* Product discovery
* E-commerce decision support

---

# 🔮 Future Improvements

Possible future improvements include:

* Deploying the API to a cloud platform
* Adding a Streamlit frontend
* Adding user authentication
* Using collaborative filtering
* Implementing hybrid recommendations
* Adding customer-specific recommendations
* Tracking recommendation performance
* Adding model monitoring
* Using a database instead of static files
* Containerizing the application with Docker
* Adding automated model retraining

---

# 👨‍💻 Author

**safi hamid**

Machine Learning / Data Science Project

---

# 📄 License

This project is intended for educational and portfolio purposes.
