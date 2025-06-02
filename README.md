# 🚚 Supply Chain Stockout Forecasting System

A comprehensive AI-powered application for simulating, predicting, and analyzing stockouts in supply chain operations. Built with `Streamlit`, `scikit-learn`, `XGBoost`, `LightGBM`, `DoWhy`, and Google's Gemini API, this system enables end-to-end analytics—from synthetic data generation to forecasting, causal inference, and what-if simulations.


## 🔧 Features

- **🧪 Synthetic Data Generator**  
  Realistically simulates weekly product-level data with seasonality, trends, promotions, supplier delays, and weather events.

- **🔍 Feature Engineering**  
  Adds lag features, rolling statistics, encoded categories, and domain-specific interaction terms.

- **📉 Stockout Forecasting**  
  Classifies stockout risks using models like Random Forest, XGBoost, LightGBM, and Logistic Regression.

- **📈 Stock Level Forecasting**  
  Estimates future inventory levels with regression models.

- **🧠 Causal Inference**  
  Understands cause-effect relationships using DoWhy or linear regression fallback.

- **🔄 Counterfactual Simulation**  
  "What-if" analysis for interventions such as reducing supplier delays or increasing order sizes.

- **💰 Business Impact Estimation**  
  Evaluates ROI from operational changes.

- **📊 Interactive Streamlit Dashboard**  
  For visual analysis, model interpretation, and real-time forecasting.



## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TejaswiMahadev/Supply-Chain-Forecast.git
cd Supply-Chain-Forecast
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Enviroment Variables

```bash
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### Run the full pipeline

```bash
python main.py
```

This will:

- Generate synthetic data

- Perform feature engineering

- Train models

- Estimate causal effects

- Run simulations

- Launch the dashboard

## 🧭 Dashboard Pages
- **Overview** – Key metrics, trends, and correlations

- **Forecasting** – Stockout and stock level predictions

- **Intervention Simulation** – Simulate scenarios

- **Model Performance** – Accuracy, F1, RMSE, MAPE

- **Causal Analysis** – Estimate & interpret causal effects

## 🤖 AI-Powered Explanations
Utilizes Google Gemini API to explain causal findings in business-friendly language 

##💡 Example Scenarios
- Forecast product stockout risk
- Estimate if supplier delays cause stockouts
- Simulate boosting orders by 15%
- Quantify stockouts prevented & ROI

## 📌 TODO
-  Real-world data ingestion
-  PDF report export
-  High-risk alert notifications




