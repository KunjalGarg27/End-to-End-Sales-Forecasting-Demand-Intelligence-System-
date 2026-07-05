# Retail Intelligence & Sales Forecasting System

An end-to-end data science pipeline and interactive dashboard built to transition retail inventory management from reactive reporting to proactive demand intelligence. This project analyzes four years of transactional data to predict future sales, automatically detect unusual purchasing behavior, and segment product lines for strategic inventory allocation.

---

## Project Objectives

* **Time Series Forecasting:** Predict future product demand up to 3 months ahead using statistical and machine learning models to prevent overstock and stockouts.
* **Anomaly Detection:** Automatically identify and flag historical sales spikes or severe drops using algorithmic outlier detection.
* **Product Segmentation:** Cluster product sub-categories based on their real-world behavior (volume, volatility, year-over-year growth) to drive distinct stocking strategies.
* **Business Intelligence:** Deploy an interactive web application allowing stakeholders to explore forecasts and operational insights dynamically.

---

## Core Features & Methodology

### 1. Advanced Exploratory Data Analysis (EDA)
* Extracted comprehensive time features (Season, Quarter, Week) and engineered logistics metrics (ShipDays).
* Conducted stationarity testing (Augmented Dickey-Fuller) and Time Series Decomposition (Trend, Seasonality, Residuals).

### 2. Predictive Modeling
Trained and evaluated three distinct mathematical approaches on a highly seasonal dataset:
* **SARIMA:** Statistical modeling utilizing differencing to achieve stationarity.
* **Facebook Prophet:** Additive modeling optimized for strong yearly and weekly seasonality. *(Selected as the production model for its superior MAE and RMSE).*
* **XGBoost:** Supervised machine learning approach utilizing custom lag features and rolling averages.

### 3. Anomaly Detection
Implemented a dual-method approach to flag unseasonal sales events (e.g., Black Friday spikes, weather disruptions):
* **Isolation Forest:** An unsupervised machine learning algorithm evaluating global structural distribution.
* **Z-Score Detection:** A statistical method identifying deviations >2 standard deviations from a 4-week rolling mean.

### 4. Demand Segmentation (Clustering)
* Applied **K-Means Clustering** to segment products into operational profiles (High Volume/Stable, Growing Demand, JIT Inventory).
* Utilized the Elbow Method for optimal K selection and **Principal Component Analysis (PCA)** for 2D visualization of the multidimensional segments.

---

## Technology Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Forecasting Models:** Statsmodels (SARIMA), Prophet, XGBoost
* **Machine Learning:** Scikit-learn (Isolation Forest, K-Means, PCA)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Community Cloud

---

## How to Run Locally

Follow these steps to deploy the interactive dashboard on your local machine.

**1. Clone the repository:**
git clone [[https://github.com/your-username/SalesForecasting_KunjalGarg.git]

2. Install required dependencies:

pip install -r requirements.txt

3. Run the Streamlit application
   
streamlit run app.py

