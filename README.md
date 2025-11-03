# Demand Forecasting Project


## Overview
This project demonstrates how to build an end to end forecasting pipeline. It uses a historical sales dataset to predict weekly demand for individual products or product store combinations.

The project utilized multiple models, including: **Prophet** and **XGBoost**


## Goals
- Aggregate transactional sales data into weekly product-level time series.
- Train and evaluate baseline forecasting models (Prophet, XGBoost).
- Evaluate forecasting accuracy using RMSE, MAE, and sMAPE.
- Communicate business insights and improvement potential.


## How to run
1. Place your dataset in `./data/sales.csv`
2. Open `notebooks/Fashion_Demand_Forecasting_Starter_Notebook.ipynb`
3. Run all cells to generate forecasts and metrics