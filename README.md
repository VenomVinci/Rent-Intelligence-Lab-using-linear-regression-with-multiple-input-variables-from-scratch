# Rent Intelligence Lab 

A machine learning project to predict house rent using Multiple Linear Regression.

## About

I built this project to understand how Linear Regression works behind the scenes rather than relying entirely on prebuilt models.

Using rental property data, the goal is to predict monthly rent based on features such as property size, number of bedrooms, bathrooms, furnishing status, and location.

I implemented the main regression algorithm from scratch using NumPy and explored how different optimization techniques affect the model.

## Dataset

**Source:** Kaggle — House Rent Prediction Dataset
**Target variable:** `Rent` (monthly rent in INR)

Features used include property size, BHK, bathrooms, floor information, city, furnishing status, tenant preferences, and locality, depending on their availability in the dataset.

## What I Implemented

* Explored the dataset and analyzed the distribution of rental prices.
* Checked for missing values, duplicates, and unusual rent values.
* Extracted numerical information from the floor column.
* Handled missing values and encoded categorical features using one-hot encoding.
* Split the dataset into training and testing sets.
* Applied feature scaling using the training data.
* Used a log transformation on rent to handle its skewed distribution.
* Implemented the prediction function and cost function from scratch.
* Calculated gradients using vectorized NumPy operations.
* Implemented Batch Gradient Descent and Mini-batch Gradient Descent.
* Visualized the cost function to understand model convergence.
* Compared different learning rates.
* Evaluated predictions using MAE, RMSE, and R².
* Compared the model against a mean-prediction baseline.
* Compared my implementation with scikit-learn Linear Regression.
* Visualized actual versus predicted rent.
* Analyzed residuals to understand prediction errors.
* Created learning curves to study how training data affects performance.
* Investigated properties with the largest prediction errors.
* Examined learned model weights to understand how features influence predictions.

## Techniques Explored

**Multiple Linear Regression · Gradient Descent · Vectorization · Feature Scaling · One-Hot Encoding · Log Transformation · Learning Curves · Residual Analysis · Model Evaluation**

## Evaluation Metrics

* **MAE:** Average absolute prediction error in rupees.
* **RMSE:** Penalizes larger prediction errors more heavily.
* **R² Score:** Measures performance relative to a mean-prediction baseline.

The model is evaluated on a held-out test set to check how well it performs on data it has not seen during training.

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn

Download the dataset from Kaggle, place the CSV file in the project directory, and open `rent_regression.ipynb` in Jupyter Notebook. Run the cells in order.

## Current Status

The project currently covers data preprocessing, a from-scratch regression implementation, gradient descent experiments, evaluation, learning curves, residual analysis, and feature-weight inspection.
