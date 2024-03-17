# House-price-prediction-using-machine-learning

Predicting house prices is a crucial task in the real estate industry, especially in booming markets like Bangalore. This repository hosts a comprehensive solution leveraging machine learning techniques to predict house prices in Bangalore. The dataset utilized for training and testing our models is sourced from Kaggle's Bangalore House Price Prediction dataset.

With the ever-growing demand for real estate in Bangalore, accurate predictions of house prices become increasingly valuable. By harnessing machine learning algorithms, we aim to provide insights into the factors influencing house prices in Bangalore and create models capable of making reliable predictions.

### Table of Contents

1. Introduction
2. Data
3. Requirements
4. Methods
5. Results
6. Conclusion
7. References

### Introduction

This repository contains code and resources for predicting house prices in Bangalore using machine learning techniques. The dataset used for training and testing the models is sourced from Kaggle's Bangalore House Price Prediction dataset. 

### Data

The dataset contains various features of houses in Bangalore along with their respective prices. The features include aspects like area, number of bedrooms, location, etc. This dataset is used for training the machine learning models to predict house prices.

### Requirements
To run the code in this repository, you'll need the following libraries:

Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

### Approach

In the Jupyter notebook, we explore the dataset, perform data preprocessing, and then train several machine learning. We evaluate the performance of each model and select the best-performing one for predicting house prices in Bangalore.

We will use three machine learning models to predict house prices:

1. Linear Regression: A simple linear regression model without regularization.
2. Lasso Regression: A linear regression model with L1 regularization.
3. Ridge Regression: A linear regression model with L2 regularization.

We will use a 70-30% train-test split to train the models and evaluate their performance.

### Results

The performance of the three models was evaluated using the R2 score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables. The R2 score for the three models was:

1. Linear Regression: 0.8233
2. Lasso Regression: 0.8128
3. Ridge Regression: 0.8234

The Ridge Regression model had the highest R2 score, indicating the best performance.

### Conclusion

This project demonstrated the use of machine learning techniques to predict house prices. The Ridge Regression model provided the best performance, indicating that regularization can improve.
