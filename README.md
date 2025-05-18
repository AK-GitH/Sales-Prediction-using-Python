# 📈 Sales Prediction using Python

This project demonstrates how to **predict product sales** based on advertising expenditure across different media channels: **TV**, **Radio**, and **Newspaper**. It uses **Linear Regression**, a foundational machine learning algorithm, to model the relationship between ad spend and resulting sales.

---

## Project Description

> **Sales prediction** means predicting how much of a product people will buy based on factors such as the amount you spend to advertise your product, the segment of people you advertise to, or the platform you advertise on.

This analysis is done using a dataset (`Advertising.csv`) that contains historical advertising data and corresponding sales figures.

---

## Files

- `Advertising.csv` — Dataset containing columns: `TV`, `Radio`, `Newspaper`, and `Sales`.
- `sales_prediction.py` — Python script that:
  - Trains a Linear Regression model
  - Prints actual vs predicted sales comparison in the terminal
  - Evaluates predictions with metrics
  - Displays a bar plot showing the influence of each advertising channel on sales

---

## Features

- **Linear Regression** model from `scikit-learn`
- **Evaluation** using Mean Squared Error and R² Score
- **Terminal output** showing actual vs predicted sales values
- **Bar Plot** visualizing feature importance (model coefficients)

---

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```