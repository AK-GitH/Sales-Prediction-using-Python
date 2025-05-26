import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

def load_data():
    df = pd.read_csv("cleaned_advertising.csv")
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    return df, X, y

def analyze_correlation(df):
    print("\nFeature Correlations:\n")
    print(df.corr())

def cross_validate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("\nCross-Validation R² Scores:", np.round(cv_scores, 3))
    print(f"Average CV R²: {cv_scores.mean():.3f}\n")

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_rounded = np.round(y_pred, 2)

    print("\nActual vs Predicted Sales:\n")
    predicted_df = pd.DataFrame({
        'Actual Sales': y_test.values,
        'Predicted Sales': y_pred_rounded
    })
    print(predicted_df.head(10))

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.3f}")

    return y_pred, y_pred_rounded, mse, r2

def plot_results(y_test, y_pred_rounded, residuals):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred_rounded)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred_rounded, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.show()

def save_model(model, filename="linear_model.pkl"):
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")

def main():
    df, X, y = load_data()
    analyze_correlation(df)
    model = LinearRegression()
    cross_validate_model(model, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    y_pred, y_pred_rounded, mse, r2 = evaluate_model(model, X_test, y_test)
    residuals = y_test - y_pred
    plot_results(y_test, y_pred_rounded, residuals)
    save_model(model)

if __name__ == "__main__":
    main()
