import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Advertising.csv")

# data allocation for training
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train) # training the model

y_pred = model.predict(X_test)

# show difference between actual and predicted sales in terminal
predicted_df = pd.DataFrame({'Actual Sales': y_test.values, 'Predicted Sales': y_pred})
print("\nActual vs Predicted Sales:\n")
print(predicted_df.head(10))

# calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# bar plot of feature importance
coef = pd.Series(model.coef_, index=X.columns)
coef.plot(kind='bar', color='skyblue')
plt.title("Feature Influence on Sales")
plt.ylabel("Coefficient Value")
plt.xlabel("Advertising Channel")
plt.tight_layout()
plt.show()