import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
# Replace 'largest_companies_sa.csv' with your actual dataset
df = pd.read_csv("largest_companies_sa.csv")

# Preview the dataset
print(df.head())

# Data Cleaning and Preparation
# Ensure 'Year' and 'Revenue' columns exist; otherwise, modify or create them.
df['Year'] = df['Year'].astype(int)  # Ensure Year is integer
df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)  # Remove commas and convert to float

# Data Visualization
sns.set(style="whitegrid")

# Bar Plot of Revenues
plt.figure(figsize=(12, 6))
sns.barplot(x="Year", y="Revenue", data=df, ci=None, palette="viridis")
plt.title("Revenue of Largest Companies in South Africa by Year")
plt.xlabel("Year")
plt.ylabel("Revenue (in billions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Regression Analysis
X = df[['Year']].values  # Independent variable (Year)
y = df['Revenue'].values  # Dependent variable (Revenue)

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Display Regression Coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Regression Line Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Year'], y=df['Revenue'], color='blue', label="Actual Data")
plt.plot(df['Year'], model.predict(X), color='red', label="Regression Line")
plt.title("Regression Analysis: Revenue vs. Year")
plt.xlabel("Year")
plt.ylabel("Revenue (in billions)")
plt.legend()
plt.show()

# Forecasting for the Next 10–20 Years
future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 21).reshape(-1, 1)
future_revenue = model.predict(future_years)

# Forecast Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Revenue'], label="Historical Data", color="blue", marker='o')
plt.plot(future_years, future_revenue, label="Forecast (10-20 years)", color="green", linestyle="--")
plt.title("Revenue Forecast for the Next 10–20 Years")
plt.xlabel("Year")
plt.ylabel("Revenue (in billions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combine Historical and Forecast Data
forecast_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Forecasted Revenue': future_revenue
})

print("Forecast for the Next 10–20 Years:")
print(forecast_df)

# Save forecasted data to a CSV file
forecast_df.to_csv("revenue_forecast.csv", index=False)
print("Forecast saved to 'revenue_forecast.csv'")
