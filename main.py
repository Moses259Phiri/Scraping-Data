import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_largest_companies_in_South_Africa"

# Send a GET request to the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print("Successfully fetched the page!")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
    exit()

# Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find the table by its class or structure
# Here, we look for the first <table> with class 'wikitable'
table = soup.find("table", {"class": "wikitable"})

if not table:
    print("No table found!")
    exit()

# Extract table headers
headers = [th.text.strip() for th in table.find_all("th")]

# Extract table rows
rows = table.find_all("tr")[1:]  # Skip the header row
data = []

for row in rows:
    cells = row.find_all(["td", "th"])
    cell_data = [cell.text.strip() for cell in cells]
    data.append(cell_data)

# Convert to a Pandas DataFrame
df = pd.DataFrame(data, columns=headers)

# Display the DataFrame
print(df)

# Optionally, save the data to a CSV file
df.to_csv("largest_companies_sa.csv", index=False)
print("Data saved to 'largest_companies_sa.csv'")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
# Replace 'largest_companies_sa.csv' with your actual dataset
df = pd.read_csv("largest_companies_sa.csv")

# Check the column names to ensure 'Year' and 'Revenue' exist
print(df.columns)

# Ensure 'Year' is an integer
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Convert to numeric, set errors as NaN if invalid

# Ensure 'Revenue' is numeric, remove commas if necessary, and convert to float
df['Revenue'] = df['Revenue'].str.replace(',', '').apply(pd.to_numeric, errors='coerce')

# Check the data types to confirm the conversion
print(df.dtypes)

# Drop rows where either 'Year' or 'Revenue' is NaN (if any)
df = df.dropna(subset=['Year', 'Revenue'])

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
