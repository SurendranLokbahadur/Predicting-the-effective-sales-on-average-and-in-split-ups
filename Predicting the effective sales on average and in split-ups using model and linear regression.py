import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample sales_data for illustration (replace with your actual data)
sales_data = {
    'Advertising_Budget': [1000, 2000, 3000, 4000, 5000],
    'Social_Media_Reach': [50000, 60000, 70000, 80000, 90000],
    'Sales_Representatives': [50, 60, 70, 80, 90],
    'Seasonality_Factor': [0, 1, 0, 1, 1],
    'Sales': [100000, 150000, 200000, 250000, 300000]
}

# Create DataFrame
df = pd.DataFrame(sales_data)

# Prepare features and target variable
X = df[['Advertising_Budget', 'Social_Media_Reach', 'Sales_Representatives', 'Seasonality_Factor']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get user input
ad_budget = float(input("Enter the advertising budget in USD: "))
social_media_reach = float(input("Enter the social media reach (in number of people): "))
num_sales_reps = int(input("Enter the number of sales representatives: "))
seasonality_factor = int(input("Enter the seasonality factor (1 for peak season, 0 for off-season): "))

# Prepare the input for prediction
input_data = pd.DataFrame([[ad_budget, social_media_reach, num_sales_reps, seasonality_factor]], 
                           columns=['Advertising_Budget', 'Social_Media_Reach', 'Sales_Representatives', 'Seasonality_Factor'])

# Make prediction
predicted_sales = model.predict(input_data)[0]

# Ensure positive prediction
predicted_sales = max(predicted_sales, 0)  # Ensure it's not negative

# Adjust based on seasonality factor
if seasonality_factor == 0:  # Off-season
    predicted_sales *= 0.5  # Reduce sales by 50% for off-season

# Get sales timeframe input
timeframe = input("Enter the sales timeframe (days, weeks, months, years): ").strip().lower()
time_amount = float(input("Enter the sales amount for the specified timeframe: "))

# Adjust predicted sales based on the timeframe
if timeframe == 'days':
    predicted_sales *= time_amount
elif timeframe == 'weeks':
    predicted_sales *= time_amount * 7
elif timeframe == 'months':
    predicted_sales *= time_amount * 30  # Approximate month length
elif timeframe == 'years':
    predicted_sales *= time_amount * 365  # Approximate year length
else:
    print("Invalid timeframe entered. Defaulting to predicted sales for 1 day.")

# Output the predicted sales
print(f"\nPredicted sales for {time_amount} {timeframe}: ${predicted_sales:,.2f}")
