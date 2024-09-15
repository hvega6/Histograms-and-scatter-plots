import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch SPY stock data
spy_data = yf.download('SPY', start='2020-01-01', end='2023-01-01')

# Calculate daily returns
spy_data['Daily Return'] = spy_data['Adj Close'].pct_change()

# Drop NaN values
daily_returns = spy_data['Daily Return'].dropna()

# Display the DataFrame (optional)
print(daily_returns)

# Plotting the histogram of daily returns
plt.figure(figsize=(10, 6))
plt.hist(daily_returns, bins=50, color='blue', alpha=0.7, edgecolor='black', range=(-0.8, 0.8))
plt.title('Histogram of SPY Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.xlim(-0.8, 0.8)  # Set x-axis limits
plt.show()
