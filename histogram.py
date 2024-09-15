import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch SPY stock data
spy_data = yf.download('SPY', start='2020-01-01', end='2023-01-01')

# Fetch NVIDIA stock data
nvidia_data = yf.download('NVDA', start='2020-01-01', end='2023-01-01')

# Calculate daily returns for SPY
spy_data['Daily Return'] = spy_data['Adj Close'].pct_change()
spy_daily_returns = spy_data['Daily Return'].dropna()

# Calculate daily returns for NVIDIA
nvidia_data['Daily Return'] = nvidia_data['Adj Close'].pct_change()
nvidia_daily_returns = nvidia_data['Daily Return'].dropna()

# Calculate standard deviation of daily returns
spy_std_dev = spy_daily_returns.std()
nvidia_std_dev = nvidia_daily_returns.std()

# Display the DataFrames (optional)
print("SPY Daily Returns:\n", spy_daily_returns)
print("NVIDIA Daily Returns:\n", nvidia_daily_returns)

# Plotting the histograms
plt.figure(figsize=(12, 6))

# Histogram for SPY
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.hist(spy_daily_returns, bins=20, color='blue', alpha=0.7, edgecolor='black', range=(-0.8, 0.8))
plt.title('Histogram of SPY Daily Returns (20 Bins)')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.xlim(-0.8, 0.8)  # Set x-axis limits

# Add a vertical line for the standard deviation
plt.axvline(spy_std_dev, color='red', linestyle='dashed', linewidth=2, label=f'SD: {spy_std_dev:.4f}')
plt.axvline(-spy_std_dev, color='red', linestyle='dashed', linewidth=2)
plt.legend()

# Histogram for NVIDIA
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.hist(nvidia_daily_returns, bins=20, color='green', alpha=0.7, edgecolor='black', range=(-0.8, 0.8))
plt.title('Histogram of NVIDIA Daily Returns (20 Bins)')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.xlim(-0.8, 0.8)  # Set x-axis limits

# Add a vertical line for the standard deviation
plt.axvline(nvidia_std_dev, color='red', linestyle='dashed', linewidth=2, label=f'SD: {nvidia_std_dev:.4f}')
plt.axvline(-nvidia_std_dev, color='red', linestyle='dashed', linewidth=2)
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
