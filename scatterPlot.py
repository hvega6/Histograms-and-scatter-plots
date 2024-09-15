import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import numpy as np

# Define the end date as today
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch stock data for SPY, NVIDIA, QQQ, and Gold
spy_data = yf.download('SPY', start='2020-01-01', end=end_date)
nvidia_data = yf.download('NVDA', start='2020-01-01', end=end_date)
qqq_data = yf.download('QQQ', start='2020-01-01', end=end_date)
gold_data = yf.download('GC=F', start='2020-01-01', end=end_date)  # Gold futures

# Calculate daily returns
spy_data['Daily Return'] = spy_data['Adj Close'].pct_change()
nvidia_data['Daily Return'] = nvidia_data['Adj Close'].pct_change()
qqq_data['Daily Return'] = qqq_data['Adj Close'].pct_change()
gold_data['Daily Return'] = gold_data['Adj Close'].pct_change()  # Gold daily returns

# Drop NaN values
spy_returns = spy_data['Daily Return'].dropna()
nvidia_returns = nvidia_data['Daily Return'].dropna()
qqq_returns = qqq_data['Daily Return'].dropna()
gold_returns = gold_data['Daily Return'].dropna()

# Align all returns to the same index (dates)
returns_df = pd.DataFrame({
    'SPY': spy_returns,
    'NVIDIA': nvidia_returns,
    'QQQ': qqq_returns,
    'Gold': gold_returns
}).dropna()  # Drop rows with NaN values

# Extract aligned returns
spy_returns = returns_df['SPY']
nvidia_returns = returns_df['NVIDIA']
qqq_returns = returns_df['QQQ']
gold_returns = returns_df['Gold']

# Function to calculate Beta
def calculate_beta(stock_returns, benchmark_returns):
    covariance = stock_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    beta = covariance / variance
    return beta

# Calculate Beta for each stock relative to SPY
beta_nvidia = calculate_beta(nvidia_returns, spy_returns)
beta_qqq = calculate_beta(qqq_returns, spy_returns)
beta_gold = calculate_beta(gold_returns, spy_returns)

# Calculate slopes using linear regression
def calculate_slope(stock_returns, benchmark_returns):
    slope, intercept = np.polyfit(benchmark_returns, stock_returns, 1)
    return slope

# Calculate slopes (betas) for NVIDIA, QQQ, and Gold relative to SPY
slope_nvidia = calculate_slope(nvidia_returns, spy_returns)
slope_qqq = calculate_slope(qqq_returns, spy_returns)
slope_gold = calculate_slope(gold_returns, spy_returns)

# Print slopes (betas) for NVIDIA, QQQ, and Gold
print(f"Slope (Beta) of NVIDIA relative to SPY: {slope_nvidia:.4f}")
print(f"Slope (Beta) of QQQ relative to SPY: {slope_qqq:.4f}")
print(f"Slope (Beta) of Gold relative to SPY: {slope_gold:.4f}")

# Create a beta matrix
beta_matrix = pd.DataFrame({
    'SPY': [1, beta_nvidia, beta_qqq, beta_gold],  # Beta of SPY to itself is 1
    'NVIDIA': [beta_nvidia, 1, calculate_beta(nvidia_returns, qqq_returns), calculate_beta(nvidia_returns, gold_returns)],
    'QQQ': [beta_qqq, calculate_beta(qqq_returns, nvidia_returns), 1, calculate_beta(qqq_returns, gold_returns)],
    'Gold': [beta_gold, calculate_beta(gold_returns, nvidia_returns), calculate_beta(gold_returns, qqq_returns), 1]
}, index=['SPY', 'NVIDIA', 'QQQ', 'Gold'])

# Print Beta Matrix
print("\nBeta Matrix:")
print(beta_matrix)

# Calculate Alpha for NVIDIA, QQQ, and Gold relative to SPY
def calculate_alpha(stock_returns, benchmark_returns):
    risk_free_rate = 0
    market_return = benchmark_returns.mean() * 252  # Annualized market return
    stock_annual_return = stock_returns.mean() * 252  # Annualized stock return
    alpha = stock_annual_return - (risk_free_rate + calculate_beta(stock_returns, benchmark_returns) * (market_return - risk_free_rate))
    return alpha

nvidia_alpha = calculate_alpha(nvidia_returns, spy_returns)
qqq_alpha = calculate_alpha(qqq_returns, spy_returns)
gold_alpha = calculate_alpha(gold_returns, spy_returns)

# Print Alpha for NVIDIA, QQQ, and Gold
print(f"Alpha of NVIDIA relative to SPY: {nvidia_alpha:.4f}")
print(f"Alpha of QQQ relative to SPY: {qqq_alpha:.4f}")
print(f"Alpha of Gold relative to SPY: {gold_alpha:.4f}")

# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot for SPY vs. NVIDIA
plt.scatter(spy_returns, nvidia_returns, color='blue', alpha=0.5, label='NVIDIA vs SPY')

# Draw the regression line for NVIDIA
x_values = np.linspace(spy_returns.min(), spy_returns.max(), 100)
y_values_nvidia = slope_nvidia * x_values  # y = mx + b (b = 0)
plt.plot(x_values, y_values_nvidia, color='blue', linestyle='dashed', linewidth=2, label=f'NVIDIA Line (Slope: {slope_nvidia:.4f})')

# Scatter plot for SPY vs. QQQ
plt.scatter(spy_returns, qqq_returns, color='green', alpha=0.5, label='QQQ vs SPY')

# Draw the regression line for QQQ
y_values_qqq = slope_qqq * x_values  # y = mx + b (b = 0)
plt.plot(x_values, y_values_qqq, color='green', linestyle='dashed', linewidth=2, label=f'QQQ Line (Slope: {slope_qqq:.4f})')

# Scatter plot for SPY vs. Gold
plt.scatter(spy_returns, gold_returns, color='gold', alpha=0.5, label='Gold vs SPY')

# Draw the regression line for Gold
y_values_gold = slope_gold * x_values  # y = mx + b (b = 0)
plt.plot(x_values, y_values_gold, color='gold', linestyle='dashed', linewidth=2, label=f'Gold Line (Slope: {slope_gold:.4f})')

# Adding titles and labels
plt.title('Scatter Plot of Daily Returns: SPY, NVIDIA, QQQ, and Gold')
plt.xlabel('SPY Daily Returns')
plt.ylabel('Daily Returns')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Vertical line at x=0
plt.grid()
plt.legend()
plt.show()