import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Black-Scholes model implementation for European call and put options
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S: current stock price
    K: strike price of the option
    T: time to expiration in years
    r: risk-free rate (annual)
    sigma: volatility of the underlying stock
    option_type: type of the option ("call" or "put")
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Load data from CSV file
def load_csv_data(file_path):
    """
    Reads options data from a CSV file.
    file_path: path to the CSV file
    """
    df = pd.read_csv(file_path)
    df['time_to_expiry'] = df['remaining'] / 365.25  # Convert remaining days to years
    return df

# Trading strategy and backtesting function with stop-loss
def backtest_strategy(options_data, risk_free_rate, stop_loss_threshold, buy_threshold=0.05, sell_threshold=0.10):
    """
    options_data: DataFrame with columns ['contractSymbol', 'strike', 'bid', 'ask', 'impliedVolatility', 
                                          'price', 'remaining', 'time_to_expiry']
    buy_threshold: float, threshold for price difference to initiate a buy
    sell_threshold: float, threshold above buy price to sell for profit
    stop_loss_threshold: float, threshold below buy price to stop loss
    """
    cash = 0
    position = None
    transactions = []

    for index, row in options_data.iterrows():
        theoretical_price = black_scholes(row['price'], row['strike'], row['time_to_expiry'],
                                          risk_free_rate, row['impliedVolatility'], 'call')  # assuming all are calls

        # Calculate the midpoint price of bid-ask spread
        market_price = (row['bid'] + row['ask']) / 2

        # Buy option if the theoretical price is significantly higher than the market price
        if position is None and (theoretical_price - market_price > buy_threshold):
            position = {'buy_price': market_price, 'theoretical_buy_price': theoretical_price,
                        'contract': row['contractSymbol'], 'buy_date': datetime.now(), 'type': 'call'}
        
        # Check for stop-loss condition
        if position and (position['buy_price'] - market_price > stop_loss_threshold*position['buy_price']):
            loss = market_price - position['buy_price']
            cash += loss
            transactions.append({
                'contract': position['contract'],
                'buy_date': position['buy_date'],
                'sell_date': datetime.now(),
                'buy_price': position['buy_price'],
                'sell_price': market_price,
                'profit': loss,
                'type': position['type']
            })
            position = None

        # Sell option either at a price above the buy price by 'sell_threshold' or at expiry
        elif position and ((market_price - position['buy_price'] > sell_threshold) or row['time_to_expiry'] <= 0):
            profit = market_price - position['buy_price']
            cash += profit
            transactions.append({
                'contract': position['contract'],
                'buy_date': position['buy_date'],
                'sell_date': datetime.now(),
                'buy_price': position['buy_price'],
                'sell_price': market_price,
                'profit': profit,
                'type': position['type']
            })
            position = None
        
    transactions_df = pd.DataFrame(transactions)
    success_ratio = (transactions_df['profit'] > 0).sum() / len(transactions_df) if len(transactions_df) > 0 else 0
    return transactions_df, cash, success_ratio


# Example usage
file_path = "datasets/tesla.csv"  # Replace with your CSV file path
risk_free_rate = 0.01  # Approximate current risk-free rate

options_data = load_csv_data(file_path)

# Function to find the best stop-loss threshold
def find_best_stop_loss(options_data, risk_free_rate, thresholds):
    results = []
    for threshold in thresholds:
        _, profit, _ = backtest_strategy(options_data, risk_free_rate, stop_loss_threshold=threshold)
        results.append((threshold, profit))
    best_threshold = max(results, key=lambda x: x[1])
    return best_threshold

# Example usage
thresholds = np.linspace(0.96, 0.9999999, 100)  # Example range of thresholds from 95% to 99%
best_threshold, best_profit = find_best_stop_loss(options_data, risk_free_rate, thresholds)

transaction_log, total_profit, success_ratio = backtest_strategy(options_data, risk_free_rate, best_threshold)
print("Basic Trading Strategy Results")
print("Total trades: ", len(transaction_log))
print("Total Profit: ", total_profit)
print("Success Ratio: ", success_ratio)
