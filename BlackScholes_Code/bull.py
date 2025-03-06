import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
import random

def black_scholes_call(S, K, T, r, sigma):
    """Simple Black-Scholes implementation"""
    try:
        T = max(T, 1e-6)
        sigma = max(sigma, 1e-6)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(price, 0)
    except Exception:
        return 0

def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['time_to_expiry'] = df['remaining'] / 365.25
        return df.dropna(subset=['strike', 'price', 'impliedVolatility', 'remaining'])
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return pd.DataFrame()

SLIPPAGE_RATE = 0.01  # Slippage rate
TRANSACTION_COST = 1.0  # Transaction cost
STOP_LOSS_PERCENTAGE = 0.20  # Stop loss percentage

def bull_spread_strategy(file_path, risk_free_rate=0.01):
    options_data = load_csv_data(file_path)
    if options_data.empty:
        return pd.DataFrame(), 0, 0

    # Use average price from CSV
    stock_price = options_data['price'].mean()
    
    # Wider price range for more opportunities
    price_range = 0.15  # 15% range
    options_data = options_data[
        (options_data['strike'] >= stock_price * (1 - price_range)) & 
        (options_data['strike'] <= stock_price * (1 + price_range))
    ]

    # Updated restrictions for bull spread
    min_strike_distance = 2  # Minimum distance between strikes
    max_strike_range = 12    # Maximum difference between strikes

    # Get unique strikes sorted
    unique_strikes = sorted(options_data['strike'].unique())

    # Create strike combinations for bull spread
    strike_combinations = [
        (K1, K2) for K1, K2 in combinations(unique_strikes, 2)
        if K2 > K1  # Ensure higher strike is greater than lower strike
        and (K2 - K1) >= min_strike_distance  # Minimum distance between strikes
        and (K2 - K1) <= max_strike_range  # Maximum range between strikes
    ]

    # print(f"Generated {len(strike_combinations)} valid strike combinations.")

    results = []
    total_profit = 0
    total_trades = 0
    successful_trades = 0

    for K1, K2 in strike_combinations:
        # Find relevant options for these strikes
        relevant_options = options_data[options_data['strike'].isin([K1, K2])]
        if relevant_options.empty:
            continue

        sigma = relevant_options['impliedVolatility'].mean()
        T = relevant_options['time_to_expiry'].mean()

        # Calculate prices using the Black-Scholes model
        long_call_price = black_scholes_call(stock_price, K1, T, risk_free_rate, sigma)
        short_call_price = black_scholes_call(stock_price, K2, T, risk_free_rate, sigma)

        # Apply slippage
        long_call_price_slippage = long_call_price * (1 + SLIPPAGE_RATE)
        short_call_price_slippage = short_call_price * (1 + SLIPPAGE_RATE)

        # Calculate spread cost (buy lower strike call, sell higher strike call)
        total_cost = long_call_price_slippage - short_call_price_slippage + TRANSACTION_COST * 2
        
        # Calculate potential profit (max profit is difference between strikes minus net cost)
        max_profit = (K2 - K1) - total_cost

        # Check if potential profit exceeds stop loss threshold
        if max_profit > total_cost * STOP_LOSS_PERCENTAGE:
            realized_profit = max_profit  # Directly use max profit as the realized profit
            total_profit += realized_profit
            successful_trades += 1
            success = True
        else:
            loss = total_cost * STOP_LOSS_PERCENTAGE
            total_profit -= loss
            success = False

        total_trades += 1

        results.append({
            'Lower_Strike (K1)': K1,
            'Higher_Strike (K2)': K2,
            'Cost': total_cost,
            'Max_Profit': max_profit,
            'Success': success,
            'Realized_PnL': realized_profit if success else -loss
        })

    success_ratio = (successful_trades / total_trades * 100) if total_trades > 0 else 0
    print("Total trades:- ", total_trades)
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No trades executed. Check your conditions and data.")
    
    return results_df, total_profit, success_ratio

# Example usage
if __name__ == "__main__":
    file_path = "datasets/apple.csv"
    
    results_df, total_profit, success_ratio = bull_spread_strategy(file_path)
    
    print("\nBull Spread Strategy Results:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Success Ratio: {success_ratio:.2f}%")
    # if not results_df.empty:
    #     print("\nDetailed Results:")
    #     print(results_df.describe())
    # else:
    #     print("No valid trades to show.")