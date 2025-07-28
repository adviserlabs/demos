#!/usr/bin/env python3
"""
Stock Performance Analyzer
Generates fake trading data using Brownian motion and analyzes performance.
Uses only Python standard library.
"""

import random
import math
import json
import sys
import os
from datetime import datetime, timedelta


def number_to_stock_symbol(num):
    """Convert a number to a 4-character uppercase stock symbol."""
    # Simple hash function to convert number to 4 letters
    hash_val = hash(str(num)) % (26**4)
    symbol = ""
    for _ in range(4):
        symbol = chr(65 + (hash_val % 26)) + symbol
        hash_val //= 26
    return symbol


def generate_brownian_motion_data(num_stocks=1500, num_days=1720, initial_price=100.0):
    """Generate stock price data using Brownian motion."""
    stocks = {}
    
    for stock_id in range(num_stocks):
        symbol = number_to_stock_symbol(stock_id)
        prices = [initial_price]
        
        for day in range(1, num_days):
            # Brownian motion: random walk with drift
            drift = 0.0005  # Small positive drift (0.05% per day)
            volatility = 0.02  # 2% daily volatility
            
            random_change = random.gauss(0, 1)  # Standard normal distribution
            price_change = drift + volatility * random_change
            
            new_price = prices[-1] * (1 + price_change)
            # Ensure price doesn't go negative
            new_price = max(new_price, 0.01)
            prices.append(new_price)
        
        stocks[symbol] = prices
    
    return stocks


def save_data_to_file(data, filename="trading_data.json"):
    """Save trading data to a JSON file."""
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved successfully!")


def load_data_from_file(filename="trading_data.json"):
    """Load trading data from a JSON file."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_performance(prices, period_days=None):
    """Calculate performance over a specific period."""
    if period_days is None:
        period_days = len(prices)
    
    start_idx = max(0, len(prices) - period_days)
    start_price = prices[start_idx]
    end_price = prices[-1]
    
    return (end_price - start_price) / start_price * 100


def analyze_stocks(stocks_data):
    """Analyze stock performance with progress indicator."""
    total_stocks = len(stocks_data)
    results = {
        'full_period': [],
        'last_30d': [],
        'last_7d': []
    }
    
    print(f"\nAnalyzing {total_stocks} stocks...")
    
    for i, (symbol, prices) in enumerate(stocks_data.items()):
        # Progress indicator
        progress = (i + 1) / total_stocks
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {progress:.1%} ({i+1}/{total_stocks})')
        sys.stdout.flush()
        
        # Calculate performance for different periods
        full_perf = calculate_performance(prices)
        last_30d_perf = calculate_performance(prices, 30)
        last_7d_perf = calculate_performance(prices, 7)
        
        results['full_period'].append((symbol, full_perf))
        results['last_30d'].append((symbol, last_30d_perf))
        results['last_7d'].append((symbol, last_7d_perf))
    
    print()  # New line after progress bar
    
    # Sort by performance (descending)
    for period in results:
        results[period].sort(key=lambda x: x[1], reverse=True)
    
    return results


def create_ascii_chart(data, title, width=60, height=20, show_worst=False):
    """Create an ASCII chart for the top/worst performers."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not data:
        print("No data available")
        return
    
    # Take top 10 or worst 5
    if show_worst:
        display_data = data[-5:]  # Last 5 (worst performers)
        display_data.reverse()  # Show worst first
        chart_title = "Worst 5 Performers:"
    else:
        display_data = data[:10]  # Top 10
        chart_title = "Top 10 Performers:"
    
    # Find min and max values for scaling
    values = [perf for _, perf in display_data]
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        max_val = min_val + 1  # Avoid division by zero
    
    print(f"\n{chart_title}")
    print(f"{'Rank':<4} {'Symbol':<10} {'Performance':<12} {'Chart'}")
    print("-" * (width + 30))
    
    for i, (symbol, perf) in enumerate(display_data, 1):
        # Scale performance to chart width
        normalized = (perf - min_val) / (max_val - min_val)
        bar_length = int(normalized * width)
        
        # Create bar with ANSI color codes
        bar_char = '█' * bar_length
        if perf >= 0:
            bar = f'\033[30m{bar_char}\033[0m'  # Black for positive
        else:
            bar = f'\033[31m{bar_char}\033[0m'  # Red for negative
        
        print(f"{i:<4} {symbol:<10} {perf:>8.2f}%   {bar}")
    
    print(f"\nRange: {min_val:.2f}% to {max_val:.2f}%")


def display_summary(results):
    """Display summary of analysis results."""
    print("\n" + "="*70)
    print("STOCK PERFORMANCE ANALYSIS SUMMARY")
    print("="*70)
    
    periods = [
        ('full_period', 'Full Period (720 days)'),
        ('last_30d', 'Last 30 Days'),
        ('last_7d', 'Last 7 Days')
    ]
    
    for period_key, period_name in periods:
        data = results[period_key]
        if data:
            best_stock, best_perf = data[0]
            worst_stock, worst_perf = data[-1]
            
            print(f"\n{period_name}:")
            print(f"  Best Performer:  {best_stock} ({best_perf:+.2f}%)")
            print(f"  Worst Performer: {worst_stock} ({worst_perf:+.2f}%)")
            
            # Create ASCII charts for top and worst performers
            create_ascii_chart(data, f"{period_name} - Top 10 Performers")
            create_ascii_chart(data, f"{period_name} - Worst 5 Performers", show_worst=True)


def main():
    """Main program execution."""
    data_file = "trading_data.json"
    
    print("Stock Performance Analyzer")
    print("=" * 30)
    
    # Check if data file exists
    stocks_data = load_data_from_file(data_file)
    
    if stocks_data is None:
        print(f"No existing data found. Generating new trading data...")
        print("Generating Brownian motion data for 500 stocks over 720 days...")
        
        # Generate data
        #stocks_data = generate_brownian_motion_data(2500, 11720, 100.0)
        stocks_data = generate_brownian_motion_data()
        
        # Save to file
        save_data_to_file(stocks_data, data_file)
    else:
        print(f"Loading existing data from {data_file}...")
        print(f"Found data for {len(stocks_data)} stocks")
    
    # Analyze performance
    results = analyze_stocks(stocks_data)
    
    # Display results
    display_summary(results)
    
    print(f"\nAnalysis complete! Data stored in: {data_file}")


if __name__ == "__main__":
    main()
