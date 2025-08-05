#!/usr/bin/env python
"""
Performance Analyzer for Trading System
Analyzes trade logs and generates detailed performance reports
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PerformanceAnalyzer:
    """Analyze trading performance from logs."""
    
    def __init__(self, trade_log_path=None, performance_log_path=None):
        """Initialize the analyzer."""
        self.trade_log_path = trade_log_path
        self.performance_log_path = performance_log_path
        self.trades_df = None
        self.performance_df = None
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self):
        """Load trade and performance data."""
        # Find latest files if not specified
        if not self.trade_log_path:
            self.trade_log_path = self._find_latest_file('trade_logs', 'trades_*.csv')
        
        if not self.performance_log_path:
            self.performance_log_path = self._find_latest_file('trade_logs', 'performance_*.csv')
        
        # Load trade data
        if self.trade_log_path and os.path.exists(self.trade_log_path):
            self.trades_df = pd.read_csv(self.trade_log_path)
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            print(f"Loaded {len(self.trades_df)} trades from {self.trade_log_path}")
        else:
            print("No trade log found")
            return False
        
        # Load performance data
        if self.performance_log_path and os.path.exists(self.performance_log_path):
            self.performance_df = pd.read_csv(self.performance_log_path)
            self.performance_df['timestamp'] = pd.to_datetime(self.performance_df['timestamp'])
            print(f"Loaded performance data from {self.performance_log_path}")
        
        return True
    
    def _find_latest_file(self, directory, pattern):
        """Find the latest file matching pattern."""
        import glob
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            return max(files, key=os.path.getctime)
        return None
    
    def analyze_overall_performance(self):
        """Analyze overall trading performance."""
        if self.trades_df is None:
            return
        
        # Filter for closed trades
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) == 0:
            print("No closed trades found")
            return
        
        # Calculate metrics
        total_trades = len(closed_trades)
        profitable_trades = closed_trades[closed_trades['profit_loss'] > 0]
        losing_trades = closed_trades[closed_trades['profit_loss'] < 0]
        
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = profitable_trades['profit_loss'].sum()
        total_loss = abs(losing_trades['profit_loss'].sum())
        net_profit = total_profit - total_loss
        
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Risk/reward ratio
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Maximum consecutive wins/losses
        results = closed_trades['profit_loss'].apply(lambda x: 1 if x > 0 else -1)
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for result in results:
            if result > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        # Print summary
        print("\n=== OVERALL PERFORMANCE SUMMARY ===")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {win_count}")
        print(f"Losing Trades: {loss_count}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"\nTotal Profit: ${total_profit:.2f}")
        print(f"Total Loss: ${total_loss:.2f}")
        print(f"Net Profit/Loss: ${net_profit:.2f}")
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"\nMax Consecutive Wins: {max_consecutive_wins}")
        print(f"Max Consecutive Losses: {max_consecutive_losses}")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def analyze_by_instrument(self):
        """Analyze performance by instrument."""
        if self.trades_df is None:
            return
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) == 0:
            return
        
        # Group by instrument
        instrument_stats = []
        
        for instrument in closed_trades['instrument'].unique():
            instrument_trades = closed_trades[closed_trades['instrument'] == instrument]
            
            total = len(instrument_trades)
            wins = len(instrument_trades[instrument_trades['profit_loss'] > 0])
            win_rate = (wins / total * 100) if total > 0 else 0
            net_profit = instrument_trades['profit_loss'].sum()
            
            instrument_stats.append({
                'Instrument': instrument,
                'Trades': total,
                'Wins': wins,
                'Win Rate %': f"{win_rate:.1f}",
                'Net P/L': f"${net_profit:.2f}"
            })
        
        # Sort by net profit
        instrument_stats = sorted(instrument_stats, key=lambda x: float(x['Net P/L'].replace('$', '')), reverse=True)
        
        print("\n=== PERFORMANCE BY INSTRUMENT ===")
        print(tabulate(instrument_stats, headers='keys', tablefmt='pretty'))
        
        return pd.DataFrame(instrument_stats)
    
    def analyze_by_market_condition(self):
        """Analyze performance by market condition."""
        if self.trades_df is None:
            return
        
        # Get trades with market condition
        trades_with_condition = self.trades_df[
            (self.trades_df['action'].isin(['BUY', 'SELL'])) & 
            (self.trades_df['market_condition'].notna())
        ]
        
        if len(trades_with_condition) == 0:
            return
        
        condition_stats = []
        
        for condition in trades_with_condition['market_condition'].unique():
            condition_trades = trades_with_condition[trades_with_condition['market_condition'] == condition]
            
            total = len(condition_trades)
            avg_signal_strength = condition_trades['signal_strength'].mean()
            
            condition_stats.append({
                'Market Condition': condition,
                'Trades': total,
                'Avg Signal Strength': f"{avg_signal_strength:.1f}"
            })
        
        print("\n=== TRADES BY MARKET CONDITION ===")
        print(tabulate(condition_stats, headers='keys', tablefmt='pretty'))
    
    def analyze_by_time(self):
        """Analyze performance by time of day."""
        if self.trades_df is None:
            return
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) == 0:
            return
        
        # Add hour column
        closed_trades['hour'] = pd.to_datetime(closed_trades['timestamp']).dt.hour
        
        hourly_stats = []
        
        for hour in range(24):
            hour_trades = closed_trades[closed_trades['hour'] == hour]
            
            if len(hour_trades) > 0:
                total = len(hour_trades)
                wins = len(hour_trades[hour_trades['profit_loss'] > 0])
                win_rate = (wins / total * 100) if total > 0 else 0
                net_profit = hour_trades['profit_loss'].sum()
                
                hourly_stats.append({
                    'Hour (GMT)': f"{hour:02d}:00",
                    'Trades': total,
                    'Win Rate %': f"{win_rate:.1f}",
                    'Net P/L': f"${net_profit:.2f}"
                })
        
        print("\n=== PERFORMANCE BY HOUR ===")
        print(tabulate(hourly_stats, headers='keys', tablefmt='pretty'))
    
    def plot_equity_curve(self):
        """Plot equity curve."""
        if self.trades_df is None:
            return
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])].copy()
        
        if len(closed_trades) == 0:
            return
        
        # Calculate cumulative P/L
        closed_trades = closed_trades.sort_values('timestamp')
        closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(closed_trades['timestamp'], closed_trades['cumulative_pnl'], linewidth=2)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P/L ($)')
        plt.grid(True, alpha=0.3)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        running_max = closed_trades['cumulative_pnl'].expanding().max()
        drawdown = (closed_trades['cumulative_pnl'] - running_max) / running_max * 100
        plt.fill_between(closed_trades['timestamp'], drawdown, 0, alpha=0.3, color='red')
        plt.plot(closed_trades['timestamp'], drawdown, color='red', linewidth=1)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
        print("\nEquity curve saved as 'equity_curve.png'")
        
        plt.show()
    
    def plot_win_loss_distribution(self):
        """Plot distribution of wins and losses."""
        if self.trades_df is None:
            return
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) == 0:
            return
        
        wins = closed_trades[closed_trades['profit_loss'] > 0]['profit_loss']
        losses = abs(closed_trades[closed_trades['profit_loss'] < 0]['profit_loss'])
        
        plt.figure(figsize=(10, 6))
        
        # Create bins
        bins = np.linspace(0, max(wins.max(), losses.max()) if len(wins) > 0 and len(losses) > 0 else 100, 30)
        
        # Plot histograms
        if len(wins) > 0:
            plt.hist(wins, bins=bins, alpha=0.6, label='Wins', color='green')
        if len(losses) > 0:
            plt.hist(losses, bins=bins, alpha=0.6, label='Losses', color='red')
        
        plt.xlabel('Profit/Loss Amount ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Wins and Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('win_loss_distribution.png', dpi=300, bbox_inches='tight')
        print("Win/Loss distribution saved as 'win_loss_distribution.png'")
        
        plt.show()
    
    def generate_report(self, output_file='performance_report.txt'):
        """Generate comprehensive performance report."""
        with open(output_file, 'w') as f:
            # Redirect print to file
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"TRADING PERFORMANCE REPORT")
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            
            # Overall performance
            overall_stats = self.analyze_overall_performance()
            
            # By instrument
            self.analyze_by_instrument()
            
            # By market condition
            self.analyze_by_market_condition()
            
            # By time
            self.analyze_by_time()
            
            # Risk metrics
            if overall_stats:
                print("\n=== RISK METRICS ===")
                print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.2f}")
                print(f"Maximum Drawdown: {self._calculate_max_drawdown():.2f}%")
                print(f"Recovery Factor: {self._calculate_recovery_factor():.2f}")
            
            # Restore stdout
            sys.stdout = original_stdout
        
        print(f"\nDetailed report saved as '{output_file}'")
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio."""
        if self.trades_df is None:
            return 0
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) < 2:
            return 0
        
        # Calculate daily returns
        daily_returns = closed_trades.groupby(closed_trades['timestamp'].dt.date)['profit_loss'].sum()
        
        if len(daily_returns) < 2:
            return 0
        
        # Assuming 252 trading days per year
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0
        
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
        return sharpe_ratio
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown."""
        if self.trades_df is None:
            return 0
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])].copy()
        
        if len(closed_trades) == 0:
            return 0
        
        closed_trades = closed_trades.sort_values('timestamp')
        closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
        
        running_max = closed_trades['cumulative_pnl'].expanding().max()
        drawdown = (closed_trades['cumulative_pnl'] - running_max) / running_max * 100
        
        return abs(drawdown.min())
    
    def _calculate_recovery_factor(self):
        """Calculate recovery factor (net profit / max drawdown)."""
        if self.trades_df is None:
            return 0
        
        closed_trades = self.trades_df[self.trades_df['action'].isin(['CLOSE', 'close'])]
        
        if len(closed_trades) == 0:
            return 0
        
        net_profit = closed_trades['profit_loss'].sum()
        max_dd = self._calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf') if net_profit > 0 else 0
        
        return net_profit / max_dd

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze trading performance')
    parser.add_argument('--trades', type=str, help='Path to trades CSV file')
    parser.add_argument('--performance', type=str, help='Path to performance CSV file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--report', type=str, default='performance_report.txt', 
                       help='Output report filename')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(args.trades, args.performance)
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data")
        return
    
    # Analyze
    print("\nAnalyzing trading performance...\n")
    
    # Generate text analysis
    analyzer.analyze_overall_performance()
    analyzer.analyze_by_instrument()
    analyzer.analyze_by_market_condition()
    analyzer.analyze_by_time()
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_equity_curve()
        analyzer.plot_win_loss_distribution()
    
    # Generate report
    analyzer.generate_report(args.report)

if __name__ == "__main__":
    main()