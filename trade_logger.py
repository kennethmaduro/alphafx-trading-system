"""
Trade Logger for OANDA Trading System

This module handles detailed trade logging, analytics, and report generation.
"""
import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Trade Logger for capturing and analyzing detailed trade information.
    
    This class provides methods for logging trade data, calculating performance
    metrics, and exporting results in various formats for analysis.
    """
    
    def __init__(self, log_dir=None):
        """
        Initialize the Trade Logger.
        
        Args:
            log_dir (str, optional): Directory to store log files. If None,
                                     logs will be stored in a 'trade_logs' subdirectory.
        """
        # Set up log directory
        if log_dir is None:
            self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_logs')
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize trade log storage
        self.trade_logs = []
        self.performance_logs = []
        
        # Performance tracking
        self.initial_balance = None
        self.current_balance = None
        self.max_balance = None
        self.current_drawdown = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Track trade count by instrument and action
        self.trade_counts = defaultdict(lambda: defaultdict(int))
        
        # Track profit by market condition and timeframe
        self.profit_by_condition = defaultdict(float)
        self.profit_by_timeframe = defaultdict(float)
        
        # Create log files
        self.trade_log_file = os.path.join(self.log_dir, f'trades_{self.session_timestamp}.csv')
        self.performance_log_file = os.path.join(self.log_dir, f'performance_{self.session_timestamp}.csv')
        self.summary_file = os.path.join(self.log_dir, f'summary_{self.session_timestamp}.json')
        
        # Create CSV file headers
        self._initialize_log_files()
        
        logger.info(f"Trade Logger initialized. Logs will be stored in {self.log_dir}")
    
    def _initialize_log_files(self):
        """Initialize log files with headers."""
        # Trade log header
        with open(self.trade_log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'instrument', 'action', 'price', 'units', 
                'profit_loss', 'signal_strength', 'market_condition', 
                'timeframe', 'indicators', 'risk_amount', 'risk_pct',
                'stop_loss', 'take_profit', 'trade_duration'
            ])
        
        # Performance log header
        with open(self.performance_log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'balance', 'drawdown', 'win_rate', 
                'profit_factor', 'win_count', 'loss_count',
                'active_positions'
            ])
    
    def set_initial_balance(self, balance):
        """
        Set the initial account balance.
        
        Args:
            balance (float): Initial account balance
        """
        self.initial_balance = balance
        self.current_balance = balance
        self.max_balance = balance
    
    def log_trade(self, trade_data):
        """
        Log a trade with detailed information.
        
        Args:
            trade_data (dict): Dictionary containing trade details
        """
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Save to memory
        self.trade_logs.append(trade_data)
        
        # Update trade counts
        instrument = trade_data.get('instrument', 'unknown')
        action = trade_data.get('action', 'unknown')
        self.trade_counts[instrument][action] += 1
        
        # Debug logging for closing trades
        if action.lower() in ['close', 'close_long', 'close_short']:
            logger.info(f"TradeLogger: Processing close trade - Action: {action}, P/L: {trade_data.get('profit_loss', 0)}, Data: {trade_data}")
        
        # Update performance metrics if this is a closing trade
        profit_loss = trade_data.get('profit_loss', 0)
        if action.lower() in ['close', 'close_long', 'close_short'] and profit_loss != 0:
            # Update win/loss counts
            if profit_loss > 0:
                self.win_count += 1
                self.total_profit += profit_loss
            else:
                self.loss_count += 1
                self.total_loss += abs(profit_loss)
            
            # Update balance
            if self.current_balance is not None:
                self.current_balance += profit_loss
                if self.current_balance > self.max_balance:
                    self.max_balance = self.current_balance
                
                # Calculate drawdown
                if self.max_balance > 0:
                    self.current_drawdown = max(0, 1 - (self.current_balance / self.max_balance))
            
            # Track profit by market condition and timeframe
            market_condition = trade_data.get('market_condition', 'unknown')
            timeframe = trade_data.get('timeframe', 'unknown')
            
            self.profit_by_condition[market_condition] += profit_loss
            self.profit_by_timeframe[timeframe] += profit_loss
        
        # Write to CSV file
        self._write_trade_to_csv(trade_data)
    
    def _write_trade_to_csv(self, trade_data):
        """
        Write a trade to the CSV log file.
        
        Args:
            trade_data (dict): Dictionary containing trade details
        """
        try:
            # Format indicators as a string
            indicators = trade_data.get('indicators', {})
            if isinstance(indicators, dict):
                indicator_str = ','.join([f"{k}={v}" for k, v in indicators.items()])
            else:
                indicator_str = str(indicators)
            
            # Prepare row data
            row = [
                trade_data.get('timestamp', ''),
                trade_data.get('instrument', ''),
                trade_data.get('action', ''),
                trade_data.get('price', 0),
                trade_data.get('units', 0),
                trade_data.get('profit_loss', 0),
                trade_data.get('signal_strength', 0),
                trade_data.get('market_condition', ''),
                trade_data.get('timeframe', ''),
                indicator_str,
                trade_data.get('risk_amount', 0),
                trade_data.get('risk_pct', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('trade_duration', 0)
            ]
            
            # Write to file
            with open(self.trade_log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Error writing trade to CSV: {str(e)}")
    
    def log_performance(self, active_positions=0):
        """
        Log current performance metrics.
        
        Args:
            active_positions (int): Number of active positions
        """
        # Calculate performance metrics
        win_rate = (self.win_count / (self.win_count + self.loss_count) * 100) if (self.win_count + self.loss_count) > 0 else 0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0
        
        # Create performance entry
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.current_balance,
            'drawdown': self.current_drawdown * 100,  # Convert to percentage
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'active_positions': active_positions
        }
        
        # Save to memory
        self.performance_logs.append(performance_data)
        
        # Write to CSV
        with open(self.performance_log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                performance_data['timestamp'],
                performance_data['balance'],
                performance_data['drawdown'],
                performance_data['win_rate'],
                performance_data['profit_factor'],
                performance_data['win_count'],
                performance_data['loss_count'],
                performance_data['active_positions']
            ])
    
    def generate_summary(self):
        """
        Generate a summary of trading performance.
        
        Returns:
            dict: Summary of trading performance
        """
        # Calculate performance metrics
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        profit_loss = self.total_profit - self.total_loss
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else float('inf')
        
        if self.initial_balance and self.initial_balance > 0:
            profit_pct = (profit_loss / self.initial_balance) * 100
        else:
            profit_pct = 0
        
        # Calculate average win and loss
        avg_win = self.total_profit / self.win_count if self.win_count > 0 else 0
        avg_loss = self.total_loss / self.loss_count if self.loss_count > 0 else 0
        
        # Get trades by instrument
        trades_by_instrument = {
            instrument: sum(counts.values())
            for instrument, counts in self.trade_counts.items()
        }
        
        # Calculate win rate by market condition
        wins_by_condition = defaultdict(int)
        losses_by_condition = defaultdict(int)
        
        for trade in self.trade_logs:
            if trade.get('action', '').lower() in ['close', 'close_long', 'close_short']:
                profit_loss = trade.get('profit_loss', 0)
                market_condition = trade.get('market_condition', 'unknown')
                
                if profit_loss > 0:
                    wins_by_condition[market_condition] += 1
                elif profit_loss < 0:
                    losses_by_condition[market_condition] += 1
        
        win_rate_by_condition = {}
        for condition in set(list(wins_by_condition.keys()) + list(losses_by_condition.keys())):
            total = wins_by_condition[condition] + losses_by_condition[condition]
            if total > 0:
                win_rate_by_condition[condition] = (wins_by_condition[condition] / total) * 100
            else:
                win_rate_by_condition[condition] = 0
        
        # Create summary dictionary
        summary = {
            'session_start': self.session_timestamp,
            'session_end': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_pct,
            'max_drawdown': self.current_drawdown * 100,  # Convert to percentage
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades_by_instrument': trades_by_instrument,
            'profit_by_market_condition': dict(self.profit_by_condition),
            'win_rate_by_market_condition': win_rate_by_condition,
            'profit_by_timeframe': dict(self.profit_by_timeframe)
        }
        
        # Save summary to file
        with open(self.summary_file, 'w') as file:
            json.dump(summary, file, indent=4)
        
        return summary
    
    def get_top_profitable_conditions(self):
        """
        Get the most profitable market conditions.
        
        Returns:
            list: List of (condition, profit) tuples sorted by profit
        """
        return sorted(self.profit_by_condition.items(), key=lambda x: x[1], reverse=True)
    
    def get_top_profitable_timeframes(self):
        """
        Get the most profitable timeframes.
        
        Returns:
            list: List of (timeframe, profit) tuples sorted by profit
        """
        return sorted(self.profit_by_timeframe.items(), key=lambda x: x[1], reverse=True)
    
    def create_trades_dataframe(self):
        """
        Convert trade logs to a pandas DataFrame for analysis.
        
        Returns:
            pd.DataFrame: DataFrame containing trade data
        """
        if not self.trade_logs:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_logs)
    
    def create_performance_dataframe(self):
        """
        Convert performance logs to a pandas DataFrame for analysis.
        
        Returns:
            pd.DataFrame: DataFrame containing performance data
        """
        if not self.performance_logs:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_logs)
    
    def export_to_csv(self, filepath=None):
        """
        Export all trade logs to a single CSV file.
        
        Args:
            filepath (str, optional): Path to the CSV file. If None, a default path is used.
            
        Returns:
            str: Path to the exported CSV file
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, f'trade_export_{self.session_timestamp}.csv')
        
        df = self.create_trades_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} trades to {filepath}")
        else:
            logger.warning("No trades to export")
        
        return filepath
    
    def print_summary(self):
        """Print a summary of trading performance to the log."""
        summary = self.generate_summary()
        
        logger.info("=== TRADING SESSION SUMMARY ===")
        logger.info(f"Session: {summary['session_start']} to {summary['session_end']}")
        logger.info(f"Initial Balance: ${summary['initial_balance']:.2f}")
        logger.info(f"Final Balance: ${summary['final_balance']:.2f}")
        logger.info(f"Profit/Loss: ${summary['profit_loss']:.2f} ({summary['profit_loss_pct']:.2f}%)")
        logger.info(f"Maximum Drawdown: {summary['max_drawdown']:.2f}%")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['win_rate']:.2f}% ({summary['win_count']} wins, {summary['loss_count']} losses)")
        logger.info(f"Profit Factor: {summary['profit_factor']:.2f}")
        logger.info(f"Average Win: ${summary['avg_win']:.2f}, Average Loss: ${summary['avg_loss']:.2f}")
        
        logger.info("Profit by Market Condition:")
        for condition, profit in self.get_top_profitable_conditions():
            logger.info(f"  {condition}: ${profit:.2f}")
        
        logger.info("Profit by Timeframe:")
        for timeframe, profit in self.get_top_profitable_timeframes():
            logger.info(f"  {timeframe}: ${profit:.2f}")
        
        return summary