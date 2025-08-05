#!/usr/bin/env python
"""
Strategy Tester for Trading System
Tests strategy logic with historical data without placing real trades
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from market_data_adapter import MarketDataAdapter
from cep_engine import CEPEngine, SignalType, MarketCondition
from technical_indicators import TechnicalIndicators
from collections import deque


class KellyCriterion:
    """Kelly Criterion implementation for optimal position sizing."""
    
    def __init__(self, lookback_trades=50, kelly_fraction=0.25):
        self.lookback_trades = lookback_trades
        self.kelly_fraction = kelly_fraction  # Use 25% of Kelly for safety
        self.trade_results = deque(maxlen=lookback_trades)
    
    def add_trade_result(self, profit_loss, risk_amount):
        """Add a trade result for Kelly calculation."""
        win = 1 if profit_loss > 0 else 0
        win_amount = abs(profit_loss) if profit_loss > 0 else 0
        loss_amount = abs(profit_loss) if profit_loss < 0 else risk_amount
        
        self.trade_results.append({
            'win': win,
            'win_amount': win_amount,
            'loss_amount': loss_amount
        })
    
    def calculate_optimal_fraction(self):
        """Calculate optimal betting fraction using Kelly Criterion."""
        if len(self.trade_results) < 20:  # Need minimum trades
            return 0.5  # Default 50% until enough data (less conservative)
        
        wins = sum(t['win'] for t in self.trade_results)
        total = len(self.trade_results)
        
        if wins == 0 or wins == total:  # All losses or all wins
            return 0.01  # Default conservative size
        
        # Calculate win probability
        p = wins / total
        
        # Calculate average win/loss ratio
        avg_win = sum(t['win_amount'] for t in self.trade_results if t['win']) / wins
        avg_loss = sum(t['loss_amount'] for t in self.trade_results if not t['win']) / (total - wins)
        
        if avg_loss == 0:
            return 0.01
        
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - p  # Loss probability
        
        # Kelly formula: f = (p * b - q) / b
        kelly_full = (p * b - q) / b
        
        # Apply safety fraction and bounds
        kelly_optimal = kelly_full * self.kelly_fraction
        
        # Bound between 0.5% and 3%
        return max(0.005, min(0.03, kelly_optimal))


class VolatilityManager:
    """Manage position sizing based on market volatility."""
    
    def __init__(self):
        self.volatility_history = {}
        self.volatility_percentiles = {}
    
    def update_volatility(self, instrument, atr_pct):
        """Update volatility tracking for an instrument."""
        if instrument not in self.volatility_history:
            self.volatility_history[instrument] = deque(maxlen=100)
        
        self.volatility_history[instrument].append(atr_pct)
        
        # Calculate percentiles
        if len(self.volatility_history[instrument]) >= 20:
            volatilities = list(self.volatility_history[instrument])
            self.volatility_percentiles[instrument] = {
                'p20': np.percentile(volatilities, 20),
                'p50': np.percentile(volatilities, 50),
                'p80': np.percentile(volatilities, 80)
            }
    
    def get_volatility_adjustment(self, instrument, current_atr_pct):
        """Get position size adjustment based on volatility."""
        if instrument not in self.volatility_percentiles:
            return 1.0
        
        percentiles = self.volatility_percentiles[instrument]
        
        if current_atr_pct < percentiles['p20']:
            # Very low volatility - normal size
            return 1.0
        elif current_atr_pct < percentiles['p50']:
            # Below median - minimal reduction
            return 0.95
        elif current_atr_pct < percentiles['p80']:
            # Above median - slight reduction (less aggressive)
            return 0.85
        else:
            # High volatility - moderate reduction (less aggressive)
            return 0.7

class StrategyTester:
    """Multi-Instrument Portfolio Backtest System - IDENTICAL to live trading system."""
    
    def __init__(self, instruments=None, initial_balance=10000):
        # Handle both single instrument (backward compatibility) and multi-instrument
        if isinstance(instruments, str):
            self.instruments = [instruments]  # Convert single string to list
        elif isinstance(instruments, list):
            self.instruments = instruments
        else:
            # Default to all 8 pairs like live trading
            self.instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'AUD_CAD', 'AUD_NZD', 'NZD_CAD']
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_balance = initial_balance
        self.daily_starting_balance = initial_balance
        self.daily_high_balance = initial_balance
        
        print(f"Portfolio Backtest System Initialized")
        print(f"   Instruments: {', '.join(self.instruments)}")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        
        # Market data and processing
        self.market_data_adapter = MarketDataAdapter()
        self.cep_engine = CEPEngine()
        self.indicators = TechnicalIndicators()
        
        # Risk management parameters (SAME AS LIVE)
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.default_risk_per_trade = config.DEFAULT_RISK_PER_TRADE
        self.max_total_positions = config.POSITION_MANAGEMENT['MAX_TOTAL_POSITIONS']
        self.max_positions_per_instrument = config.POSITION_MANAGEMENT['MAX_POSITIONS_PER_INSTRUMENT']
        self.min_time_between_trades = config.POSITION_MANAGEMENT['MIN_TIME_BETWEEN_TRADES']
        
        # Portfolio-wide position tracking (SAME AS LIVE)
        self.open_positions = {}  # instrument -> position_data
        self.position_details = {}  # instrument -> detailed_data
        self.last_trade_time = {}  # instrument -> last_trade_timestamp
        
        # Per-instrument Kelly Criterion and volatility management (SAME AS LIVE)
        self.kelly_criterion = {instrument: KellyCriterion() for instrument in self.instruments}
        self.volatility_manager = VolatilityManager()
        
        # Portfolio performance tracking (SAME AS LIVE)
        self.daily_drawdown = 0
        self.max_drawdown = 0
        
        # Trade tracking (SAME AS LIVE)
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Correlation tracking (SAME AS LIVE)
        self.active_pairs = set()
        self.correlation_matrix = pd.DataFrame()
        
        # Recovery mode (SAME AS LIVE)
        self.recovery_mode = False
        
        # Session tracking (SAME AS LIVE)
        self.session_start_time = datetime.now()
        self.trade_entry_times = {}
        
        # Portfolio performance metrics
        self.equity_curve = []
        self.signals_generated = []
        
        # Per-instrument signal tracking
        self.signal_history = {instrument: [] for instrument in self.instruments}
        
    async def fetch_historical_data_multi_instrument(self, days=5):
        """Fetch historical data for all instruments in parallel - OPTIMIZED FOR HIGH-PERFORMANCE HARDWARE."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        start_time = time.time()
        print(f"\n*** PARALLEL DATA FETCHING: {days} days for {len(self.instruments)} instruments...")
        print(f"*** Hardware Optimization: Using multi-threaded parallel processing")
        
        async def fetch_single_instrument(instrument):
            """Fetch data for a single instrument asynchronously."""
            try:
                # Direct API call without nested tester
                adapter = MarketDataAdapter()
                
                # Fetch all timeframes in parallel for this instrument
                async def fetch_timeframe_data(tf_key, tf_config):
                    if tf_key == "M1":
                        candles_needed = days * 24 * 60
                    elif tf_key == "M5":
                        candles_needed = days * 24 * 12
                    elif tf_key == "M15":
                        candles_needed = days * 24 * 4
                    else:
                        candles_needed = tf_config['count']
                    
                    df = await adapter.fetch_candles(
                        instrument=instrument,
                        granularity=tf_config['granularity'],
                        count=min(candles_needed, 5000)
                    )
                    
                    if df is not None and not df.empty:
                        df = self.indicators.add_all_indicators(df)
                        return tf_key, df
                    return tf_key, None
                
                # Fetch all timeframes for this instrument in parallel
                tf_tasks = [fetch_timeframe_data(tf_key, tf_config) 
                           for tf_key, tf_config in config.TIMEFRAMES.items()]
                tf_results = await asyncio.gather(*tf_tasks)
                
                # Build timeframe data dict
                instrument_data = {}
                for tf_key, df in tf_results:
                    if df is not None:
                        instrument_data[tf_key] = df
                
                return instrument, instrument_data if instrument_data else None
                
            except Exception as e:
                print(f"ERROR {instrument}: {str(e)}")
                return instrument, None
        
        # Create concurrent tasks for all instruments
        tasks = [fetch_single_instrument(instrument) for instrument in self.instruments]
        
        # Execute all tasks in parallel with limited concurrency to respect API limits
        semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent requests
        
        async def limited_fetch(task):
            async with semaphore:
                return await task
        
        print(f"*** Launching {len(tasks)} parallel data fetch tasks...")
        results = await asyncio.gather(*[limited_fetch(task) for task in tasks])
        
        # Process results
        all_instrument_data = {}
        successful_instruments = []
        failed_instruments = []
        
        for instrument, data in results:
            if data:
                all_instrument_data[instrument] = data
                candle_count = len(data.get('M5', pd.DataFrame()))
                successful_instruments.append(f"{instrument}({candle_count})")
                print(f"SUCCESS {instrument}: {candle_count} M5 candles")
            else:
                failed_instruments.append(instrument)
                print(f"FAILED {instrument}: Failed")
        
        elapsed_time = time.time() - start_time
        print(f"\n*** PARALLEL FETCH COMPLETE in {elapsed_time:.1f}s")
        print(f"*** Success: {len(successful_instruments)}/{len(self.instruments)} instruments")
        if successful_instruments:
            print(f"   {' | '.join(successful_instruments)}")
        if failed_instruments:
            print(f"*** Failed: {', '.join(failed_instruments)}")
        
        if len(all_instrument_data) == 0:
            print(f"WARNING: No data fetched for any instrument!")
            return None
        
        return all_instrument_data
        
    async def fetch_historical_data(self, days=5):
        """Fetch historical data with parallel timeframe processing - OPTIMIZED FOR PERFORMANCE."""
        import asyncio
        
        # This method now works with the first instrument for backward compatibility
        instrument = self.instruments[0] if self.instruments else 'EUR_USD'
        
        async def fetch_timeframe_data(tf_key, tf_config):
            """Fetch data for a single timeframe asynchronously."""
            try:
                # Calculate required candles
                if tf_key == "M1":
                    candles_needed = days * 24 * 60
                elif tf_key == "M5":
                    candles_needed = days * 24 * 12
                elif tf_key == "M15":
                    candles_needed = days * 24 * 4
                else:
                    candles_needed = tf_config['count']
                
                # Handle large requests with multiple API calls
                if candles_needed <= 5000:
                    # Single API call
                    df = await self.market_data_adapter.fetch_candles(
                        instrument, 
                        tf_config['granularity'], 
                        candles_needed
                    )
                else:
                    # Multiple API calls needed
                    all_dataframes = []
                    call_count = 0
                    remaining_candles = candles_needed
                    
                    while remaining_candles > 0 and call_count < 10:  # Safety limit
                        call_count += 1
                        current_batch = min(remaining_candles, 5000)
                        
                        df_batch = await self.market_data_adapter.fetch_candles(
                            instrument,
                            tf_config['granularity'], 
                            current_batch
                        )
                        
                        if df_batch is not None and not df_batch.empty:
                            all_dataframes.append(df_batch)
                            batch_size = len(df_batch)
                            remaining_candles -= batch_size
                            
                            # If we got fewer candles than requested, we've reached the end
                            if batch_size < current_batch:
                                break
                        else:
                            break
                        
                        # Minimal delay between API calls
                        await asyncio.sleep(0.05)  # Reduced from 0.1s
                    
                    # Combine all dataframes
                    if all_dataframes:
                        df = pd.concat(all_dataframes).drop_duplicates().sort_index()
                    else:
                        df = None
                
                if df is not None and not df.empty:
                    # Add indicators
                    df = self.indicators.add_all_indicators(df)
                    return tf_key, df, len(df)
                else:
                    return tf_key, None, 0
                    
            except Exception as e:
                return tf_key, None, 0
        
        # Create parallel tasks for all timeframes
        tasks = [fetch_timeframe_data(tf_key, tf_config) 
                for tf_key, tf_config in config.TIMEFRAMES.items()]
        
        # Execute all timeframe fetches in parallel
        results = await asyncio.gather(*tasks)
        
        # Process results
        timeframe_data = {}
        for tf_key, df, candle_count in results:
            if df is not None:
                timeframe_data[tf_key] = df
        
        return timeframe_data
    
    async def run_portfolio_backtest(self, all_timeframe_data, risk_per_trade=0.03):
        """Run portfolio backtest across all instruments with portfolio-wide risk management."""
        print(f"\nStarting Portfolio Backtest")
        print(f"   Instruments: {len(all_timeframe_data)} of {len(self.instruments)}")
        print(f"   Risk per trade: {risk_per_trade:.1%}")
        print(f"   Initial balance: ${self.initial_balance:,.2f}")
        
        # Validate we have data for at least some instruments
        if not all_timeframe_data:
            print("ERROR: No timeframe data provided")
            return
        
        # Get primary timeframe (M1) from all instruments  
        primary_tf = "M1"
        instrument_dataframes = {}
        
        for instrument, timeframe_data in all_timeframe_data.items():
            if primary_tf in timeframe_data and not timeframe_data[primary_tf].empty:
                instrument_dataframes[instrument] = timeframe_data[primary_tf]
            else:
                print(f"WARNING: No {primary_tf} data for {instrument}")
        
        if not instrument_dataframes:
            print(f"ERROR: No {primary_tf} data available for any instrument")
            return
        
        print(f"Processing {len(instrument_dataframes)} instruments with {primary_tf} data")
        
        # Create unified timeline from all instruments
        all_timestamps = set()
        for df in instrument_dataframes.values():
            all_timestamps.update(df.index)
        
        unified_timeline = sorted(all_timestamps)
        print(f"   Unified timeline: {len(unified_timeline)} timestamps")
        print(f"   Period: {unified_timeline[0]} to {unified_timeline[-1]}")
        
        # Portfolio backtest loop with performance optimizations
        signals_processed = 0
        trades_executed = 0
        
        # Progress tracking for large datasets
        total_timestamps = len(unified_timeline)
        progress_interval = max(1, total_timestamps // 20)  # Show progress every 5%
        
        for i, timestamp in enumerate(unified_timeline):
            # Progress reporting
            if i % progress_interval == 0 or i == total_timestamps - 1:
                progress_pct = (i + 1) / total_timestamps * 100
                print(f"   Progress: {progress_pct:.1f}% ({i+1:,}/{total_timestamps:,} timestamps)")
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.current_balance,
                'equity': self.current_balance,  # Simplified for backtest
                'drawdown': (self.max_balance - self.current_balance) / self.max_balance if self.max_balance > 0 else 0
            })
            
            # Check for time-based position exits FIRST (before processing new signals)
            positions_to_close = []
            for instrument, position in self.open_positions.items():
                entry_time = position['timestamp']
                entry_price = position['price']
                units = position['units']
                
                # Get current price for this instrument if available
                if instrument in instrument_dataframes:
                    df = instrument_dataframes[instrument]
                    if timestamp in df.index:
                        row = df.loc[timestamp]
                        current_price = (row['high'] + row['low']) / 2
                        
                        # Check maximum hold time (4 hours = 14400 seconds)
                        time_held = (timestamp - entry_time).total_seconds()
                        if time_held >= config.POSITION_MANAGEMENT.get('MAX_POSITION_HOLD_TIME', 14400):
                            positions_to_close.append((instrument, current_price, "Max hold time"))
                            continue
                        
                        # Check profit target (1.5%)
                        if units > 0:  # Long position
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:  # Short position
                            pnl_pct = (entry_price - current_price) / entry_price
                        
                        profit_target = config.POSITION_MANAGEMENT.get('PROFIT_TARGET_CLOSE', 0.015)
                        stop_loss = config.POSITION_MANAGEMENT.get('STOP_LOSS_CLOSE', -0.02)
                        
                        if pnl_pct >= profit_target:
                            positions_to_close.append((instrument, current_price, "Profit target"))
                        elif pnl_pct <= stop_loss:
                            positions_to_close.append((instrument, current_price, "Stop loss"))
            
            # Close positions that meet time/profit/loss criteria
            for instrument, close_price, reason in positions_to_close:
                self._close_position_portfolio(instrument, timestamp, close_price, 0, reason)
            
            # Process each instrument at this timestamp
            for instrument in instrument_dataframes.keys():
                df = instrument_dataframes[instrument]
                
                # Skip if no data for this timestamp
                if timestamp not in df.index:
                    continue
                
                row = df.loc[timestamp]
                current_price = (row['high'] + row['low']) / 2  # Mid price
                
                # Get all indicator data for this row
                indicator_data = {
                    'atr': row.get('atr', 0.001),
                    'sma_20': row.get('sma_20', current_price),
                    'ema_12': row.get('ema_12', current_price),
                    'ema_26': row.get('ema_26', current_price),
                    'rsi': row.get('rsi', 50),
                    'macd': row.get('macd', 0),
                    'macd_signal': row.get('macd_signal', 0),
                    'bb_upper': row.get('bb_upper', current_price * 1.01),
                    'bb_lower': row.get('bb_lower', current_price * 0.99),
                    'stoch_k': row.get('stoch_k', 50),
                    'stoch_d': row.get('stoch_d', 50)
                }
                
                # Generate signals using CEP engine
                # Prepare ALL timeframe data like live system (M1, M5, M15)
                current_timeframe_data = {}
                for timeframe, tf_data in all_timeframe_data[instrument].items():
                    if tf_data is not None and not tf_data.empty:
                        # Get data up to current timestamp with sufficient history
                        current_data = tf_data.loc[tf_data.index <= timestamp].tail(100)
                        if not current_data.empty:
                            current_timeframe_data[timeframe] = current_data
                
                signal_type, signal_strength, market_condition, _ = self.cep_engine.process_timeframe_data(
                    current_timeframe_data
                )
                
                # Process the signal if it's actionable
                if signal_type != SignalType.NO_ACTION:
                    signals_processed += 1
                    print(f"Signal generated for {instrument}: {signal_type.value}, strength: {signal_strength:.2f}")
                    
                    # Process the signal using portfolio-wide logic
                    trade_executed = self._process_signal_portfolio(
                        instrument=instrument,
                        timestamp=timestamp,
                        price=current_price,
                        signal_type=signal_type,
                        signal_strength=signal_strength,
                        indicator_data=indicator_data,
                        market_condition=market_condition,
                        risk_per_trade=risk_per_trade,
                        all_timeframe_data=all_timeframe_data
                    )
                    
                    if trade_executed:
                        trades_executed += 1
            
            # Update max balance for drawdown calculation
            if self.current_balance > self.max_balance:
                self.max_balance = self.current_balance
            
            # Progress update every 1000 timestamps
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / len(unified_timeline) * 100
                print(f"   Progress: {progress:.1f}% ({i+1:,}/{len(unified_timeline):,} timestamps)")
        
        print(f"\nPortfolio Backtest Complete")
        print(f"   Signals processed: {signals_processed:,}")
        print(f"   Trades executed: {trades_executed:,}")
        print(f"   Final balance: ${self.current_balance:,.2f}")
        print(f"   Total return: {((self.current_balance - self.initial_balance) / self.initial_balance * 100):+.2f}%")
        
    async def run_backtest(self, timeframe_data, risk_per_trade=0.01):
        """Run backtest on historical data."""
        print(f"\nRunning backtest with {risk_per_trade:.1%} risk per trade...")
        
        # Use M1 as primary timeframe for unified timeline (like live system)
        primary_tf = 'M1'
        if primary_tf not in timeframe_data or timeframe_data[primary_tf] is None:
            print("Error: No M1 data available for backtesting")
            return
        
        primary_df = timeframe_data[primary_tf]
        
        # Ensure we have enough data
        min_lookback = 50
        if len(primary_df) < min_lookback:
            print(f"Error: Insufficient data. Need at least {min_lookback} candles")
            return
        
        # Iterate through data
        for i in range(min_lookback, len(primary_df)):
            current_time = primary_df.index[i]
            current_price = primary_df.iloc[i]['close']
            
            # Prepare data for signal generation
            current_data = {}
            for tf_key, df in timeframe_data.items():
                # Get data up to current time
                mask = df.index <= current_time
                if mask.sum() > 0:
                    current_data[tf_key] = df[mask].copy()
            
            # DEBUG: Check data being passed to CEP (first few iterations)
            if i < 5:
                print(f"DEBUG: Timeframe data keys: {list(current_data.keys())}")
                for tf, df in current_data.items():
                    if len(df) > 0:
                        print(f"DEBUG: {tf} data shape: {df.shape}, latest close: {df['close'].iloc[-1]:.5f}")
                        if 'sma_20' in df.columns:
                            print(f"DEBUG: {tf} SMA_20: {df['sma_20'].iloc[-1]:.5f}")
                        break
            
            # Generate signal using CEP engine
            signal_type, signal_strength, market_condition, indicator_data = (
                self.cep_engine.process_timeframe_data(current_data)
            )
            
            # DEBUG: Check CEP output (first few iterations)
            if i < 5:
                print(f"DEBUG: CEP output - Type: {signal_type}, Strength: {signal_strength}, Condition: {market_condition}")
            
            # Store signal
            self.signals_generated.append({
                'timestamp': current_time,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'market_condition': market_condition
            })
            
            # Process signal using EXACT SAME LOGIC as live system
            self._process_signal(current_time, current_price, signal_type, signal_strength, indicator_data, market_condition)
            
            # Check exit conditions for all open positions
            self._check_exit_conditions(current_time, current_price)
            
            # Update equity curve
            equity = self.current_balance
            # Calculate unrealized P/L for all open positions
            total_unrealized_pnl = 0
            for instrument, position in self.open_positions.items():
                if position['units'] > 0:  # Long position
                    unrealized_pnl = (current_price - position['price']) * position['units']
                else:  # Short position
                    unrealized_pnl = (position['price'] - current_price) * abs(position['units'])
                total_unrealized_pnl += unrealized_pnl
            
            equity += total_unrealized_pnl
            
            self.equity_curve.append({
                'time': current_time,
                'balance': self.current_balance,
                'equity': equity
            })
            
            # Already checking exit conditions above in the main loop
        
        # Close any remaining positions
        remaining_instruments = list(self.open_positions.keys())
        for instrument in remaining_instruments:
            self._close_position(instrument, current_time, current_price, "End of test")
        
        print(f"\nBacktest completed. Processed {len(self.signals_generated)} signals")
        print(f"DEBUG: Open positions: {len(self.open_positions)}, Trade history: {len(self.trade_history)}")
        print(f"DEBUG: Active pairs: {self.active_pairs}")
        if self.open_positions:
            for instrument, pos in self.open_positions.items():
                print(f"DEBUG: Open position {instrument}: {pos['units']} units at {pos['price']:.5f}")
    
    def _record_signal(self, instrument, signal_type, signal_strength):
        """Record signal for analysis - SAME AS LIVE SYSTEM."""
        signal_record = {
            'instrument': instrument,
            'timestamp': pd.Timestamp.now(),
            'signal_type': signal_type,
            'signal_strength': signal_strength
        }
        
        self.signal_history[instrument].append(signal_record)
        
        # Keep only last 100 signals per instrument
        if len(self.signal_history[instrument]) > 100:
            self.signal_history[instrument] = self.signal_history[instrument][-100:]
    
    def _process_signal_portfolio(self, instrument, timestamp, price, signal_type, signal_strength, 
                                 indicator_data, market_condition, risk_per_trade, all_timeframe_data):
        """Process trading signal for portfolio backtest - IDENTICAL to live system logic."""
        
        # Check if we should stop trading (SAME AS LIVE)
        if self.recovery_mode and self.daily_drawdown >= config.CRITICAL_DRAWDOWN:
            return False
        
        # Check global position limits (SAME AS LIVE)
        if len(self.open_positions) >= self.max_total_positions:
            return False
        
        # Record signal for analysis (SAME AS LIVE)
        self._record_signal(instrument, signal_type, signal_strength)
        
        # Check position status (SAME AS LIVE)
        has_position = instrument in self.open_positions
        
        # Skip weak signals (SAME AS LIVE)
        min_strength = config.SIGNAL_THRESHOLDS.get('MIN_SIGNAL_STRENGTH', 30)
        if abs(signal_strength) < min_strength:
            return False
        
        # Signal thresholds (SAME AS LIVE)
        new_long_threshold = config.SIGNAL_THRESHOLDS.get('NEW_LONG', 40)
        new_short_threshold = config.SIGNAL_THRESHOLDS.get('NEW_SHORT', -40)
        close_threshold = config.SIGNAL_THRESHOLDS.get('CLOSE_POSITION', 20)
        
        print(f"DEBUG {instrument}: signal={signal_type.value}, strength={signal_strength:.2f}, thresholds: long>={new_long_threshold}, short<={new_short_threshold}")
        
        # Check if we should open new position (SAME AS LIVE)
        if not has_position:
            # Check instrument-specific position limits
            instrument_positions = sum(1 for pos_inst in self.open_positions.keys() if pos_inst == instrument)
            if signal_type == SignalType.BUY and signal_strength >= new_long_threshold:
                return self._open_position_portfolio(
                    instrument, True, timestamp, price, signal_strength, 
                    indicator_data, market_condition, risk_per_trade
                )
            elif signal_type == SignalType.SELL and signal_strength <= new_short_threshold:
                return self._open_position_portfolio(
                    instrument, False, timestamp, price, signal_strength, 
                    indicator_data, market_condition, risk_per_trade
                )
        else:
            # Handle position closing and reversals (SAME AS LIVE)
            current_position = self.open_positions[instrument]
            is_long = current_position['units'] > 0  # Positive units = long, negative = short
            
            # Close on explicit CLOSE signal
            if signal_type == SignalType.CLOSE and abs(signal_strength) >= close_threshold:
                return self._close_position_portfolio(instrument, timestamp, price, 
                                                    signal_strength, "SIGNAL_CLOSE")
            
            # Close on OPPOSITE signal (SELL when LONG, BUY when SHORT)
            elif ((is_long and signal_type == SignalType.SELL and signal_strength <= new_short_threshold) or
                  (not is_long and signal_type == SignalType.BUY and signal_strength >= new_long_threshold)):
                return self._close_position_portfolio(instrument, timestamp, price, 
                                                    signal_strength, "OPPOSITE_SIGNAL")
    
        return False
    
    def _process_signal(self, timestamp, price, signal_type, signal_strength, indicator_data, market_condition):
        """Process trading signal - EXACT SAME LOGIC AS LIVE SYSTEM."""
        instrument = self.instrument
        
        # Check if we should stop trading (SAME AS LIVE)
        if self.recovery_mode and self.daily_drawdown >= config.CRITICAL_DRAWDOWN:
            print(f"Trading halted due to critical drawdown: {self.daily_drawdown:.2%}")
            return False
        
        # Check global position limits (SAME AS LIVE)
        if len(self.open_positions) >= self.max_total_positions:
            print(f"Maximum total positions ({self.max_total_positions}) reached")
            return False
        
        # Record signal for analysis (SAME AS LIVE)
        self._record_signal(instrument, signal_type, signal_strength)
        
        # DEBUG: Log signal details
        if len(self.signals_generated) <= 10:  # Only log first 10 to avoid spam
            print(f"DEBUG: Processing signal - Type: {signal_type.value}, Strength: {signal_strength}, Instrument: {instrument}")
        
        # Check position status (SAME AS LIVE)
        has_position = instrument in self.open_positions
        current_position_count = self.position_count.get(instrument, 0)
        
        # Skip weak signals (SAME AS LIVE)
        min_strength = config.SIGNAL_THRESHOLDS.get('MIN_SIGNAL_STRENGTH', 30)
        if abs(signal_strength) < min_strength:
            if len(self.signals_generated) <= 10:
                print(f"DEBUG: Signal too weak - Strength: {signal_strength}, Min required: {min_strength}")
            return False
        
        # Check if we should open new position (SAME AS LIVE)
        if not has_position:
            # Check instrument-specific position limits
            if current_position_count >= self.max_positions_per_instrument:
                return False
            
            # Check correlation limits
            if not self._check_correlation_limits(instrument, signal_type):
                return False
            
            # Check signal thresholds for new positions
            new_long_threshold = config.SIGNAL_THRESHOLDS['NEW_LONG']
            new_short_threshold = config.SIGNAL_THRESHOLDS['NEW_SHORT']
            
            if len(self.signals_generated) <= 10:
                print(f"DEBUG: Signal threshold check - Type: {signal_type.value}, Strength: {signal_strength}")
                print(f"DEBUG: Thresholds - NEW_LONG: {new_long_threshold}, NEW_SHORT: {new_short_threshold}")
            
            if signal_type == SignalType.BUY and signal_strength >= new_long_threshold:
                if len(self.signals_generated) <= 10:
                    print(f"DEBUG: Opening LONG position - Signal strength {signal_strength} >= {new_long_threshold}")
                return self._open_new_position(instrument, True, signal_strength, market_condition, indicator_data, price, timestamp)
            elif signal_type == SignalType.SELL and signal_strength <= new_short_threshold:
                if len(self.signals_generated) <= 10:
                    print(f"DEBUG: Opening SHORT position - Signal strength {signal_strength} <= {new_short_threshold}")
                return self._open_new_position(instrument, False, signal_strength, market_condition, indicator_data, price, timestamp)
            else:
                if len(self.signals_generated) <= 10:
                    if signal_type == SignalType.BUY:
                        print(f"DEBUG: BUY signal too weak - {signal_strength} < {new_long_threshold}")
                    elif signal_type == SignalType.SELL:
                        print(f"DEBUG: SELL signal too weak - {signal_strength} > {new_short_threshold}")
        
        elif has_position:
            # Check if we should close based on signal (SAME AS LIVE)
            position = self.open_positions[instrument]
            position_units = self._get_position_units(position)
            
            if position_units > 0 and signal_type == SignalType.SELL:
                print(f"Closing long position for {instrument} due to opposite signal")
                return self._close_position(instrument, timestamp, price, "Opposite signal")
            elif position_units < 0 and signal_type == SignalType.BUY:
                print(f"Closing short position for {instrument} due to opposite signal")
                return self._close_position(instrument, timestamp, price, "Opposite signal")
        
        return False
    
    def _record_signal(self, instrument, signal_type, signal_strength):
        """Record signal for analysis - SAME AS LIVE."""
        signal_data = {
            'timestamp': datetime.now(),
            'signal_type': signal_type,
            'signal_strength': signal_strength
        }
        self.signal_history[instrument].append(signal_data)
    
    def _check_correlation_limits(self, instrument, signal_type):
        """Check if opening this position violates correlation limits - SAME AS LIVE."""
        # Get base currency from instrument
        base_currency = instrument.split('_')[0]
        quote_currency = instrument.split('_')[1]
        
        # Check if we have correlated positions
        for open_instrument in self.open_positions:
            open_base = open_instrument.split('_')[0]
            open_quote = open_instrument.split('_')[1]
            
            # Skip same instrument
            if open_instrument == instrument:
                continue
            
            # Check for currency overlap (correlation)
            if base_currency in [open_base, open_quote] or quote_currency in [open_base, open_quote]:
                # Allow only if total correlated risk is within limits
                total_correlated_positions = len([pos for pos in self.open_positions 
                                                if base_currency in pos or quote_currency in pos])
                
                max_correlated = config.MAX_CORRELATED_RISK
                if total_correlated_positions >= max_correlated:
                    print(f"Correlation limit reached for {instrument} (max: {max_correlated})")
                    return False
        
        return True
    
    def _get_position_units(self, position):
        """Get position units - SAME AS LIVE."""
        return position.get('units', 0)
    
    def _open_new_position(self, instrument, is_long, signal_strength, market_condition, indicator_data, price, timestamp):
        """Open new position - SAME AS LIVE SYSTEM LOGIC."""
        try:
            # Calculate position size using same logic as live system
            position_size = self._calculate_position_size(instrument, is_long, indicator_data)
            
            if position_size <= 0:
                print(f"Position size too small for {instrument}")
                return False
            
            # Calculate stop loss and take profit using ATR (SAME AS LIVE)
            atr = indicator_data.get('atr', 0.001)
            atr_multiplier_sl = 1.5  # Same as live system
            atr_multiplier_tp = 2.0  # Same as live system
            
            if is_long:
                stop_loss = price - (atr * atr_multiplier_sl)
                take_profit = price + (atr * atr_multiplier_tp)
                units = position_size
            else:
                stop_loss = price + (atr * atr_multiplier_sl)
                take_profit = price - (atr * atr_multiplier_tp)
                units = -position_size
            
            # Create position record (SAME AS LIVE)
            position = {
                'instrument': instrument,
                'units': units,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': timestamp,
                'signal_strength': signal_strength,
                'market_condition': market_condition,
                'unrealized_pl': 0
            }
            
            # Update tracking (SAME AS LIVE)
            self.open_positions[instrument] = position
            self.position_count[instrument] = 1
            self.position_details[instrument] = position
            self.last_trade_time[instrument] = timestamp
            self.active_pairs.add(instrument)
            
            # Log the trade
            print(f"OPENED {('LONG' if is_long else 'SHORT')} position: {instrument} @ {price:.5f}")
            print(f"  Units: {units}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
            print(f"  Signal strength: {signal_strength}, Market: {market_condition}")
            
            return True
            
        except Exception as e:
            print(f"Error opening position for {instrument}: {e}")
            return False
    
    def _calculate_position_size(self, instrument, is_long, indicator_data):
        """Calculate position size using EXACT SAME logic as live system."""
        try:
            # Get ATR for volatility adjustment
            atr = indicator_data.get('atr', 0.001)
            current_price = indicator_data.get('close', 1.0)
            atr_pct = (atr / current_price) * 100
            
            # Update volatility manager (SAME AS LIVE)
            self.volatility_manager.update_volatility(instrument, atr_pct)
            vol_adjustment = self.volatility_manager.get_volatility_adjustment(instrument, atr_pct)
            
            # Get Kelly Criterion adjustment (SAME AS LIVE)
            kelly_fraction = self.kelly_criterion.calculate_optimal_fraction()
            
            # Base risk calculation (SAME AS LIVE)
            base_risk = self.current_risk_per_trade
            adjusted_risk = base_risk * kelly_fraction * vol_adjustment
            
            # Calculate risk amount in account currency
            risk_amount = self.current_balance * adjusted_risk
            
            # Calculate stop distance
            atr_multiplier = 1.5  # Same as live system
            stop_distance = atr * atr_multiplier
            stop_distance_pips = stop_distance * 10000  # Convert to pips
            
            # Position size calculation (SAME AS LIVE)
            if stop_distance_pips > 0:
                # Calculate position size based on risk
                # Correct pip values: For majors, 1 pip on 10K units = $1, so 1 pip on 1 unit = $0.0001
                pip_value = 0.001 if 'JPY' in instrument else 0.0001  # Correct pip values
                position_size = risk_amount / (stop_distance_pips * pip_value)
                
                # Apply leverage (SAME AS LIVE)
                position_size = min(position_size, self.current_balance * self.leverage / current_price)
                
                # Maximum position size bounds (removed minimum constraint)
                max_size = self.current_balance * 0.5 / current_price  # Max 50% of balance
            
                position_size = min(position_size, max_size)
                
                print(f"Position sizing for {instrument}:")
                print(f"  Risk amount: ${risk_amount:.2f}")
                print(f"  Kelly fraction: {kelly_fraction:.3f}")
                print(f"  Vol adjustment: {vol_adjustment:.3f}")
                print(f"  Stop distance: {stop_distance_pips:.1f} pips")
                print(f"  Position size: {position_size:.0f} units")
                
                return int(position_size)
            
            return 0
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    def _open_position(self, timestamp, price, position_type, signal_strength, risk_per_trade):
        """Open a new position."""
        # Use proper position sizing calculation
        units = self._calculate_position_size(self.instrument, price, signal_strength, risk_per_trade)
        
        # Calculate stop distance for SL/TP (15 pips)
        pip_size = 0.0001 if 'JPY' not in self.instrument else 0.01
        stop_distance = 15 * pip_size
        
        # Set stop loss and take profit
        if position_type == 'LONG':
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * 1.5)  # 1.5:1 risk/reward
        else:
            stop_loss = price + stop_distance
            take_profit = price - (stop_distance * 1.5)
        
        self.open_position = {
            'type': position_type,
            'entry_time': timestamp,
            'entry_price': price,
            'units': units,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_strength': signal_strength
        }
    
    def _check_exit_conditions(self, timestamp, price):
        """Check if positions should be closed - SAME AS LIVE SYSTEM."""
        # Check all open positions for exit conditions
        positions_to_close = []
        
        for instrument, position in self.open_positions.items():
            units = position['units']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if units > 0:  # Long position
                if price <= stop_loss:
                    positions_to_close.append((instrument, "Stop loss"))
                elif price >= take_profit:
                    positions_to_close.append((instrument, "Take profit"))
            else:  # Short position
                if price >= stop_loss:
                    positions_to_close.append((instrument, "Stop loss"))
                elif price <= take_profit:
                    positions_to_close.append((instrument, "Take profit"))
        
        # Close positions that hit exit conditions
        for instrument, reason in positions_to_close:
            self._close_position(instrument, timestamp, price, reason)
    
    def _close_position(self, instrument, timestamp, price, reason):
        """Close position - SAME AS LIVE SYSTEM."""
        if instrument not in self.open_positions:
            return False
        
        try:
            position = self.open_positions[instrument]
            units = position['units']
            entry_price = position['price']
            
            # Calculate P&L (SAME AS LIVE)
            if units > 0:  # Long position
                pnl = (price - entry_price) * units
            else:  # Short position
                pnl = (entry_price - price) * abs(units)
            
            # Update balance and tracking (SAME AS LIVE)
            self.current_balance += pnl
            self.max_balance = max(self.max_balance, self.current_balance)
            
            # Update drawdown tracking (SAME AS LIVE)
            current_drawdown = (self.max_balance - self.current_balance) / self.max_balance
            self.current_drawdown = current_drawdown
            
            # Update daily drawdown (SAME AS LIVE) 
            daily_drawdown = (self.daily_high_balance - self.current_balance) / self.daily_high_balance
            self.daily_drawdown = daily_drawdown
            
            # Update performance metrics (SAME AS LIVE)
            if pnl > 0:
                self.win_count += 1
                self.total_profit += pnl
            else:
                self.loss_count += 1
                self.total_loss += abs(pnl)
            
            # Add to Kelly Criterion (SAME AS LIVE)
            risk_amount = self.current_balance * self.current_risk_per_trade
            self.kelly_criterion.add_trade_result(pnl, risk_amount)
            
            # Record trade (SAME AS LIVE)
            trade = {
                'entry_time': position['timestamp'],
                'exit_time': timestamp,
                'instrument': instrument,
                'type': 'LONG' if units > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': price,
                'units': units,
                'pnl': pnl,
                'reason': reason,
                'signal_strength': position.get('signal_strength', 0),
                'market_condition': position.get('market_condition', 'UNKNOWN')
            }
            
            self.trade_history.append(trade)
            
            # Update position tracking (SAME AS LIVE)
            del self.open_positions[instrument]
            self.position_count[instrument] = 0
            self.position_details[instrument] = None
            self.active_pairs.discard(instrument)
            
            # Check for recovery mode (SAME AS LIVE)
            if self.daily_drawdown >= config.DRAWDOWN_THRESHOLD:
                self.recovery_mode = True
                print(f"Recovery mode activated due to drawdown: {self.daily_drawdown:.2%}")
            elif self.current_balance > self.daily_high_balance * 0.98:
                self.recovery_mode = False
            
            print(f"CLOSED {trade['type']} position: {instrument} @ {price:.5f}")
            print(f"  P&L: ${pnl:.2f}, Reason: {reason}")
            print(f"  Balance: ${self.current_balance:.2f}, Drawdown: {current_drawdown:.2%}")
            
            return True
            
        except Exception as e:
            print(f"Error closing position for {instrument}: {e}")
            return False
    
    def _open_position_portfolio(self, instrument, is_long, timestamp, price, signal_strength, 
                               indicator_data, market_condition, risk_per_trade):
        """Open position for portfolio backtest - IDENTICAL to live system logic."""
        try:
            # Calculate position size using Kelly Criterion for this instrument
            kelly_criterion = self.kelly_criterion[instrument]
            kelly_fraction = kelly_criterion.calculate_optimal_fraction()
            
            # Get ATR for volatility adjustment
            atr = indicator_data.get('atr', 0.001)
            current_price = price
            atr_pct = (atr / current_price) * 100
            
            # Volatility adjustment
            vol_adjustment = self.volatility_manager.get_volatility_adjustment(instrument, atr_pct)
            self.volatility_manager.update_volatility(instrument, atr_pct)
            
            # Calculate risk amount
            base_risk = risk_per_trade
            adjusted_risk = base_risk * kelly_fraction * vol_adjustment
            risk_amount = self.current_balance * adjusted_risk
            
            # Get pip value (hardcoded for backtest reliability)
            pip_value = 0.001 if 'JPY' in instrument else 0.0001
            
            # Calculate stop distance in pips
            atr_multiplier_sl = 1.5
            stop_distance_pips = (atr * atr_multiplier_sl) / pip_value
            
            # Calculate position size
            position_size = risk_amount / (stop_distance_pips * pip_value)
            
            # Apply leverage constraints
            leverage = config.DEFAULT_LEVERAGE
            max_size = (self.current_balance * leverage) / current_price
            position_size = min(position_size, max_size)
            
            if position_size <= 0:
                return False
            
            # Calculate stop loss and take profit
            atr_multiplier_tp = 3.0  # Higher reward-to-risk ratio
            
            if is_long:
                stop_loss = price - (atr * atr_multiplier_sl)
                take_profit = price + (atr * atr_multiplier_tp)
                units = int(position_size)
            else:
                stop_loss = price + (atr * atr_multiplier_sl)
                take_profit = price - (atr * atr_multiplier_tp)
                units = -int(position_size)
            
            # Create position record
            position = {
                'instrument': instrument,
                'units': units,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': timestamp,
                'signal_strength': signal_strength,
                'market_condition': market_condition,
                'unrealized_pl': 0
            }
            
            # Update tracking
            self.open_positions[instrument] = position
            self.position_details[instrument] = position
            self.last_trade_time[instrument] = timestamp
            self.active_pairs.add(instrument)
            
            print(f"POSITION OPENED: {instrument} {'LONG' if is_long else 'SHORT'} {units} units at {price:.5f}")
            print(f"Active positions: {len(self.open_positions)}, Trade history: {len(self.trade_history)}")
            
            return True
            
        except Exception as e:
            print(f"Error opening portfolio position for {instrument}: {e}")
            return False
    def _close_position_portfolio(self, instrument, timestamp, price, signal_strength, reason):
        """Close position for portfolio backtest - IDENTICAL to live system logic."""
        try:
            if instrument not in self.open_positions:
                return False
            
            position = self.open_positions[instrument]
            units = position['units']
            entry_price = position['price']
            
            # Calculate P&L
            if units > 0:  # Long position
                pnl = (price - entry_price) * units
            else:  # Short position
                pnl = (entry_price - price) * abs(units)
            
            # Update balance and tracking
            self.current_balance += pnl
            self.max_balance = max(self.max_balance, self.current_balance)
            
            # Update drawdown tracking
            current_drawdown = (self.max_balance - self.current_balance) / self.max_balance
            
            # Update daily drawdown
            daily_drawdown = (self.daily_high_balance - self.current_balance) / self.daily_high_balance
            self.daily_drawdown = daily_drawdown
            
            # Update performance metrics
            if pnl > 0:
                self.win_count += 1
                self.total_profit += pnl
            else:
                self.loss_count += 1
                self.total_loss += abs(pnl)
            
            # Add to Kelly Criterion for this instrument
            risk_amount = self.current_balance * self.default_risk_per_trade
            self.kelly_criterion[instrument].add_trade_result(pnl, risk_amount)
            
            # Record trade
            trade = {
                'entry_time': position['timestamp'],
                'exit_time': timestamp,
                'instrument': instrument,
                'type': 'LONG' if units > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': price,
                'units': units,
                'pnl': pnl,
                'signal_strength': signal_strength,
                'market_condition': position.get('market_condition', 'NORMAL'),
                'reason': reason
            }
            
            self.trade_history.append(trade)
            
            # Clean up position tracking
            del self.open_positions[instrument]
            del self.position_details[instrument]
            self.active_pairs.discard(instrument)
            
            # Check recovery mode
            if self.daily_drawdown >= config.RECOVERY_MODE_THRESHOLD:
                self.recovery_mode = True
            elif current_drawdown <= 0.05:  # Exit recovery when drawdown < 5%
                self.recovery_mode = False
            
            return True
            
        except Exception as e:
            print(f"Error closing portfolio position for {instrument}: {e}")
            return False
    
    def generate_report(self):
        """Generate backtest report - SAME AS LIVE SYSTEM."""
        total_trades = len(self.trade_history)
        
        if total_trades == 0:
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print("No trades executed during backtest.")
            print(f"Signals Generated: {len(self.signals_generated)}")
            return
        
        # Calculate performance metrics (SAME AS LIVE)
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        avg_win = self.total_profit / self.win_count if self.win_count > 0 else 0
        avg_loss = self.total_loss / self.loss_count if self.loss_count > 0 else 0
        
        net_profit = self.total_profit - self.total_loss
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        final_balance = self.current_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        max_drawdown = (self.max_balance - min(self.current_balance, min([t['pnl'] for t in self.trade_history], default=final_balance))) / self.max_balance
        
        # Calculate additional metrics
        trade_pnls = [t['pnl'] for t in self.trade_history]
        if len(trade_pnls) > 1:
            import numpy as np
            sharpe_ratio = np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0
            sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Recovery factor
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown > 0 else float('inf')
        
        print("\n" + "="*60)
        print("ENHANCED BACKTEST RESULTS (SAME AS LIVE SYSTEM)")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Max Balance: ${self.max_balance:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Net P&L: ${net_profit:,.2f}")
        
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {self.win_count} ({win_rate:.1%})")
        print(f"Losing Trades: {self.loss_count}")
        
        print(f"\nPerformance Metrics:")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Recovery Factor: {recovery_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\nRisk Management:")
        print(f"Kelly Fraction: {self.kelly_criterion.calculate_optimal_fraction():.3f}")
        print(f"Current Risk per Trade: {self.current_risk_per_trade:.2%}")
        print(f"Recovery Mode: {'Active' if self.recovery_mode else 'Inactive'}")
        print(f"Current Drawdown: {self.current_drawdown:.2%}")
        
        print(f"\nSignal Analysis:")
        print(f"Signals Generated: {len(self.signals_generated)}")
        strong_signals = len([s for s in self.signals_generated if abs(s['signal_strength']) >= 40])
        print(f"Strong Signals (>=40): {strong_signals}")
        print(f"Signal-to-Trade Ratio: {total_trades/len(self.signals_generated)*100:.1f}%")
        
        trade_summary = []
        for trade in self.trade_history[-10:]:  # Last 10 trades
            trade_summary.append({
                'Entry Time': trade['entry_time'].strftime('%Y-%m-%d %H:%M'),
                'Type': trade['type'],
                'Entry': f"{trade['entry_price']:.5f}",
                'Exit': f"{trade['exit_price']:.5f}",
                'P/L': f"${trade['pnl']:.2f}",
                'Reason': trade['reason']
            })
        
        print("\nLast 10 Trades:")
        print(tabulate(trade_summary, headers='keys', tablefmt='pretty'))
        
        # Signal analysis
        if self.signals_generated:
            df_signals = pd.DataFrame(self.signals_generated)
            signal_counts = df_signals['signal_type'].value_counts()
            
            print("\nSignal Distribution:")
            for signal, count in signal_counts.items():
                print(f"  {signal.value}: {count} ({count/len(df_signals)*100:.1f}%)")
    
    def plot_results(self):
        """Plot backtest results."""
        if not self.equity_curve:
            print("No data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Convert to DataFrame for easier plotting
        df_equity = pd.DataFrame(self.equity_curve)
        df_trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # 1. Equity curve
        ax1 = axes[0]
        ax1.plot(df_equity['time'], df_equity['balance'], label='Balance', linewidth=2)
        ax1.plot(df_equity['time'], df_equity['equity'], label='Equity', linewidth=1, alpha=0.7)
        ax1.set_title('Account Balance and Equity')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[1]
        peak = df_equity['equity'].expanding().max()
        drawdown = (df_equity['equity'] - peak) / peak * 100
        ax2.fill_between(df_equity['time'], drawdown, 0, alpha=0.3, color='red')
        ax2.plot(df_equity['time'], drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade distribution
        ax3 = axes[2]
        if not df_trades.empty:
            wins = df_trades[df_trades['pnl'] > 0]['pnl']
            losses = abs(df_trades[df_trades['pnl'] < 0]['pnl'])
            
            bins = 20
            if len(wins) > 0:
                ax3.hist(wins, bins=bins, alpha=0.6, label='Wins', color='green')
            if len(losses) > 0:
                ax3.hist(losses, bins=bins, alpha=0.6, label='Losses', color='red')
            
            ax3.set_title('Trade P/L Distribution')
            ax3.set_xlabel('P/L ($)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'backtest_results_{self.instrument}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved as '{filename}'")
        
        plt.show()
    
    def generate_portfolio_report(self):
        """Generate comprehensive portfolio backtest report."""
        total_trades = len(self.trade_history)
        
        print("\n" + "="*60)
        print("PORTFOLIO BACKTEST RESULTS")
        print("="*60)
        
        # Portfolio overview
        print(f"\n[Portfolio Overview]")
        print(f"   Instruments: {', '.join(self.instruments)}")
        print(f"   Total instruments: {len(self.instruments)}")
        print(f"   Active pairs: {len(self.active_pairs)}")
        
        # Account summary
        initial_balance = self.initial_balance
        final_balance = self.current_balance
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        print(f"\n[Account Summary]")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Final Balance: ${final_balance:,.2f}")
        print(f"   Net Profit: ${final_balance - initial_balance:+,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Balance: ${self.max_balance:,.2f}")
        print(f"   Max Drawdown: {self.max_drawdown:.2%}")
        
        if total_trades == 0:
            print(f"\n[WARNING] No trades executed during backtest.")
            print(f"   Signals Generated: {len(self.signals_generated)}")
            return
        
        # Trading statistics
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        avg_win = self.total_profit / self.win_count if self.win_count > 0 else 0
        avg_loss = self.total_loss / self.loss_count if self.loss_count > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        print(f"\n[Trading Performance]")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {self.win_count} ({win_rate:.1%})")
        print(f"   Losing Trades: {self.loss_count} ({(1-win_rate):.1%})")
        print(f"   Average Win: ${avg_win:.2f}")
        print(f"   Average Loss: ${avg_loss:.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        
        # Per-instrument breakdown
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            instrument_stats = df_trades.groupby('instrument').agg({
                'pnl': ['count', 'sum', 'mean'],
                'type': lambda x: f"{sum(x=='LONG')}/{sum(x=='SHORT')}"
            }).round(2)
            
            print(f"\n[Per-Instrument Performance]")
            for instrument in instrument_stats.index:
                count = int(instrument_stats.loc[instrument, ('pnl', 'count')])
                total_pnl = instrument_stats.loc[instrument, ('pnl', 'sum')]
                avg_pnl = instrument_stats.loc[instrument, ('pnl', 'mean')]
                long_short = instrument_stats.loc[instrument, ('type', '<lambda>0>')]
                
                print(f"   {instrument}: {count} trades, ${total_pnl:+.2f} total, ${avg_pnl:+.2f} avg (L/S: {long_short})")
        
        # Risk analysis
        print(f"\n[Risk Analysis]")
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            max_loss = df_trades['pnl'].min()
            max_win = df_trades['pnl'].max()
            print(f"   Largest Win: ${max_win:.2f}")
            print(f"   Largest Loss: ${max_loss:.2f}")
            print(f"   Risk/Reward Ratio: {abs(avg_loss/avg_win):.2f}" if avg_win > 0 else "   Risk/Reward Ratio: N/A")
        
        # Kelly Criterion summary
        print(f"\n[Kelly Criterion Summary]")
        for instrument in self.instruments:
            kelly_fraction = self.kelly_criterion[instrument].calculate_optimal_fraction()
            trades_count = len(self.kelly_criterion[instrument].trade_results)
            print(f"   {instrument}: {kelly_fraction:.1%} (based on {trades_count} trades)")
        
        # Signal analysis
        if self.signals_generated:
            print(f"\n[Signal Analysis]")
            print(f"   Total Signals: {len(self.signals_generated)}")
            print(f"   Trades Executed: {total_trades}")
            print(f"   Execution Rate: {total_trades/len(self.signals_generated)*100:.1f}%")
        
        print("\n" + "="*60)
    
    def plot_portfolio_results(self):
        """Plot comprehensive portfolio backtest results."""
        if not self.equity_curve:
            print("No equity data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Backtest Results', fontsize=16, fontweight='bold')
        
        # Convert to DataFrame
        df_equity = pd.DataFrame(self.equity_curve)
        df_trades = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        
        # 1. Portfolio equity curve
        ax1 = axes[0, 0]
        ax1.plot(df_equity['timestamp'], df_equity['balance'], label='Balance', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial')
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        if len(df_equity) > 0:
            peak = df_equity['balance'].expanding().max()
            drawdown = (df_equity['balance'] - peak) / peak * 100
            ax2.fill_between(df_equity['timestamp'], drawdown, 0, alpha=0.3, color='red')
            ax2.plot(df_equity['timestamp'], drawdown, color='red', linewidth=1)
        ax2.set_title('Portfolio Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Per-instrument performance
        ax3 = axes[1, 0]
        if not df_trades.empty:
            instrument_pnl = df_trades.groupby('instrument')['pnl'].sum().sort_values(ascending=True)
            colors = ['red' if x < 0 else 'green' for x in instrument_pnl.values]
            ax3.barh(range(len(instrument_pnl)), instrument_pnl.values, color=colors, alpha=0.7)
            ax3.set_yticks(range(len(instrument_pnl)))
            ax3.set_yticklabels(instrument_pnl.index)
            ax3.axvline(x=0, color='black', linewidth=0.8)
            ax3.set_title('P&L by Instrument')
            ax3.set_xlabel('P&L ($)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade distribution
        ax4 = axes[1, 1]
        if not df_trades.empty:
            wins = df_trades[df_trades['pnl'] > 0]['pnl']
            losses = df_trades[df_trades['pnl'] < 0]['pnl']
            
            if len(wins) > 0:
                ax4.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
            if len(losses) > 0:
                ax4.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
            
            ax4.axvline(x=0, color='black', linewidth=0.8)
            ax4.set_title('Trade P&L Distribution')
            ax4.set_xlabel('P&L ($)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'portfolio_backtest_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPortfolio results plot saved as '{filename}'")
        
        plt.show()
    
    def generate_portfolio_report(self):
        """Generate comprehensive portfolio backtest report."""
        print("\n" + "="*60)
        print("PORTFOLIO BACKTEST SUMMARY")
        print("="*60)
        
        # Basic portfolio metrics
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        net_profit = self.current_balance - self.initial_balance
        
        print(f"\n[Portfolio Overview]")
        print(f"   Instruments: {', '.join(self.instruments)}")
        print(f"   Total instruments: {len(self.instruments)}")
        print(f"   Active positions: {len(self.open_positions)}")
        
        print(f"\n[Account Summary]")
        print(f"   Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   Final Balance: ${self.current_balance:,.2f}")
        print(f"   Net Profit: ${net_profit:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        
        # Trade statistics
        if self.trade_history:
            completed_trades = [t for t in self.trade_history if 'pnl' in t]
            total_trades = len(completed_trades)
            
            if total_trades > 0:
                winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in completed_trades if t.get('pnl', 0) < 0]
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
                
                avg_win = sum(t['pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
                avg_loss = sum(t['pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
                
                profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if loss_count > 0 and sum(t['pnl'] for t in losing_trades) != 0 else 0
                
                print(f"\n[Trading Performance]")
                print(f"   Total Trades: {total_trades}")
                print(f"   Winning Trades: {win_count} ({win_rate:.1f}%)")
                print(f"   Losing Trades: {loss_count} ({100-win_rate:.1f}%)")
                print(f"   Average Win: ${avg_win:.2f}")
                print(f"   Average Loss: ${avg_loss:.2f}")
                print(f"   Profit Factor: {profit_factor:.2f}")
                
                # Per-instrument breakdown
                print(f"\n[Per-Instrument Performance]")
                instrument_pnl = {}
                for trade in completed_trades:
                    instrument = trade.get('instrument', 'Unknown')
                    pnl = trade.get('pnl', 0)
                    if instrument not in instrument_pnl:
                        instrument_pnl[instrument] = {'pnl': 0, 'trades': 0}
                    instrument_pnl[instrument]['pnl'] += pnl
                    instrument_pnl[instrument]['trades'] += 1
                
                for instrument, stats in instrument_pnl.items():
                    print(f"   {instrument}: ${stats['pnl']:.2f} ({stats['trades']} trades)")
            else:
                print(f"\n[Trading Performance]")
                print(f"   No completed trades found")
        else:
            print(f"\n[Trading Performance]")
            print(f"   No trade history available")
        
        print("\n" + "="*60)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test trading strategy - Multi-Instrument Portfolio')
    parser.add_argument('--instruments', type=str, nargs='*', 
                       default=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'AUD_CAD', 'AUD_NZD', 'NZD_CAD'],
                       help='Instruments to test (default: all 8 pairs like live trading)')
    parser.add_argument('--days', type=int, default=5,
                       help='Days of historical data to test')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial balance for testing')
    parser.add_argument('--risk', type=float, default=3.0,
                       help='Risk per trade percentage (default 3%)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    print(f"\n*** MULTI-INSTRUMENT PORTFOLIO BACKTEST ***")
    print(f"Instruments: {', '.join(args.instruments)}")
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Risk per Trade: {args.risk}%")
    print(f"Historical Data: {args.days} days")
    
    # Create tester with multiple instruments
    tester = StrategyTester(args.instruments, args.balance)
    
    # Fetch historical data for all instruments
    all_timeframe_data = await tester.fetch_historical_data_multi_instrument(args.days)
    
    if not all_timeframe_data:
        print("Failed to fetch historical data for instruments")
        return
    
    # Run portfolio backtest
    await tester.run_portfolio_backtest(all_timeframe_data, args.risk / 100)
    
    # Generate comprehensive report
    tester.generate_portfolio_report()
    
    # Plot results if requested
    if args.plot:
        tester.plot_portfolio_results()

if __name__ == "__main__":
    asyncio.run(main())