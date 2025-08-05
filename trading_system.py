"""
Multi-Timeframe Automated Trading System
Enhanced with research-based optimizations for profitable trading
"""
import asyncio
import logging
import time
import sys
import os
import signal
import argparse
from datetime import datetime, timedelta
import pandas as pd

import config
from market_data_adapter import MarketDataAdapter
from cep_engine import CEPEngine, SignalType, MarketCondition
from order_routing import OrderRoutingSystem
from trade_logger import TradeLogger

# Configure logging
def setup_logging():
    """Configure the logging system."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_system_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from OANDA library if configured
    if config.LOGGING_CONFIG.get('REDUCE_OANDA_LOGGING', True):
        logging.getLogger('oandapyV20.oandapyV20').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced logging initialized. Log file: {log_file}")
    
    return logger

class TradingSystem:
    """
    Enhanced Multi-Timeframe Automated Trading System for profitable forex trading.
    """
    
    def __init__(self, run_duration=None):
        """Initialize the enhanced Trading System."""
        self.logger = logging.getLogger(__name__)
        self.market_data_adapter = MarketDataAdapter()
        
        # Create CEP engines for each instrument
        self.cep_engines = {}
        self.order_routing = OrderRoutingSystem()
        
        # Get enabled instruments
        self.instruments = [inst for inst, details in config.INSTRUMENTS.items() 
                           if details.get('enabled', True)]
        
        # Initialize CEP engine for each instrument
        for instrument in self.instruments:
            self.cep_engines[instrument] = CEPEngine()
        
        # System state
        self.is_running = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.start_time = None
        self.last_update_time = None
        self.update_count = 0
        self.signals_generated = 0
        self.trades_executed = 0
        
        # Session management
        self.current_session = None
        self.session_performance = {}
        
        # Run duration
        self.run_duration = run_duration
        self.end_time = None
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Optimized update scheduling
        self.next_update_times = {
            'M1': datetime.now(),
            'M5': datetime.now(),
            'M15': datetime.now()
        }
        
        # Enhanced logging tracking
        self.last_summary_time = datetime.now()
        self.last_performance_time = datetime.now()
        self.active_signals = {}  # Track current signals per instrument
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received shutdown signal ({sig}). Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize(self):
        """Initialize all system components."""
        self.logger.info("Initializing enhanced trading system...")
        
        # Initialize order routing
        if not await self.order_routing.initialize():
            self.logger.error("Failed to initialize Order Routing System")
            return False
        
        # Log system configuration
        self.logger.info(f"System Configuration:")
        self.logger.info(f"  Instruments: {', '.join(self.instruments)}")
        self.logger.info(f"  Risk per trade: {config.DEFAULT_RISK_PER_TRADE:.2%}")
        self.logger.info(f"  Max drawdown: {config.CRITICAL_DRAWDOWN:.2%}")
        self.logger.info(f"  Update interval: {config.UPDATE_INTERVAL}s")
        self.logger.info(f"  Kelly Criterion: Enabled (25% fraction)")
        self.logger.info(f"  Market regime detection: Machine Learning")
        
        return True
    
    def _get_current_session(self):
        """Determine current trading session."""
        current_hour = datetime.now().hour
        
        for session_name, session_info in config.TRADING_SESSIONS.items():
            if session_info['start'] <= current_hour < session_info['end']:
                return session_name, session_info
        
        return None, None
    
    def _is_optimal_trading_time(self):
        """Check if current time is optimal for trading."""
        # Check if weekend
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday or Sunday
            if not config.WEEKEND_TRADING['ENABLED']:
                return False
        
        # Check trading sessions
        session_name, session_info = self._get_current_session()
        if session_name is None:
            self.logger.debug("Outside active trading sessions")
            return True  # Allow trading but with reduced confidence
        
        # Update current session
        if self.current_session != session_name:
            self.current_session = session_name
            self.logger.info(f"Entered {session_name} trading session")
        
        return True
    
    def _should_update_timeframe(self, timeframe):
        """Determine if a timeframe needs updating."""
        now = datetime.now()
        
        if timeframe not in self.next_update_times:
            return True
        
        return now >= self.next_update_times[timeframe]
    
    def _schedule_next_update(self, timeframe):
        """Schedule next update for a timeframe."""
        now = datetime.now()
        
        if timeframe == 'M1':
            # Update every minute
            next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            self.next_update_times[timeframe] = next_minute
        elif timeframe == 'M5':
            # Update every 5 minutes
            minutes = (now.minute // 5 + 1) * 5
            if minutes >= 60:
                next_update = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_update = now.replace(minute=minutes, second=0, microsecond=0)
            self.next_update_times[timeframe] = next_update
        elif timeframe == 'M15':
            # Update every 15 minutes
            minutes = (now.minute // 15 + 1) * 15
            if minutes >= 60:
                next_update = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_update = now.replace(minute=minutes, second=0, microsecond=0)
            self.next_update_times[timeframe] = next_update
    
    def _should_log_summary(self):
        """Check if we should log a summary update."""
        now = datetime.now()
        interval = config.LOGGING_CONFIG.get('SUMMARY_INTERVAL_MINUTES', 5)
        return (now - self.last_summary_time).total_seconds() >= interval * 60
    
    def _should_log_performance(self):
        """Check if we should log performance update."""
        now = datetime.now()
        interval = config.LOGGING_CONFIG.get('PERFORMANCE_INTERVAL_MINUTES', 15)
        return (now - self.last_performance_time).total_seconds() >= interval * 60
    
    def _should_log_signal(self, signal_strength):
        """Check if we should log this signal based on strength."""
        threshold = config.LOGGING_CONFIG.get('MIN_SIGNAL_STRENGTH_LOG', 30)
        return signal_strength >= threshold
    
    async def start(self):
        """Start the enhanced trading system."""
        # Initialize components
        if not await self.initialize():
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info(f"Trading system started at {self.start_time}")
        
        if self.run_duration:
            self.end_time = self.start_time + timedelta(minutes=self.run_duration)
            self.logger.info(f"System will run until {self.end_time}")
        
        # Main trading loop
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Check run duration
                if self.run_duration and datetime.now() >= self.end_time:
                    self.logger.info("Run duration expired")
                    break
                
                self.last_update_time = datetime.now()
                
                # Check if optimal trading time
                if not self._is_optimal_trading_time():
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue
                
                # Update account info
                await self.order_routing.update_account_info()
                
                # Check if we should stop due to drawdown
                performance = self.order_routing.get_performance_summary()
                if performance['daily_drawdown'] >= config.CRITICAL_DRAWDOWN * 100:
                    self.logger.warning(f"Critical daily drawdown: {performance['daily_drawdown']:.2f}%")
                    self.logger.warning("Trading halted for the day")
                    # Wait until next day
                    tomorrow = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
                    wait_seconds = (tomorrow - datetime.now()).total_seconds()
                    await asyncio.sleep(wait_seconds)
                    continue
                
                # Determine which timeframes need updating
                timeframes_to_update = []
                for tf in config.TIMEFRAMES.keys():
                    if self._should_update_timeframe(tf):
                        timeframes_to_update.append(tf)
                        self._schedule_next_update(tf)
                
                if not timeframes_to_update and self.update_count > 0:
                    # No updates needed right now
                    await asyncio.sleep(1)
                    continue
                
                # Fetch market data - only log if significant updates needed
                if self._should_log_summary():
                    self.logger.info(f"Processing {len(self.instruments)} instruments for timeframes: {timeframes_to_update}")
                
                # Process each instrument
                for instrument in self.instruments:
                    try:
                        # Get cached data first
                        timeframe_data = {}
                        
                        # Fetch only needed timeframes
                        fetch_tasks = []
                        for tf_key in timeframes_to_update:
                            tf_config = config.TIMEFRAMES[tf_key]
                            task = self.market_data_adapter.fetch_candles(
                                instrument, 
                                tf_config['granularity'], 
                                tf_config['count']
                            )
                            fetch_tasks.append((tf_key, task))
                        
                        # Get updated data
                        for tf_key, task in fetch_tasks:
                            df = await task
                            if df is not None:
                                timeframe_data[tf_key] = df
                        
                        # Add cached data for non-updated timeframes
                        if hasattr(self.market_data_adapter, 'candle_data'):
                            if instrument in self.market_data_adapter.candle_data:
                                for tf_key in config.TIMEFRAMES:
                                    if tf_key not in timeframe_data:
                                        cached_df = self.market_data_adapter.candle_data[instrument].get(tf_key)
                                        if cached_df is not None:
                                            timeframe_data[tf_key] = cached_df
                        
                        # Skip if insufficient data
                        if not timeframe_data:
                            self.logger.warning(f"No data available for {instrument}")
                            continue
                        
                        # Process signals
                        signal_type, signal_strength, market_condition, indicator_data = (
                            self.cep_engines[instrument].process_timeframe_data(timeframe_data)
                        )
                        
                        self.signals_generated += 1
                        
                        # Track signal changes and log only significant ones
                        current_signal = f"{signal_type.value}_{signal_strength:.0f}"
                        previous_signal = self.active_signals.get(instrument)
                        
                        # Log if signal changed or strength is significant
                        if (current_signal != previous_signal or 
                            self._should_log_signal(signal_strength)):
                            self.logger.info(f"{instrument}: {signal_type.value} "
                                           f"(strength: {signal_strength:.1f}, "
                                           f"regime: {market_condition.value})")
                            self.active_signals[instrument] = current_signal
                        
                        # Execute trade if signal is strong enough
                        if signal_type in [SignalType.BUY, SignalType.SELL, SignalType.CLOSE]:
                            traded = await self.order_routing.process_signal(
                                instrument, signal_type, signal_strength, 
                                market_condition, indicator_data
                            )
                            
                            if traded:
                                self.trades_executed += 1
                                self.logger.info(f"Trade executed for {instrument}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {instrument}: {str(e)}")
                
                # Update tracking
                self.update_count += 1
                consecutive_errors = 0
                
                # Update summary log timestamp if we logged
                if self._should_log_summary():
                    self.last_summary_time = datetime.now()
                
                # Log performance based on time intervals
                if self._should_log_performance():
                    self._log_performance()
                    self.last_performance_time = datetime.now()
                
                # Brief pause before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical(f"Too many consecutive errors ({consecutive_errors})")
                    break
                
                wait_time = min(30, 5 * (2 ** (consecutive_errors - 1)))
                await asyncio.sleep(wait_time)
    
    async def stop(self):
        """Stop the trading system gracefully."""
        self.logger.info("Stopping trading system...")
        self.is_running = False
        
        # Final performance log
        self._log_performance(final=True)
        
        # Generate final report
        summary = self.order_routing.get_trade_summary()
        
        if summary:
            self.logger.info("=== FINAL TRADING SUMMARY ===")
            self.logger.info(f"Session duration: {datetime.now() - self.start_time}")
            self.logger.info(f"Initial balance: ${summary.get('initial_balance', 0):.2f}")
            self.logger.info(f"Final balance: ${summary.get('final_balance', 0):.2f}")
            self.logger.info(f"Total P/L: ${summary.get('profit_loss', 0):.2f} "
                           f"({summary.get('profit_loss_pct', 0):.2f}%)")
            self.logger.info(f"Max drawdown: {summary.get('max_drawdown', 0):.2f}%")
            self.logger.info(f"Total trades: {summary.get('total_trades', 0)}")
            self.logger.info(f"Win rate: {summary.get('win_rate', 0):.2f}%")
            self.logger.info(f"Profit factor: {summary.get('profit_factor', 0):.2f}")
            
            # Export logs
            csv_path = self.order_routing.export_trade_log()
            self.logger.info(f"Trade log exported to: {csv_path}")
        
        self.logger.info("Trading system stopped")
    
    def _log_performance(self, final=False):
        """Log system performance metrics."""
        if self.start_time is None:
            return
        
        runtime = datetime.now() - self.start_time
        performance = self.order_routing.get_performance_summary()
        
        if final:
            self.logger.info("=== FINAL PERFORMANCE ===")
        else:
            self.logger.info("=== PERFORMANCE UPDATE ===")
        
        self.logger.info(f"Runtime: {runtime}")
        self.logger.info(f"Updates: {self.update_count}")
        self.logger.info(f"Signals generated: {self.signals_generated}")
        self.logger.info(f"Trades executed: {self.trades_executed}")
        self.logger.info(f"Current balance: ${performance['current_balance']:.2f}")
        self.logger.info(f"P/L: ${performance['profit_loss']:.2f} ({performance['profit_loss_pct']:.2f}%)")
        self.logger.info(f"Win rate: {performance['win_rate']:.2f}%")
        self.logger.info(f"Profit factor: {performance['profit_factor']:.2f}")
        self.logger.info(f"Daily drawdown: {performance['daily_drawdown']:.2f}%")
        self.logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
        self.logger.info(f"Kelly optimal: {performance['kelly_optimal']:.2f}%")
        self.logger.info(f"Current risk: {performance['current_risk_pct']:.2f}%")
        
        if performance['recovery_mode']:
            self.logger.warning("System in recovery mode")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Timeframe Trading System')
    
    parser.add_argument('--duration', type=int, default=None,
                        help='Run duration in minutes')
    parser.add_argument('--practice', action='store_true', default=True,
                        help='Run in practice mode (default)')
    parser.add_argument('--live', action='store_true',
                        help='Run in live mode (USE WITH CAUTION)')
    parser.add_argument('--instruments', type=str,
                        help='Comma-separated list of instruments')
    parser.add_argument('--risk', type=float,
                        help='Risk per trade percentage (e.g., 1.0 for 1%)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Enhanced Multi-Timeframe Trading System")
    
    # Update environment
    if args.live:
        os.environ['OANDA_ENVIRONMENT'] = 'live'
        logger.warning("!!! LIVE TRADING MODE - REAL MONEY !!!")
        
        # Confirm live trading
        confirmation = input("Type 'CONFIRM' to proceed with LIVE trading: ")
        if confirmation != 'CONFIRM':
            logger.info("Live trading not confirmed. Exiting.")
            return
    
    # Update configuration
    if args.instruments:
        # Enable only specified instruments
        for instrument in config.INSTRUMENTS:
            config.INSTRUMENTS[instrument]['enabled'] = False
        
        for instrument in args.instruments.split(','):
            if instrument in config.INSTRUMENTS:
                config.INSTRUMENTS[instrument]['enabled'] = True
    
    if args.risk:
        config.DEFAULT_RISK_PER_TRADE = args.risk / 100
    
    # Create and start system
    system = TradingSystem(run_duration=args.duration)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    finally:
        await system.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())