"""
Order Routing System for OANDA Trading System
Enhanced with research-based risk management and position sizing
"""
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import deque

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.accounts import AccountSummary, AccountDetails
from oandapyV20.endpoints.orders import OrderCreate, OrderDetails, OrderCancel
from oandapyV20.endpoints.positions import OpenPositions, PositionDetails, PositionClose
from oandapyV20.endpoints.trades import OpenTrades, TradeDetails, TradeClose, TradeCRCDO
from oandapyV20.endpoints.pricing import PricingInfo

import config
from cep_engine import SignalType, MarketCondition
from trade_logger import TradeLogger

logger = logging.getLogger(__name__)

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

class OrderRoutingSystem:
    """
    Enhanced Order Routing System with advanced risk management.
    """
    
    def __init__(self):
        """Initialize the enhanced Order Routing System."""
        self.api = API(access_token=config.OANDA_API_KEY, environment=config.OANDA_ENVIRONMENT)
        self.account_id = config.OANDA_ACCOUNT_ID
        
        # Risk management parameters
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.current_risk_per_trade = config.DEFAULT_RISK_PER_TRADE
        self.leverage = config.DEFAULT_LEVERAGE
        
        # Enhanced components
        self.kelly_criterion = KellyCriterion()
        self.volatility_manager = VolatilityManager()
        self.trade_logger = TradeLogger()
        
        # Position tracking
        self.open_positions = {}
        self.position_details = {}
        self.signal_history = {}
        self.last_trade_time = {}
        self.position_count = {}
        
        # Enhanced position limits
        self.max_positions_per_instrument = config.POSITION_MANAGEMENT['MAX_POSITIONS_PER_INSTRUMENT']
        self.max_total_positions = config.POSITION_MANAGEMENT['MAX_TOTAL_POSITIONS']
        self.min_time_between_trades = config.POSITION_MANAGEMENT['MIN_TIME_BETWEEN_TRADES']
        
        # Performance tracking
        self.initial_balance = None
        self.current_balance = None
        self.max_balance = None
        self.daily_high_balance = None
        self.daily_starting_balance = None
        self.current_drawdown = 0
        self.daily_drawdown = 0
        
        # Trade tracking
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Correlation tracking
        self.active_pairs = set()
        self.correlation_matrix = pd.DataFrame()
        
        # Recovery mode
        self.recovery_mode = False
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.trade_entry_times = {}
        
        # Transaction tracking for closed position detection
        self.last_transaction_id = None
    
    async def initialize(self):
        """Initialize account information and risk parameters."""
        try:
            # Get account summary
            account_summary = await self._execute_request(AccountSummary(accountID=self.account_id))
            account = account_summary.get('account', {})
            
            self.initial_balance = float(account.get('balance', 0))
            self.current_balance = self.initial_balance
            self.max_balance = self.initial_balance
            self.daily_starting_balance = self.initial_balance
            self.daily_high_balance = self.initial_balance
            
            # Initialize trade logger
            self.trade_logger.set_initial_balance(self.initial_balance)
            
            # Get current positions
            await self._update_open_positions()
            
            # Initialize tracking for each instrument
            for instrument in config.INSTRUMENTS:
                self.signal_history[instrument] = []
                self.last_trade_time[instrument] = datetime.min
                self.position_count[instrument] = 0
                self.position_details[instrument] = None
            
            logger.info(f"Account initialized. Balance: ${self.current_balance:.2f}")
            logger.info(f"Using effective leverage: {self.leverage}:1")
            
            return True
            
        except V20Error as e:
            logger.error(f"OANDA API error initializing account: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error initializing account: {str(e)}")
            return False
    
    async def update_account_info(self):
        """Update account information and check risk limits."""
        try:
            # Get account summary
            account_summary = await self._execute_request(AccountSummary(accountID=self.account_id))
            account = account_summary.get('account', {})
            
            # Update balance
            previous_balance = self.current_balance
            self.current_balance = float(account.get('balance', 0))
            
            # Update max balance
            if self.current_balance > self.max_balance:
                self.max_balance = self.current_balance
            
            # Update daily high
            if self.current_balance > self.daily_high_balance:
                self.daily_high_balance = self.current_balance
            
            # Calculate drawdowns
            if self.max_balance > 0:
                self.current_drawdown = (self.max_balance - self.current_balance) / self.max_balance
            
            if self.daily_starting_balance > 0:
                self.daily_drawdown = (self.daily_starting_balance - self.current_balance) / self.daily_starting_balance
            
            # Check for new day
            if datetime.now().date() > self.session_start_time.date():
                self._reset_daily_metrics()
            
            # Check drawdown limits
            if self.daily_drawdown >= config.CRITICAL_DRAWDOWN:
                logger.warning(f"Critical daily drawdown reached: {self.daily_drawdown:.2%}")
                self.recovery_mode = True
            elif self.daily_drawdown >= config.DRAWDOWN_THRESHOLD:
                logger.warning(f"Daily drawdown threshold reached: {self.daily_drawdown:.2%}")
                self._reduce_risk_parameters()
            elif self.current_drawdown >= config.RECOVERY_MODE_THRESHOLD:
                logger.info(f"Account in recovery mode. Drawdown: {self.current_drawdown:.2%}")
                self.recovery_mode = True
            else:
                self.recovery_mode = False
            
            # Update position sizing based on Kelly Criterion
            self._update_position_sizing()
            
            # Update open positions
            await self._update_open_positions()
            
            # Log performance if balance changed
            if self.current_balance != previous_balance:
                self.trade_logger.log_performance(active_positions=len(self.open_positions))
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating account info: {str(e)}")
            return False
    
    def _reset_daily_metrics(self):
        """Reset daily tracking metrics."""
        self.daily_starting_balance = self.current_balance
        self.daily_high_balance = self.current_balance
        self.daily_drawdown = 0
        self.session_start_time = datetime.now()
        logger.info(f"Daily metrics reset. Starting balance: ${self.daily_starting_balance:.2f}")
    
    def _reduce_risk_parameters(self):
        """Reduce risk when drawdown thresholds are hit."""
        # Reduce position size
        self.current_risk_per_trade = self.current_risk_per_trade * 0.5
        
        # Reduce leverage
        self.leverage = max(1, self.leverage // 2)
        
        logger.info(f"Risk parameters reduced. Risk per trade: {self.current_risk_per_trade:.2%}, Leverage: {self.leverage}:1")
    
    def _update_position_sizing(self):
        """Update position sizing based on Kelly Criterion and performance."""
        # Get Kelly optimal fraction
        kelly_fraction = self.kelly_criterion.calculate_optimal_fraction()
        
        # Adjust for recovery mode
        if self.recovery_mode:
            kelly_fraction *= 0.5
        
        # Adjust for daily performance
        if hasattr(self, 'daily_starting_balance') and self.daily_starting_balance > 0:
            daily_return = (self.current_balance - self.daily_starting_balance) / self.daily_starting_balance
            
            if daily_return >= config.PERFORMANCE_TARGETS['DAILY_PROFIT_TARGET']:
                # Hit daily target - reduce size
                kelly_fraction *= 0.7
                logger.info("Daily profit target reached. Reducing position sizes.")
        
        # Update risk per trade
        self.current_risk_per_trade = min(
            config.MAX_RISK_PER_TRADE,
            kelly_fraction
        )
        
        logger.debug(f"Position sizing updated. Risk per trade: {self.current_risk_per_trade:.2%}")
    
    async def process_signal(self, instrument, signal_type, signal_strength, market_condition, indicator_data):
        """
        Process trading signal with enhanced risk management.
        """
        # Check if we should stop trading
        if self.recovery_mode and self.daily_drawdown >= config.CRITICAL_DRAWDOWN:
            logger.warning(f"Trading halted due to critical drawdown: {self.daily_drawdown:.2%}")
            return False
        
        # Check global position limits
        if len(self.open_positions) >= self.max_total_positions:
            logger.info(f"Maximum total positions ({self.max_total_positions}) reached")
            return False
        
        # Update account info first
        await self.update_account_info()
        
        # Check correlation limits
        if not self._check_correlation_limits(instrument, signal_type):
            logger.info(f"Correlation limits prevent trading {instrument}")
            return False
        
        # Record signal
        self._record_signal(instrument, signal_type, signal_strength)
        
        # Get current position
        position = self.open_positions.get(instrument)
        has_position = position is not None
        
        # Get current price
        current_price = await self._get_current_price(instrument)
        if current_price is None:
            logger.error(f"Could not get current price for {instrument}")
            return False
        
        # Process based on signal type
        if signal_type == SignalType.BUY and not has_position:
            # Check signal quality
            if signal_strength >= config.SIGNAL_THRESHOLDS['NEW_LONG']:
                return await self._open_new_position(
                    instrument, True, signal_strength, market_condition, 
                    indicator_data, current_price
                )
        
        elif signal_type == SignalType.SELL and not has_position:
            if signal_strength >= abs(config.SIGNAL_THRESHOLDS['NEW_SHORT']):
                return await self._open_new_position(
                    instrument, False, signal_strength, market_condition, 
                    indicator_data, current_price
                )
        
        elif signal_type == SignalType.CLOSE and has_position:
            return await self.close_position(instrument)
        
        elif has_position:
            # Check if we should close based on signal
            position_units = self._get_position_units(position)
            
            if position_units > 0 and signal_type == SignalType.SELL:
                logger.info(f"Closing long position for {instrument} due to opposite signal")
                return await self.close_position(instrument)
            elif position_units < 0 and signal_type == SignalType.BUY:
                logger.info(f"Closing short position for {instrument} due to opposite signal")
                return await self.close_position(instrument)
        
        return False
    
    def _get_position_units(self, position):
        """Get net position units."""
        long_units = int(str(position.get('long', {}).get('units', 0)))
        short_units = int(str(position.get('short', {}).get('units', 0)))
        return long_units + short_units
    
    def _check_correlation_limits(self, instrument, signal_type):
        """Check if correlation limits allow trading."""
        # Count correlated positions
        base_currency = instrument.split('_')[0]
        quote_currency = instrument.split('_')[1]
        
        correlated_risk = 0
        for active_instrument in self.active_pairs:
            if base_currency in active_instrument or quote_currency in active_instrument:
                correlated_risk += self.current_risk_per_trade
        
        # Check if adding this position would exceed limits
        if correlated_risk + self.current_risk_per_trade > config.MAX_CORRELATED_RISK:
            return False
        
        return True
    
    def _record_signal(self, instrument, signal_type, signal_strength):
        """Record signal for analysis."""
        if len(self.signal_history[instrument]) >= 10:
            self.signal_history[instrument].pop(0)
        
        self.signal_history[instrument].append({
            'timestamp': datetime.now(),
            'type': signal_type,
            'strength': signal_strength
        })
    
    async def _calculate_position_size(self, instrument, is_long, signal_strength, market_condition, indicator_data, current_price):
        """
        Calculate position size using volatility-adjusted formula.
        """
        if current_price is None or current_price <= 0:
            logger.error(f"Invalid price for {instrument} position sizing")
            return 0
        
        # Get ATR for volatility-based sizing
        atr = self._get_atr_from_indicator_data(indicator_data)
        if atr is None or atr <= 0:
            atr = current_price * 0.001
        
        # Calculate ATR percentage
        atr_pct = (atr / current_price) * 100
        
        # Update volatility manager
        self.volatility_manager.update_volatility(instrument, atr_pct)
        
        # Get pip value
        pip_value = config.INSTRUMENTS.get(instrument, {}).get('pip_value', 0.0001)
        
        # Determine stop loss distance using ATR multiplier
        atr_multiplier = config.INDICATOR_SETTINGS['ATR_MULTIPLIER_STOP']
        
        # Adjust multiplier based on market condition
        if 'high_volatility' in market_condition.value:
            atr_multiplier *= 1.2
        elif 'low_volatility' in market_condition.value:
            atr_multiplier *= 0.8
        
        # Calculate stop distance
        stop_distance = atr * atr_multiplier
        stop_distance_pips = stop_distance / pip_value
        
        # Apply min/max stop limits
        min_stop_pips = 5
        max_stop_pips = 20  # Tighter for scalping
        stop_distance_pips = max(min_stop_pips, min(max_stop_pips, stop_distance_pips))
        
        # Calculate risk amount
        risk_amount = self.current_balance * self.current_risk_per_trade
        
        # Apply volatility adjustment
        volatility_adjustment = self.volatility_manager.get_volatility_adjustment(instrument, atr_pct)
        risk_amount *= volatility_adjustment
        
        # Apply signal strength adjustment
        signal_factor = signal_strength / 100
        risk_amount *= (0.5 + 0.5 * signal_factor)  # 50% to 100% based on signal
        
        # Apply market condition adjustment
        if market_condition in [MarketCondition.RANGING_HIGH_VOL, MarketCondition.RANGING_LOW_VOL]:
            risk_amount *= 0.8  # Reduce risk in ranging markets
        
        # Apply instrument-specific volatility factor
        instrument_config = config.INSTRUMENTS.get(instrument, {})
        volatility_factor = instrument_config.get('volatility_factor', 1.0)
        risk_amount *= volatility_factor
        
        # Calculate position size using formula
        # Position Size = Risk Amount / (Stop Distance in Pips × Pip Value)
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply leverage constraint
        max_position_by_leverage = (self.current_balance * self.leverage) / current_price
        position_size = min(position_size, max_position_by_leverage)
        
        # Round position size (no artificial constraints)
        position_size = int(position_size)
        
        # Apply maximum position size bounds (removed minimum constraint)
        max_size = self.current_balance * 0.5 / current_price  # Max 50% of balance
        position_size = min(position_size, max_size)
        
        # Apply direction
        units = position_size if is_long else -position_size
        
        logger.info(f"Position size calculation for {instrument}:")
        logger.info(f"  Risk amount: ${risk_amount:.2f}")
        logger.info(f"  Stop distance: {stop_distance_pips:.1f} pips")
        logger.info(f"  Volatility adjustment: {volatility_adjustment:.2f}")
        logger.info(f"  Units: {units}")
        
        return units
    
    def _get_atr_from_indicator_data(self, indicator_data):
        """Extract ATR from indicator data."""
        for tf_key in ["M5", "M1", "M15"]:
            if tf_key in indicator_data and indicator_data[tf_key] is not None:
                df = indicator_data[tf_key]
                if not df.empty and 'atr' in df.columns:
                    return df['atr'].iloc[-1]
        return None
    
    async def _open_new_position(self, instrument, is_long, signal_strength, market_condition, indicator_data, current_price):
        """
        Open a new position with ATR-based stops and targets.
        """
        try:
            # Check time since last trade
            time_since_last = (datetime.now() - self.last_trade_time.get(instrument, datetime.min)).total_seconds()
            if time_since_last < self.min_time_between_trades:
                logger.info(f"Too soon to trade {instrument}. Wait {self.min_time_between_trades - time_since_last:.0f}s")
                return False
            
            # Calculate position size
            units = await self._calculate_position_size(
                instrument, is_long, signal_strength, market_condition, 
                indicator_data, current_price
            )
            
            if units == 0:
                logger.error(f"Position size calculation resulted in zero units for {instrument}")
                return False
            
            # Calculate stop loss and take profit using ATR
            atr = self._get_atr_from_indicator_data(indicator_data)
            if atr is None:
                atr = current_price * 0.001
            
            # Get parameters
            atr_multiplier_stop = config.INDICATOR_SETTINGS['ATR_MULTIPLIER_STOP']
            atr_multiplier_target = config.INDICATOR_SETTINGS['ATR_MULTIPLIER_TARGET']
            
            # Adjust for market conditions
            if 'high_volatility' in market_condition.value:
                atr_multiplier_stop *= 1.2
                atr_multiplier_target *= 1.3
            
            # Calculate levels
            stop_distance = atr * atr_multiplier_stop
            target_distance = atr * atr_multiplier_target
            
            if is_long:
                stop_loss_price = current_price - stop_distance
                take_profit_price = current_price + target_distance
            else:
                stop_loss_price = current_price + stop_distance
                take_profit_price = current_price - target_distance
            
            # Validate take profit and stop loss logic
            if is_long:
                if take_profit_price <= current_price:
                    logger.error(f"Invalid take profit for BUY {instrument}: TP {take_profit_price:.5f} <= Current {current_price:.5f}")
                    logger.error(f"ATR: {atr:.5f}, Target distance: {target_distance:.5f}")
                    return False
                if stop_loss_price >= current_price:
                    logger.error(f"Invalid stop loss for BUY {instrument}: SL {stop_loss_price:.5f} >= Current {current_price:.5f}")
                    return False
            else:  # Short position
                if take_profit_price >= current_price:
                    logger.error(f"Invalid take profit for SELL {instrument}: TP {take_profit_price:.5f} >= Current {current_price:.5f}")
                    logger.error(f"ATR: {atr:.5f}, Target distance: {target_distance:.5f}")
                    return False
                if stop_loss_price <= current_price:
                    logger.error(f"Invalid stop loss for SELL {instrument}: SL {stop_loss_price:.5f} <= Current {current_price:.5f}")
                    return False
            
            # Log order levels for verification
            logger.info(f"Order levels for {instrument} {'BUY' if is_long else 'SELL'}:")
            logger.info(f"  Current price: {current_price:.5f}")
            logger.info(f"  Take profit: {take_profit_price:.5f} ({(take_profit_price-current_price)*10000 if is_long else (current_price-take_profit_price)*10000:+.1f} pips)")
            logger.info(f"  Stop loss: {stop_loss_price:.5f} ({(current_price-stop_loss_price)*10000 if is_long else (stop_loss_price-current_price)*10000:+.1f} pips)")
            logger.info(f"  Risk:Reward = 1:{abs(take_profit_price-current_price)/abs(current_price-stop_loss_price):.2f}")
            
            # Determine precision
            if instrument.endswith('JPY'):
                price_precision = 3
            else:
                price_precision = 5
            
            # Create order data
            data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": str(round(stop_loss_price, price_precision)),
                        "timeInForce": "GTC"
                    },
                    "takeProfitOnFill": {
                        "price": str(round(take_profit_price, price_precision)),
                        "timeInForce": "GTC"
                    }
                }
            }
            
            # Execute order
            response = await self._execute_request(OrderCreate(accountID=self.account_id, data=data))
            
            # Log full response for debugging
            logger.info(f"Order response for {instrument}: {response}")
            
            # Check for order rejection
            if 'orderRejectTransaction' in response:
                reject_reason = response['orderRejectTransaction'].get('rejectReason', 'Unknown')
                logger.error(f"Order rejected for {instrument}: {reject_reason}")
                logger.error(f"Full rejection details: {response['orderRejectTransaction']}")
                return False
            
            # Check for order cancellation
            if 'orderCancelTransaction' in response:
                cancel_reason = response['orderCancelTransaction'].get('reason', 'Unknown')
                logger.error(f"Order canceled for {instrument}: {cancel_reason}")
                logger.error(f"Full cancellation details: {response['orderCancelTransaction']}")
                return False
            
            # Extract order info
            order_id = response.get('orderCreateTransaction', {}).get('id')
            fill_transaction = response.get('orderFillTransaction')
            
            if not fill_transaction:
                logger.error(f"No fill transaction found for {instrument}. Response: {response}")
                return False
            
            # Update tracking
            self.last_trade_time[instrument] = datetime.now()
            self.position_count[instrument] = 1
            self.active_pairs.add(instrument)
            self.trade_entry_times[instrument] = datetime.now()
            
            # Log trade
            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'instrument': instrument,
                'action': 'BUY' if is_long else 'SELL',
                'price': current_price,
                'units': units,
                'signal_strength': signal_strength,
                'market_condition': market_condition.value,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'risk_amount': abs(units) * abs(current_price - stop_loss_price) * config.INSTRUMENTS[instrument]['pip_value'],
                'risk_pct': self.current_risk_per_trade * 100,
                'order_id': order_id
            }
            
            self.trade_history.append(trade_info)
            self.trade_logger.log_trade(trade_info)
            
            logger.info(f"Opened {'long' if is_long else 'short'} position for {instrument}: {units} units at {current_price}")
            
            return True
            
        except V20Error as e:
            logger.error(f"OANDA API error opening position for {instrument}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error opening position for {instrument}: {str(e)}")
            return False
    
    async def close_position(self, instrument):
        """Close position with partial profit taking."""
        try:
            position = self.open_positions.get(instrument)
            if not position:
                logger.warning(f"No position found for {instrument}")
                return False
            
            # Get current price
            current_price = await self._get_current_price(instrument)
            
            # Calculate P&L
            profit_loss = 0.0
            if current_price is not None:
                long_units = int(str(position.get('long', {}).get('units', 0)))
                short_units = int(str(position.get('short', {}).get('units', 0)))
                
                if long_units > 0:
                    entry_price = float(position.get('long', {}).get('averagePrice', 0))
                    profit_loss = (current_price - entry_price) * long_units * config.INSTRUMENTS[instrument]['pip_value']
                elif short_units < 0:
                    entry_price = float(position.get('short', {}).get('averagePrice', 0))
                    profit_loss = (entry_price - current_price) * abs(short_units) * config.INSTRUMENTS[instrument]['pip_value']
            
            # Update Kelly Criterion with result
            risk_amount = self.current_balance * self.current_risk_per_trade
            self.kelly_criterion.add_trade_result(profit_loss, risk_amount)
            
            # Close position
            data = {"longUnits": "ALL", "shortUnits": "ALL"}
            response = await self._execute_request(
                PositionClose(accountID=self.account_id, instrument=instrument, data=data)
            )
            
            # Update tracking
            if profit_loss > 0:
                self.win_count += 1
                self.total_profit += profit_loss
            else:
                self.loss_count += 1
                self.total_loss += abs(profit_loss)
            
            # Log trade
            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'instrument': instrument,
                'action': 'CLOSE',
                'price': current_price if current_price is not None else 0.0,
                'units': 0,
                'profit_loss': profit_loss
            }
            
            self.trade_history.append(trade_info)
            self.trade_logger.log_trade(trade_info)
            
            # Clean up tracking
            self.position_count[instrument] = 0
            if instrument in self.active_pairs:
                self.active_pairs.remove(instrument)
            if instrument in self.trade_entry_times:
                del self.trade_entry_times[instrument]
            
            logger.info(f"Closed position for {instrument} with P/L: ${profit_loss:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {instrument}: {str(e)}")
            return False
    
    async def _update_open_positions(self):
        """Update open positions tracking and detect closed positions."""
        try:
            # Get previous positions for comparison
            previous_positions = set(self.open_positions.keys())
            
            # Get current positions from OANDA
            positions = await self._get_open_positions()
            
            # Update current positions
            self.open_positions = {}
            self.position_count = {instrument: 0 for instrument in config.INSTRUMENTS}
            self.active_pairs = set()
            
            current_positions = set()
            for position in positions:
                instrument = position.get('instrument')
                if instrument:
                    self.open_positions[instrument] = position
                    self.position_count[instrument] = 1
                    self.active_pairs.add(instrument)
                    current_positions.add(instrument)
            
            # Detect closed positions
            closed_positions = previous_positions - current_positions
            if closed_positions:
                logger.info(f"Detected closed positions: {closed_positions}")
                logger.info(f"About to call _handle_closed_positions with: {closed_positions}")
                await self._handle_closed_positions(closed_positions)
                logger.info(f"Completed _handle_closed_positions call")
            
            return True
        except Exception as e:
            logger.error(f"Error updating open positions: {str(e)}")
            return False
    
    async def _get_open_positions(self):
        """Get current open positions."""
        try:
            response = await self._execute_request(OpenPositions(accountID=self.account_id))
            return response.get('positions', [])
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    async def _handle_closed_positions(self, closed_instruments):
        """Handle positions that were closed automatically by OANDA (TP/SL)."""
        logger.info(f"=== STARTING _handle_closed_positions for: {closed_instruments} ===")
        
        try:
            # Use trades endpoint to get recently closed trades (more reliable)
            from oandapyV20.endpoints.trades import TradesList
            from datetime import datetime, timedelta
            
            logger.info("Step 1: Importing required modules - SUCCESS")
            
            # Get all trades (including recently closed ones)
            params = {
                'state': 'CLOSED',  # Get closed trades
                'count': 50  # Get last 50 closed trades
            }
            
            logger.info(f"Step 2: Prepared params for trades endpoint: {params}")
            logger.info(f"Step 3: About to call TradesList endpoint for account: {self.account_id}")
            
            response = await self._execute_request(
                TradesList(accountID=self.account_id, params=params)
            )
            
            logger.info(f"Step 4: Received response from trades endpoint: {type(response)}")
            
            # Get closed trades from response
            closed_trades = response.get('trades', [])
            logger.info(f"Step 5: Found {len(closed_trades)} closed trades to analyze")
            
            # Filter for trades of the instruments that were just closed
            relevant_trades = []
            for trade in closed_trades:
                instrument = trade.get('instrument')
                logger.info(f"Step 6: Analyzing trade - Instrument: {instrument}, State: {trade.get('state')}, P/L: {trade.get('realizedPL', 'N/A')}")
                
                # Check if this trade is for one of our recently closed instruments
                if instrument in closed_instruments:
                    # Check if trade was closed recently (within last hour)
                    close_time_str = trade.get('closingTransactionIDs', [None])[-1] if trade.get('closingTransactionIDs') else None
                    if close_time_str:
                        relevant_trades.append(trade)
                        logger.info(f"Step 7: Found relevant closed trade for {instrument}")
            
            logger.info(f"Step 8: Processing {len(relevant_trades)} relevant closed trades")
            
            for trade in relevant_trades:
                instrument = trade.get('instrument')
                logger.info(f"Step 9: Processing closed trade for {instrument}")
                
                # Extract trade details from closed trade
                units = abs(float(trade.get('initialUnits', 0)))
                price = float(trade.get('price', 0))  # Opening price
                close_price = float(trade.get('averageClosePrice', 0))  # Closing price
                profit_loss = float(trade.get('realizedPL', 0))
                trade_id = trade.get('id')
                close_reason = 'AUTOMATIC_CLOSE'  # Since these are auto-closed positions
                
                logger.info(f"Step 10: Found closed trade: {instrument} - P/L: {profit_loss}, Reason: {close_reason}")
                
                # Create trade info for logging
                trade_info = {
                    'timestamp': datetime.now(),
                    'instrument': instrument,
                    'action': 'CLOSE',
                    'price': close_price,  # Use closing price
                    'units': units,
                    'signal_strength': 0,  # Not applicable for automated closes
                    'market_condition': 'automated_close',
                    'stop_loss': None,
                    'take_profit': None,
                    'risk_amount': 0,
                    'risk_pct': 0,
                    'order_id': trade_id,
                    'profit_loss': profit_loss,
                    'close_reason': close_reason
                }
                
                logger.info(f"Step 11: Created trade info for TradeLogger: {trade_info}")
                
                # Log to TradeLogger
                self.trade_logger.log_trade(trade_info)
                logger.info(f"Step 12: Logged trade to TradeLogger")
                
                # Update OrderRoutingSystem metrics
                if profit_loss > 0:
                    self.win_count += 1
                    self.total_profit += profit_loss
                    logger.info(f"Step 13: Updated win count: {self.win_count}, total profit: {self.total_profit}")
                elif profit_loss < 0:
                    self.loss_count += 1
                    self.total_loss += abs(profit_loss)
                    logger.info(f"Step 13: Updated loss count: {self.loss_count}, total loss: {self.total_loss}")
            
            # Transaction processing complete
            logger.info(f"=== COMPLETED _handle_closed_positions processing ===")
            
        except Exception as e:
            import traceback
            logger.error(f"=== ERROR in _handle_closed_positions ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:")
            logger.error(traceback.format_exc())
            logger.error(f"=== END ERROR DETAILS ===")
    
    async def _get_current_price(self, instrument):
        """Get current market price."""
        try:
            params = {"instruments": instrument}
            response = await self._execute_request(PricingInfo(accountID=self.account_id, params=params))
            
            prices = response.get('prices', [])
            if not prices:
                return None
            
            price_data = prices[0]
            bid = float(price_data.get('bids', [{}])[0].get('price', 0))
            ask = float(price_data.get('asks', [{}])[0].get('price', 0))
            
            return (bid + ask) / 2
            
        except Exception as e:
            logger.error(f"Error getting current price for {instrument}: {str(e)}")
            return None
    
    async def _execute_request(self, request):
        """Execute API request asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.api.request(request))
        return response
    
    def get_performance_summary(self):
        """Get current performance summary."""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else float('inf')
        
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'profit_loss': self.current_balance - self.initial_balance,
            'profit_loss_pct': ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0,
            'max_drawdown': self.current_drawdown * 100,
            'daily_drawdown': self.daily_drawdown * 100,
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': self.total_profit / self.win_count if self.win_count > 0 else 0,
            'average_loss': self.total_loss / self.loss_count if self.loss_count > 0 else 0,
            'kelly_optimal': self.kelly_criterion.calculate_optimal_fraction() * 100,
            'current_risk_pct': self.current_risk_per_trade * 100,
            'recovery_mode': self.recovery_mode
        }
    
    def get_trade_summary(self):
        """Get trade summary for reporting."""
        return self.trade_logger.generate_summary()
    
    def export_trade_log(self, filepath=None):
        """Export trade log to CSV."""
        return self.trade_logger.export_to_csv(filepath)