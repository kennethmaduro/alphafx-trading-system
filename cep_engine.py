"""
Complex Event Processing (CEP) Engine for OANDA Trading System
Enhanced with research-based market regime detection and signal generation
"""
import pandas as pd
import numpy as np
import logging
from enum import Enum
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config
from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Enhanced market conditions with volatility states."""
    TRENDING_UP_HIGH_VOL = "trending_up_high_volatility"
    TRENDING_UP_LOW_VOL = "trending_up_low_volatility"
    TRENDING_DOWN_HIGH_VOL = "trending_down_high_volatility"
    TRENDING_DOWN_LOW_VOL = "trending_down_low_volatility"
    RANGING_HIGH_VOL = "ranging_high_volatility"
    RANGING_LOW_VOL = "ranging_low_volatility"
    TRANSITIONAL = "transitional"
    UNKNOWN = "unknown"

class SignalType(Enum):
    """Enum for different signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    NO_ACTION = "no_action"

class MarketRegimeDetector:
    """Advanced market regime detection using machine learning."""
    
    def __init__(self):
        self.gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.regime_history = []
        self.is_fitted = False
        
    def detect_regime(self, df):
        """Detect market regime using Gaussian Mixture Model."""
        if df is None or len(df) < 50:
            return MarketCondition.UNKNOWN
            
        try:
            # Calculate features for regime detection
            features = self._calculate_regime_features(df)
            
            if features is None:
                return MarketCondition.UNKNOWN
            
            # Fit or predict
            if not self.is_fitted and len(features) >= 100:
                scaled_features = self.scaler.fit_transform(features)
                self.gmm.fit(scaled_features)
                self.is_fitted = True
            
            if self.is_fitted:
                scaled_features = self.scaler.transform(features[-1:])
                regime = self.gmm.predict(scaled_features)[0]
                probabilities = self.gmm.predict_proba(scaled_features)[0]
                
                # Map GMM clusters to market conditions
                return self._map_regime_to_condition(regime, features.iloc[-1], probabilities)
            else:
                # Fallback to rule-based detection
                return self._rule_based_regime_detection(features.iloc[-1])
                
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return MarketCondition.UNKNOWN
    
    def _calculate_regime_features(self, df):
        """Calculate features for regime detection."""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['returns'] = df['close'].pct_change()
            features['returns_std'] = features['returns'].rolling(20).std()
            features['returns_skew'] = features['returns'].rolling(20).skew()
            features['returns_kurt'] = features['returns'].rolling(20).kurt()
            
            # Trend features
            if 'adx' in df.columns:
                features['adx'] = df['adx']
                features['trend_strength'] = df['adx'] / 50  # Normalized
            
            # Volatility features
            if 'atr_pct' in df.columns:
                features['volatility'] = df['atr_pct']
                features['volatility_ma'] = df['atr_pct'].rolling(20).mean()
            
            # Volume features
            if 'volume' in df.columns:
                features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Remove NaN values
            features = features.dropna()
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {str(e)}")
            return None
    
    def _map_regime_to_condition(self, regime, features, probabilities):
        """Map GMM regime to market condition."""
        confidence = max(probabilities)
        
        # High confidence threshold
        if confidence < 0.6:
            return MarketCondition.TRANSITIONAL
        
        # Determine volatility level
        volatility = features.get('volatility', 0.1)
        is_high_vol = volatility > config.MARKET_CONDITION['VOLATILITY_HIGH']
        
        # Determine trend direction and strength
        adx = features.get('adx', 20)
        returns = features.get('returns', 0)
        
        # Map regimes (this mapping should be calibrated based on backtesting)
        if regime == 0:  # Assumed trending up
            if is_high_vol:
                return MarketCondition.TRENDING_UP_HIGH_VOL
            else:
                return MarketCondition.TRENDING_UP_LOW_VOL
        elif regime == 1:  # Assumed trending down
            if is_high_vol:
                return MarketCondition.TRENDING_DOWN_HIGH_VOL
            else:
                return MarketCondition.TRENDING_DOWN_LOW_VOL
        elif regime == 2:  # Assumed ranging
            if is_high_vol:
                return MarketCondition.RANGING_HIGH_VOL
            else:
                return MarketCondition.RANGING_LOW_VOL
        else:  # Transitional
            return MarketCondition.TRANSITIONAL
    
    def _rule_based_regime_detection(self, features):
        """Fallback rule-based regime detection."""
        adx = features.get('adx', 20)
        volatility = features.get('volatility', 0.1)
        returns = features.get('returns', 0)
        
        is_high_vol = volatility > config.MARKET_CONDITION['VOLATILITY_HIGH']
        
        if adx > config.MARKET_CONDITION['TRENDING_ADX_THRESHOLD']:
            if returns > 0:
                return MarketCondition.TRENDING_UP_HIGH_VOL if is_high_vol else MarketCondition.TRENDING_UP_LOW_VOL
            else:
                return MarketCondition.TRENDING_DOWN_HIGH_VOL if is_high_vol else MarketCondition.TRENDING_DOWN_LOW_VOL
        elif adx < config.MARKET_CONDITION['RANGING_ADX_THRESHOLD']:
            return MarketCondition.RANGING_HIGH_VOL if is_high_vol else MarketCondition.RANGING_LOW_VOL
        else:
            return MarketCondition.TRANSITIONAL

class CEPEngine:
    """
    Enhanced Complex Event Processing Engine with multi-indicator confirmation
    and adaptive parameter adjustment.
    """
    
    def __init__(self):
        """Initialize the enhanced CEP Engine."""
        self.indicators_calculator = TechnicalIndicators()
        self.regime_detector = MarketRegimeDetector()
        self.market_condition = MarketCondition.UNKNOWN
        self.signal_history = []
        self.last_market_conditions = []
        self.last_signal_time = datetime.min
        
        # Signal stabilization parameters
        self.min_signal_interval = 60  # Minimum seconds between signals
        self.max_consecutive_signals = 2  # Maximum same direction signals
        
        # Consistency tracking
        self.last_signals = []
        
        # Pattern recognition
        self.pattern_history = []
        
        # Performance tracking for adaptive adjustment
        self.signal_performance = []
        self.parameter_performance = {}
    
    def process_timeframe_data(self, timeframe_data):
        """
        Enhanced process with multi-indicator confirmation and regime adaptation.
        """
        if not timeframe_data:
            logger.warning("No timeframe data to process")
            return SignalType.NO_ACTION, 0, MarketCondition.UNKNOWN, {}
        
        try:
            # Step 1: Advanced market regime detection
            primary_tf = self._get_primary_timeframe_data(timeframe_data)
            if primary_tf is not None:
                new_market_condition = self.regime_detector.detect_regime(primary_tf)
            else:
                new_market_condition = MarketCondition.UNKNOWN
            
            # Stabilize market condition
            self.last_market_conditions.append(new_market_condition)
            if len(self.last_market_conditions) > 5:
                self.last_market_conditions.pop(0)
            
            self.market_condition = self._get_stable_market_condition()
            logger.info(f"Market regime: {self.market_condition.value}")
            
            # Step 2: Adapt parameters based on regime
            adapted_params = self.adapt_indicator_parameters(self.market_condition, timeframe_data)
            
            # Step 3: Calculate indicators with adapted parameters
            indicator_data = {}
            for tf_key, df in timeframe_data.items():
                if df is not None and not df.empty:
                    try:
                        # Use special RSI period for 1-minute charts
                        params = adapted_params.copy()
                        if tf_key == "M1":
                            params['RSI_PERIOD'] = config.INDICATOR_SETTINGS['RSI_PERIOD_1MIN']
                        
                        indicator_data[tf_key] = self.indicators_calculator.add_all_indicators(df, params)
                    except Exception as e:
                        logger.error(f"Error calculating indicators for {tf_key}: {str(e)}")
                        continue
            
            # Step 4: Multi-timeframe pattern detection
            pattern_signals = self._detect_multi_timeframe_patterns(indicator_data)
            
            # Step 5: Generate primary signal with multi-indicator confirmation
            if indicator_data:
                signal_type, signal_strength = self.generate_enhanced_signal(
                    indicator_data, self.market_condition, pattern_signals
                )
                
                # Apply signal quality filters
                signal_type, signal_strength = self._apply_signal_filters(
                    signal_type, signal_strength, indicator_data
                )
                
                # Log signal
                logger.info(f"Generated signal: {signal_type.value}, "
                          f"Strength: {signal_strength:.2f}, "
                          f"Regime: {self.market_condition.value}")
                
                # Save to history
                self.signal_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'signal_type': signal_type.value,
                    'signal_strength': signal_strength,
                    'market_condition': self.market_condition.value
                })
                
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]
                
                return signal_type, signal_strength, self.market_condition, indicator_data
            else:
                return SignalType.NO_ACTION, 0, self.market_condition, {}
                
        except Exception as e:
            logger.error(f"Error processing timeframe data: {str(e)}")
            return SignalType.NO_ACTION, 0, self.market_condition, {}
    
    def _get_primary_timeframe_data(self, timeframe_data):
        """Get primary timeframe data for analysis."""
        for tf in ["M5", "M15", "M1"]:
            if tf in timeframe_data and timeframe_data[tf] is not None:
                return timeframe_data[tf]
        return None
    
    def _detect_multi_timeframe_patterns(self, indicator_data):
        """Enhanced pattern detection across multiple timeframes."""
        patterns = {
            'bullish': 0,
            'bearish': 0,
            'strength': 0
        }
        
        # Check each timeframe for patterns
        for tf, weight in [("M1", 0.4), ("M5", 0.35), ("M15", 0.25)]:
            if tf not in indicator_data or indicator_data[tf] is None:
                continue
                
            df = indicator_data[tf]
            if len(df) < 20:
                continue
            
            # EMA ribbon alignment
            if all(col in df.columns for col in ['ema_fast', 'ema_slow']):
                if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
                    patterns['bullish'] += weight * 20
                else:
                    patterns['bearish'] += weight * 20
            
            # RSI momentum
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    patterns['bullish'] += weight * 15
                elif rsi > 70:
                    patterns['bearish'] += weight * 15
            
            # Bollinger Band position
            if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                close = df['close'].iloc[-1]
                upper = df['bb_upper'].iloc[-1]
                lower = df['bb_lower'].iloc[-1]
                
                bb_position = (close - lower) / (upper - lower) if upper > lower else 0.5
                
                if bb_position < 0.2:
                    patterns['bullish'] += weight * 10
                elif bb_position > 0.8:
                    patterns['bearish'] += weight * 10
            
            # MACD momentum
            if 'macd_hist' in df.columns:
                hist = df['macd_hist'].iloc[-1]
                hist_prev = df['macd_hist'].iloc[-2]
                
                if hist > 0 and hist > hist_prev:
                    patterns['bullish'] += weight * 10
                elif hist < 0 and hist < hist_prev:
                    patterns['bearish'] += weight * 10
        
        # Determine overall pattern signal
        if patterns['bullish'] > patterns['bearish'] + 10:
            return SignalType.BUY, patterns['bullish']
        elif patterns['bearish'] > patterns['bullish'] + 10:
            return SignalType.SELL, patterns['bearish']
        else:
            return SignalType.NO_ACTION, 0
    
    def generate_enhanced_signal(self, indicator_data, market_condition, pattern_signals):
        """
        Generate signals using multi-indicator confirmation system.
        """
        # Initialize scoring system
        buy_indicators = 0
        sell_indicators = 0
        total_indicators = 0
        signal_details = []
        
        # Get the primary timeframe (M5 for main signals)
        primary_tf = "M5" if "M5" in indicator_data else "M1"
        if primary_tf not in indicator_data or indicator_data[primary_tf] is None:
            return SignalType.NO_ACTION, 0
        
        df = indicator_data[primary_tf]
        if df.empty or len(df) < 20:
            return SignalType.NO_ACTION, 0
        
        latest = df.iloc[-1]
        
        # 1. EMA Ribbon Signal (5-8-13)
        if all(col in latest for col in ['ema_fast', 'ema_medium', 'ema_slow']):
            total_indicators += 1
            
            # Check for ribbon alignment
            if latest['ema_fast'] > latest['ema_medium'] > latest['ema_slow']:
                buy_indicators += 1
                signal_details.append("EMA ribbon bullish")
            elif latest['ema_fast'] < latest['ema_medium'] < latest['ema_slow']:
                sell_indicators += 1
                signal_details.append("EMA ribbon bearish")
        
        # 2. RSI Signal (Optimized thresholds)
        if 'rsi' in latest:
            total_indicators += 1
            rsi = latest['rsi']
            
            if market_condition in [MarketCondition.RANGING_HIGH_VOL, MarketCondition.RANGING_LOW_VOL]:
                # Mean reversion in ranging markets
                if rsi < 20:
                    buy_indicators += 1
                    signal_details.append(f"RSI oversold: {rsi:.1f}")
                elif rsi > 80:
                    sell_indicators += 1
                    signal_details.append(f"RSI overbought: {rsi:.1f}")
            else:
                # Momentum in trending markets
                if 30 < rsi < 50 and 'trending_up' in market_condition.value:
                    buy_indicators += 1
                    signal_details.append(f"RSI pullback in uptrend: {rsi:.1f}")
                elif 50 < rsi < 70 and 'trending_down' in market_condition.value:
                    sell_indicators += 1
                    signal_details.append(f"RSI pullback in downtrend: {rsi:.1f}")
        
        # 3. MACD Signal (6-13-4 optimized)
        if all(col in latest for col in ['macd', 'macd_signal', 'macd_hist']):
            total_indicators += 1
            
            if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
                buy_indicators += 1
                signal_details.append("MACD bullish")
            elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
                sell_indicators += 1
                signal_details.append("MACD bearish")
        
        # 4. Bollinger Bands Signal
        if all(col in latest for col in ['close', 'bb_upper', 'bb_lower', 'bb_b']):
            total_indicators += 1
            
            if market_condition in [MarketCondition.RANGING_HIGH_VOL, MarketCondition.RANGING_LOW_VOL]:
                # Mean reversion signals
                if latest['bb_b'] < 0.2:
                    buy_indicators += 1
                    signal_details.append("Price near lower BB")
                elif latest['bb_b'] > 0.8:
                    sell_indicators += 1
                    signal_details.append("Price near upper BB")
            else:
                # Breakout signals in trending markets
                if latest['close'] > latest['bb_upper']:
                    if 'trending_up' in market_condition.value:
                        buy_indicators += 1
                        signal_details.append("BB breakout up")
                elif latest['close'] < latest['bb_lower']:
                    if 'trending_down' in market_condition.value:
                        sell_indicators += 1
                        signal_details.append("BB breakout down")
        
        # 5. Stochastic Signal
        if all(col in latest for col in ['stoch_k', 'stoch_d']):
            total_indicators += 1
            
            if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
                buy_indicators += 1
                signal_details.append("Stochastic oversold crossover")
            elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
                sell_indicators += 1
                signal_details.append("Stochastic overbought crossover")
        
        # 6. Volume Confirmation
        if 'volume' in df.columns:
            total_indicators += 1
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if current_volume > avg_volume * config.SIGNAL_FILTERS['VOLUME_THRESHOLD']:
                if buy_indicators > sell_indicators:
                    buy_indicators += 0.5
                    signal_details.append("Volume confirmation")
                elif sell_indicators > buy_indicators:
                    sell_indicators += 0.5
                    signal_details.append("Volume confirmation")
        
        # Calculate signal strength
        min_agreement = config.SIGNAL_FILTERS['MIN_INDICATOR_AGREEMENT']
        
        # DEBUG: Log signal calculation details
        logger.info(f"DEBUG: Signal calculation - Buy: {buy_indicators}, Sell: {sell_indicators}, Total: {total_indicators}")
        logger.info(f"DEBUG: Min agreement required: {min_agreement}")
        logger.info(f"DEBUG: Signal details: {signal_details}")
        
        if buy_indicators >= min_agreement and buy_indicators > sell_indicators:
            strength = (buy_indicators / total_indicators) * 100
            logger.info(f"Buy signal: {signal_details}")
            return SignalType.BUY, min(100, strength)
        elif sell_indicators >= min_agreement and sell_indicators > buy_indicators:
            strength = (sell_indicators / total_indicators) * 100
            logger.info(f"Sell signal: {signal_details}")
            # SELL signals should have negative strength
            return SignalType.SELL, -min(100, strength)
        else:
            logger.info(f"DEBUG: No action - insufficient agreement (buy: {buy_indicators}, sell: {sell_indicators}, min: {min_agreement})")
            return SignalType.NO_ACTION, 0
    
    def _apply_signal_filters(self, signal_type, signal_strength, indicator_data):
        """Apply quality filters to signals."""
        if signal_type == SignalType.NO_ACTION:
            return signal_type, signal_strength
        
        # Get primary timeframe data
        primary_tf = "M5" if "M5" in indicator_data else "M1"
        if primary_tf not in indicator_data:
            return SignalType.NO_ACTION, 0
        
        df = indicator_data[primary_tf]
        latest = df.iloc[-1]
        
        # 1. Volatility filter
        if 'atr_pct' in latest:
            volatility = latest['atr_pct']
            min_vol = config.SIGNAL_FILTERS['MIN_VOLATILITY']
            max_vol = config.SIGNAL_FILTERS['MAX_VOLATILITY']
            logger.info(f"DEBUG: Volatility check - ATR: {volatility:.6f}, Min: {min_vol:.6f}, Max: {max_vol:.6f}")
            
            if volatility < min_vol:
                logger.info(f"Signal filtered: Volatility too low ({volatility:.6f} < {min_vol:.6f})")
                return SignalType.NO_ACTION, 0
            elif volatility > max_vol:
                logger.info(f"Signal filtered: Volatility too high ({volatility:.6f} > {max_vol:.6f})")
                return SignalType.NO_ACTION, 0
        
        # 2. Time-based filter for session trading
        current_hour = datetime.now().hour
        in_active_session = False
        session_weight = 1.0
        
        for session, details in config.TRADING_SESSIONS.items():
            if details['start'] <= current_hour < details['end']:
                in_active_session = True
                session_weight = details['weight']
                break
        
        if not in_active_session:
            signal_strength *= 0.7  # Reduce strength outside active sessions
        else:
            signal_strength *= session_weight
        
        # 3. Trend strength filter (only for trend-following in trending markets)
        # Skip ADX filter for ranging markets since they don't need strong trends
        if 'adx' in latest and self.market_condition.value.startswith('trending'):
            adx_value = latest['adx']
            logger.info(f"DEBUG: ADX check - ADX: {adx_value:.2f}, Min required: 20 (trending market)")
            if adx_value < 20:
                logger.debug(f"Signal filtered: Weak trend in trending market (ADX: {adx_value:.2f} < 20)")
                return SignalType.NO_ACTION, 0
        elif 'adx' in latest:
            adx_value = latest['adx']
            logger.info(f"DEBUG: ADX check - ADX: {adx_value:.2f} (ranging market - no minimum required)")
        
        # 4. Minimum signal interval
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        if time_since_last < self.min_signal_interval:
            return SignalType.NO_ACTION, 0
        
        # Update last signal time if passing all filters
        if signal_type in [SignalType.BUY, SignalType.SELL]:
            self.last_signal_time = datetime.now()
        
        return signal_type, signal_strength
    
    def adapt_indicator_parameters(self, market_condition, timeframe_data):
        """
        Dynamically adapt indicator parameters based on market regime.
        """
        params = config.INDICATOR_SETTINGS.copy()
        
        # Adapt based on specific market conditions
        if market_condition in [MarketCondition.TRENDING_UP_HIGH_VOL, MarketCondition.TRENDING_DOWN_HIGH_VOL]:
            # High volatility trending - use wider parameters
            params['BBANDS_STD'] = config.INDICATOR_SETTINGS['BBANDS_STD_TREND']
            params['ATR_MULTIPLIER_STOP'] = 2.0  # Wider stops in volatile trends
            params['RSI_OVERBOUGHT'] = 80
            params['RSI_OVERSOLD'] = 20
            
        elif market_condition in [MarketCondition.TRENDING_UP_LOW_VOL, MarketCondition.TRENDING_DOWN_LOW_VOL]:
            # Low volatility trending - standard parameters
            params['BBANDS_STD'] = config.INDICATOR_SETTINGS['BBANDS_STD_TREND']
            params['ATR_MULTIPLIER_STOP'] = 1.5
            
        elif market_condition in [MarketCondition.RANGING_HIGH_VOL, MarketCondition.RANGING_LOW_VOL]:
            # Ranging markets - tighter parameters
            params['BBANDS_STD'] = config.INDICATOR_SETTINGS['BBANDS_STD']
            params['ATR_MULTIPLIER_STOP'] = 1.2  # Tighter stops in ranges
            params['RSI_OVERBOUGHT'] = 70
            params['RSI_OVERSOLD'] = 30
            
        else:  # Transitional or unknown
            # Use default parameters
            pass
        
        logger.debug(f"Adapted parameters for {market_condition.value}")
        return params
    
    def _get_stable_market_condition(self):
        """Get stable market condition from recent history."""
        if not self.last_market_conditions:
            return MarketCondition.UNKNOWN
        
        # Count occurrences
        condition_counts = {}
        for condition in self.last_market_conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # Find most common
        max_count = 0
        most_common = self.last_market_conditions[-1]
        
        for condition, count in condition_counts.items():
            if count > max_count:
                max_count = count
                most_common = condition
        
        return most_common