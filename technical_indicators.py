"""
Technical Indicators Module
Enhanced with research-based optimizations for short-term trading
"""
import pandas as pd
import numpy as np
import talib
import logging
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Enhanced technical indicators for profitable short-term trading."""
    
    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        self.indicator_cache = {}
        
    @staticmethod
    def add_all_indicators(df, params=None):
        """
        Add all technical indicators optimized for scalping.
        """
        if df is None or df.empty:
            logger.warning("Cannot add indicators to empty dataframe")
            return df
            
        if params is None:
            params = config.INDICATOR_SETTINGS
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Apply each indicator function in order
        df = TechnicalIndicators.add_moving_averages(df, params)
        df = TechnicalIndicators.add_rsi(df, params)
        df = TechnicalIndicators.add_macd(df, params)
        df = TechnicalIndicators.add_bollinger_bands(df, params)
        df = TechnicalIndicators.add_adx(df, params)
        df = TechnicalIndicators.add_atr(df, params)
        df = TechnicalIndicators.add_stochastic(df, params)
        df = TechnicalIndicators.add_cci(df, params)
        df = TechnicalIndicators.add_obv(df, params)
        df = TechnicalIndicators.add_linear_regression(df)
        df = TechnicalIndicators.add_volume_indicators(df, params)
        df = TechnicalIndicators.add_price_action_features(df)
        
        return df
    
    @staticmethod
    def add_moving_averages(df, params):
        """Add optimized moving averages including 5-8-13 EMA ribbon."""
        # EMA Ribbon (5-8-13) for scalping
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=params['EMA_FAST'])
        df['ema_medium'] = talib.EMA(df['close'], timeperiod=params['EMA_MEDIUM'])
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=params['EMA_SLOW'])
        
        # EMA Ribbon alignment signal
        df['ema_ribbon_bullish'] = (
            (df['ema_fast'] > df['ema_medium']) & 
            (df['ema_medium'] > df['ema_slow'])
        ).astype(int)
        
        df['ema_ribbon_bearish'] = (
            (df['ema_fast'] < df['ema_medium']) & 
            (df['ema_medium'] < df['ema_slow'])
        ).astype(int)
        
        # EMA crossover signals
        df['ema_fast_cross_medium'] = np.where(
            (df['ema_fast'] > df['ema_medium']) & 
            (df['ema_fast'].shift(1) <= df['ema_medium'].shift(1)), 
            1,
            np.where(
                (df['ema_fast'] < df['ema_medium']) & 
                (df['ema_fast'].shift(1) >= df['ema_medium'].shift(1)),
                -1, 0
            )
        )
        
        # Price position relative to EMAs
        df['price_above_emas'] = (
            (df['close'] > df['ema_fast']) & 
            (df['close'] > df['ema_medium']) & 
            (df['close'] > df['ema_slow'])
        ).astype(int)
        
        # Traditional SMAs for comparison
        df['sma_fast'] = talib.SMA(df['close'], timeperiod=params['EMA_FAST'])
        df['sma_slow'] = talib.SMA(df['close'], timeperiod=params['EMA_SLOW'])
        
        # Adaptive Moving Average (KAMA)
        df['kama'] = talib.KAMA(df['close'], timeperiod=params['EMA_FAST'])
        
        return df
    
    @staticmethod
    def add_rsi(df, params):
        """Add RSI with optimized parameters for scalping."""
        # Standard RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=params['RSI_PERIOD'])
        
        # RSI extremes for scalping
        df['rsi_oversold'] = (df['rsi'] < params['RSI_OVERSOLD']).astype(int)
        df['rsi_overbought'] = (df['rsi'] > params['RSI_OVERBOUGHT']).astype(int)
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi'] - df['rsi'].shift(1)
        
        # RSI divergence detection
        if len(df) >= 10:
            # Price momentum
            df['price_momentum'] = df['close'].pct_change(5)
            df['rsi_change'] = df['rsi'].diff(5)
            
            # Bullish divergence: price down, RSI up
            df['rsi_bull_div'] = (
                (df['price_momentum'] < -0.001) & 
                (df['rsi_change'] > 2) & 
                (df['rsi'] < 40)
            ).astype(int)
            
            # Bearish divergence: price up, RSI down
            df['rsi_bear_div'] = (
                (df['price_momentum'] > 0.001) & 
                (df['rsi_change'] < -2) & 
                (df['rsi'] > 60)
            ).astype(int)
        
        # RSI signal based on thresholds
        df['rsi_signal'] = np.where(
            df['rsi_oversold'], 1,
            np.where(df['rsi_overbought'], -1, 0)
        )
        
        # Stochastic RSI for additional confirmation
        df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(
            df['close'], 
            timeperiod=params['RSI_PERIOD'],
            fastk_period=3,
            fastd_period=3
        )
        
        return df
    
    @staticmethod
    def add_macd(df, params):
        """Add MACD with optimized 6-13-4 settings for scalping."""
        macd, signal, hist = talib.MACD(
            df['close'], 
            fastperiod=params['MACD_FAST'], 
            slowperiod=params['MACD_SLOW'], 
            signalperiod=params['MACD_SIGNAL']
        )
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # MACD crossover signals
        df['macd_crossover'] = np.where(
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 
            1,  # Bullish crossover
            np.where(
                (df['macd'] < df['macd_signal']) & 
                (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                -1,  # Bearish crossover
                0
            )
        )
        
        # MACD histogram momentum
        df['macd_hist_momentum'] = df['macd_hist'] - df['macd_hist'].shift(1)
        df['macd_hist_increasing'] = (df['macd_hist_momentum'] > 0).astype(int)
        
        # MACD zero line crossover
        df['macd_zero_cross'] = np.where(
            (df['macd'] > 0) & (df['macd'].shift(1) <= 0), 1,
            np.where(
                (df['macd'] < 0) & (df['macd'].shift(1) >= 0), -1, 0
            )
        )
        
        # MACD divergence
        if len(df) >= 10:
            df['macd_bull_div'] = (
                (df['price_momentum'] < -0.001) & 
                (df['macd_hist'] > df['macd_hist'].shift(5)) & 
                (df['macd_hist'] < 0)
            ).astype(int)
            
            df['macd_bear_div'] = (
                (df['price_momentum'] > 0.001) & 
                (df['macd_hist'] < df['macd_hist'].shift(5)) & 
                (df['macd_hist'] > 0)
            ).astype(int)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df, params):
        """Add Bollinger Bands with adaptive standard deviation."""
        # Standard Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'], 
            timeperiod=params['BBANDS_PERIOD'], 
            nbdevup=params['BBANDS_STD'], 
            nbdevdn=params['BBANDS_STD']
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Bollinger Band metrics
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_bandwidth'] = df['bb_width'] / df['bb_middle']
        
        # Bollinger %B (position within bands)
        df['bb_b'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Band touch signals
        df['bb_upper_touch'] = (df['high'] >= df['bb_upper']).astype(int)
        df['bb_lower_touch'] = (df['low'] <= df['bb_lower']).astype(int)
        
        # Squeeze detection (low volatility)
        df['bb_squeeze'] = (
            df['bb_bandwidth'] < df['bb_bandwidth'].rolling(20).quantile(0.2)
        ).astype(int)
        
        # Band breakout signals
        df['bb_breakout_up'] = (
            (df['close'] > df['bb_upper']) & 
            (df['close'].shift(1) <= df['bb_upper'].shift(1))
        ).astype(int)
        
        df['bb_breakout_down'] = (
            (df['close'] < df['bb_lower']) & 
            (df['close'].shift(1) >= df['bb_lower'].shift(1))
        ).astype(int)
        
        # Mean reversion signals
        df['bb_signal'] = np.where(
            df['bb_b'] < 0.05, 1,  # Near lower band
            np.where(df['bb_b'] > 0.95, -1, 0)  # Near upper band
        )
        
        return df
    
    @staticmethod
    def add_adx(df, params):
        """Add ADX for trend strength measurement."""
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=params['ADX_PERIOD'])
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=params['ADX_PERIOD'])
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=params['ADX_PERIOD'])
        
        # Trend strength
        df['trend_strength'] = np.where(
            df['adx'] > params['ADX_THRESHOLD'], 1, 0
        )
        
        # Trend direction
        df['trend_direction'] = np.where(
            df['plus_di'] > df['minus_di'], 1,
            np.where(df['plus_di'] < df['minus_di'], -1, 0)
        )
        
        # DI crossovers
        df['di_bullish_cross'] = (
            (df['plus_di'] > df['minus_di']) & 
            (df['plus_di'].shift(1) <= df['minus_di'].shift(1))
        ).astype(int)
        
        df['di_bearish_cross'] = (
            (df['plus_di'] < df['minus_di']) & 
            (df['plus_di'].shift(1) >= df['minus_di'].shift(1))
        ).astype(int)
        
        # ADX momentum
        df['adx_increasing'] = (df['adx'] > df['adx'].shift(1)).astype(int)
        
        return df
    
    @staticmethod
    def add_atr(df, params):
        """Add ATR for volatility measurement and stop placement."""
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=params['ATR_PERIOD'])
        
        # ATR as percentage of price
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Volatility classification
        df['volatility_high'] = (
            df['atr_pct'] > config.MARKET_CONDITION['VOLATILITY_HIGH']
        ).astype(int)
        
        df['volatility_low'] = (
            df['atr_pct'] < config.MARKET_CONDITION['VOLATILITY_LOW']
        ).astype(int)
        
        # ATR bands for dynamic support/resistance
        df['atr_upper'] = df['close'] + (df['atr'] * params['ATR_MULTIPLIER_STOP'])
        df['atr_lower'] = df['close'] - (df['atr'] * params['ATR_MULTIPLIER_STOP'])
        
        # Volatility change
        df['atr_change'] = df['atr'].pct_change()
        df['volatility_increasing'] = (df['atr_change'] > 0.05).astype(int)
        
        return df
    
    @staticmethod
    def add_stochastic(df, params):
        """Add Stochastic oscillator optimized for scalping."""
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], 
            df['low'], 
            df['close'], 
            fastk_period=params['STOCH_K_PERIOD'],
            slowk_period=3,
            slowd_period=params['STOCH_D_PERIOD']
        )
        
        # Stochastic extremes
        df['stoch_oversold'] = (
            (df['stoch_k'] < params['STOCH_OVERSOLD']) & 
            (df['stoch_d'] < params['STOCH_OVERSOLD'])
        ).astype(int)
        
        df['stoch_overbought'] = (
            (df['stoch_k'] > params['STOCH_OVERBOUGHT']) & 
            (df['stoch_d'] > params['STOCH_OVERBOUGHT'])
        ).astype(int)
        
        # Stochastic crossovers in extreme zones
        df['stoch_bullish_cross'] = (
            (df['stoch_k'] > df['stoch_d']) & 
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & 
            (df['stoch_oversold'])
        ).astype(int)
        
        df['stoch_bearish_cross'] = (
            (df['stoch_k'] < df['stoch_d']) & 
            (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & 
            (df['stoch_overbought'])
        ).astype(int)
        
        # Stochastic signal
        df['stoch_signal'] = np.where(
            df['stoch_bullish_cross'], 1,
            np.where(df['stoch_bearish_cross'], -1, 0)
        )
        
        return df
    
    @staticmethod
    def add_cci(df, params):
        """Add Commodity Channel Index."""
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=params['CCI_PERIOD'])
        
        # CCI extremes
        df['cci_oversold'] = (df['cci'] < -100).astype(int)
        df['cci_overbought'] = (df['cci'] > 100).astype(int)
        
        # CCI zero line crossover
        df['cci_bullish'] = (
            (df['cci'] > 0) & (df['cci'].shift(1) <= 0)
        ).astype(int)
        
        df['cci_bearish'] = (
            (df['cci'] < 0) & (df['cci'].shift(1) >= 0)
        ).astype(int)
        
        # CCI signal
        df['cci_signal'] = np.where(
            df['cci_oversold'], 1,
            np.where(df['cci_overbought'], -1, 0)
        )
        
        return df
    
    @staticmethod
    def add_obv(df, params):
        """Add On-Balance Volume and volume analysis."""
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # OBV moving average
        df['obv_ma'] = df['obv'].rolling(params['VOLUME_MA_PERIOD']).mean()
        
        # OBV trend
        df['obv_trend'] = np.where(
            df['obv'] > df['obv_ma'], 1,
            np.where(df['obv'] < df['obv_ma'], -1, 0)
        )
        
        # Volume price trend
        df['vpt'] = talib.OBV(df['close'], df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)))
        
        # OBV divergence
        if len(df) >= 10:
            df['obv_change'] = df['obv'].diff(5)
            
            df['obv_bull_div'] = (
                (df['price_momentum'] < -0.001) & 
                (df['obv_change'] > 0)
            ).astype(int)
            
            df['obv_bear_div'] = (
                (df['price_momentum'] > 0.001) & 
                (df['obv_change'] < 0)
            ).astype(int)
        
        df['obv_divergence'] = 0
        if 'obv_bull_div' in df.columns:
            df['obv_divergence'] = df['obv_bull_div'] - df['obv_bear_div']
        
        return df
    
    @staticmethod
    def add_volume_indicators(df, params):
        """Add additional volume-based indicators."""
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(params['VOLUME_MA_PERIOD']).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High volume flag
        df['high_volume'] = (
            df['volume_ratio'] > config.SIGNAL_FILTERS['VOLUME_THRESHOLD']
        ).astype(int)
        
        # Volume-weighted average price (VWAP) - simplified daily
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Price-volume trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        # Accumulation/Distribution Line
        df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
        
        return df
    
    @staticmethod
    def add_price_action_features(df):
        """Add price action features for pattern recognition."""
        # Candlestick patterns
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Candlestick type
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        
        # Pin bar detection
        df['pin_bar_bullish'] = (
            (df['lower_wick'] > 2 * df['candle_body']) & 
            (df['upper_wick'] < 0.5 * df['candle_body'])
        ).astype(int)
        
        df['pin_bar_bearish'] = (
            (df['upper_wick'] > 2 * df['candle_body']) & 
            (df['lower_wick'] < 0.5 * df['candle_body'])
        ).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift(1) < df['open'].shift(1)) & 
            (df['close'] > df['open'].shift(1)) & 
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) & 
            (df['close'].shift(1) > df['open'].shift(1)) & 
            (df['close'] < df['open'].shift(1)) & 
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        # Support/Resistance levels (simplified)
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        # Price relative to support/resistance
        df['close_to_resistance'] = (
            (df['resistance'] - df['close']) / df['close'] < 0.001
        ).astype(int)
        
        df['close_to_support'] = (
            (df['close'] - df['support']) / df['close'] < 0.001
        ).astype(int)
        
        return df
    
    @staticmethod
    def add_linear_regression(df, period=20):
        """Add linear regression for trend analysis."""
        df['lr_time'] = np.arange(len(df))
        
        # Initialize columns
        df['lr_slope'] = np.nan
        df['lr_intercept'] = np.nan
        df['lr_r2'] = np.nan
        df['lr_slope_norm'] = np.nan
        df['lr_trend'] = 0
        
        if len(df) >= period:
            # Calculate rolling linear regression
            for i in range(period - 1, len(df)):
                # Get window
                y = df['close'].values[i - period + 1:i + 1]
                x = np.arange(period)
                
                # Normalize x
                x_norm = (x - np.mean(x)) / np.std(x) if np.std(x) != 0 else x - np.mean(x)
                
                # Calculate slope and intercept
                slope, intercept = np.polyfit(x_norm, y, 1)
                
                # Calculate R-squared
                y_pred = slope * x_norm + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Store results
                df.loc[df.index[i], 'lr_slope'] = slope
                df.loc[df.index[i], 'lr_intercept'] = intercept
                df.loc[df.index[i], 'lr_r2'] = r2
            
            # Normalize slope
            df['lr_slope_norm'] = df['lr_slope'] / df['close'] * 100
            
            # Trend classification
            df['lr_trend'] = np.where(
                df['lr_slope_norm'] > 0.02, 1,  # Uptrend
                np.where(df['lr_slope_norm'] < -0.02, -1, 0)  # Downtrend
            )
            
            # Strong trend detection
            df['lr_strong_trend'] = (
                (abs(df['lr_slope_norm']) > 0.05) & 
                (df['lr_r2'] > 0.7)
            ).astype(int)
        
        return df