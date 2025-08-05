"""
Market Data Adapter for OANDA Trading System

This module handles retrieving and preprocessing candle data from OANDA.
"""
import asyncio
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.instruments import InstrumentsCandles

import config

logger = logging.getLogger(__name__)

class MarketDataAdapter:
    """Handles retrieving and preprocessing market data from OANDA."""
    
    def __init__(self):
        """Initialize the Market Data Adapter."""
        self.api = API(access_token=config.OANDA_API_KEY, environment=config.OANDA_ENVIRONMENT)
        self.timeframes = config.TIMEFRAMES
        self.candle_data = {}  # Structure: {instrument: {timeframe: dataframe}}
        self.last_fetch_time = {}  # Track last fetch time for each instrument/timeframe
        
    async def fetch_candles(self, instrument, granularity, count):
        """
        Fetch candles for a specific instrument and timeframe.
        
        Args:
            instrument (str): The instrument to fetch (e.g., 'EUR_USD')
            granularity (str): The granularity of candles (e.g., 'S30', 'M1', 'M5')
            count (int): Number of candles to retrieve
            
        Returns:
            pd.DataFrame: DataFrame containing the candle data
        """
        # For faster timeframes, optimize the count to reduce data transfer
        if granularity == "S30":
            optimized_count = min(count, 500)  # Reduce count for 30-second data
        else:
            optimized_count = count
            
        params = {
            "granularity": granularity,
            "count": optimized_count,
            "price": "M"  # Midpoint data
        }
        
        # Check if we can use fromTime to optimize data retrieval
        cache_key = f"{instrument}_{granularity}"
        if cache_key in self.last_fetch_time and self.candle_data.get(instrument, {}).get(granularity) is not None:
            # Calculate how many candles we expect since last fetch
            last_time = self.last_fetch_time[cache_key]
            seconds_since_last = (datetime.now() - last_time).total_seconds()
            
            # Only use fromTime optimization if we have recent data
            if seconds_since_last < 900:  # Less than 15 minutes
                # Get the most recent timestamp and add a small buffer
                latest_candle = self.candle_data[instrument][granularity].index[-1]
                # Convert to OANDA format
                from_time = latest_candle.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
                params["from"] = from_time
                del params["count"]  # Remove count when using fromTime
                
                logger.debug(f"Using fromTime optimization for {instrument} {granularity}")
        
        for attempt in range(config.MAX_API_RETRIES):
            try:
                request = InstrumentsCandles(instrument=instrument, params=params)
                response = await self._execute_request(request)
                
                # Extract candle data
                candles = response.get('candles', [])
                if not candles:
                    logger.warning(f"No candles returned for {instrument} {granularity}")
                    return None
                
                # Convert to DataFrame
                df = self._process_candles(candles)
                
                # Save fetch time
                self.last_fetch_time[cache_key] = datetime.now()
                
                # If we already have data in the cache, append new data
                if instrument in self.candle_data and granularity in self.candle_data[instrument]:
                    existing_df = self.candle_data[instrument][granularity]
                    if not existing_df.empty and not df.empty:
                        # Combine DataFrames and drop duplicates
                        combined_df = pd.concat([existing_df, df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        # Sort by index
                        combined_df.sort_index(inplace=True)
                        # Keep only the most recent 'count' candles
                        if len(combined_df) > count:
                            combined_df = combined_df.iloc[-count:]
                        df = combined_df
                
                # Only log if we want to debug data fetching issues
                if config.LOGGING_CONFIG.get('LOG_DATA_FETCH_SUCCESS', False):
                    logger.info(f"Successfully fetched {len(df)} {granularity} candles for {instrument}")
                return df
                
            except V20Error as e:
                logger.error(f"OANDA API error for {instrument} {granularity}: {str(e)}")
                if attempt < config.MAX_API_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    return None
            except Exception as e:
                logger.error(f"Error fetching candles for {instrument} {granularity}: {str(e)}")
                return None
    
    async def _execute_request(self, request):
        """
        Execute an API request asynchronously.
        
        Args:
            request: OANDA API request object
            
        Returns:
            dict: API response
        """
        # We need to wrap the synchronous API call in a thread to make it async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.api.request(request))
        return response
    
    def _process_candles(self, candles):
        """
        Process raw candle data from OANDA into a pandas DataFrame.
        
        Args:
            candles (list): List of candle dicts from OANDA API
            
        Returns:
            pd.DataFrame: Processed candle data
        """
        data = []
        
        for candle in candles:
            # Skip incomplete candles
            if candle.get('complete', False) is False:
                continue
                
            # Extract OHLC data
            time_str = candle['time'].replace('.000000000Z', '')
            timestamp = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            
            # Get bid and ask if available
            if 'bid' in candle:
                mid = candle['bid']
            elif 'ask' in candle:
                mid = candle['ask']
            else:
                mid = candle['mid']
            
            data.append({
                'timestamp': timestamp,
                'open': float(mid['o']),
                'high': float(mid['h']),
                'low': float(mid['l']),
                'close': float(mid['c']),
                'volume': int(candle['volume'])
            })
        
        # Create DataFrame and set timestamp as index
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
        return df
    
    async def get_all_timeframe_data(self, instruments=None):
        """
        Fetch data for all configured timeframes for specified instruments.
        
        Args:
            instruments (list, optional): List of instruments to fetch data for.
                If None, fetch data for all enabled instruments.
        
        Returns:
            dict: Nested dictionary with instrument and timeframe keys
                  {instrument: {timeframe: dataframe}}
        """
        # If no instruments specified, use all enabled instruments
        if instruments is None:
            instruments = [inst for inst, details in config.INSTRUMENTS.items() 
                          if details.get('enabled', True)]
        elif isinstance(instruments, str):
            # If a single instrument is passed as a string, convert to list
            instruments = [instruments]
            
        tasks = []
        for instrument in instruments:
            for tf_key, tf_config in self.timeframes.items():
                task = self.fetch_candles(instrument, tf_config['granularity'], tf_config['count'])
                tasks.append((instrument, tf_key, task))
        
        # Run all fetch tasks concurrently
        results = {}
        for instrument, tf_key, task in tasks:
            if instrument not in results:
                results[instrument] = {}
                
            results[instrument][tf_key] = await task
            
        # Update the cached data
        for instrument, timeframes in results.items():
            if instrument not in self.candle_data:
                self.candle_data[instrument] = {}
                
            for tf_key, df in timeframes.items():
                if df is not None:
                    self.candle_data[instrument][tf_key] = df
        
        return self.candle_data