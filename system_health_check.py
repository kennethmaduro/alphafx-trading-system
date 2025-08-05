#!/usr/bin/env python
"""
System Health Check for Trading System
Verifies configuration, connectivity, and readiness for trading
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.accounts import AccountSummary
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.endpoints.instruments import InstrumentsCandles
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from technical_indicators import TechnicalIndicators
from cep_engine import CEPEngine, MarketCondition

class SystemHealthCheck:
    """Comprehensive health check for the trading system."""
    
    def __init__(self):
        self.api = API(access_token=config.OANDA_API_KEY, environment=config.OANDA_ENVIRONMENT)
        self.account_id = config.OANDA_ACCOUNT_ID
        self.check_results = {}
        self.warnings = []
        self.errors = []
    
    async def run_all_checks(self):
        """Run all system health checks."""
        print("=" * 60)
        print("TRADING SYSTEM HEALTH CHECK")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run checks
        await self.check_api_connectivity()
        await self.check_account_status()
        await self.check_instrument_availability()
        await self.check_market_data()
        await self.check_indicators()
        await self.check_signal_generation()
        self.check_risk_parameters()
        self.check_file_system()
        
        # Summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    async def check_api_connectivity(self):
        """Check OANDA API connectivity."""
        print("\n1. Checking API Connectivity...")
        try:
            response = await self._execute_request(AccountSummary(accountID=self.account_id))
            if response:
                print("   ✓ OANDA API connection successful")
                print(f"   ✓ Environment: {config.OANDA_ENVIRONMENT}")
                self.check_results['api_connectivity'] = True
            else:
                self.errors.append("Failed to connect to OANDA API")
                self.check_results['api_connectivity'] = False
        except V20Error as e:
            print(f"   ✗ API Error: {str(e)}")
            self.errors.append(f"API Error: {str(e)}")
            self.check_results['api_connectivity'] = False
        except Exception as e:
            print(f"   ✗ Connection Error: {str(e)}")
            self.errors.append(f"Connection Error: {str(e)}")
            self.check_results['api_connectivity'] = False
    
    async def check_account_status(self):
        """Check account status and balance."""
        print("\n2. Checking Account Status...")
        if not self.check_results.get('api_connectivity', False):
            print("   ⚠ Skipping - API not connected")
            return
        
        try:
            response = await self._execute_request(AccountSummary(accountID=self.account_id))
            account = response.get('account', {})
            
            balance = float(account.get('balance', 0))
            margin_available = float(account.get('marginAvailable', 0))
            margin_used = float(account.get('marginUsed', 0))
            open_positions = int(account.get('openPositionCount', 0))
            open_trades = int(account.get('openTradeCount', 0))
            
            print(f"   ✓ Account ID: {self.account_id}")
            print(f"   ✓ Balance: ${balance:,.2f}")
            print(f"   ✓ Margin Available: ${margin_available:,.2f}")
            print(f"   ✓ Margin Used: ${margin_used:,.2f}")
            print(f"   ✓ Open Positions: {open_positions}")
            print(f"   ✓ Open Trades: {open_trades}")
            
            # Warnings
            if balance < 1000:
                self.warnings.append(f"Low account balance: ${balance:.2f}")
                print(f"   ⚠ Warning: Low account balance")
            
            if margin_available < balance * 0.5:
                self.warnings.append("Low margin available")
                print(f"   ⚠ Warning: Low margin available")
            
            self.check_results['account_status'] = True
            self.check_results['account_balance'] = balance
            
        except Exception as e:
            print(f"   ✗ Error checking account: {str(e)}")
            self.errors.append(f"Account check error: {str(e)}")
            self.check_results['account_status'] = False
    
    async def check_instrument_availability(self):
        """Check if configured instruments are tradeable."""
        print("\n3. Checking Instrument Availability...")
        if not self.check_results.get('api_connectivity', False):
            print("   ⚠ Skipping - API not connected")
            return
        
        enabled_instruments = [inst for inst, details in config.INSTRUMENTS.items() 
                             if details.get('enabled', True)]
        
        tradeable_instruments = []
        spread_info = []
        
        for instrument in enabled_instruments:
            try:
                # Get current pricing
                params = {"instruments": instrument}
                response = await self._execute_request(
                    PricingInfo(accountID=self.account_id, params=params)
                )
                
                prices = response.get('prices', [])
                if prices:
                    price_data = prices[0]
                    status = price_data.get('status', 'unknown')
                    
                    if status == 'tradeable':
                        bid = float(price_data.get('bids', [{}])[0].get('price', 0))
                        ask = float(price_data.get('asks', [{}])[0].get('price', 0))
                        spread_pips = (ask - bid) / config.INSTRUMENTS[instrument]['pip_value']
                        
                        tradeable_instruments.append(instrument)
                        spread_info.append({
                            'Instrument': instrument,
                            'Bid': f"{bid:.5f}",
                            'Ask': f"{ask:.5f}",
                            'Spread (pips)': f"{spread_pips:.1f}",
                            'Status': '✓ Tradeable'
                        })
                        
                        if spread_pips > config.SIGNAL_FILTERS['SPREAD_FILTER_PIPS']:
                            self.warnings.append(f"{instrument} spread too high: {spread_pips:.1f} pips")
                    else:
                        spread_info.append({
                            'Instrument': instrument,
                            'Status': f'✗ {status}'
                        })
                        self.warnings.append(f"{instrument} not tradeable: {status}")
                        
            except Exception as e:
                spread_info.append({
                    'Instrument': instrument,
                    'Status': f'✗ Error: {str(e)}'
                })
                self.warnings.append(f"{instrument} check failed: {str(e)}")
        
        print(f"\n   Enabled instruments: {len(enabled_instruments)}")
        print(f"   Tradeable instruments: {len(tradeable_instruments)}")
        
        if spread_info:
            print("\n   Current spreads:")
            print(tabulate(spread_info, headers='keys', tablefmt='pretty'))
        
        self.check_results['tradeable_instruments'] = tradeable_instruments
    
    async def check_market_data(self):
        """Check market data availability and quality."""
        print("\n4. Checking Market Data...")
        if not self.check_results.get('api_connectivity', False):
            print("   ⚠ Skipping - API not connected")
            return
        
        # Test with EUR_USD
        test_instrument = "EUR_USD"
        
        for tf_key, tf_config in config.TIMEFRAMES.items():
            try:
                params = {
                    "granularity": tf_config['granularity'],
                    "count": 50,
                    "price": "M"
                }
                
                request = InstrumentsCandles(instrument=test_instrument, params=params)
                response = await self._execute_request(request)
                
                candles = response.get('candles', [])
                complete_candles = [c for c in candles if c.get('complete', False)]
                
                print(f"   ✓ {tf_key}: {len(complete_candles)} candles available")
                
                if len(complete_candles) < 20:
                    self.warnings.append(f"Insufficient {tf_key} data: only {len(complete_candles)} candles")
                    
            except Exception as e:
                print(f"   ✗ {tf_key}: Error - {str(e)}")
                self.errors.append(f"Market data error for {tf_key}: {str(e)}")
        
        self.check_results['market_data'] = True
    
    async def check_indicators(self):
        """Check technical indicator calculations."""
        print("\n5. Checking Technical Indicators...")
        
        try:
            # Create sample data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
            sample_data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 1.1000,
                'high': np.random.randn(100).cumsum() + 1.1010,
                'low': np.random.randn(100).cumsum() + 1.0990,
                'close': np.random.randn(100).cumsum() + 1.1000,
                'volume': np.random.randint(1000, 5000, 100)
            }, index=dates)
            
            # Calculate indicators
            indicators = TechnicalIndicators()
            result = indicators.add_all_indicators(sample_data)
            
            # Check key indicators
            required_indicators = [
                'ema_fast', 'ema_medium', 'ema_slow',
                'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'adx', 'atr'
            ]
            
            missing_indicators = []
            for indicator in required_indicators:
                if indicator not in result.columns:
                    missing_indicators.append(indicator)
                elif result[indicator].isna().all():
                    missing_indicators.append(f"{indicator} (all NaN)")
            
            if missing_indicators:
                print(f"   ✗ Missing indicators: {', '.join(missing_indicators)}")
                self.errors.append(f"Missing indicators: {', '.join(missing_indicators)}")
            else:
                print(f"   ✓ All {len(required_indicators)} core indicators calculated successfully")
                print(f"   ✓ Total indicators available: {len(result.columns)}")
            
            self.check_results['indicators'] = len(missing_indicators) == 0
            
        except Exception as e:
            print(f"   ✗ Error calculating indicators: {str(e)}")
            self.errors.append(f"Indicator calculation error: {str(e)}")
            self.check_results['indicators'] = False
    
    async def check_signal_generation(self):
        """Check signal generation engine."""
        print("\n6. Checking Signal Generation...")
        
        try:
            # Create CEP engine
            cep_engine = CEPEngine()
            
            # Create sample multi-timeframe data
            timeframe_data = {}
            
            for tf_key in config.TIMEFRAMES:
                dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
                df = pd.DataFrame({
                    'open': np.random.randn(200).cumsum() + 1.1000,
                    'high': np.random.randn(200).cumsum() + 1.1010,
                    'low': np.random.randn(200).cumsum() + 1.0990,
                    'close': np.random.randn(200).cumsum() + 1.1000,
                    'volume': np.random.randint(1000, 5000, 200)
                }, index=dates)
                
                # Add indicators
                indicators = TechnicalIndicators()
                timeframe_data[tf_key] = indicators.add_all_indicators(df)
            
            # Generate signal
            signal_type, signal_strength, market_condition, indicator_data = (
                cep_engine.process_timeframe_data(timeframe_data)
            )
            
            print(f"   ✓ Signal generated: {signal_type.value}")
            print(f"   ✓ Signal strength: {signal_strength:.2f}")
            print(f"   ✓ Market condition: {market_condition.value}")
            
            self.check_results['signal_generation'] = True
            
        except Exception as e:
            print(f"   ✗ Error generating signals: {str(e)}")
            self.errors.append(f"Signal generation error: {str(e)}")
            self.check_results['signal_generation'] = False
    
    def check_risk_parameters(self):
        """Check risk management parameters."""
        print("\n7. Checking Risk Parameters...")
        
        # Check risk limits
        checks = [
            ('Default risk per trade', config.DEFAULT_RISK_PER_TRADE, 0.005, 0.02),
            ('Max risk per trade', config.MAX_RISK_PER_TRADE, 0.01, 0.05),
            ('Drawdown threshold', config.DRAWDOWN_THRESHOLD, 0.02, 0.05),
            ('Critical drawdown', config.CRITICAL_DRAWDOWN, 0.05, 0.10),
            ('Max total risk', config.MAX_TOTAL_RISK, 0.02, 0.05),
        ]
        
        all_valid = True
        for name, value, min_val, max_val in checks:
            if min_val <= value <= max_val:
                print(f"   ✓ {name}: {value:.1%}")
            else:
                print(f"   ✗ {name}: {value:.1%} (outside range {min_val:.1%}-{max_val:.1%})")
                self.warnings.append(f"{name} outside recommended range")
                all_valid = False
        
        # Check leverage
        if config.DEFAULT_LEVERAGE <= 10:
            print(f"   ✓ Leverage: {config.DEFAULT_LEVERAGE}:1")
        else:
            print(f"   ⚠ Leverage: {config.DEFAULT_LEVERAGE}:1 (high)")
            self.warnings.append(f"High leverage: {config.DEFAULT_LEVERAGE}:1")
        
        self.check_results['risk_parameters'] = all_valid
    
    def check_file_system(self):
        """Check file system and directories."""
        print("\n8. Checking File System...")
        
        directories = ['logs', 'trade_logs']
        all_exist = True
        
        for directory in directories:
            if os.path.exists(directory):
                print(f"   ✓ Directory '{directory}' exists")
                
                # Check write permissions
                test_file = os.path.join(directory, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    print(f"   ✓ Write permissions OK for '{directory}'")
                except Exception as e:
                    print(f"   ✗ Cannot write to '{directory}': {str(e)}")
                    self.errors.append(f"Cannot write to '{directory}'")
                    all_exist = False
            else:
                print(f"   ✗ Directory '{directory}' not found")
                self.warnings.append(f"Directory '{directory}' not found")
                
                # Try to create it
                try:
                    os.makedirs(directory)
                    print(f"   ✓ Created directory '{directory}'")
                except Exception as e:
                    print(f"   ✗ Cannot create '{directory}': {str(e)}")
                    self.errors.append(f"Cannot create '{directory}'")
                    all_exist = False
        
        self.check_results['file_system'] = all_exist
    
    def print_summary(self):
        """Print summary of health check results."""
        print("\n" + "=" * 60)
        print("HEALTH CHECK SUMMARY")
        print("=" * 60)
        
        # Overall status
        total_checks = len(self.check_results)
        passed_checks = sum(1 for v in self.check_results.values() if v)
        
        if len(self.errors) == 0:
            print(f"\n✅ SYSTEM READY FOR TRADING")
        else:
            print(f"\n❌ SYSTEM NOT READY - {len(self.errors)} ERRORS FOUND")
        
        print(f"\nChecks passed: {passed_checks}/{total_checks}")
        
        # Errors
        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   ❌ {error}")
        
        # Warnings
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        # Trading recommendations
        print("\nRECOMMENDATIONS:")
        
        if len(self.errors) > 0:
            print("   1. Fix all errors before starting trading")
            print("   2. Check API credentials and network connection")
            print("   3. Ensure all required packages are installed")
        elif len(self.warnings) > 5:
            print("   1. Review warnings and adjust configuration if needed")
            print("   2. Consider starting with reduced risk")
            print("   3. Monitor closely during initial trading")
        else:
            print("   1. System is ready for trading")
            print("   2. Start with small position sizes")
            print("   3. Monitor performance closely")
            
            if self.check_results.get('account_balance', 0) < 5000:
                print("   4. Consider using micro lots due to account size")
    
    async def _execute_request(self, request):
        """Execute API request asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.api.request(request))
        return response

async def main():
    """Main function."""
    print("Starting Trading System Health Check...\n")
    
    # Create health checker
    checker = SystemHealthCheck()
    
    # Run all checks
    success = await checker.run_all_checks()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Health check PASSED - System ready for trading!")
    else:
        print("❌ Health check FAILED - Please fix errors before trading")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    # Run the health check
    result = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)