#!/usr/bin/env python
"""
Script to start the Multi-Timeframe Automated Trading System.

This script sets up the environment, loads credentials, and starts the trading system.
"""
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Add parent directory to path so we can import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import TradingSystem, setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Multi-Timeframe Automated Trading System')
    
    parser.add_argument('--practice', action='store_true', default=True,
                        help='Run in practice mode (default)')
    parser.add_argument('--live', action='store_true',
                        help='Run in live mode (USE WITH CAUTION)')
    parser.add_argument('--api-key', type=str,
                        help='OANDA API key (overrides environment variable)')
    parser.add_argument('--account-id', type=str,
                        help='OANDA account ID (overrides environment variable)')
    parser.add_argument('--instruments', type=str,
                        help='Comma-separated list of instruments to trade (e.g., EUR_USD,GBP_USD)')
    parser.add_argument('--weekend', action='store_true',
                        help='Enable weekend trading (not recommended)')
    parser.add_argument('--risk', type=float, 
                        help='Risk per trade as a percentage (e.g., 2.0 for 2%)')
    parser.add_argument('--max-risk', type=float,
                        help='Maximum risk per trade as a percentage')
    parser.add_argument('--total-risk', type=float,
                        help='Maximum total risk across all instruments as a percentage')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--update-interval', type=int, default=30,
                        help='Update interval in seconds (default: 30)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Run duration in minutes (default: run indefinitely)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.live and args.practice:
        parser.error("Cannot specify both --practice and --live")
    
    if args.risk is not None and (args.risk <= 0 or args.risk > 10):
        parser.error("Risk percentage must be between 0 and 10")
    
    if args.max_risk is not None and (args.max_risk <= 0 or args.max_risk > 10):
        parser.error("Maximum risk percentage must be between 0 and 10")
    
    if args.total_risk is not None and (args.total_risk <= 0 or args.total_risk > 20):
        parser.error("Total risk percentage must be between 0 and 20")
    
    if args.update_interval is not None and (args.update_interval < 1 or args.update_interval > 300):
        parser.error("Update interval must be between 1 and 300 seconds")
    
    if args.duration is not None and args.duration <= 0:
        parser.error("Duration must be a positive number of minutes")
    
    return args

def get_available_instruments():
    """Get a list of available instruments from config."""
    import config
    available = []
    for instrument, details in config.INSTRUMENTS.items():
        available.append(instrument)
    return available

async def main():
    """Main function."""
    # Load environment variables from .env file (if exists)
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    
    # Update environment based on command line arguments
    if args.api_key:
        os.environ['OANDA_API_KEY'] = args.api_key
    
    if args.account_id:
        os.environ['OANDA_ACCOUNT_ID'] = args.account_id
    
    if args.live:
        os.environ['OANDA_ENVIRONMENT'] = 'live'
        logger.warning("!!! RUNNING IN LIVE MODE - REAL MONEY WILL BE TRADED !!!")
    else:
        os.environ['OANDA_ENVIRONMENT'] = 'practice'
        logger.info("Running in practice mode")
    
    # Update config values from command line
    import config
    
    # Update update interval if specified
    if args.update_interval:
        config.UPDATE_INTERVAL = args.update_interval
    
    # Enable specific instruments
    if args.instruments:
        requested_instruments = args.instruments.split(',')
        # Get available instruments
        available_instruments = get_available_instruments()
        
        # Validate requested instruments
        for instrument in requested_instruments:
            if instrument not in available_instruments:
                logger.warning(f"Unknown instrument: {instrument}. Available instruments: {', '.join(available_instruments)}")
                return
        
        # Disable all instruments by default
        for instrument in config.INSTRUMENTS:
            config.INSTRUMENTS[instrument]['enabled'] = False
        
        # Enable only the requested instruments
        for instrument in requested_instruments:
            if instrument in config.INSTRUMENTS:
                config.INSTRUMENTS[instrument]['enabled'] = True
                logger.info(f"Enabled trading for {instrument}")
    
    # Update risk parameters
    if args.risk is not None:
        config.DEFAULT_RISK_PER_TRADE = args.risk / 100  # Convert percentage to decimal
    
    if args.max_risk is not None:
        config.MAX_RISK_PER_TRADE = args.max_risk / 100  # Convert percentage to decimal
    
    if args.total_risk is not None:
        config.MAX_TOTAL_RISK = args.total_risk / 100  # Convert percentage to decimal
    
    # Update weekend trading settings
    if args.weekend:
        config.WEEKEND_TRADING['ENABLED'] = True
        logger.warning("Weekend trading enabled (use with caution - reduced liquidity and potential gaps)")
    
    config.LOGGING_LEVEL = args.log_level
    
    # Get list of enabled instruments
    enabled_instruments = [inst for inst, details in config.INSTRUMENTS.items() 
                         if details.get('enabled', True)]
    
    # Print configuration summary
    logger.info("=== Configuration Summary ===")
    logger.info(f"OANDA Environment: {os.environ.get('OANDA_ENVIRONMENT', 'Unknown')}")
    logger.info(f"Account ID: {os.environ.get('OANDA_ACCOUNT_ID', 'Unknown')}")
    logger.info(f"Enabled Instruments: {', '.join(enabled_instruments)}")
    logger.info(f"Risk per trade: {config.DEFAULT_RISK_PER_TRADE:.2%}")
    logger.info(f"Max risk per trade: {config.MAX_RISK_PER_TRADE:.2%}")
    logger.info(f"Total risk limit: {config.MAX_TOTAL_RISK:.2%}")
    logger.info(f"Weekend trading: {'Enabled' if config.WEEKEND_TRADING['ENABLED'] else 'Disabled'}")
    logger.info(f"Update interval: {config.UPDATE_INTERVAL} seconds")
    logger.info(f"Timeframes: {', '.join(config.TIMEFRAMES.keys())}")
    logger.info(f"Run duration: {args.duration if args.duration else 'Indefinite'} minutes")
    logger.info("===========================")
    
    # Verify required environment variables
    if not os.environ.get('OANDA_API_KEY'):
        logger.error("OANDA API key not provided. Exiting.")
        return
    
    if not os.environ.get('OANDA_ACCOUNT_ID'):
        logger.error("OANDA account ID not provided. Exiting.")
        return
    
    # Request user confirmation for live mode
    if os.environ.get('OANDA_ENVIRONMENT') == 'live':
        confirmation = input("You are about to trade with REAL MONEY. Type 'I CONFIRM' to proceed: ")
        if confirmation != "I CONFIRM":
            logger.info("Live mode not confirmed. Exiting.")
            return
    
    # Create and start the trading system
    system = TradingSystem(run_duration=args.duration)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping gracefully...")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    finally:
        await system.stop()
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())