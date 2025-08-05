#!/usr/bin/env python
"""
Script to monitor active trades and account status.

This script connects to the OANDA API and displays real-time information about 
open positions, account balance, and recent trades.
"""
import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate
import time
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.accounts import AccountSummary, AccountDetails
from oandapyV20.endpoints.positions import OpenPositions
from oandapyV20.endpoints.transactions import TransactionsSinceID
from oandapyV20.endpoints.pricing import PricingInfo

# Add parent directory to path so we can import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor active trades and account status')
    
    parser.add_argument('--api-key', type=str,
                        help='OANDA API key (overrides environment variable)')
    parser.add_argument('--account-id', type=str,
                        help='OANDA account ID (overrides environment variable)')
    parser.add_argument('--environment', type=str, choices=['practice', 'live'],
                        default='practice', help='OANDA environment (default: practice)')
    parser.add_argument('--refresh', type=int, default=5,
                        help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    return args

async def execute_request(api, request):
    """Execute an API request asynchronously."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: api.request(request))
    return response

async def get_account_summary(api, account_id):
    """Get account summary information."""
    try:
        response = await execute_request(api, AccountSummary(accountID=account_id))
        return response.get('account', {})
    except V20Error as e:
        print(f"Error getting account summary: {str(e)}")
        return {}

async def get_open_positions(api, account_id):
    """Get open positions."""
    try:
        response = await execute_request(api, OpenPositions(accountID=account_id))
        return response.get('positions', [])
    except V20Error as e:
        print(f"Error getting open positions: {str(e)}")
        return []

async def get_current_prices(api, account_id, instruments):
    """Get current prices for instruments."""
    results = {}
    if not instruments:
        return results
        
    try:
        params = {"instruments": ",".join(instruments)}
        response = await execute_request(api, PricingInfo(accountID=account_id, params=params))
        
        for price_info in response.get('prices', []):
            instrument = price_info.get('instrument')
            bid = float(price_info.get('bids', [{}])[0].get('price', 0))
            ask = float(price_info.get('asks', [{}])[0].get('price', 0))
            results[instrument] = {'bid': bid, 'ask': ask, 'mid': (bid + ask) / 2}
        
        return results
    except V20Error as e:
        print(f"Error getting prices: {str(e)}")
        return results

async def get_recent_transactions(api, account_id, since_id=None, limit=100):
    """Get recent transactions."""
    try:
        params = {}
        if since_id:
            params['sinceTransactionID'] = since_id
        
        params['type'] = 'ORDER_FILL,MARKET_ORDER,TAKE_PROFIT_ORDER,STOP_LOSS_ORDER'
        params['pageSize'] = limit
        
        response = await execute_request(api, TransactionsSinceID(accountID=account_id, params=params))
        return response.get('transactions', [])
    except V20Error as e:
        print(f"Error getting transactions: {str(e)}")
        return []

async def calculate_performance_metrics(api, account_id):
    """Calculate performance metrics for today's trading session."""
    try:
        # Get account summary for current balance
        account = await get_account_summary(api, account_id)
        current_balance = float(account.get('balance', 0))
        
        # Get transactions for today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Parse transactions to calculate metrics
        transactions = await get_recent_transactions(api, account_id, limit=200)
        
        today_transactions = []
        for transaction in transactions:
            time_str = transaction.get('time', '').replace('.000000000Z', '')
            timestamp = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            
            if timestamp >= today:
                today_transactions.append(transaction)
        
        # Calculate metrics
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for transaction in today_transactions:
            if transaction.get('type') == 'ORDER_FILL':
                pl = float(transaction.get('pl', 0))
                
                if pl != 0:
                    total_trades += 1
                    
                    if pl > 0:
                        winning_trades += 1
                        total_profit += pl
                    else:
                        losing_trades += 1
                        total_loss += abs(pl)
        
        # Compute metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = (total_profit / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        net_profit = total_profit - total_loss
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'net_profit': net_profit
        }
    except Exception as e:
        print(f"Error calculating performance metrics: {str(e)}")
        return {}

async def display_monitor(api, account_id, refresh_interval=5):
    """Display real-time monitor of trades and account status."""
    # Track the latest transaction ID we've seen
    latest_transaction_id = None
    
    # Track startup time to display duration
    start_time = datetime.now()
    
    while True:
        try:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Calculate runtime
            runtime = datetime.now() - start_time
            hours, remainder = divmod(runtime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"===== OANDA TRADE MONITOR =====")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Running for: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print()
            
            # Get account summary
            account = await get_account_summary(api, account_id)
            
            # Display account information
            print(f"==== Account Summary ====")
            print(f"Account ID: {account_id}")
            print(f"Balance: ${float(account.get('balance', 0)):.2f}")
            print(f"Unrealized PL: ${float(account.get('unrealizedPL', 0)):.2f}")
            print(f"Realized PL: ${float(account.get('realizedPL', 0)):.2f}")
            print(f"Margin Used: ${float(account.get('marginUsed', 0)):.2f}")
            print(f"Margin Available: ${float(account.get('marginAvailable', 0)):.2f}")
            print()
            
            # Calculate and display today's performance metrics
            metrics = await calculate_performance_metrics(api, account_id)
            
            print(f"==== Today's Performance ====")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}% ({metrics.get('winning_trades', 0)} wins, {metrics.get('losing_trades', 0)} losses)")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Net Profit: ${metrics.get('net_profit', 0):.2f}")
            print(f"Avg Win: ${metrics.get('avg_win', 0):.2f}, Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
            print()
            
            # Get open positions
            positions = await get_open_positions(api, account_id)
            
            # Get current prices for all instruments with open positions
            instruments = [p.get('instrument') for p in positions]
            prices = await get_current_prices(api, account_id, instruments)
            
            # Display open positions
            print(f"==== Open Positions ({len(positions)}) ====")
            if positions:
                position_data = []
                for position in positions:
                    instrument = position.get('instrument')
                    
                    # Get long and short units
                    long_units = int(position.get('long', {}).get('units', 0))
                    short_units = int(position.get('short', {}).get('units', 0))
                    
                    # Determine if long or short
                    is_long = long_units > 0
                    units = long_units if is_long else short_units
                    direction = "LONG" if is_long else "SHORT"
                    
                    # Get entry price
                    entry_price = float(position.get('long' if is_long else 'short', {}).get('averagePrice', 0))
                    
                    # Get current price
                    current_price = 0
                    if instrument in prices:
                        current_price = prices[instrument]['ask' if is_long else 'bid']
                    
                    # Calculate pip value (assume 0.0001 for most currency pairs)
                    pip_value = 0.0001
                    if instrument.endswith('JPY'):
                        pip_value = 0.01
                    
                    # Calculate profit/loss
                    price_diff = current_price - entry_price if is_long else entry_price - current_price
                    pips = price_diff / pip_value
                    pl = float(position.get('unrealizedPL', 0))
                    
                    # Get stop loss and take profit levels
                    stop_loss = position.get('long' if is_long else 'short', {}).get('stopLossOrder', {}).get('price', '-')
                    take_profit = position.get('long' if is_long else 'short', {}).get('takeProfitOrder', {}).get('price', '-')
                    
                    position_data.append([
                        instrument,
                        direction,
                        units,
                        f"{entry_price:.5f}",
                        f"{current_price:.5f}",
                        f"{pips:.1f}",
                        f"${pl:.2f}",
                        stop_loss,
                        take_profit
                    ])
                
                headers = ["Instrument", "Direction", "Units", "Entry", "Current", "Pips", "P/L", "Stop Loss", "Take Profit"]
                print(tabulate(position_data, headers=headers, tablefmt="pretty"))
            else:
                print("No open positions")
            print()
            
            # Get recent transactions
            if latest_transaction_id is None:
                # First run, just get the latest transaction ID
                transactions = await get_recent_transactions(api, account_id, limit=1)
                if transactions:
                    latest_transaction_id = transactions[0].get('id')
            else:
                # Get transactions since the last check
                transactions = await get_recent_transactions(api, account_id, since_id=latest_transaction_id)
                
                # Update latest transaction ID
                if transactions:
                    latest_transaction_id = transactions[0].get('id')
                
                # Display recent trading activity
                print(f"==== Recent Trading Activity ====")
                if transactions:
                    # Filter for order fills
                    fills = [t for t in transactions if t.get('type') == 'ORDER_FILL']
                    
                    if fills:
                        transaction_data = []
                        for fill in fills:
                            time_str = fill.get('time', '').replace('.000000000Z', '')
                            timestamp = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                            
                            instrument = fill.get('instrument')
                            units = int(fill.get('units', 0))
                            price = float(fill.get('price', 0))
                            pl = float(fill.get('pl', 0))
                            
                            # Determine trade type
                            trade_type = "BUY" if units > 0 else "SELL"
                            if fill.get('reason') == 'STOP_LOSS_ORDER':
                                trade_type += " (SL)"
                            elif fill.get('reason') == 'TAKE_PROFIT_ORDER':
                                trade_type += " (TP)"
                            
                            transaction_data.append([
                                timestamp.strftime('%H:%M:%S'),
                                instrument,
                                trade_type,
                                abs(units),
                                f"{price:.5f}",
                                f"${pl:.2f}"
                            ])
                        
                        headers = ["Time", "Instrument", "Type", "Units", "Price", "P/L"]
                        print(tabulate(transaction_data, headers=headers, tablefmt="pretty"))
                    else:
                        print("No recent trades")
                else:
                    print("No recent trading activity")
            
            # Wait for next refresh
            print()
            print(f"Refreshing in {refresh_interval} seconds... (Press Ctrl+C to exit)")
            await asyncio.sleep(refresh_interval)
            
        except V20Error as e:
            print(f"OANDA API error: {str(e)}")
            await asyncio.sleep(30)  # Wait longer on API errors
        except KeyboardInterrupt:
            print("\nMonitor stopped by user.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            await asyncio.sleep(30)  # Wait longer on other errors

async def main():
    """Main function."""
    # Load environment variables from .env file (if exists)
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get API credentials
    api_key = args.api_key or os.environ.get('OANDA_API_KEY')
    account_id = args.account_id or os.environ.get('OANDA_ACCOUNT_ID')
    environment = args.environment or os.environ.get('OANDA_ENVIRONMENT', 'practice')
    
    # Verify required credentials
    if not api_key:
        print("Error: OANDA API key not provided")
        return
    
    if not account_id:
        print("Error: OANDA account ID not provided")
        return
    
    # Initialize API
    api = API(access_token=api_key, environment=environment)
    
    # Start monitor
    try:
        await display_monitor(api, account_id, args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")

if __name__ == "__main__":
    asyncio.run(main())