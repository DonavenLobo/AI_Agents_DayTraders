import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Type, List, Optional, Literal # Added Literal

# Import crewai BaseTool and Pydantic
from crewai.tools import BaseTool
from pydantic.v1 import BaseModel, Field, ConfigDict # Use pydantic.v1 if CrewAI requires it, else use pydantic

# Import alpaca-py modules
try:
    import alpaca
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest,
        TrailingStopOrderRequest, TakeProfitRequest, StopLossRequest,
        GetOrdersRequest, GetAssetsRequest, ClosePositionRequest
    )
    from alpaca.trading.enums import (
        OrderSide, OrderType, TimeInForce, OrderClass, QueryOrderStatus, AssetStatus
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.requests import StockBarsRequest
    from alpaca.common.exceptions import APIError
except ImportError:
    raise ImportError("Please install alpaca-py: pip install alpaca-py")

# Configure logging (can be configured globally in your main script)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use a logger instance

# Load environment variables (consider loading once in your main script)
load_dotenv()

# --- Helper Function/Class for Client Initialization (Optional but recommended) ---
# This avoids initializing clients in every tool instance if tools are created frequently.
# Alternatively, initialize clients once in main.py/crew.py and pass them in.

_trading_client = None
_stock_data_client = None

def get_alpaca_clients():
    """Initializes and returns Alpaca clients (singleton pattern)."""
    global _trading_client, _stock_data_client
    if _trading_client is None or _stock_data_client is None:
        try:
            api_key = os.getenv("ALPACA_API_KEY_ID")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            paper_trading = os.getenv("ALPACA_PAPER", "true").lower() == "true"

            if not api_key or not secret_key:
                raise ValueError("Alpaca API Key ID or Secret Key not found.")

            log.info(f"Initializing Alpaca clients (Paper Trading: {paper_trading})")
            _trading_client = TradingClient(api_key, secret_key, paper=paper_trading)
            _stock_data_client = StockHistoricalDataClient(api_key, secret_key)

            # Verify connection
            account = _trading_client.get_account()
            log.info(f"Alpaca Trading client initialized. Account Status: {account.status}")

        except APIError as e:
            log.error(f"Alpaca API Error during initialization: {e}")
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}") from e
        except Exception as e:
            log.error(f"Failed to initialize Alpaca clients: {e}")
            raise RuntimeError(f"Error initializing Alpaca clients: {e}") from e
    return _trading_client, _stock_data_client


# --- Tool Definition Base Class (MODIFIED) ---
class BaseAlpacaTool(BaseTool):
    """Base class for Alpaca tools to handle client injection and Pydantic config."""
    # Define fields for clients
    trading_client: TradingClient = None
    stock_data_client: StockHistoricalDataClient = None

    # Add Pydantic Config to allow arbitrary types like TradingClient
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
         super().__init__(**kwargs)
         # Ensure clients are initialized and assigned
         if self.trading_client is None or self.stock_data_client is None:
              self.trading_client, self.stock_data_client = get_alpaca_clients()


# --- Individual Tool Classes ---

# === Account Information Tools ===

class GetAccountDetailsToolInput(BaseModel):
    """Input schema is empty as this tool takes no arguments."""
    pass

class GetAccountDetailsTool(BaseAlpacaTool):
    name: str = "Get Alpaca Account Details"
    description: str = (
        "Fetches current Alpaca paper trading account details (status, equity, buying power, cash, etc.). "
        "Use this to check available funds or account status."
    )
    args_schema: Type[BaseModel] = GetAccountDetailsToolInput

    def _run(self) -> str:
        """Executes the tool to get account details."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.info("Fetching Alpaca account details...")
            account = self.trading_client.get_account()
            log.info(f"Account details fetched: Status={account.status}, Equity={account.equity}")
            return (
                f"Account Status: {account.status}, ID: {account.account_number}, Currency: {account.currency}, "
                f"Equity: ${account.equity}, Buying Power: ${account.buying_power}, Cash: ${account.cash}, "
                f"Portfolio Value: ${account.portfolio_value}, Daytrade Count: {account.daytrade_count}, "
                f"Pattern Day Trader: {account.pattern_day_trader}"
            )
        except APIError as e:
            log.error(f"API Error fetching account details: {e}")
            return f"Error fetching account details: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching account details: {e}")
            return f"Unexpected error fetching account details: {e}"

class GetPositionsToolInput(BaseModel):
    """Input schema is empty as this tool takes no arguments."""
    pass

class GetPositionsTool(BaseAlpacaTool):
    name: str = "Get Current Alpaca Positions"
    description: str = (
        "Fetches all currently open positions in the Alpaca paper trading account. "
        "Use this to see what stocks are currently held, their quantities, and profit/loss."
    )
    args_schema: Type[BaseModel] = GetPositionsToolInput

    def _run(self) -> str:
        """Executes the tool to get open positions."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.info("Fetching current Alpaca positions...")
            positions = self.trading_client.get_all_positions()
            if not positions:
                log.info("No open positions found.")
                return "No open positions."

            position_summaries = []
            for pos in positions:
                summary = (
                    f"Symbol: {pos.symbol}, Asset ID: {pos.asset_id}, Qty: {pos.qty}, Side: {pos.side}, "
                    f"Market Value: ${pos.market_value}, Avg Entry Price: ${pos.avg_entry_price}, "
                    f"Unrealized P/L: ${pos.unrealized_pl} ({pos.unrealized_plpc:.4f}%), "
                    f"Cost Basis: ${pos.cost_basis}"
                )
                position_summaries.append(summary)
            log.info(f"Found {len(positions)} open positions.")
            return "\n".join(position_summaries)
        except APIError as e:
            log.error(f"API Error fetching positions: {e}")
            return f"Error fetching positions: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching positions: {e}")
            return f"Unexpected error fetching positions: {e}"


# === Market Data Tools ===

class GetStockBarsToolInput(BaseModel):
    """Input schema for GetStockBarsTool."""
    symbol: str = Field(..., description="The stock ticker symbol (e.g., 'AAPL', 'TSLA').")
    timeframe_str: Literal['1Min', '5Min', '15Min', '1Hour', '1Day'] = Field(default='1Min', description="Time interval for bars ('1Min', '5Min', '15Min', '1Hour', '1Day').")
    limit: int = Field(default=100, description="Maximum number of bars to return (default 100, max depends on subscription).")
    start_days_ago: int = Field(default=5, description="How many days back from now to start fetching data (default 5).")

class GetStockBarsTool(BaseAlpacaTool):
    name: str = "Get Historical Stock Bars"
    description: str = (
        "Fetches recent historical stock bar data (OHLCV - Open, High, Low, Close, Volume) for a specific symbol. "
        "Useful for technical analysis or understanding recent price action."
    )
    args_schema: Type[BaseModel] = GetStockBarsToolInput

    def _run(self, symbol: str, timeframe_str: str, limit: int, start_days_ago: int) -> str:
        """Executes the tool to fetch stock bars."""
        if not self.stock_data_client:
            return "Error: Alpaca Stock Data client not initialized."

        timeframe_map = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute), '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute), '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
            '1Day': TimeFrame(1, TimeFrameUnit.Day)
        }
        timeframe = timeframe_map.get(timeframe_str) # Already validated by Pydantic Literal

        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            start_time = now - timedelta(days=start_days_ago)
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol], timeframe=timeframe, start=start_time,
                limit=limit, adjustment='raw'
            )
            log.info(f"Fetching {timeframe_str} bars for {symbol} (limit {limit}, start {start_days_ago} days ago)...")
            bars = self.stock_data_client.get_stock_bars(request_params)

            if not bars or symbol not in bars.data or not bars.data[symbol]:
                 log.warning(f"No bar data found for {symbol} with the specified parameters.")
                 return f"No bar data found for {symbol} with timeframe {timeframe_str}."

            bar_list = bars.data[symbol]
            summary_lines = [f"Fetched {len(bar_list)} bars for {symbol}:"]
            bars_to_show = bar_list if len(bar_list) <= 4 else bar_list[:2] + bar_list[-2:]
            if len(bar_list) > 4:
                 summary_lines.append("(Showing first 2 and last 2)")
            for bar in bars_to_show:
                 summary_lines.append(f"  T={bar.timestamp}, O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}")

            log.info(f"Successfully fetched {len(bar_list)} bars for {symbol}.")
            return "\n".join(summary_lines)

        except APIError as e:
            log.error(f"API Error fetching stock bars for {symbol}: {e}")
            return f"Error fetching stock bars for {symbol}: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching stock bars for {symbol}: {e}")
            return f"Unexpected error fetching stock bars for {symbol}: {e}"


# === Order Management Tools ===

class SubmitOrderToolInput(BaseModel):
    """Input schema for SubmitOrderTool."""
    symbol: str = Field(..., description="Ticker symbol of the stock to trade.")
    side: Literal['buy', 'sell'] = Field(..., description="Order side: 'buy' or 'sell'.")
    order_type: Literal['market', 'limit', 'stop', 'stop_limit', 'trailing_stop'] = Field(default='market', description="Type of order.")
    time_in_force: Literal['day', 'gtc', 'opg', 'cls', 'ioc', 'fok'] = Field(default='day', description="Time the order remains active.")
    qty: Optional[float] = Field(default=None, description="Number of shares to trade. Specify EITHER 'qty' OR 'notional'.")
    notional: Optional[float] = Field(default=None, description="Dollar amount to trade (for market/limit DAY orders). Specify EITHER 'qty' OR 'notional'.")
    limit_price: Optional[float] = Field(default=None, description="Required price for 'limit' and 'stop_limit' orders.")
    stop_price: Optional[float] = Field(default=None, description="Required price for 'stop' and 'stop_limit' orders.")
    trail_percent: Optional[float] = Field(default=None, description="Required percentage offset for 'trailing_stop' orders (e.g., 1.5 for 1.5%). Specify EITHER 'trail_percent' OR 'trail_price'.")
    trail_price: Optional[float] = Field(default=None, description="Required price offset for 'trailing_stop' orders. Specify EITHER 'trail_percent' OR 'trail_price'.")
    stop_loss_price: Optional[float] = Field(default=None, description="Creates a bracket/OTO order with a stop loss at this price.")
    take_profit_price: Optional[float] = Field(default=None, description="Creates a bracket order with a take profit limit order at this price.")

class SubmitOrderTool(BaseAlpacaTool):
    name: str = "Submit Alpaca Trading Order"
    description: str = (
        "Submits a trading order (buy/sell) to the Alpaca paper trading account. Supports various order types "
        "(market, limit, stop, etc.) and complex orders (stop loss, take profit, bracket). Ensure all required "
        "price fields (limit_price, stop_price) are provided for the chosen order_type. Specify quantity OR notional value."
    )
    args_schema: Type[BaseModel] = SubmitOrderToolInput

    # We need to map all fields from Pydantic model to _run method signature
    def _run(self, symbol: str, side: str, order_type: str, time_in_force: str,
             qty: Optional[float] = None, notional: Optional[float] = None,
             limit_price: Optional[float] = None, stop_price: Optional[float] = None,
             trail_percent: Optional[float] = None, trail_price: Optional[float] = None,
             stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None) -> str:
        """Executes the tool to submit an order."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."

        # Enums are handled internally based on validated Literal strings
        order_side_enum = OrderSide(side)
        order_type_enum = OrderType(order_type)
        time_in_force_enum = TimeInForce(time_in_force)

        # --- Input Validation (Pydantic handles basic types/literals, add logic here) ---
        if qty is None and notional is None: return "Error: Must specify either 'qty' or 'notional'."
        if qty is not None and notional is not None: return "Error: Cannot specify both 'qty' and 'notional'."
        if qty is not None and qty <= 0: return "Error: 'qty' must be positive."
        if notional is not None and notional <= 0: return "Error: 'notional' must be positive."
        # Price checks based on order type
        if order_type_enum in [OrderType.LIMIT, OrderType.STOP_LIMIT] and limit_price is None: return f"Error: limit_price required for {order_type}."
        if order_type_enum in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None: return f"Error: stop_price required for {order_type}."
        if order_type_enum == OrderType.TRAILING_STOP and trail_percent is None and trail_price is None: return "Error: trail_percent or trail_price required for trailing_stop."
        if trail_percent is not None and trail_price is not None: return "Error: Cannot specify both trail_percent and trail_price."
        # --- End Validation ---

        order_data = {
            "symbol": symbol, "side": order_side_enum, "type": order_type_enum, "time_in_force": time_in_force_enum,
            "qty": qty, "notional": notional, "limit_price": limit_price, "stop_price": stop_price,
            "trail_percent": trail_percent, "trail_price": trail_price,
            "order_class": None, "take_profit": None, "stop_loss": None,
        }

        # Determine Order Class
        if take_profit_price and stop_loss_price:
            order_data["order_class"] = OrderClass.BRACKET
            order_data["take_profit"] = TakeProfitRequest(limit_price=take_profit_price)
            order_data["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
        elif stop_loss_price:
            order_data["order_class"] = OrderClass.OTO
            order_data["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
        elif take_profit_price:
             order_data["order_class"] = OrderClass.BRACKET # Or OTO? Assume Bracket for now
             order_data["take_profit"] = TakeProfitRequest(limit_price=take_profit_price)

        request_class_map = {
            OrderType.MARKET: MarketOrderRequest, OrderType.LIMIT: LimitOrderRequest,
            OrderType.STOP: StopOrderRequest, OrderType.STOP_LIMIT: StopLimitOrderRequest,
            OrderType.TRAILING_STOP: TrailingStopOrderRequest
        }
        RequestClass = request_class_map.get(order_type_enum)
        if not RequestClass: return f"Error: Order type '{order_type}' not supported."

        order_data_cleaned = {k: v for k, v in order_data.items() if v is not None}

        try:
            order_request = RequestClass(**order_data_cleaned)
            log.info(f"Submitting order request: {order_request}")
            order = self.trading_client.submit_order(order_data=order_request)
            log.info(f"Order submitted successfully: ID={order.id}, Status={order.status}")
            return (
                f"Order submitted: ID={order.id}, ClientOrderID={order.client_order_id}, Symbol={order.symbol}, "
                f"Qty={order.qty}, Side={order.side}, Type={order.type}, Status={order.status}, "
                f"LimitPrice={order.limit_price}, StopPrice={order.stop_price}, OrderClass={order.order_class}"
            )
        except APIError as e:
            log.error(f"API Error submitting order: {e}. Request: {order_data_cleaned}")
            return f"Error submitting order: {e} (Request: {order_data_cleaned})"
        except Exception as e:
            log.error(f"Unexpected error submitting order: {e}. Request: {order_data_cleaned}")
            return f"Unexpected error submitting order: {e} (Request: {order_data_cleaned})"


class GetOrderStatusToolInput(BaseModel):
    """Input schema for GetOrderStatusTool."""
    order_id: str = Field(..., description="The unique ID of the order to check.")

class GetOrderStatusTool(BaseAlpacaTool):
    name: str = "Get Alpaca Order Status by ID"
    description: str = (
        "Checks the status (filled, open, canceled, etc.) and details of a specific order using its ID. "
        "Useful for tracking order execution after submission."
    )
    args_schema: Type[BaseModel] = GetOrderStatusToolInput

    def _run(self, order_id: str) -> str:
        """Executes the tool to get order status."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.info(f"Fetching status for order ID: {order_id}...")
            order = self.trading_client.get_order_by_id(order_id)
            log.info(f"Order status fetched: ID={order.id}, Status={order.status}")
            details = (
                 f"Order ID: {order.id}, ClientOrderID: {order.client_order_id}, Symbol: {order.symbol}, Status: {order.status}, "
                 f"Qty: {order.qty}, Filled Qty: {order.filled_qty}, Type: {order.type}, Side: {order.side}, "
                 f"TimeInForce: {order.time_in_force}, OrderClass: {order.order_class}, "
                 f"Filled Avg Price: {order.filled_avg_price}, LimitPrice: {order.limit_price}, StopPrice: {order.stop_price}, "
                 f"Submitted: {order.submitted_at}, Filled: {order.filled_at}, Expired: {order.expired_at}, Canceled: {order.canceled_at}"
            )
            if hasattr(order, 'legs') and order.legs:
                 leg_details = [f"Related Orders ({len(order.legs)}):"]
                 for leg in order.legs: leg_details.append(f"  Leg ID: {leg.id}, Status: {leg.status}, Symbol: {leg.symbol}, Type: {leg.type}")
                 details += "\n" + "\n".join(leg_details)
            return details
        except APIError as e:
            if "order not found" in str(e).lower() or e.status_code == 404: return f"Error: Order with ID {order_id} not found."
            log.error(f"API Error fetching order status for {order_id}: {e}")
            return f"Error fetching order status for {order_id}: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching order status for {order_id}: {e}")
            return f"Unexpected error fetching order status for {order_id}: {e}"


class ListOpenOrdersToolInput(BaseModel):
    """Input schema is empty as this tool takes no arguments."""
    pass

class ListOpenOrdersTool(BaseAlpacaTool):
    name: str = "List Open Alpaca Orders"
    description: str = (
        "Retrieves a list of all currently open (not filled, canceled, or expired) orders in the account. "
        "Useful for seeing pending trades."
    )
    args_schema: Type[BaseModel] = ListOpenOrdersToolInput

    def _run(self) -> str:
        """Executes the tool to list open orders."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.info("Fetching open orders...")
            request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100, nested=True)
            orders = self.trading_client.get_orders(filter=request_params)
            if not orders: return "No open orders."

            order_summaries = [f"Found {len(orders)} open orders:"]
            for order in orders:
                 summary = (f"  ID: {order.id}, Symbol: {order.symbol}, Status: {order.status}, Qty: {order.qty}, Type: {order.type}, Side: {order.side}, Class: {order.order_class}")
                 order_summaries.append(summary)
                 if order.legs:
                      for leg in order.legs: order_summaries.append(f"    Leg ID: {leg.id}, Status: {leg.status}, Type: {leg.type}")
            log.info(f"Found {len(orders)} open orders.")
            return "\n".join(order_summaries)
        except APIError as e:
            log.error(f"API Error listing open orders: {e}")
            return f"Error listing open orders: {e}"
        except Exception as e:
            log.error(f"Unexpected error listing open orders: {e}")
            return f"Unexpected error listing open orders: {e}"

class CancelAllOrdersToolInput(BaseModel):
    """Input schema is empty as this tool takes no arguments."""
    pass

class CancelAllOrdersTool(BaseAlpacaTool):
    name: str = "Cancel ALL Open Alpaca Orders"
    description: str = (
        "Cancels ALL open orders currently in the account. THIS IS A BULK ACTION AND SHOULD BE USED WITH EXTREME CAUTION. "
        "Useful for quickly clearing the order book, e.g., at end of day or in response to major market events."
    )
    args_schema: Type[BaseModel] = CancelAllOrdersToolInput

    def _run(self) -> str:
        """Executes the tool to cancel all open orders."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.warning("Attempting to cancel ALL open orders...")
            cancel_responses = self.trading_client.cancel_orders()
            if not cancel_responses: return "No open orders were found to cancel."

            canceled_count = sum(1 for resp in cancel_responses if 200 <= resp.status < 300)
            failed_count = len(cancel_responses) - canceled_count
            log.info(f"Cancel all orders response: {len(cancel_responses)} attempts. Success: {canceled_count}, Failed: {failed_count}")
            return f"Attempted to cancel {len(cancel_responses)} orders. Success: {canceled_count}, Failed: {failed_count}."
        except APIError as e:
            log.error(f"API Error canceling all orders: {e}")
            return f"Error canceling all orders: {e}"
        except Exception as e:
            log.error(f"Unexpected error canceling all orders: {e}")
            return f"Unexpected error canceling all orders: {e}"

class GetDailyPerformanceToolInput(BaseModel):
    """Input schema is empty."""
    pass

class GetDailyPerformanceTool(BaseAlpacaTool): # Inherits client access
    name: str = "Get Alpaca Account Daily Performance"
    description: str = (
        "Fetches the account's profit/loss details for the current trading day from Alpaca. "
        "Useful for monitoring overall daily performance."
    )
    args_schema: Type[BaseModel] = GetDailyPerformanceToolInput

    def _run(self) -> str:
        """Fetches daily P/L from account details."""
        if not self.trading_client:
            return "Error: Alpaca Trading client not initialized."
        try:
            log.info("Fetching Alpaca account details for daily P/L...")
            account = self.trading_client.get_account()
            # Extract P/L related fields (Note: Fields might vary, check Account model)
            # Example fields - adjust based on actual 'Account' object attributes
            equity = float(account.equity)
            last_equity = float(account.last_equity) # Equity at previous day's close
            daily_pl = equity - last_equity
            daily_pl_percent = (daily_pl / last_equity) * 100 if last_equity else 0.0

            return (
                f"Daily Performance Summary: Current Equity=${equity:.2f}, "
                f"Change Today=${daily_pl:.2f} ({daily_pl_percent:.2f}%)"
                # Add other relevant fields like daytrade_count if needed
            )
        except APIError as e:
            log.error(f"API Error fetching account performance details: {e}")
            return f"Error fetching account performance: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching account performance: {e}")
            return f"Unexpected error fetching account performance: {e}"

# --- Example Instantiation (in crew.py / main.py) ---
#
# from src.day_traders.tools.alpaca_trade_tools import (
#     GetAccountDetailsTool, GetPositionsTool, GetStockBarsTool,
#     SubmitOrderTool, GetOrderStatusTool, ListOpenOrdersTool, CancelAllOrdersTool
# )
#
# # Instantiate each tool - they will initialize/reuse clients via get_alpaca_clients()
# account_details_tool = GetAccountDetailsTool()
# positions_tool = GetPositionsTool()
# stock_bars_tool = GetStockBarsTool()
# submit_order_tool = SubmitOrderTool()
# order_status_tool = GetOrderStatusTool()
# list_orders_tool = ListOpenOrdersTool()
# cancel_all_tool = CancelAllOrdersTool()
#
# # Assign tools to agents
# market_data_analyst_agent = Agent(..., tools=[stock_bars_tool])
# risk_manager_agent = Agent(..., tools=[account_details_tool, positions_tool])
# order_executor_agent = Agent(..., tools=[
#     submit_order_tool,
#     order_status_tool,
#     list_orders_tool,
#     cancel_all_tool # Be careful assigning this one!
# ])
#
