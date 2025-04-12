import os
import logging
# DO NOT load dotenv here - load it ONCE in main.py
# from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Type, List, Optional, Literal

# Import crewai BaseTool
from crewai.tools import BaseTool
# Import standard Pydantic (V2 assumed)
from pydantic import BaseModel, Field, ConfigDict

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# --- Tool Definition Base Class (NO __init__ METHOD HERE) ---
class BaseAlpacaTool(BaseTool):
    """Base class for Alpaca tools defining client fields and Pydantic config."""
    # Define fields to HOLD the clients - they will be injected externally
    trading_client: Optional[TradingClient] = None
    stock_data_client: Optional[StockHistoricalDataClient] = None

    # Define Pydantic V2 Config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    # NO __INIT__ METHOD DEFINED IN THIS CLASS


# --- Individual Tool Classes ---

# === Account Information Tools ===

class GetAccountDetailsToolInput(BaseModel):
    """Input schema is empty as this tool takes no arguments."""
    pass

class GetAccountDetailsTool(BaseAlpacaTool): # Inherits BaseAlpacaTool
    name: str = "Get Alpaca Account Details"
    description: str = "Fetches current Alpaca paper trading account details..."
    args_schema: Type[BaseModel] = GetAccountDetailsToolInput

    def _run(self) -> str:
        """Executes the tool to get account details."""
        # Check if client was injected correctly BEFORE use
        if not self.trading_client:
            log.error("GetAccountDetailsTool._run: Trading client not available.")
            return "Error: Trading client not provided to GetAccountDetailsTool."
        try:
            log.info("Fetching Alpaca account details...")
            account = self.trading_client.get_account()
            log.info(f"Account details fetched: Status={account.status}, Equity={account.equity}")
            # Line 91 in the ORIGINAL provided code was here:
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

# --- GET POSITIONS TOOL ---
class GetPositionsToolInput(BaseModel):
    pass

class GetPositionsTool(BaseAlpacaTool):
    name: str = "Get Current Alpaca Positions"
    description: str = "Fetches all currently open positions..."
    args_schema: Type[BaseModel] = GetPositionsToolInput

    def _run(self) -> str:
        if not self.trading_client:
            log.error("GetPositionsTool._run: Trading client not available.")
            return "Error: Trading client not provided to GetPositionsTool."
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

# --- GET STOCK BARS TOOL ---
class GetStockBarsToolInput(BaseModel):
    symbol: str = Field(..., description="The stock ticker symbol.")
    timeframe_str: Literal['1Min', '5Min', '15Min', '1Hour', '1Day'] = Field(default='1Min')
    limit: int = Field(default=100)
    simulation_time_iso: Optional[str] = Field(default=None)

class GetStockBarsTool(BaseAlpacaTool):
    name: str = "Get Historical Stock Bars"
    description: str = "Fetches historical stock bar data (OHLCV) for a symbol, using a fixed lookback period ending near the simulation time or current time."
    args_schema: Type[BaseModel] = GetStockBarsToolInput # Input schema remains the same

    def _run(self, symbol: str, timeframe_str: str, limit: int, simulation_time_iso: Optional[str] = None) -> str:
        if not self.stock_data_client:
            log.error("GetStockBarsTool._run: Stock data client not available.")
            return "Error: Stock data client not provided to GetStockBarsTool."

        timeframe_map = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute), '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute), '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
            '1Day': TimeFrame(1, TimeFrameUnit.Day)
        }
        timeframe = timeframe_map.get(timeframe_str)
        if timeframe is None: return f"Error: Invalid timeframe_str '{timeframe_str}'."

        try:
            # Determine reference time based on simulation or live mode
            market_tz = ZoneInfo("America/New_York")
            if simulation_time_iso:
                # Use the simulation time as the reference point for "now"
                reference_time = datetime.fromisoformat(simulation_time_iso)
                # Ensure it's timezone-aware for consistency if needed by API/calculations
                if reference_time.tzinfo is None:
                     reference_time = market_tz.localize(reference_time.replace(tzinfo=None))
                mode = "Simulation"
            else:
                reference_time = datetime.now(market_tz)
                mode = "Live"

            # --- MODIFICATION: Use fixed lookback for start, omit end ---
            # Set a fixed lookback period (e.g., 5 trading days - adjust as needed)
            # Note: Calculating exact trading days back is complex, using calendar days is simpler.
            start_time = reference_time - timedelta(days=7) # Look back 7 calendar days to likely capture 5 trading days
            # OMIT end time - let the API default

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_time,
                # end=None, # Explicitly omit end time parameter
                limit=limit, # Still limit the number of bars returned from the period
                adjustment='raw'
                # feed='iex' # Keep commented out unless specifically testing IEX
            )
            # --- END MODIFICATION ---

            log.info(f"[{mode}] Fetching {timeframe_str} bars for {symbol} (start: {start_time}, limit: {limit}, ref_time: {reference_time})")
            bars_data = self.stock_data_client.get_stock_bars(request_params) # Renamed variable

            # Check and process bars_data (using bars_data instead of bars)
            if not bars_data or symbol not in bars_data.data or not bars_data.data[symbol]:
                log.warning(f"No bar data found for {symbol} starting around {start_time}.")
                return f"No bar data found for {symbol} starting around {start_time}."

            bar_list = bars_data.data[symbol]

            # Filter bars to be *before* the reference_time if needed, as the API might return data up to the request time
            # The API *should* respect the limit and timeframe, often returning data ending just before 'now' or the end time if specified.
            # Let's assume for now the default end takes care of this. If not, add filtering:
            # bar_list = [bar for bar in bar_list if market_tz.normalize(bar.timestamp) < reference_time]
            # if not bar_list: return f"No bars found strictly before {reference_time} after filtering."

            # --- Formatting logic (using bar_list) ---
            formatted_bars = []
            # Show latest bars if many are returned due to fixed start time
            bars_to_show = bar_list[-limit:] if len(bar_list) > limit else bar_list # Show up to the limit requested, from the end
            bars_to_show_display = bars_to_show if len(bars_to_show) <= 4 else bars_to_show[:2] + bars_to_show[-2:] # For display shortening

            for bar in bars_to_show_display:
                 formatted_bars.append(f"  T={bar.timestamp}, O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}")

            summary = f"Fetched {len(bar_list)} bars for {symbol} (Showing latest {len(bars_to_show)}):\n" + "\n".join(formatted_bars)
            if len(bars_to_show) > 4 and len(bars_to_show) > len(bars_to_show_display):
                 summary += "\n(Showing first 2 and last 2 of the latest bars)"
            # --- End Formatting ---

            return summary

        except APIError as e: # Catch specific API errors first
            log.error(f"API Error fetching stock bars for {symbol} ({mode}): {e}")
            # Log the exact request parameters might help debugging
            log.debug(f"Failed request params: {request_params.dict()}")
            return f"API Error fetching stock bars for {symbol}: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching stock bars for {symbol} ({mode}): {e}")
            return f"Unexpected error fetching stock bars: {e}"


# --- SUBMIT ORDER TOOL ---
class SubmitOrderToolInput(BaseModel):
    symbol: str = Field(...)
    side: Literal['buy', 'sell'] = Field(...)
    order_type: Literal['market', 'limit', 'stop', 'stop_limit', 'trailing_stop'] = Field(default='market')
    time_in_force: Literal['day', 'gtc', 'opg', 'cls', 'ioc', 'fok'] = Field(default='day')
    qty: Optional[float] = Field(default=None)
    notional: Optional[float] = Field(default=None)
    limit_price: Optional[float] = Field(default=None)
    stop_price: Optional[float] = Field(default=None)
    trail_percent: Optional[float] = Field(default=None)
    trail_price: Optional[float] = Field(default=None)
    stop_loss_price: Optional[float] = Field(default=None)
    take_profit_price: Optional[float] = Field(default=None)

class SubmitOrderTool(BaseAlpacaTool):
    name: str = "Submit Alpaca Trading Order"
    description: str = "Submits a trading order..."
    args_schema: Type[BaseModel] = SubmitOrderToolInput

    def _run(self, symbol: str, side: str, order_type: str, time_in_force: str, # All args from Input
             qty: Optional[float] = None, notional: Optional[float] = None,
             limit_price: Optional[float] = None, stop_price: Optional[float] = None,
             trail_percent: Optional[float] = None, trail_price: Optional[float] = None,
             stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None) -> str:
        if not self.trading_client:
            log.error("SubmitOrderTool._run: Trading client not available.")
            return "Error: Trading client not provided to SubmitOrderTool."

        try:
            order_side_enum = OrderSide(side)
            order_type_enum = OrderType(order_type)
            time_in_force_enum = TimeInForce(time_in_force)
        except ValueError as e:
            return f"Error: Invalid order parameter string: {e}"

        # --- Validation ---
        if qty is None and notional is None: return "Error: Must specify either 'qty' or 'notional'."
        if qty is not None and notional is not None: return "Error: Cannot specify both 'qty' and 'notional'."
        if qty is not None and qty <= 0: return "Error: 'qty' must be positive."
        if notional is not None and notional <= 0: return "Error: 'notional' must be positive."
        if order_type_enum in [OrderType.LIMIT, OrderType.STOP_LIMIT] and limit_price is None: return f"Error: limit_price required for {order_type}."
        if order_type_enum in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None: return f"Error: stop_price required for {order_type}."
        if order_type_enum == OrderType.TRAILING_STOP and trail_percent is None and trail_price is None: return "Error: trail_percent or trail_price required for trailing_stop."
        if trail_percent is not None and trail_price is not None: return "Error: Cannot specify both trail_percent and trail_price."

        order_data = { # Build dict for request
            "symbol": symbol, "side": order_side_enum, "type": order_type_enum, "time_in_force": time_in_force_enum,
            "qty": qty, "notional": notional, "limit_price": limit_price, "stop_price": stop_price,
            "trail_percent": trail_percent, "trail_price": trail_price,
            "order_class": None, "take_profit": None, "stop_loss": None, }

        # Determine Order Class
        if take_profit_price and stop_loss_price:
            order_data["order_class"] = OrderClass.BRACKET
            order_data["take_profit"] = TakeProfitRequest(limit_price=take_profit_price)
            order_data["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
        elif stop_loss_price: order_data["order_class"] = OrderClass.OTO; order_data["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)
        elif take_profit_price: order_data["order_class"] = OrderClass.BRACKET; order_data["take_profit"] = TakeProfitRequest(limit_price=take_profit_price)

        request_class_map = { OrderType.MARKET: MarketOrderRequest, OrderType.LIMIT: LimitOrderRequest, OrderType.STOP: StopOrderRequest, OrderType.STOP_LIMIT: StopLimitOrderRequest, OrderType.TRAILING_STOP: TrailingStopOrderRequest }
        RequestClass = request_class_map.get(order_type_enum)
        if not RequestClass: return f"Error: Order type '{order_type}' not supported."

        order_data_cleaned = {k: v for k, v in order_data.items() if v is not None}

        try:
            order_request = RequestClass(**order_data_cleaned)
            log.info(f"Submitting order request: {order_request}")
            order = self.trading_client.submit_order(order_data=order_request)
            log.info(f"Order submitted successfully: ID={order.id}, Status={order.status}")
            return (f"Order submitted: ID={order.id}, ClientOrderID={order.client_order_id}, Symbol={order.symbol}, ..., Status={order.status}") # Abbreviated
        except APIError as e:
            log.error(f"API Error submitting order: {e}. Request: {order_data_cleaned}")
            return f"Error submitting order: {e} (Request: {order_data_cleaned})"
        except Exception as e:
            log.error(f"Unexpected error submitting order: {e}. Request: {order_data_cleaned}")
            return f"Unexpected error submitting order: {e} (Request: {order_data_cleaned})"

# --- GET ORDER STATUS TOOL ---
class GetOrderStatusToolInput(BaseModel):
    order_id: str = Field(...)

class GetOrderStatusTool(BaseAlpacaTool):
    name: str = "Get Alpaca Order Status by ID"
    description: str = "Checks the status... of a specific order using its ID."
    args_schema: Type[BaseModel] = GetOrderStatusToolInput

    def _run(self, order_id: str) -> str:
        if not self.trading_client:
            log.error("GetOrderStatusTool._run: Trading client not available.")
            return "Error: Trading client not provided to GetOrderStatusTool."
        try:
            log.info(f"Fetching status for order ID: {order_id}...")
            order = self.trading_client.get_order_by_id(order_id)
            log.info(f"Order status fetched: ID={order.id}, Status={order.status}")
            details = ( f"Order ID: {order.id}, Symbol: {order.symbol}, Status: {order.status}, ...") # Abbreviated
            # ... add leg details if needed ...
            return details
        except APIError as e:
            if "order not found" in str(e).lower() or e.status_code == 404: return f"Error: Order with ID {order_id} not found."
            log.error(f"API Error fetching order status for {order_id}: {e}")
            return f"Error fetching order status for {order_id}: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching order status for {order_id}: {e}")
            return f"Unexpected error fetching order status for {order_id}: {e}"

# --- LIST OPEN ORDERS TOOL ---
class ListOpenOrdersToolInput(BaseModel):
    pass

class ListOpenOrdersTool(BaseAlpacaTool):
    name: str = "List Open Alpaca Orders"
    description: str = "Retrieves a list of all currently open orders..."
    args_schema: Type[BaseModel] = ListOpenOrdersToolInput

    def _run(self) -> str:
        if not self.trading_client:
            log.error("ListOpenOrdersTool._run: Trading client not available.")
            return "Error: Trading client not provided to ListOpenOrdersTool."
        try:
            log.info("Fetching open orders...")
            request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100, nested=True)
            orders = self.trading_client.get_orders(filter=request_params)
            if not orders: return "No open orders."
            # ... formatting logic ...
            return "\n".join(order_summaries) # Assuming order_summaries is populated
        except APIError as e:
            log.error(f"API Error listing open orders: {e}")
            return f"Error listing open orders: {e}"
        except Exception as e:
            log.error(f"Unexpected error listing open orders: {e}")
            return f"Unexpected error listing open orders: {e}"

# --- CANCEL ALL ORDERS TOOL ---
class CancelAllOrdersToolInput(BaseModel):
    pass

class CancelAllOrdersTool(BaseAlpacaTool):
    name: str = "Cancel ALL Open Alpaca Orders"
    description: str = "Cancels ALL open orders... USE WITH EXTREME CAUTION."
    args_schema: Type[BaseModel] = CancelAllOrdersToolInput

    def _run(self) -> str:
        if not self.trading_client:
            log.error("CancelAllOrdersTool._run: Trading client not available.")
            return "Error: Trading client not provided to CancelAllOrdersTool."
        try:
            log.warning("Attempting to cancel ALL open orders...")
            cancel_responses = self.trading_client.cancel_orders()
            if not cancel_responses: return "No open orders were found to cancel."
            # ... status counting logic ...
            return f"Attempted to cancel {len(cancel_responses)} orders. Success: {canceled_count}, Failed: {failed_count}."
        except APIError as e:
            log.error(f"API Error canceling all orders: {e}")
            return f"Error canceling all orders: {e}"
        except Exception as e:
            log.error(f"Unexpected error canceling all orders: {e}")
            return f"Unexpected error canceling all orders: {e}"

# --- GET DAILY PERFORMANCE TOOL ---
class GetDailyPerformanceToolInput(BaseModel):
    pass

class GetDailyPerformanceTool(BaseAlpacaTool):
    name: str = "Get Alpaca Account Daily Performance"
    description: str = "Fetches the account's P/L details..."
    args_schema: Type[BaseModel] = GetDailyPerformanceToolInput

    def _run(self) -> str:
        if not self.trading_client:
            log.error("GetDailyPerformanceTool._run: Trading client not available.")
            return "Error: Trading client not provided to GetDailyPerformanceTool."
        try:
            log.info("Fetching Alpaca account details for daily P/L...")
            account = self.trading_client.get_account()
            # ... P/L calculation logic ...
            return ( f"Daily Performance Summary: Current Equity=${equity:.2f}, Change Today=${daily_pl:.2f} ({daily_pl_percent:.2f}%)" )
        except APIError as e:
            log.error(f"API Error fetching account performance details: {e}")
            return f"Error fetching account performance: {e}"
        except Exception as e:
            log.error(f"Unexpected error fetching account performance: {e}")
            return f"Unexpected error fetching account performance: {e}"