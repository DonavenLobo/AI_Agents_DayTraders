#!/usr/bin/env python
import time
import datetime
import pytz
import json
import os
import warnings
from dotenv import load_dotenv

# Import your Crew class
from crew import DayTradersCrew

# Load environment variables from .env file
load_dotenv()

# Ignore specific warnings if necessary (though generally good to understand them)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="crewai.*") # Example


# --- Configuration ---
# Define Market Hours (ET) - Adjust slightly to avoid edge cases at exact open/close? Maybe run 9:31 to 15:59?
MARKET_OPEN_TIME = datetime.time(9, 30, 0)
MARKET_CLOSE_TIME = datetime.time(16, 0, 0)
MARKET_TIMEZONE = pytz.timezone('America/New_York')

# Define Crew Execution Frequency (e.g., run every 5 minutes)
# Be mindful of API rate limits and cost!
CYCLE_DELAY_SECONDS = 300 # 5 minutes
MARKET_CLOSED_DELAY_SECONDS = 600 # 10 minutes check when market is closed

# --- State Management (Basic Example using a file) ---
STATE_FILE = 'trading_state.json'

def load_state():
    """Loads persistent state from a JSON file."""
    default_state = {
        "cumulative_pnl_estimate": 0.0, # Note: This is an estimate based on cycle results, not live Alpaca P/L
        "trades_attempted_today": 0,
        "trades_filled_today": 0,
        "last_manager_guidance": "Start of day, monitor initial price action.",
        # Add any other state needed between cycles
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Ensure all expected keys exist, merge with defaults if needed
                for key, value in default_state.items():
                    state.setdefault(key, value)
                print(f"Loaded previous state from {STATE_FILE}")
                return state
        except json.JSONDecodeError:
            print(f"Warning: Error decoding {STATE_FILE}. Starting with default state.")
            return default_state
        except Exception as e:
            print(f"Warning: Could not load state from {STATE_FILE}. Error: {e}. Starting with default state.")
            return default_state
    else:
        print(f"State file {STATE_FILE} not found. Starting with default state.")
        return default_state

def save_state(state):
    """Saves the current state to a JSON file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        # print(f"Saved current state to {STATE_FILE}") # Optional: Can be verbose
    except Exception as e:
        print(f"Error: Could not save state to {STATE_FILE}. Error: {e}")

# --- Market Hours Check ---
def is_market_open():
    """Checks if the current time is within NYSE trading hours."""
    now_local = datetime.datetime.now(MARKET_TIMEZONE)
    # Simple check (add holiday/weekend checks for production)
    if now_local.weekday() >= 5: # Saturday or Sunday
        return False
    # TODO: Add check for market holidays using a library like 'pandas_market_calendars'
    # Example:
    # import pandas_market_calendars as mcal
    # nyse = mcal.get_calendar('NYSE')
    # schedule = nyse.schedule(start_date=now_local.date(), end_date=now_local.date())
    # if schedule.empty: # Holiday or weekend according to calendar
    #     return False
    # return schedule.iloc[0].market_open <= now_local <= schedule.iloc[0].market_close

    # Using simple time comparison for now:
    return MARKET_OPEN_TIME <= now_local.time() < MARKET_CLOSE_TIME

# --- Main Execution Loop ---
def run_trading_loop():
    """Main loop to run the trading crew during market hours."""
    print(f"[{datetime.datetime.now(MARKET_TIMEZONE)}] Initializing Day Trading Crew Loop...")
    print(f"Market Hours: {MARKET_OPEN_TIME} - {MARKET_CLOSE_TIME} {MARKET_TIMEZONE.zone}")
    print(f"Cycle Delay: {CYCLE_DELAY_SECONDS} seconds")

    current_state = load_state()
    # Reset daily counters if needed (e.g., based on date change)
    # TODO: Add logic to reset daily counters if the date changes since last run

    while True:
        market_now_open = is_market_open()

        if market_now_open:
            print(f"\n[{datetime.datetime.now(MARKET_TIMEZONE)}] Market Open. Starting Trading Cycle...")
            try:
                # --- Prepare Inputs for this Cycle ---
                # These inputs need to match ALL placeholders in agents.yaml and tasks.yaml
                inputs = {
                    # Dynamic/Configurable Inputs
                    'list_of_symbols': ["AAPL", "TSLA", "NVDA"], # Example List
                    'data_timeframe': '5Min',
                    'data_limit': 60, # e.g., 5 hours of 5min data
                    'news_time_window': '4 hours',
                    'risk_rules_summary': "Max 1% equity risk/trade, Max 3 concurrent positions, No new trades after 3:45 PM ET",
                    'max_risk_per_trade_percent': 1.0,

                    # Placeholders for strategy/manager (even if dynamic)
                    'trading_strategy_name': 'Adaptive Analysis V1', # Give the dynamic approach a name
                    'trading_strategy_description': 'Dynamically identify setups based on technicals, PA, and news context. Adapt based on performance feedback.',

                    # State passed to the Crew/Manager
                    'current_performance_summary': json.dumps(current_state) # Pass state as JSON string
                }
                # ------------------------------------

                # --- Execute CrewAI Cycle ---
                # Instantiate the crew for this cycle
                # Consider if DayTradersCrew needs methods to update internal state or if passing via inputs is sufficient
                trading_crew_instance = DayTradersCrew()
                # Kickoff the crew defined in @crew method
                result = trading_crew_instance.crew().kickoff(inputs=inputs)
                # ---------------------------

                print(f"--- Cycle Result ---")
                # The result is often the output of the LAST task in the sequence/hierarchy
                print(result) # Print the raw result (likely manager's guidance)
                print("--------------------")


                # --- Process Result & Update State ---
                # This part is CRUCIAL but depends heavily on the *actual* structure of the 'result' string
                # We assume the result is the string output from the Manager's task

                # 1. Update manager guidance in state
                if isinstance(result, str) and "Guidance:" in result:
                     # Extract guidance part if possible, otherwise store the whole result
                     current_state["last_manager_guidance"] = result.split("Guidance:", 1)[-1].strip() if "Guidance:" in result else result


                # 2. Update P/L and Trade Counts (This is tricky without parsing specific trade results)
                #    A more robust way would be for the monitoring task to output structured data
                #    about filled orders and their P/L, which the manager then aggregates.
                #    For now, we'll just increment 'attempted' trades based on keywords.
                if isinstance(result, str) and "Order Submitted: ID=" in result: # Check if execution task output is passed through
                     current_state["trades_attempted_today"] += result.count("Order Submitted: ID=")
                # TODO: Need a reliable way to track filled trades and P/L from cycle results.
                # This might require modifying task outputs or adding specific parsing logic here.
                # For now, cumulative_pnl_estimate won't be updated reliably.

                # Save the updated state for the next cycle
                save_state(current_state)
                # ------------------------------------

            except Exception as e:
                print(f"!!!!!! ERROR during trading cycle: {e} !!!!!!")
                # Consider adding more specific error handling or backoff delays

            print(f"Waiting {CYCLE_DELAY_SECONDS} seconds for next cycle...")
            time.sleep(CYCLE_DELAY_SECONDS)

        else:
            # Reset state flags if end-of-day occurred
            if datetime.datetime.now(MARKET_TIMEZONE).time() >= MARKET_CLOSE_TIME and \
               (datetime.datetime.now(MARKET_TIMEZONE) - timedelta(seconds=MARKET_CLOSED_DELAY_SECONDS)).time() < MARKET_CLOSE_TIME:
                print(f"\n[{datetime.datetime.now(MARKET_TIMEZONE)}] Market Closed. Performing End-of-Day actions...")
                # --- End-of-Day Logic ---
                try:
                    # Example: Use the CancelAllOrdersTool (if safe and desired)
                    print("Attempting to cancel any remaining open orders...")
                    # Need to instantiate the crew/agent/tool again for this specific action
                    eod_crew = DayTradersCrew()
                    # Directly invoke the tool via an agent requires careful setup,
                    # Or create a simple EOD task/crew. Simplest might be direct tool call if feasible.
                    # This assumes the tool can be called directly after instantiation:
                    cancel_result = eod_crew.cancel_all_tool._run() # Be cautious with direct _run calls
                    print(f"Cancel all orders result: {cancel_result}")

                    # Reset daily state counters
                    current_state["trades_attempted_today"] = 0
                    current_state["trades_filled_today"] = 0
                    # Keep cumulative PNL estimate? Or reset daily? Decide based on need.
                    current_state["last_manager_guidance"] = "Market closed. Reset for next trading day."
                    save_state(current_state)
                    print("End-of-day state reset.")

                except Exception as e:
                    print(f"Error during End-of-Day tasks: {e}")
                # --- End EOD Logic ---

            print(f"[{datetime.datetime.now(MARKET_TIMEZONE)}] Market Closed / Pre-Market. Waiting {MARKET_CLOSED_DELAY_SECONDS} seconds...")
            time.sleep(MARKET_CLOSED_DELAY_SECONDS)


# --- Script Entry Point ---
if __name__ == "__main__":
    # Validate necessary environment variables are set
    required_vars = ["OPENAI_API_KEY", "SERPER_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure they are set in your .env file or system environment.")
    else:
        print("Environment variables seem loaded.")
        run_trading_loop()