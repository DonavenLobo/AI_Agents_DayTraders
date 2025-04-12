#!/usr/bin/env python
import time
import datetime
import pytz
import json
import os
import warnings
import argparse # Import argparse
from dotenv import load_dotenv

# Import your Crew class
from crew import DayTradersCrew

# Load environment variables from .env file
load_dotenv()

# Ignore specific warnings if necessary
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="crewai.*")


# --- Configuration ---
MARKET_OPEN_TIME = datetime.time(9, 30, 0)
MARKET_CLOSE_TIME = datetime.time(16, 0, 0)
MARKET_TIMEZONE = pytz.timezone('America/New_York')
CYCLE_DELAY_SECONDS = 300 # Use this for step duration in replay too
MARKET_CLOSED_DELAY_SECONDS = 600

# --- State Management (Remains the same) ---
STATE_FILE = 'trading_state.json'
# load_state() and save_state() functions remain as before...
def load_state():
    """Loads persistent state from a JSON file."""
    default_state = {
        "cumulative_pnl_estimate": 0.0,
        "trades_attempted_today": 0,
        "trades_filled_today": 0,
        "last_manager_guidance": "Start of day / simulation.",
        # Add simulated state for replay
        "simulated_cash": 100000.0, # Example starting cash for simulation
        "simulated_positions": {} # Dict like {'AAPL': {'qty': 10, 'entry_price': 150.0}, ...}
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for key, value in default_state.items():
                    state.setdefault(key, value)
                print(f"Loaded previous state from {STATE_FILE}")
                return state
        except Exception as e:
            print(f"Warning: Could not load state from {STATE_FILE}. Error: {e}. Starting with default state.")
            return default_state
    else:
        print(f"State file {STATE_FILE} not found. Starting with default state.")
        return default_state

def save_state(state):
    """Saves the current state to a JSON file."""
    try:
        # Don't save simulated state if running live? Or use different files?
        # For now, save everything.
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error: Could not save state to {STATE_FILE}. Error: {e}")


# --- Market Hours Check (Remains the same) ---
def is_market_open():
     # ... (function as before) ...
    now_local = datetime.datetime.now(MARKET_TIMEZONE)
    if now_local.weekday() >= 5: # Saturday or Sunday
        return False
    return MARKET_OPEN_TIME <= now_local.time() < MARKET_CLOSE_TIME


# --- Live Trading Loop (Remains the same) ---
def run_trading_loop():
    # ... (function largely as before, loads/saves state) ...
    print(f"[{datetime.datetime.now(MARKET_TIMEZONE)}] Initializing LIVE Day Trading Crew Loop...")
    current_state = load_state() # Load potentially including sim state, ok for now

    while True:
        market_now_open = is_market_open()
        if market_now_open:
            print(f"\n[{datetime.datetime.now(MARKET_TIMEZONE)}] Market Open. Starting LIVE Trading Cycle...")
            try:
                # --- Prepare LIVE Inputs ---
                # NO simulation_time passed here
                inputs = {
                    'list_of_symbols': ["AAPL", "TSLA", "NVDA"],
                    'data_timeframe': '5Min',
                    'data_limit': 60,
                    'news_time_window': '4 hours',
                    'risk_rules_summary': "Max 1% equity risk/trade, Max 3 concurrent positions...",
                    'max_risk_per_trade_percent': 1.0,
                    'trading_strategy_name': 'Adaptive Analysis V1',
                    'trading_strategy_description': 'Dynamically identify setups...',
                    'current_performance_summary': json.dumps(current_state), # Pass live state
                    'historical_date': None, # Explicitly None for live mode
                    'simulation_time': None, # Explicitly None for live mode
                    'simulated_account_state': None, # Explicitly None for live mode
                }
                trading_crew_instance = DayTradersCrew()
                result = trading_crew_instance.crew().kickoff(inputs=inputs)
                print(f"--- LIVE Cycle Result ---\n{result}\n--------------------")

                # --- Process LIVE Result & Update State ---
                # (Update state logic as before, potentially refining P/L tracking)
                if isinstance(result, str) and "Guidance:" in result:
                     current_state["last_manager_guidance"] = result.split("Guidance:", 1)[-1].strip()
                # Increment live counters if needed
                save_state(current_state)

            except Exception as e:
                print(f"!!!!!! ERROR during LIVE trading cycle: {e} !!!!!!")

            print(f"Waiting {CYCLE_DELAY_SECONDS} seconds for next LIVE cycle...")
            time.sleep(CYCLE_DELAY_SECONDS)

        else:
            # ... (Market closed logic as before, including potential EOD actions) ...
            print(f"[{datetime.datetime.now(MARKET_TIMEZONE)}] Market Closed / Pre-Market. Waiting {MARKET_CLOSED_DELAY_SECONDS} seconds...")
            time.sleep(MARKET_CLOSED_DELAY_SECONDS)


# --- NEW: Historical Replay Function ---
def test_replay(historical_date_str: str):
    """Runs the crew simulation over a specified historical trading day."""
    print(f"\n--- Starting Historical Replay for Date: {historical_date_str} ---")

    try:
        historical_date = datetime.datetime.strptime(historical_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    # Initialize simulated state for this replay run
    sim_state = {
        "simulated_cash": 100000.0,
        "simulated_buying_power": 100000.0, # Start same as cash, adjust later if needed
        "simulated_positions": {}, # symbol: {'qty': float, 'entry_price': float}
        "simulated_daily_pnl": 0.0,
        "trades_attempted_today": 0,
        "trades_filled_today": 0,
        "last_manager_guidance": f"Start of simulation for {historical_date_str}.",
    }

    # Instantiate crew once outside the loop for efficiency
    trading_crew_instance = DayTradersCrew()

    # --- Time Iteration Loop ---
    start_datetime_et = MARKET_TIMEZONE.localize(datetime.datetime.combine(historical_date, MARKET_OPEN_TIME))
    end_datetime_et = MARKET_TIMEZONE.localize(datetime.datetime.combine(historical_date, MARKET_CLOSE_TIME))
    current_sim_time = start_datetime_et

    print(f"Simulating from {start_datetime_et} to {end_datetime_et}")

    while current_sim_time < end_datetime_et:
        print(f"\n[{current_sim_time}] Starting Simulation Cycle...")

        try:
            # --- Prepare REPLAY Inputs ---
            # Pass simulation time and simulated state
            sim_account_state_for_input = {
                "cash": sim_state["simulated_cash"],
                "buying_power": sim_state["simulated_buying_power"],
                "positions": sim_state["simulated_positions"]
            }
            inputs = {
                'list_of_symbols': ["AAPL", "TSLA", "NVDA"], # Use same symbols for consistency
                'data_timeframe': '5Min',
                'data_limit': 60, # How much history *before* current_sim_time to fetch
                'news_time_window': '4 hours', # Search news relevant to this window before sim time
                'risk_rules_summary': "Max 1% equity risk/trade, Max 3 concurrent positions...",
                'max_risk_per_trade_percent': 1.0,
                'trading_strategy_name': 'Adaptive Analysis V1 (Replay)',
                'trading_strategy_description': f'Dynamically identify setups for {historical_date_str}.',
                'current_performance_summary': json.dumps({ # Pass sim performance snapshot
                    "daily_pnl": sim_state["simulated_daily_pnl"],
                    "trades_attempted": sim_state["trades_attempted_today"],
                    "trades_filled": sim_state["trades_filled_today"],
                    "guidance": sim_state["last_manager_guidance"]
                }),
                'historical_date': historical_date_str, # Pass the target date string
                'simulation_time': current_sim_time.isoformat(), # Pass current simulation time
                'simulated_account_state': json.dumps(sim_account_state_for_input) # Pass needed sim state
            }

            # --- Execute CrewAI Cycle ---
            result = trading_crew_instance.crew().kickoff(inputs=inputs)
            # ---------------------------

            print(f"--- Simulation Cycle Result ---\n{result}\n--------------------")

            # --- Process Result & Update SIMULATED State ---
            # 1. Update guidance
            if isinstance(result, str) and "Guidance:" in result:
                sim_state["last_manager_guidance"] = result.split("Guidance:", 1)[-1].strip()

            # 2. Simulate Trade Execution & Update State (VERY SIMPLIFIED)
            #    This needs parsing the output of the execution/monitoring task if it contains fill info.
            #    Let's assume 'result' might contain a string like "Order Submitted: ID=... Symbol=AAPL, Qty=10, Side=buy..."
            #    And another task might output "Order Status Update: ID=... (AAPL buy 10): filled @ $PRICE."
            if isinstance(result, str):
                 # Crude check for fills - NEEDS robust parsing based on actual task output structure
                if "filled @" in result and "symbol=" in result: # Hypothetical check
                    sim_state["trades_filled_today"] += 1
                    # TODO: Implement simple P/L update based on simulated fill price vs current 'mark-to-market' price
                    # This requires getting a price point for the *current* sim_time via GetStockBarsTool again
                    # Example: Get price for AAPL at current_sim_time + 1 min
                    # Calculate P/L based on fill price from result and current mark price
                    # Update sim_state["simulated_daily_pnl"]
                    # Update sim_state["simulated_cash"] / sim_state["simulated_positions"]
                    print("TODO: Implement P/L tracking and position update based on simulated fill.")
                elif "Order Submitted: ID=" in result:
                     sim_state["trades_attempted_today"] += result.count("Order Submitted: ID=")


            # Optional: Print current simulated state
            print(f"Simulated State: Cash={sim_state['simulated_cash']:.2f}, Positions={len(sim_state['simulated_positions'])}, PNL={sim_state['simulated_daily_pnl']:.2f}")


        except Exception as e:
            print(f"!!!!!! ERROR during simulation cycle at {current_sim_time}: {e} !!!!!!")

        # --- Advance Simulation Time ---
        current_sim_time += datetime.timedelta(seconds=CYCLE_DELAY_SECONDS)

    print(f"\n--- Historical Replay for {historical_date_str} Completed ---")
    print(f"Final Simulated State: {sim_state}")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DayTraders Crew in live or test replay mode.")
    parser.add_argument(
        "mode",
        choices=['run', 'test'],
        help="Mode to run: 'run' for live trading loop, 'test' for historical replay."
    )
    parser.add_argument(
        "-d", "--date",
        metavar="YYYY-MM-DD",
        help="The historical date for test replay mode (required if mode is 'test')."
    )

    args = parser.parse_args()

    # Validate necessary environment variables
    required_vars = ["OPENAI_API_KEY", "SERPER_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    else:
        print("Environment variables loaded.")
        if args.mode == 'run':
            run_trading_loop()
        elif args.mode == 'test':
            if not args.date:
                parser.error("The --date argument (YYYY-MM-DD) is required for test mode.")
            else:
                test_replay(args.date)
        else:
            parser.print_help() # Should not happen due to choices constraint