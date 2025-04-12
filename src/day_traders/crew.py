import os
from dotenv import load_dotenv

# Load environment variables from .env file at the beginning
load_dotenv()

import logging
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI # Use langchain_openai for LLM definition

# --- Import Standard Tools ---
from crewai_tools import SerperDevTool

# --- Import Custom Tools ---
from tools.alpaca_trade_tool import (
    GetAccountDetailsTool,
    GetPositionsTool,
    GetStockBarsTool,
    SubmitOrderTool,
    GetOrderStatusTool,
    ListOpenOrdersTool,
    CancelAllOrdersTool,
    GetDailyPerformanceTool # Import the performance tool
)

# Option 2: Define client init logic here (Preferred)
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.common.exceptions import APIError

log = logging.getLogger(__name__) # Use logger

def initialize_alpaca_clients():
    """Initializes and returns Alpaca clients."""
    # Moved initialization logic here
    trading_client = None
    stock_data_client = None
    try:
        api_key = os.getenv("ALPACA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper_trading = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        if not api_key or not secret_key:
            raise ValueError("Alpaca API Key ID or Secret Key not found.")

        log.info(f"Initializing Alpaca clients in crew.py (Paper Trading: {paper_trading})")
        trading_client = TradingClient(api_key, secret_key, paper=paper_trading)
        stock_data_client = StockHistoricalDataClient(api_key, secret_key)

        # Verify connection
        account = trading_client.get_account()
        log.info(f"Alpaca Trading client initialized. Account Status: {account.status}")
        return trading_client, stock_data_client

    except APIError as e:
        log.error(f"Alpaca API Error during initialization: {e}")
        raise ConnectionError(f"Failed to connect to Alpaca API: {e}") from e
    except Exception as e:
        log.error(f"Failed to initialize Alpaca clients: {e}")
        raise RuntimeError(f"Error initializing Alpaca clients: {e}") from e

@CrewBase
class DayTradersCrew():
	"""DayTradersCrew orchestrates the team of trading agents."""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		# === Initialize Clients ONCE ===
		self._trading_client, self._stock_data_client = initialize_alpaca_clients()

		# === Instantiate Tools ===
		self.search_tool = SerperDevTool()

		# --- Instantiate Alpaca & Performance Tools ---
		self.account_details_tool = GetAccountDetailsTool()
		self.positions_tool = GetPositionsTool()
		self.stock_bars_tool = GetStockBarsTool()
		self.submit_order_tool = SubmitOrderTool()
		self.order_status_tool = GetOrderStatusTool()
		self.list_orders_tool = ListOpenOrdersTool()
		self.cancel_all_tool = CancelAllOrdersTool()
		self.performance_tool = GetDailyPerformanceTool()

		# === Inject Clients into Tools ===
		# Assign the correct client(s) needed by each tool instance
		self.account_details_tool.trading_client = self._trading_client
		self.positions_tool.trading_client = self._trading_client
		self.stock_bars_tool.stock_data_client = self._stock_data_client
		self.submit_order_tool.trading_client = self._trading_client
		self.order_status_tool.trading_client = self._trading_client
		self.list_orders_tool.trading_client = self._trading_client
		self.cancel_all_tool.trading_client = self._trading_client
		self.performance_tool.trading_client = self._trading_client # Needs trading client

		# Define the Manager LLM
		self.manager_llm = LLM(model="openai/o3-mini", temperature=0.7)

	# === Agent Definitions ===

	@agent
	def market_data_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['market_data_analyst'],
			tools=[self.stock_bars_tool], # Assign the specific tool instance
			verbose=True,
            allow_delegation=False # This agent primarily executes a specific tool call
		)

	@agent
	def news_sentiment_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['news_sentiment_analyst'],
			tools=[self.search_tool], # Assign the search tool instance
			verbose=True,
            allow_delegation=False # Focused task using its tool
		)

	@agent
	def trading_strategy_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['trading_strategy_analyst'],
			# No external tools needed; relies on context and internal reasoning
			tools=[],
			verbose=True,
            # Allow delegation? Maybe not, its primary role is analysis based on context.
            # If it needed complex calculations, maybe delegate to a calculator agent? For now, False.
            allow_delegation=False
		)

	@agent
	def risk_management_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['risk_management_analyst'],
			tools=[
                self.account_details_tool,
                self.positions_tool
                # Add Calculator Tool instance here if needed/created
            ],
			verbose=True,
            allow_delegation=False # Focused on applying rules using its tools/context
		)

	@agent
	def order_execution_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['order_execution_specialist'],
			tools=[
                self.submit_order_tool,
                self.order_status_tool,
                self.list_orders_tool,
                self.cancel_all_tool # Assign with caution enabled
            ],
			verbose=True,
            allow_delegation=False # Primarily executes specific tool calls based on input
		)

	@agent
	def crew_manager(self) -> Agent:
		return Agent(
			config=self.agents_config['crew_manager'],
			llm=self.manager_llm,
			verbose=True,
            allow_delegation=True # MUST allow delegation for hierarchical manager
		)

	# === Task Definitions ===

	@task
	def fetch_market_data(self) -> Task: # RENAMED from fetch_market_data_task
		return Task(
			config=self.tasks_config['fetch_market_data'], # Matches YAML key
			agent=self.market_data_analyst()
		)

	@task
	def fetch_news_sentiment(self) -> Task: # RENAMED from fetch_news_sentiment_task
		return Task(
			config=self.tasks_config['fetch_news_sentiment'], # Matches YAML key
			agent=self.news_sentiment_analyst()
		)

	@task
	def analyze_market_and_propose_trade(self) -> Task: # RENAMED
		return Task(
			config=self.tasks_config['analyze_market_and_propose_trade'], # Matches YAML key
			agent=self.trading_strategy_analyst(),
            # Context now refers to the RENAMED methods
			# context=[
            #     self.fetch_market_data, # Reference renamed method
            #     self.fetch_news_sentiment # Reference renamed method
            # ]
		)

	@task
	def assess_trade_risk(self) -> Task: # RENAMED
		return Task(
			config=self.tasks_config['assess_trade_risk'], # Matches YAML key
			agent=self.risk_management_analyst(),
            # Context refers to the RENAMED method
			# context=[self.analyze_market_and_propose_trade] # Reference renamed method
		)

	@task
	def execute_trade_order(self) -> Task: # RENAMED
		return Task(
			config=self.tasks_config['execute_trade_order'], # Matches YAML key
			agent=self.order_execution_specialist(),
            # Context refers to the RENAMED method
			# context=[self.assess_trade_risk] # Reference renamed method
		)

	@task
	def monitor_order_status(self) -> Task: # RENAMED
		return Task(
			config=self.tasks_config['monitor_order_status'], # Matches YAML key
			agent=self.order_execution_specialist(),
            # Context refers to the RENAMED method
			# context=[self.execute_trade_order] # Reference renamed method
		)

	@task
	def review_performance_and_provide_guidance(self) -> Task: # RENAMED
		return Task(
			config=self.tasks_config['review_performance_and_provide_guidance'], # Matches YAML key
			agent=self.crew_manager(),
            # Context refers to the RENAMED method
			# context=[self.monitor_order_status] # Reference renamed method
		)

	# === Crew Definition ===

	@crew
	def crew(self) -> Crew:
		"""Creates and configures the DayTraders Crew."""

		# Get the manager agent instance explicitly
		manager = self.crew_manager()

        # Get the list of all agents collected by @CrewBase
		all_agents = self.agents

        # Filter out the manager from the list of agents to be managed
		worker_agents = [agent for agent in all_agents if agent != manager]

		return Crew(
            # Use the agents and tasks automatically gathered by the decorators
			agents=worker_agents,
			tasks=self.tasks,
			process=Process.hierarchical, # Explicitly set hierarchical process
			manager_llm=self.manager_llm, # Assign the pre-configured manager LLM
            # CrewAI infers the manager agent if not specified AND process is hierarchical,
            # BUT it's better practice to be explicit if you have a dedicated manager role.
            # Check CrewAI docs if manager_agent assignment here is needed/preferred over inference.
            # Based on recent practices, often you define the manager agent and Crew infers it,
            # or you pass manager_agent=self.crew_manager() explicitly. Let's be explicit.
            manager_agent=manager,
			verbose=True, 
            # memory=True # Consider enabling memory for context persistence within a single cycle run
		)
