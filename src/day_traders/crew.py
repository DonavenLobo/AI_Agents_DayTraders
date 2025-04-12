import os
from dotenv import load_dotenv

# Load environment variables from .env file at the beginning
load_dotenv()

from crewai import Agent, Crew, Process, Task
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


@CrewBase
class DayTradersCrew():
	"""DayTradersCrew orchestrates the team of trading agents."""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		# Instantiate tools ONCE to be shared across agent methods
		self.search_tool = SerperDevTool()
		# Alpaca & Performance Tools
		self.account_details_tool = GetAccountDetailsTool()
		self.positions_tool = GetPositionsTool()
		self.stock_bars_tool = GetStockBarsTool()
		self.submit_order_tool = SubmitOrderTool()
		self.order_status_tool = GetOrderStatusTool()
		self.list_orders_tool = ListOpenOrdersTool()
		self.cancel_all_tool = CancelAllOrdersTool()
		self.performance_tool = GetDailyPerformanceTool()

		# Define the Manager LLM (using o3-mini as planned)
		self.manager_llm = ChatOpenAI(
			model="o3-mini", # Or another powerful model
			temperature=0.7 # Adjust temperature as needed
		)

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

	def crew_manager(self) -> Agent:
		return Agent(
			config=self.agents_config['crew_manager'],
			tools=[
                self.performance_tool # Manager needs performance data
                # Does the manager need other tools directly? Maybe not if delegating properly.
            ],
			verbose=True,
            allow_delegation=True # MUST allow delegation for hierarchical manager
		)

	# === Task Definitions ===

	@task
	def fetch_market_data_task(self) -> Task:
		return Task(
			config=self.tasks_config['fetch_market_data'],
			agent=self.market_data_analyst() # Assign agent instance
            # No context needed for the first task
		)

	@task
	def fetch_news_sentiment_task(self) -> Task:
		return Task(
			config=self.tasks_config['fetch_news_sentiment'],
			agent=self.news_sentiment_analyst()
            # No context needed
		)

	@task
	def analyze_market_and_propose_trade_task(self) -> Task:
		return Task(
			config=self.tasks_config['analyze_market_and_propose_trade'],
			agent=self.trading_strategy_analyst(),
            # Define context dependencies using task methods
			context=[
                self.fetch_market_data_task(),
                self.fetch_news_sentiment_task()
            ]
		)

	@task
	def assess_trade_risk_task(self) -> Task:
		return Task(
			config=self.tasks_config['assess_trade_risk'],
			agent=self.risk_management_analyst(),
			context=[self.analyze_market_and_propose_trade_task()]
		)

	@task
	def execute_trade_order_task(self) -> Task:
		return Task(
			config=self.tasks_config['execute_trade_order'],
			agent=self.order_execution_specialist(),
			context=[self.assess_trade_risk_task()]
		)

	@task
	def monitor_order_status_task(self) -> Task:
		return Task(
			config=self.tasks_config['monitor_order_status'],
			agent=self.order_execution_specialist(),
			context=[self.execute_trade_order_task()]
		)

	@task
	def review_performance_and_provide_guidance_task(self) -> Task:
		return Task(
			config=self.tasks_config['review_performance_and_provide_guidance'],
			agent=self.crew_manager(), # Manager performs this review/guidance task
			context=[
                self.monitor_order_status_task(),
                # Could potentially add other contexts if manager needs direct data access
                # e.g., self.fetch_market_data_task() if manager needs raw data? Less ideal.
            ]
		)

	# === Crew Definition ===

	@crew
	def crew(self) -> Crew:
		"""Creates and configures the DayTraders Crew."""
		return Crew(
            # Use the agents and tasks automatically gathered by the decorators
			agents=self.agents,
			tasks=self.tasks,
			process=Process.hierarchical, # Explicitly set hierarchical process
			manager_llm=self.manager_llm, # Assign the pre-configured manager LLM
            # CrewAI infers the manager agent if not specified AND process is hierarchical,
            # BUT it's better practice to be explicit if you have a dedicated manager role.
            # Check CrewAI docs if manager_agent assignment here is needed/preferred over inference.
            # Based on recent practices, often you define the manager agent and Crew infers it,
            # or you pass manager_agent=self.crew_manager() explicitly. Let's be explicit.
            manager_agent=self.crew_manager(),
			verbose=True, 
            # memory=True # Consider enabling memory for context persistence within a single cycle run
		)
