import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)                # Initialize logger

class AgentState(TypedDict):                        # Agent graph state
    messages: Annotated[list, lambda x: x]          # List of messages (HumanMessage, AIMessage)

class Agent:                                        # Initializes the agent with Google Gemini LLM.
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not provided.")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        logger.info("Gemini LLM initialized.")

    def call_llm(self, state: AgentState) -> AgentState:        # Invokes the LLM with the current conversation messages.
        messages = state['messages']
        logger.info(f"Calling LLM with messages: {messages}")
        try:
            response = self.llm.invoke(messages)
            logger.info(f"LLM response received: {response.content[:100]}...")
            return {"messages": messages + [response]}
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return {"messages": messages + [AIMessage(content=f"Error: Could not get a response from the AI. {e}")]}

    def build_graph(self):                      # Builds LangGraph state machine.

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("call_llm", self.call_llm)

        # Entry point
        workflow.set_entry_point("call_llm")

        # Exit point 
        workflow.add_edge("call_llm", END)

        app = workflow.compile()
        logger.info("LangGraph workflow compiled.")
        return app

