import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)            # Initialize logger

class AgentState(TypedDict):                                        # Define the state of our graph
    messages: Annotated[list, lambda left, right: left + right]     # List of messages (HumanMessage, AIMessage)

class Agent:                                    # Initializes the agent with the Gemini LLM.
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not provided.")

        logger.info("Initializing Gemini LLM")
        try:
            self.llm = ChatGoogleGenerativeAI(              # Initialize ChatGoogleGenerativeAI with the provided API key
                model="gemini-1.5-flash",                   # Using gemini-flash for general text generation
                google_api_key=gemini_api_key
            )
            logger.info("Gemini LLM initialized.")
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {e}", exc_info=True)
            raise                                           # Re-raise the exception to prevent the app from starting incorrectly

    def call_llm(self, state: AgentState) -> AgentState:    # Invokes the LLM with the current conversation messages.

        messages = state['messages']
        logger.info(f"Calling LLM with messages: '{messages}'")
        try:
            response = self.llm.invoke(messages)            # Invoke the Gemini LLM with the list of messages
            logger.info(f"LLM response received: {response.content[:100]}...")
            return {"messages": [response]}      # Append the AI's response to the conversation history
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return {"messages": [AIMessage(content=f"Error: Could not get a response from the AI. {e}")]}

    def build_graph(self):                                  # Graph building

        workflow = StateGraph(AgentState)

        workflow.add_node("call_llm", self.call_llm)        # Define the 'call_llm' node which executes the self.call_llm method
        workflow.set_entry_point("call_llm")                # Set 'call_llm' as the entry point of the graph
        workflow.add_edge("call_llm", END)                  # Define an edge from 'call_llm' to END, meaning the graph finishes after the LLM call

        app = workflow.compile()                            # Compile the workflow into a runnable LangGraph application
        logger.info("LangGraph workflow compiled.")
        return app

