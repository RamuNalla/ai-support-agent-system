import logging
import time
from typing import TypedDict, Annotated, List, Union, Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document 
from langchain_core.tools import Tool 
from app.rag.embeddings import EmbeddingsGenerator # custom module for embeddings
from app.rag.vector_db import VectorDBManager # custom module for FAISS
from app.config.settings import settings # settings to get FAISS_INDEX_PATH
import json # Json for parsing tool arguments
from app.observability.metrics import RAG_RETRIEVAL_LATENCY, TOOL_CALL_COUNTER, CHAT_ERRORS_TOTAL

logger = logging.getLogger(__name__) # Initialize logger

class AgentState(TypedDict):                                            # Define the state of our graph
    messages: Annotated[list, lambda left, right: left + right]         # List of messages 
    relevant_docs: List[Document]                                       # to store retrieved documents (LangChain Document objects)
    tool_calls: List[Dict[str, Any]]                                    # To store tool calls from LLM
    tool_output: Any                                                    # To store the result of a tool execution
    clarifying_question: Optional[str]                                  # Field to hold a clarifying question
class Agent:                                                            # Initializes the agent with the Gemini LLM.
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not provided.")

        logger.info("Initializing Gemini LLM and RAG, TOoling components")
        try:
            self.calculator_tool = Tool.from_function(                  # Define the Calculator Tool
                func=self._execute_calculator,
                name="calculator",
                description="""Performs basic arithmetic calculations.
                Input should be a JSON string with a 'expression' key, where the value is a string representing the mathematical expression to evaluate.
                Example: {"expression": "2 + 2 * 3"}
                Returns the numerical result of the expression.
                Use this tool for mathematical calculations.
                """
            )

            self.weather_tool = Tool.from_function(                     # Define the Weather Tool (Mock Implementation)
                func=self._execute_weather,
                name="weather",
                description="""Retrieves current weather information for a specified city.
                Input should be a JSON string with a 'city' key, where the value is a string representing the city name.
                Example: {"city": "Mumbai"}
                Returns a string with weather data (e.g., temperature, conditions).
                Use this tool to get current weather conditions.
                """
            )

            self.tools = [self.calculator_tool, self.weather_tool]      # List of all tools available to the LLM
            
            
            self.llm = ChatGoogleGenerativeAI(                          # Initialize ChatGoogleGenerativeAI with the provided API key
                model="gemini-1.5-flash",                               # Using gemini-flash for general text generation
                google_api_key=gemini_api_key,
                tools = self.tools
            )
            logger.info("Gemini LLM initialized.")
            
            self.embeddings_generator = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")  # Initialize Embeddings Generator.
            self.embeddings_model = self.embeddings_generator.get_embeddings_model()
            logger.info("Embeddings model initialized.")

            self.vector_db_manager = VectorDBManager(index_path=settings.FAISS_INDEX_PATH)  # Initialize FAISS Vector DB Manager
            logger.info("FAISS Vector DB Manager initialized and index loaded.")

        except Exception as e:
            logger.error(f"Error initializing Agent components: {e}", exc_info=True)
            raise                                                       # Re-raise the exception to prevent the app from starting incorrectly


    def _execute_calculator(self, expression: str) -> str:                      # Internal helper to execute the calculator tool. Evaluates a mathematical expression.

        logger.info(f"Executing Calculator with expression: '{expression}'")
        try:
            result = str(eval(expression))                               # Using eval() for simplicity
            logger.info(f"Calculator result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing Calculator tool for expression '{expression}': {e}", exc_info=True)
            return f"Error: Could not calculate. {e}"



    def _execute_weather(self, city: str) -> str:                           # Internal helper to execute the weather tool. Provides mock weather data for a given city.

        logger.info(f"Executing Weather tool for city: '{city}'")
        
        mock_weather_data = {                                               # This is a mock implementation. In a real scenario, you'd call an external weather API.
            "Hyderabad": "Sunny, 25°C, Light breeze",
            "Mumbai": "Cloudy, 20°C, Chance of rain",
            "Chennai": "Partly cloudy, 28°C, High humidity",
            "Bengaluru": "Monsoon showers, 23°C, Moderate wind"
        }
        weather_info = mock_weather_data.get(city, "Weather data not available for this city. Please try London, New York, Tokyo, or Bengaluru.")
        logger.info(f"Weather tool result for {city}: {weather_info}")
        return weather_info



    def retrieve_documents(self, state: AgentState) -> AgentState:              # Langgraph node to retrive relevant documents from the FAISS index
        start_time = time.time()
        latest_human_message = None
        for msg in reversed(state['messages']):                                 # Find the most recent human message in the conversation history.
            if isinstance(msg, HumanMessage):
                latest_human_message = msg.content
                break

        if not latest_human_message:
            logger.warning("No human message found for retrieval. Skipping retrieval.")
            RAG_RETRIEVAL_LATENCY.observe(0)
            return {
                "relevant_docs": [], 
                "messages": state["messages"], 
                "tool_calls": [], 
                "tool_output": None, 
                "clarifying_question": None
            }                                                                       # If no human message, return an empty list of documents.

        logger.info(f"Retrieving documents for query: '{latest_human_message[:50]}...'")
        
        try:
            query_vector = self.embeddings_model.embed_query(latest_human_message)          # Generate an embedding for the user's query.
            search_results = self.vector_db_manager.search_vectors(query_vector, limit=5)   # Search the FAISS index for the top 3 most similar documents.

            relevant_docs = []
            for res in search_results:                                                      # Convert the raw search results (payloads) into LangChain Document objects. This makes the retrieved information consistent and easy to pass to the LLM.
                content = res.get('content', '')
                source = res.get('source', 'unknown')
                score = res['score']
                relevant_docs.append(Document(page_content=content, metadata={"source": source, "score": score}))

            logger.info(f"Retrieved {len(relevant_docs)} documents.")
            return {
                "relevant_docs": relevant_docs, 
                "messages": state["messages"], 
                "tool_calls": [], 
                "tool_output": None, 
                "clarifying_question": None
            }                                                                               # Return the retrieved documents to update the 'relevant_docs' channel in the AgentState.
        
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            CHAT_ERRORS_TOTAL.labels(error_type="rag_retrieval_error").inc()                # Increment error counter
            RAG_RETRIEVAL_LATENCY.observe(time.time() - start_time)
            return {
                "relevant_docs": [], 
                "messages": state["messages"], 
                "tool_calls": [], 
                "tool_output": None, 
                "clarifying_question": None
            } # Return empty list on error 
        


    def generate_response_or_tool_call(self, state: AgentState) -> AgentState:               # LangGraph Node: Generates a response using the LLM, incorporating RAG context, or generates a tool call if the LLM decides to use a tool.

        messages = state['messages']
        relevant_docs = state['relevant_docs']
        tool_output = state.get('tool_output')                                                  # Get tool output if available from previous step

        context_str = ""                                                                        # Prepare context from retrieved documents
        if relevant_docs:
            context_str = "\n\nRelevant Context:\n"
            for i, doc in enumerate(relevant_docs):
                context_str += f"--- Document {i+1} ---\n"                                      # Simplified separator
                context_str += doc.page_content + "\n"
            context_str += "\n"
            logger.info(f"Adding {len(relevant_docs)} documents to LLM context (without messy metadata).")


        tool_output_str = ""                                                        # Add tool output to context if available
        if tool_output:
            tool_output_str = f"\n\nTool Output:\n{tool_output}\n"                  # Ensure tool_output is a string or can be safely converted
            logger.info(f"Adding tool output to LLM context: {tool_output_str[:100]}...")

        system_prompt = (                                           # Construct the system prompt - RAG context and tool output
            "You are an expert AI Support Agent specializing in Kubernetes and general technical inquiries. "
            "Your core objective is to provide precise, concise, and actionable answers to user questions. "
            "Always prioritize clarity and helpfulness.\n\n"
            "**Available Tools:**\n"
            "1. `calculator`: Performs basic arithmetic calculations. Input: `{\"expression\": \"2 + 2 * 3\"}`. Use this for mathematical calculations.\n"
            "2. `weather`: Retrieves current weather information for a specified city. Input: `{\"city\": \"London\"}`. Use this to get current weather conditions.\n\n"
            "**Instructions:**\n"
            "1. **Leverage Provided Context:** Always prioritize answering questions using the 'Relevant Context' provided from the internal knowledge base. Synthesize information from these documents to formulate your response.\n"
            "2. **Strategic Tool Utilization:** If the 'Relevant Context' is insufficient, or if the query explicitly requires real-time data or computations, utilize the available tools.\n"
            "   - **Strict Tool Call Format:** When invoking a tool, use the exact function call syntax and JSON input format specified in the tool's description. Do not deviate from this format.\n"
            "   - Example tool call: `calculator.calculate(expression='15 * 3')`\n"
            "   - Example tool call: `weather.get_current(city='New York')`\n"
            "3. **Integrate Tool Results:** If a tool is executed, its output will be provided. Incorporate this output directly and clearly into your final answer to address the user's original query.\n"
            "4. **Handle Ambiguity/Missing Information:** If a query is unclear, lacks sufficient detail for a precise answer, or cannot be fully resolved with current context/tools, you **MUST** ask a clarifying question. Prefix all clarifying questions with 'CLARIFY: '.\n"
            "   Example: `CLARIFY: What is the application you are looking for (e.g., android, ios, web-app)?`\n"
            "5. **Graceful Fallback:** If, after attempting to use tools and seeking clarification, you still cannot provide a complete answer, politely state your limitations and suggest escalating to a human support agent.\n\n"
            f"{context_str}"        # Inject RAG context
            f"{tool_output_str}"    # Inject tool output
        )

        llm_messages = [SystemMessage(content=system_prompt)] + messages            # Construct the full list of messages for the LLM. The system message with context comes first, followed by the actual conversation history.

        logger.info(f"Calling LLM for response or tool call decision. Messages: {llm_messages}")
        try:
            response = self.llm.invoke(llm_messages)
            logger.info(f"LLM response received. Type: {type(response).__name__}, Content: {response.content[:100]}...")

            if isinstance(response, AIMessage) and response.content.startswith("CLARIFY: "):        # Store the clarifying question in the state and return it
                logger.info("LLM responded with a clarifying question.")
                return {"messages": [response], "clarifying_question": response.content.replace("CLARIFY: ", "").strip(), "relevant_docs": relevant_docs, "tool_calls": [], "tool_output": None} # Preserve relevant_docs when clarifying
            
            return {"messages": messages + [response], "relevant_docs": relevant_docs, "tool_calls": [], "tool_output": None, "clarifying_question": None} # The LLM's response might be a direct answer or a tool call.
        except Exception as e:
            logger.error(f"Error calling LLM for response or tool call: {e}", exc_info=True)
            CHAT_ERRORS_TOTAL.labels(error_type="llm_call_error").inc()
            return {"messages": messages + [AIMessage(content=f"Error: Could not get a response from the AI. {e}")], "relevant_docs": relevant_docs, "tool_calls": [], "tool_output": None, "clarifying_question": None}


    def execute_tool(self, state: AgentState) -> AgentState:                # LangGraph Node: Executes the tool calls generated by the LLM.

        latest_ai_message = state['messages'][-1]
        tool_calls = latest_ai_message.tool_calls                           # Access tool_calls from the AI message

        if not tool_calls:
            logger.warning("No tool calls found in the latest AI message. Skipping tool execution.")
            return {
                "tool_output": None, 
                "messages": state["messages"], 
                "relevant_docs": state["relevant_docs"], 
                "tool_calls": [], 
                "clarifying_question": None
            }

        logger.info(f"Executing tool calls: {tool_calls}")
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']                                   # This is already a dictionary from LLM

            try:
                if tool_name == self.calculator_tool.name:                  # Calculator expects an 'expression' key with a string
                    expression = tool_args.get("expression")
                    if expression is None:
                        raise ValueError("Calculator tool requires an 'expression' argument.")
                    output = self.calculator_tool.invoke({"expression": expression})
                elif tool_name == self.weather_tool.name:                   # Weather tool expects a 'city' key with a string
                    city = tool_args.get("city")
                    if city is None:
                        raise ValueError("Weather tool requires a 'city' argument.")
                    output = self.weather_tool.invoke({"city": city})
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                tool_outputs.append(output)
                TOOL_CALL_COUNTER.labels(tool_name=tool_name, status="success").inc()
                logger.info(f"Tool '{tool_name}' executed successfully. Output: {str(output)[:100]}...")
            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}' with args {tool_args}: {e}"
                logger.error(error_msg, exc_info=True)
                tool_outputs.append({"error": error_msg})
                TOOL_CALL_COUNTER.labels(tool_name=tool_name, status="error").inc()         # Increment tool call error counter
                CHAT_ERRORS_TOTAL.labels(error_type="tool_execution_error").inc()
        
        tool_message = ToolMessage(                                                         # Add a ToolMessage to the conversation history with the tool's output
            content=json.dumps(tool_outputs),                                               # Convert list of outputs to JSON string for content
            tool_call_id=latest_ai_message.tool_calls[0]['id']                              # Link to the first tool call
        )
        logger.info(f"ToolMessage created: {tool_message.content[:100]}...")
        
        return {"messages": state["messages"] + [tool_message], "tool_output": tool_outputs, "relevant_docs": state["relevant_docs"], "tool_calls": [], "clarifying_question": None} # Return the tool output to update the state.


    def should_continue(self, state: AgentState) -> str:                                    # LangGraph Conditional Edge - If the latest message from the LLM contains tool calls, continue to execute tools.

        latest_message = state['messages'][-1]

        if state.get('clarifying_question'):
            logger.info("LLM asked a clarifying question. Ending graph to await user input.")
            return "clarify"                                                                    # path for clarifying questions

        if latest_message.tool_calls:                                                           # Check if the latest message has tool_calls (indicating the LLM wants to use a tool)
            logger.info("LLM requested tool call. Continuing to execute_tool node.")
            return "continue"
        else:
            logger.info("LLM provided a final answer. Ending graph.")
            return "end"



    def build_graph(self): # Graph building
        """
        Flow:
        1. Human Input (Entry Point)
        2. Retrieve Documents (RAG)
        3. Generate Response OR Tool Call (LLM decision)
        4. Conditional Edge:
            - If Clarifying Question -> END (await user input)
            - If Tool Call -> Execute Tool
            - If Final Answer -> END
        5. After Tool Execution -> Loop back to Generate Response (to synthesize tool output)
        """
        
        workflow = StateGraph(AgentState) 

        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("generate_response_or_tool_call", self.generate_response_or_tool_call) 
        workflow.add_node("execute_tool", self.execute_tool)
        
        workflow.set_entry_point("retrieve_documents") 
        
        workflow.add_edge("retrieve_documents", "generate_response_or_tool_call")
        
        workflow.add_conditional_edges( # Conditional edge from LLM decision node
            "generate_response_or_tool_call",
            self.should_continue,
            {
                "continue": "execute_tool", # If tool call, go to execute_tool
                "clarify": END,
                "end": END # If final answer, end
            }
        )
        
        workflow.add_edge("execute_tool", "generate_response_or_tool_call") # Define an edge from 'call_llm' to END, meaning the graph finishes after the LLM call

        app = workflow.compile() # Compile the workflow into a runnable LangGraph application
        logger.info("LangGraph workflow compiled.")
        return app
