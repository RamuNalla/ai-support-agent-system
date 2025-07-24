import streamlit as st
import logging
from typing import Dict, Any, List
import os  

from services.agent_api_client import AgentAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')      # Configure logging for Streamlit app
logger = logging.getLogger(__name__)

AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8000")         # Get agent service URL from environment variable
agent_client = AgentAPIClient(base_url=AGENT_SERVICE_URL)

st.set_page_config(page_title="AI Support Agent Chat", layout="centered")           # Streamlit App Setup 
st.title("💬 AI Support Agent")
st.caption("Powered by LangGraph, FastAPI, and Streamlit")

if "messages" not in st.session_state:                                              # Initialize chat history in session state (if the very first run for a new user session) if not already present
    st.session_state["messages"] = []

for msg in st.session_state.messages:                                   # Display existing chat messages
    if msg["type"] == "human":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["type"] == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["type"] == "clarifying_question":
        with st.chat_message("assistant"):
            st.markdown(f"**Clarification Needed:** {msg['content']}")


if prompt := st.chat_input("Ask me anything..."):                               # Chat input
    st.session_state.messages.append({"type": "human", "content": prompt})      # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):                                          # API interaction logic
        with st.spinner("Thinking..."):
            # Prepare chat history for the API call
            # We only send human and AI messages, not tool messages or clarifying questions
            # as the agent's core.py will rebuild the full state.
            api_chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude the current human message
                if msg["type"] == "human" or msg["type"] == "ai":
                    # For AI messages, pass tool_calls if they exist
                    api_msg = {"type": msg["type"], "content": msg["content"]}
                    if msg["type"] == "ai" and "tool_calls" in msg:
                        api_msg["tool_calls"] = msg["tool_calls"]
                    api_chat_history.append(api_msg)
                

            # Call the agent API
            logger.info(f"Calling agent API with prompt: '{prompt[:50]}...' and history length: {len(api_chat_history)}")
            api_response = agent_client.chat(message=prompt, chat_history=api_chat_history)
            logger.info(f"API response received: {api_response.get('response', '')[:50]}..., Clarifying: {api_response.get('clarifying_question')}")

            # Handle different types of agent responses
            if api_response.get("clarifying_question"):
                clarifying_q = api_response["clarifying_question"]
                st.markdown(f"**Clarification Needed:** {clarifying_q}")
                st.session_state.messages.append({"type": "clarifying_question", "content": clarifying_q})
            elif api_response.get("response"):
                agent_response_content = api_response["response"]
                st.markdown(agent_response_content)
                # Add AI response to chat history
                st.session_state.messages.append({"type": "ai", "content": agent_response_content})
                # If the AI response contained tool calls (e.g., from an intermediate step),
                # store them in the session state for the next turn's history.
                if "tool_calls" in api_response and api_response["tool_calls"]:
                    st.session_state.messages[-1]["tool_calls"] = api_response["tool_calls"]
            else:
                st.error("Received an empty or invalid response from the agent.")
                st.session_state.messages.append({"type": "ai", "content": "Error: Received an empty or invalid response."})


# --- UI Enhancements (Feedback Buttons - Placeholder) ---
# You can add a simple feedback mechanism here.
# For example, thumbs up/down buttons after each AI response.
# This would require another API endpoint in agent_service/app/api/v1/agent_api.py
# and a corresponding method in AgentAPIClient.

# Example:
# if st.session_state.messages and st.session_state.messages[-1]["type"] == "ai":
#     col1, col2 = st.columns([1, 10])
#     with col1:
#         if st.button("👍", key="thumbs_up"):
#             st.success("Thanks for the feedback!")
#             # Call agent_client.send_feedback(last_ai_message, "positive")
#     with col2:
#         if st.button("👎", key="thumbs_down"):
#             st.error("Sorry to hear that. How can I improve?")
#             # Call agent_client.send_feedback(last_ai_message, "negative")

# --- Display Sources/Citations (Placeholder) ---
# If your agent's response includes source information (e.g., from RAG),
# you would parse it here and display it. This requires the agent_api to return sources.
# Example:
# if "sources" in api_response and api_response["sources"]:
#     st.subheader("Sources:")
#     for source in api_response["sources"]:
#         st.write(f"- {source.get('title', 'N/A')}: {source.get('url', 'N/A')}")

