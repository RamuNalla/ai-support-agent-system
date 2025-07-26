import streamlit as st
import requests
import json
import os
import uuid

# Configuration for the agent service URL
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Support Agent", page_icon="🤖")

st.title("🤖 Kubernetes AI Support Agent")
st.caption("Powered by Langgraph, FastAPI and Streamlit, Developed by RN")


# Initialize session ID if not already present
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Generate a unique session ID

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to send feedback to the FastAPI agent service
def send_feedback(message_content: str, feedback_type: str, comment: str = None):
    """
    Sends user feedback (like/dislike) to the FastAPI agent service.
    """
    payload = {
        "session_id": st.session_state.session_id, # Use the consistent session ID
        "message_content": message_content,
        "feedback_type": feedback_type,
        "comment": comment
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(f"{AGENT_SERVICE_URL}/api/v1/feedback", json=payload, headers=headers)
        response.raise_for_status()
        st.success(f"Feedback '{feedback_type}' submitted!")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the AI Agent Service at {AGENT_SERVICE_URL}. Feedback not sent.")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error sending feedback: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred while sending feedback: {e}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Document {i+1}:**")
                    st.write(f"Content: {source.get('page_content', 'N/A')}")
                    if source.get('metadata'):
                        st.write(f"Metadata: {source['metadata']}")
                    st.markdown("---")
        
        # Add feedback buttons only for assistant messages
        if message["role"] == "assistant":
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if st.button("👍", key=f"like_{message['content'][:30]}_{len(st.session_state.messages)}"): # Unique key
                    send_feedback(message["content"], "positive")
            with col2:
                if st.button("👎", key=f"dislike_{message['content'][:30]}_{len(st.session_state.messages)}"): # Unique key
                    send_feedback(message["content"], "negative")


# Function to call the FastAPI agent service
def call_agent_service(prompt: str, chat_history: list):

    try:
        # Format chat history for the agent service
        formatted_chat_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_chat_history.append({"type": "human", "content": msg["content"]})
            elif msg["role"] == "assistant":
                ai_msg_content = msg["content"]
                ai_msg_tool_calls = msg.get("tool_calls", [])
                formatted_chat_history.append({"type": "ai", "content": ai_msg_content, "tool_calls": ai_msg_tool_calls})
            elif msg["role"] == "tool":
                formatted_chat_history.append({"type": "tool", "content": msg["content"], "tool_call_id": msg["tool_call_id"]})
            elif msg["role"] == "system":
                formatted_chat_history.append({"type": "system", "content": msg["content"]})


        payload = {
            "message": prompt,
            "chat_history": formatted_chat_history
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{AGENT_SERVICE_URL}/api/v1/chat", json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the AI Agent Service at {AGENT_SERVICE_URL}. Please ensure it is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error from AI Agent Service: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Chat input from user
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent service and get response
    with st.spinner("Thinking..."):
        agent_response_data = call_agent_service(prompt, st.session_state.messages)

    if agent_response_data:
        response_content = agent_response_data.get("response", "")
        clarifying_question = agent_response_data.get("clarifying_question")
        relevant_docs = agent_response_data.get("relevant_docs", []) # Get relevant docs

        if clarifying_question:
            ai_message_content = clarifying_question
        else:
            ai_message_content = response_content

        # Add AI message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_message_content,
            "sources": relevant_docs # Store sources with the message
        })

        # Display AI message in chat UI (will be rendered by the loop above)
        st.rerun() # Rerun to display the new message and buttons
