import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)                # Initialize logger
class AgentState(TypedDict):                        # Define the state of our graph
    messages: Annotated[list, lambda x: x]          # List of messages (HumanMessage, AIMessage)

class Agent:                                        # Initializes the agent with the Hugging Face LLM.
    def __init__(self, model_id: str, hf_token: str):
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not provided.")
        if not model_id:
            raise ValueError("HUGGINGFACE_MODEL_ID is not provided.")

        import os
        os.environ["HF_TOKEN"] = hf_token

        logger.info(f"Initializing Hugging Face model: {model_id}")
        try:
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)         # Load the tokenizer and model
            model = AutoModelForCausalLM.from_pretrained(               # device_map="auto" will attempt to use GPU if available
                model_id,
                torch_dtype=torch.bfloat16,                             # Use bfloat16 for memory efficiency
                device_map="auto",
                trust_remote_code=True                                  
            )

            self.pipe = pipeline(                                       # Hugging Face text generation pipeline
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,                                     # Max tokens to generate in response
                do_sample=True,                                         # Enable sampling for more creative responses
                temperature=0.3,                                        # Controls randomness
                top_k=50,                                               # Top-k sampling
                top_p=0.95,                                             # Nucleus sampling
                return_full_text=False                                  # Only return the generated text, not the full prompt
            )

            self.llm = HuggingFacePipeline(pipeline=self.pipe)          # Wrap the pipeline with LangChain's HuggingFacePipeline
            logger.info(f"Hugging Face LLM '{model_id}' initialized.")

            self.prompt = ChatPromptTemplate.from_messages([            # Define a simple chat prompt template
                ("system", "You are a helpful AI support agent that answers questions. Be concise and accurate."),
                ("human", "{question}"),
            ])

        except Exception as e:
            logger.error(f"Error initializing Hugging Face LLM: {e}", exc_info=True)
            raise

    def call_llm(self, state: AgentState) -> AgentState:                # Invokes the LLM with the current conversation messages.

        messages = state['messages']
        # we pass the last human message content and rely on the prompt template to format it correctly.  LangChain's ChatPromptTemplate will handle the conversion from BaseMessage to string.
        question = messages[-1].content
        logger.info(f"Calling LLM with question: '{question}'")
        try:
            chain = self.prompt | self.llm                              # Create a chain with the prompt and LLM
            response = chain.invoke({"question": question})
            logger.info(f"LLM response received: {response[:100]}...")
            return {"messages": messages + [AIMessage(content=response)]}
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return {"messages": messages + [AIMessage(content=f"Error: Could not get a response from the AI. {e}")]}



    def build_graph(self):                                              # Builds LangGraph state machine.

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

