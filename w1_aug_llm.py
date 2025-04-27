from dotenv import load_dotenv
import os
import ast
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),  # Use azure_endpoint instead of openai_api_base
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    deployment_name=os.getenv('OPENAI_LLM_DEPLOYMENT_NAME'),  # deployment_name is fine
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_type=os.getenv('OPENAI_API_TYPE'),
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm=chat
# Schema for structured output
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")   ##None come default value
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(output)

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

# Get the tool call
print(msg.tool_calls)