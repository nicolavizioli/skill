from dotenv import load_dotenv
import os
import ast
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Annotated

from langchain.chat_models import init_chat_model

from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

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

model=chat
embeddings=embedding_model
tavily_api_key=os.getenv('TAVILY_API_KEY')

class State(TypedDict):
    messages: Annotated[list, add_messages] 

graph_builder=StateGraph(State)

# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"] 

@tool
def human_assistance(query: str)-> str:
    """Request assistance from a human."""
    human_response=interrupt({'query':query})
    return human_response['data']

search=TavilySearchResults(max_results=2)

tools=[search, human_assistance]

llm_with_tools=model.bind_tools(tools)

def chatbot(state: State)-> State:
    response=llm_with_tools.invoke(state['messages'])
    assert len(response.tool_calls)<=1
    return{
        'messages' : [response]
    }

graph_builder.add_node('chatbot', chatbot)
tool_node=ToolNode(tools=tools)
graph_builder.add_node('tools', tool_node)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition)
graph_builder.add_edge('tools', 'chatbot')

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot.next)

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()