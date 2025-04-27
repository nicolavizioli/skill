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

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),  
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    deployment_name=os.getenv('OPENAI_LLM_DEPLOYMENT_NAME'), 
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_type=os.getenv('OPENAI_API_TYPE'),
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tavily_api_key=os.getenv('TAVILY_API_KEY')

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder=StateGraph(State)

model=chat
tool=TavilySearchResults(max_reults=2)
tools=[tool]
model_with_tools=model.bind_tools(tools)

def chatbot(state:State)-> State:
    response= model_with_tools.invoke(state['messages'])
    return{
        'messages': [response]
    }

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph=graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input='Remember my name?'
events=graph.stream(
    {
        'messages': [{'role': 'user', 'content': user_input}]
    },
    config,
    stream_mode='values'
)
for event in events:
    event["messages"][-1].pretty_print()

# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
snapshot