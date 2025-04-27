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

class State(TypedDict):
    mesages: Annotated[list, add_messages]

graph_builder=StateGraph(State)

model=chat
embeddings=embedding_model
tavily_api_key=os.getenv('TAVILY_API_KEY')

tool = TavilySearchResults(max_results=2)
tools=[tool]
model_with_tools=model.bind_tools(tools)

def chatbot(state:State)->State:
    response=model_with_tools.invoke(state['mesages'])
    return {
        'mesages': [response]
    }

graph_builder.add_node('chatbot', chatbot)

tool_node=ToolNode(tools=tools)
graph_builder.add_node('tools', tool_node)

graph_builder.add_edge(START, 'chatbot')

graph_builder.add_conditional_edges('chatbot', tools_condition )

graph_builder.add_edge('tools', 'chatbot')

graph=graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about the death of Papa Franceco?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
    
    
