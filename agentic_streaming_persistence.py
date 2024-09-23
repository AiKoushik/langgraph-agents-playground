from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

import os

#necessary for Tavily. on earlier commits, was also necessary for OpenAI
_ = load_dotenv(find_dotenv())

#setting up prompt, model, tool and memory checkpointer for persistence
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want, but give the tool enough context. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
#model = ChatOpenAI(model = "gpt-4o-mini")    #gpt-4o-mini #gpt-4o #gpt-4-turbo #gpt-3.5-turbo
model = ChatOllama(
    model = "llama3.1", #llama3.1   #llama3-groq-tool-use
    temperature = 0)
tool = TavilySearchResults(max_results = 2)

#os.getenv() is necessary here, but not for API keys since libraries regarding API keys automatically looks for those in Python environment where it's loaded after calling load_env()
DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:{os.getenv('POSTGRES_DB_PORT')}/{os.getenv('POSTGRES_DB_NAME')}?sslmode=disable"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
thread_id = "area_comparison_dhaka_khulna_5"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system = ""):
        self.system = system
        
        #initialise graph wiht state
        graph = StateGraph(AgentState)
        
        #adding nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        #adding conditional edge starting from llm node based on result from exists_action() method's return
        graph.add_conditional_edges("llm", self.exists_aciton,
            {
                True: "action",
                False: END
            }
        )
        #adding edge back to llm after tool calling
        graph.add_edge("action", "llm")
        
        #setting entrypoint into graph
        graph.set_entry_point("llm")

        #compiling graph with checkpointer to save state
        self.graph = graph.compile(checkpointer = checkpointer)
        
        #binding tools to model
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    #node: llm; calling large language model
    def call_openai(self, state: AgentState):
        #getting all messages
        messages = state['messages']
        
        #appending System Message from before
        if self.system:
            messages = [SystemMessage(content = self.system)] + messages
        
        #fetching and returning new message obtained from llm call
        message = self.model.invoke(messages)
        return {'messages': [message]}

    #node: action; performing tool calling for all suggested tools
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"[tool call]: calling {t}")
            #performing tool calling by fetching tool name and args
            result = self.tools[t['name']].invoke(t['args'])

            #formatting and appending tool_message to results
            tool_message = ToolMessage(tool_call_id = t['id'], name = t['name'], content = str(result))
            results.append(tool_message)

        return {'messages': results}

    #conditional_edge; determining if tool calling is necessary    
    def exists_aciton(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0


def ask_question(question, thread_id):
    messages = [HumanMessage(content = question)]
    thread = {"configurable": {"thread_id": thread_id}}
    #streaming output after every message
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print("\n\n")
            print(v['messages'])


#creating and invoking the agent with postgres memory checkpointer with connection pooling
with ConnectionPool(
    conninfo = DB_URI,
    max_size = 20,
    kwargs = connection_kwargs
) as pool:
    checkpointer = PostgresSaver(pool)
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    checkpointer.setup()
    
    abot = Agent(model = model, tools = [tool], checkpointer = checkpointer, system = prompt)

    ask_question("What is the area  of Dhaka City?", thread_id = thread_id)
    ask_question("What about that of Khulna city?", thread_id = thread_id)
    ask_question("Which is bigger and how many times??", thread_id = thread_id)

