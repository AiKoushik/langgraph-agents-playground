from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
#from langchain_community.llms import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.sqlite import SqliteSaver

_ = load_dotenv(find_dotenv())

tool = TavilySearchResults(max_results = 2)
memory = SqliteSaver.from_conn_string(":memory")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system = ""):
        self.system = system
        
        #initialise graph wiht state
        graph = StateGraph()
        
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
        retun {'messages': [message]}

    #node: action; performing tool calling for all suggested tools
    def take_action(self, state: AgentState):
        tool_calls = state['message'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"[tool call]: calling {t}")
            #performing tool calling by fetching tool name and args
            result = self.tools[t['name']].invoke(t['args'])

            #formatting and appending tool_message to results
            tool_message = ToolMessage(tool_call_id = t['id'], name =)t['name'], content = str(result)
            results.append(tool_message)

        return {'messages': results}

    #determining if tool calling is necessary    
    def exists_aciton(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tools_calls) > 0
