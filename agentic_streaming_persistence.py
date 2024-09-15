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

