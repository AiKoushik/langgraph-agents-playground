from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.vectorstores import FAISS

model_name = "llama3.1"

#setting up vectorstore and retriever
vectorstore = FAISS.from_texts(["I love reading Science Fictions and Detective Novels. My favourite writes in these genres are Agatha Christie and Isaac Asimov"],
                               embedding = OllamaEmbeddings(model = model_name))
data_retriever = vectorstore.as_retriever()

#setting up the template with context and question
template = """
Answer the question asked from the give context: {context}
Question: {question}
"""

#setting up the prompt
prompt = ChatPromptTemplate.from_template(template)

def glorify(text):
    #returning as AIMessage since the text is also received by this function as AIMessage
    return AIMessage("\n>>> " + text.content + " <<<\n")

#setting up the chain
chain = RunnableParallel(
    {"context": data_retriever, "question": RunnablePassthrough()}) | prompt | ChatOllama(model = model_name) | RunnableLambda(glorify) | StrOutputParser()


output = chain.invoke("Who are my favourite writers in which genres?")
print(output)
