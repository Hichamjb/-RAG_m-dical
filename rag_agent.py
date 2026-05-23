from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import os

from database import vector_store
ma_cle_xai = os.getenv("XAI_API_KEY") 
    


# 1. Définition de l'outil de recherche (Retriever Tool)
@tool
def retrieve_patient_records(patient_id: str, query: str) -> str:
    """Récupère les dossiers médicaux d'un patient spécifique. À utiliser pour répondre aux questions."""
    print(f"   [Tool Call] 🔍 Recherche pour le Patient : {patient_id} | Requête : {query}")
    
    retriever = vector_store.as_retriever(
        search_kwargs={
            "filter": {"patient_id": patient_id}, 
            "k": 3
        }
    )
    docs = retriever.invoke(query)
    
    if not docs:
        return "Aucun document médical trouvé pour ce patient."
    
    context = "\n\n".join([f"Document (Fichier : {d.metadata['file_id']}):\n{d.page_content}" for d in docs])
    return context

# 2. Configuration du LLM
tools = [retrieve_patient_records]
llm = ChatXAI(xai_api_key=ma_cle_xai, model="grok-4.20-0309-non-reasoning", temperature=0.01).bind_tools(tools)

# 3. Définition de l'état
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 4. Nœuds du graphe
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def take_action(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "retrieve_patient_records":
            result = retrieve_patient_records.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(result), name=tool_call["name"], tool_call_id=tool_call["id"])
            )
            
    return {"messages": results}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, 'tool_calls', None):
        return True
    return False

# 5. Construction du Graphe
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()
print("✅ Agent LangGraph RAG compilé avec succès !")