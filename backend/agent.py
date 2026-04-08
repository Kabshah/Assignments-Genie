from typing import TypedDict,List,Literal
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from pydantic import BaseModel,Field
from config import TAVILY_API_KEY,PINECONE_API_KEY,GROQ_API_KEY
from langgraph.graph import StateGraph,END
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
# List[BaseMessage] ka matlab hai:

# 👉 messages ek list hogi
# 👉 us list ke andar BaseMessage type ke objects honge
# Real example (LangGraph agent state)
# state = {
#     "messages": [
#         HumanMessage(content="What's the weather?"),
#         AIMessage(content="Let me check...")
#     ]
# }
#state → messages → LLM → tool → updated state

#Base msg sy ap convo maintain krr paty pori in list
# messages = [
#    HumanMessage, User ka message
#    AIMessage, AI ka reply
#    SystemMessage, System instruction
#    ToolMessage, Tool ka output
# ]

os.environ["GROQ_API_KEY"]=GROQ_API_KEY

class RouteDecision(BaseModel):
    route:Literal["rag","web","answer","end"]
    reply:str | None = Field(None, description="Filled only when route == 'end'.")

class JudgeRag(BaseModel):
    #embedding sy jo vectors receive howy are they even sufficient h yaa nai
    sufficient:bool = Field(...,description="True if restrieved information is sufficient to answer the user's question,False otherwise")

router_llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0).with_structured_output(RouteDecision)
judge_llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0).with_structured_output(JudgeRag)
answer_llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

# total=False When you write: Keys are optional now.You can create a dict that only has some of these keys.

# Example: state = {
#     "messages": [],
#     "route": "rag"}
# So in your RAG agent, this makes sense because:
# Some nodes might only add "rag" or "web" temporarily.
# Not all keys are guaranteed to exist at every point in the pipeline.
# Using total=False avoids type-checker errors when you construct partial states during routing.
class AgentState(TypedDict,total=False):
    messages:List[BaseMessage]
    route: Literal["rag","web","answer","end"]
    rag :str # output from rag node
    web:str # output from web search
    web_search_tool:bool

# Nodes
# node1:router node for decision node
def router_node(state:AgentState)->AgentState:
    print("Entering router node")
    # we need instance of human meesage
    # we need latest and last wala human message

    query = next((m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")
#     messages list
#       ↓
# reverse
#       ↓
# scan from bottom
#       ↓
# first HumanMessage milte hi stop
#       ↓
# content return

    # if isinstance(m,HumanMessage):
    #     for m in reversed(state["messages"]):
    #         next(m.content)
    # else:
    #     ""
    #by default value is true for web search
    web_search_enabled=state.get("web_search_tool",True)
    print(f"Router node received web search info : {web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )
    
    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions:"
        "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
    )

    messages=[
        ("system",system_prompt),
        ("user",query)
    ]
    result : RouteDecision=router_llm.invoke(messages)
    initial_router_decision = result.route
    router_result_overriden_reason=None

    # over ride the router decison to go for web search if user has disbaled websearch
    if not web_search_enabled and result.route=="web":
        result.route=="rag"
        router_result_overriden_reason="Web search disabled by user; redirected to rag"
        print(f"Router decision overriden changed from 'web' to 'rag'.")
    
    print(f"Router final decison:{result.route},reply (if 'end'):{result.reply}")


    out= {
        "messages":state['messages'],
        "route":result.route,
        "web_search_enabled":web_search_enabled
    }

    # Add override info for tracing
    if router_result_overriden_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_result_overriden_reason"]=router_result_overriden_reason

    if result.route == "end":
        out["messages"] = state["messages"]+[AIMessage(content=result.reply or "Hello!")]

    print("Exititng the router node")
    return out

from vectorstore import get_retriever
# Tools
@tool
def rag_search_tool(query:str)->str:
    """Search for documents in the vector store using RAG. Retrieves the top 3 most relevant documents."""
    try:
        retriever_instance=get_retriever()
        embeddings=retriever_instance.invoke(query,k=3)
        
        return "\n\n".join(e.page_content for e in embeddings) if embeddings else ""
    
    except Exception as e:
        return f"RAG ERROR: {e}"

def rag_lookup_node(state:AgentState)->AgentState:
    print("Entering the rag node")

    #yeh actual user kee query hold krr rha h
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")
    
    web_search_enabled=state.get("web_search_tool",True)
    print(f"Router node received web search info : {web_search_enabled}")

    chunks = rag_search_tool.invoke(query)

    if chunks:
        print(f"Retrived RAG chunks: {chunks[:500]}")
    else:
        print("No rag chunks retrived from KB")


    if chunks.startswith("RAG_ERROR:"):
        print(f"RAG ERROR:{chunks}. Checking web search enabled status.")
        # if rag fails and web search is enabled, try web, otherwise go to answer
        if web_search_enabled:
            next_route="web"
        else :"answer"
        return {**state, "rag":"","route":next_route}
    
    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **relevant** to the user's question. "
            "Your job is to determine if we have pertinent document content that could help answer the query."
            "\n\nJudging Criteria:"
            "\n- SUFFICIENT: If relevant document content was retrieved (not empty). The answer node can process/summarize/extract from this content."
            "\n- NOT SUFFICIENT: If NO relevant information was retrieved OR the content is completely unrelated to the query."
            "\n\nImportant: Do NOT judge if the answer is 'perfect' or 'complete' - judge if we have RELEVANT CONTENT to work with."
            "\n- 'summarize the pdf' + retrieved marketing content = SUFFICIENT (content exists, answer node can summarize it)"
            "\n- 'What is X?' + retrieved definition of X = SUFFICIENT (relevant content exists)"
            "\n- 'How to Y?' + empty result = NOT SUFFICIENT (no content to work with)"
            "\n- 'Tell me about Z' + completely unrelated content = NOT SUFFICIENT (irrelevant)"
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this relevant content to work with?")
    ]
#     Tum jo use karti thi
# age = 20 # Yahan Python khud type guess karta hai.

# Type hint ke sath
# age: int = 20
# Yani age ek integer hona chahiye

    verdict: JudgeRag=judge_llm.invoke(judge_messages)
    print(f"RAG Judge Verdict: {verdict.sufficient}")
    print("Exiting rag node")

    if verdict.sufficient:
        next_route="answer"
    else:
        next_route="web" if web_search_enabled else "answer"
        print()

    return{**state,"rag":chunks, "route":next_route, "web_search_tool":web_search_enabled}



from langchain_tavily import TavilySearch   
os.environ["TAVILY_API_KEY"]=TAVILY_API_KEY
tavily=TavilySearch(max_results=3,topic="general")
@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"
    

# Node 3 : Web Search
def web_node(state:AgentState)->AgentState:
    print("Entering web node")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled=state.get("web_search_tool",True)

    if not web_search_enabled:
        print("Web search aint enabled by user.")
        return{**state,"web":"web search disbaled by user","route":"answer"}
    
    print(f"Web search query:{query}")
    snippets=web_search_tool.invoke(query)

    if snippets.startswith("WEB_ERROR:"):
        print(f"Web Error:{snippets}.Proceding to answer node with limited info")
        return{**state,"web":"","route":"answer"}
    
    print(f"Web snippets retrieved:{snippets[:200]}")
    print("Exiting web node")
    return {**state,"web":snippets,"route":"answer"}

#Node 4: Final answer
def answer_node(state:AgentState)->AgentState:
    print("Entering the answer node")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx_parts=[]
    if state.get("rag"):
        ctx_parts.append("Knowledge base information: \n"+state["rag"])
    if state.get("web") and not state["web"].startswith("web search was disbaled"):
        ctx_parts.append("Web search results: \n"+state["web"])

    context="\n\n".join(ctx_parts)
    if not context.strip():
        context="No external context was found for this query use general knowledge"

    prompt = f"""Please answer the user's question using the provided context.If the context is empty or irrelevant, 
    try to answer based on your general knowledge.
    Question: {query}
    Context:{context}
    Provide a helpful, accurate, and concise response based on the available information."""
    print("Prompt sent to answer llm: {prompt[:500]}")
    ans=answer_llm.invoke([HumanMessage(content=prompt)]).content
    print(f"Final answer: {ans[:200]}")
    print("Exiting answer node")
    return{**state,"messages":state["messages"]+[AIMessage(content=ans)]}

#Routing helpers

#Python ke typing module ka ek type hai.
#Literal ka matlab hota hai: Variable sirf specific fixed values hi le sakta hai.

def from_router(st:AgentState) -> Literal["rag","web","answer","end"]:
    return st["route"]
def after_rag(st:AgentState)->Literal["answer","web"]:
    return st["route"]
def after_web(_) -> Literal["answer"]:
    return "answer"

#Build Graph
def build_agent():
    graph=StateGraph(AgentState)
    graph.add_node("router",router_node)
    graph.add_node("rag_lookup",rag_lookup_node)
    graph.add_node("web_search",web_node)
    graph.add_node("answer",answer_node)
    graph.set_entry_point("router")

    graph.add_conditional_edges("router",from_router,{
        "rag":"rag_lookup",
        "web":"web_search",
        "answer":"answer",
        "end":END
    })

    graph.add_conditional_edges("rag_lookup",after_rag,{
        "web":"web_search",
        "answer":"answer",
    })

    graph.add_conditional_edges("web_search",after_web,{"answer":"answer"})
    graph.add_edge("answer",END)
    from langgraph.checkpoint.memory import MemorySaver
    agent=graph.compile(checkpointer=MemorySaver())
    return agent
rag_agent=build_agent()