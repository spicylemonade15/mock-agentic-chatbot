import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, TypedDict
import json
import time

# Custom state with extra "subtasks" field
class MyState(MessagesState):
    subtasks: List[str]


# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

# Mock agents
def booking_agent(task): return f"📌 Booking venue: {task}"
def communication_agent(task): return f"✉️ Sending emails: {task}"
def design_agent(task): return f"🎨 Designing: {task}"
def logistics_agent(task): return f"🚚 Logistics: {task}"
def outreach_agent(task): return f"📣 Outreach: {task}"

def generate_subtask_list(state: MyState):
    query = state["messages"][-1].content
    
    subtask_prompt = f"""
    You are a subtask generator.

    Available agents: booking, communication, design, logistics, outreach.

    Return ONLY a valid JSON array of objects in this exact format:
    [
      {{ "agent": "booking", "task": "Reserve the venue" }},
      {{ "agent": "design", "task": "Create posters" }}
    ]

    Do not include explanations, markdown fences, or extra text.

    User query: "{query}"
    """

    response = llm.invoke(subtask_prompt)
    raw = response.content.strip()

    # --- Sanitize common Gemini outputs ---
    if raw.startswith("```json"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    elif raw.startswith("```"):
        raw = raw.strip("`").strip()

    try:
        subtasks = json.loads(raw)
    except Exception as e:
        print("JSON parse failed:", e, "\nRaw output:", raw)
        subtasks = []

    return {"subtasks": subtasks}


# Subtask router consumes subtasks
from langchain_core.messages import AIMessage

def subtask_router(state: MyState):
    results = []
    for subtask in state["subtasks"]:
        agent = subtask.get("agent", "generic").lower()
        task = subtask.get("task", "")

        if agent == "booking":
            agent_output = booking_agent(task)
        elif agent == "communication":
            agent_output = communication_agent(task)
        elif agent == "design":
            agent_output = design_agent(task)
        elif agent == "logistics":
            agent_output = logistics_agent(task)
        elif agent == "outreach":
            agent_output = outreach_agent(task)
        else:
            agent_output = f"⚡ Generic agent handled: {task}"

        results.append(AIMessage(content=agent_output))

    return {"messages": results}


# Build graph
graph = StateGraph(MyState)   # ✅ use custom state
graph.add_node("subtask_generator", generate_subtask_list)
graph.add_node("subtask_router", subtask_router)

graph.add_edge(START, "subtask_generator")
graph.add_edge("subtask_generator", "subtask_router")
graph.add_edge("subtask_router", END)

app = graph.compile()

# Streamlit UI
st.set_page_config(page_title="LangGraph + Gemini", page_icon="🤖")
st.title("🤖 LangGraph Gemini Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_memory.append({"role": "user", "content": prompt})

    # Run through LangGraph
    response_state = app.invoke({"messages": st.session_state.chat_memory})

    for msg in response_state["messages"]:
        st.chat_message("assistant").markdown(msg.content)
        st.session_state.messages.append({"role": "assistant", "content": msg.content})
        st.session_state.chat_memory.append({"role": "assistant", "content": msg.content})

        # add a 1 second delay for each message 
        time.sleep(1)

