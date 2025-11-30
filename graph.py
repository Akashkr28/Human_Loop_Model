import json, time, sys
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage

load_dotenv()
DB_URI = "mongodb://admin:admin@mongodb:27017"


# --- 1. Logic & Tools ---
@tool
def human_help(query: str) -> str:
    """Signals human assistance is needed."""
    return json.dumps({"need_human": True, "query": query})


tools = [human_help]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def chatbot(state):
    return {"messages": [llm.invoke([{
        "role": "system",
        "content": "If user needs help, call human_help tool."
    }] + state["messages"])]}


def human_node(state):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and json.loads(last.content).get("need_human"):
        query = json.loads(last.content)["query"]
        print(f"\n[SYSTEM] Paused for Admin. Query: {query}")

        # Pause here. Resume with admin input.
        solution = interrupt({"query": query})

        return {"messages": [HumanMessage(f"SYSTEM: Admin resolved this: {solution}")]}
    return {"messages": []}


# --- 2. Graph Definition ---
def get_app(checkpointer):
    workflow = StateGraph(TypedDict("State", {"messages": Annotated[list, add_messages]}))
    workflow.add_node("bot", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("human", human_node)

    workflow.add_edge(START, "bot")
    workflow.add_conditional_edges("bot", tools_condition, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "human")
    workflow.add_edge("human", "bot")
    return workflow.compile(checkpointer=checkpointer)


# --- 3. Execution ---
def run_chat(thread_id="t1"):
    with MongoDBSaver.from_conn_string(DB_URI) as cp:
        app = get_app(cp)
        config = {"configurable": {"thread_id": thread_id}}

        print(f"--- Chat {thread_id} ---")

        # Initialize message count
        last_msg_count = len(app.get_state(config).values.get("messages", []))

        while True:
            # 1. Fetch latest state
            state = app.get_state(config)
            msgs = state.values.get("messages", [])

            # 2. Print ANY new messages (AI or System/Human)
            if len(msgs) > last_msg_count:
                for msg in msgs[last_msg_count:]:
                    if isinstance(msg, HumanMessage) and msg.name != "user":
                        # This captures the "System" message from the Admin
                        print(f"\nSYSTEM: {msg.content}")
                    elif hasattr(msg, "content") and msg.content and msg.type == "ai":
                        print(f"\nAI: {msg.content}")
                last_msg_count = len(msgs)

            # 3. Check for Interrupts (Pause)
            if state.tasks and state.tasks[0].interrupts:
                print("Waiting for admin...", end="\r")
                time.sleep(2)
                continue  # Restart loop to keep checking for updates

            # 4. User Input
            try:
                txt = input("\nUser: ")
                if txt.lower() in ["q", "exit"]:
                    break

                # Send input and run graph
                app.invoke({"messages": [("user", txt)]}, config)
            except KeyboardInterrupt:
                break


def run_admin(thread_id="t1"):
    with MongoDBSaver.from_conn_string(DB_URI) as cp:
        app = get_app(cp)
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)

        if state.tasks and state.tasks[0].interrupts:
            query = state.tasks[0].interrupts[0].value["query"]
            print(f"User asks: {query}")
            ans = input("Admin Answer: ")

            # Resume graph
            for event in app.stream(Command(resume=ans), config, stream_mode="values"):
                msg = event["messages"][-1]
                if msg.type == "ai" and msg.content:
                    print(f"AI (Resumed): {msg.content}")
        else:
            print("No pending tasks.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if mode == "admin":
        run_admin()
    else:
        run_chat()
