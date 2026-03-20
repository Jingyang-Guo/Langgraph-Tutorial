from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from tqdm import tqdm

def llm_call(agent, prompt, thread_id):
    response = agent.invoke(
        {"messages": [prompt]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return response

def main():
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat"
    )

    conn = sqlite3.connect("sync.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent = create_agent(
        model=model,
        system_prompt=SystemMessage("You are a helpful assistant."),
        checkpointer=checkpointer
    )

    propmts = [
        HumanMessage("What is the weather in beijing?"),
        HumanMessage("Please introduce yourself briefly."),
        HumanMessage("What is the weather in shanghai?"),
        HumanMessage("Hello"),
        HumanMessage("How to calculate 6 + 8 = ?"),
        # HumanMessage("Please explain 'python' briefly.")
    ]

    response_list = []
    for id, prompt in tqdm(enumerate(propmts, start=1), total=len(propmts), desc="LLM Calls"):
        response = llm_call(agent, prompt, str(id))
        response_list.append(response)
    
    conn.close()
    for response in response_list:
        print("\n\n")
        for message in response["messages"]:
            message.pretty_print()

main()