from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
import asyncio
from tqdm.asyncio import tqdm

async def llm_call(agent, prompt, thread_id, sem):
    async with sem:
        response = await agent.ainvoke(
            {"messages": [prompt]},
            config={"configurable": {"thread_id": str(thread_id)}}
        )
    return response

async def main():
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat"
    )

    conn = aiosqlite.connect("async.db", check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn)

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

    sem = asyncio.Semaphore(2)
    tasks = [llm_call(agent, prompt, id, sem) for id, prompt in enumerate(propmts, start=1)]
    responses = await tqdm.gather(*tasks, total=len(tasks), desc="LLM Calls...")
    await conn.close()
    for response in responses:
        print("\n\n")
        for message in response["messages"]:
            message.pretty_print()

asyncio.run(main())