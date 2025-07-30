import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
from autogen_agentchat.ui import Console
# 配置列表
model_client = OllamaChatCompletionClient(
            model="qwen3:30b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'structured_output': False,
                'family': "qwen",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "6000")),
                'stream': True,  # 开启流式输出
            }
        )

# 创建助理代理
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="你是一个有帮助的 AI 助理，擅长讲故事。"
)

# 创建用户代理
user_proxy = UserProxyAgent(
    name="user_proxy"
)

async def assistant_run_stream() -> None:
    # Option 1: read each message from the stream (as shown in the previous example).
    # async for message in agent.run_stream(task="Find information on AutoGen"):
    #     print(message)

    # Option 2: use Console to print all messages as they appear.
    await Console(
        assistant.run_stream(task="Find information on AutoGen"),
        output_stats=True,  # Enable stats printing.
    )


# Use asyncio.run(assistant_run_stream()) when running in a script.
await assistant_run_stream()
