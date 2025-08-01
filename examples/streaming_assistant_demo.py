import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os


async def main():
    # 创建 Ollama 模型客户端，开启流式输出
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

    # 创建流式输出助手
    streaming_assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
        model_client_stream=True,  # Enable streaming tokens.
    )

    print("开始流式输出 streaming_assistant 的结果...")
    print("=" * 50)

    # 方法1: 使用 Console 显示流式输出（推荐）
    print("方法1: 使用 Console 显示流式输出")
    await Console(
        streaming_assistant.run_stream(task="简要介绍人工智能的发展历史"),
        output_stats=True,
    )

    print("\n" + "=" * 50)
    
    # 方法2: 手动处理流式输出
    print("方法2: 手动处理流式输出")
    stream = streaming_assistant.run_stream(task="简要介绍人工智能的发展历史")
    async for chunk in stream:
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end='', flush=True)
    print("\n")

    # 关闭客户端连接
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())