import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

async def main():
    # 创建 Ollama qwen3:7b 模型客户端，开启流式输出
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
            'num_ctx': 4096,
            'stream': True,
        }
    )

    # 创建用户代理和AI助手
    user_proxy = UserProxyAgent("user")
    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="你是一个友好的AI助手，请用中文简要介绍一下大模型推理的原理。"
    )

    # 组建团队
    team = MagenticOneGroupChat([user_proxy, assistant], model_client)

    # 定义任务
    task = "请简要介绍大模型推理的原理。"

    await team.reset()
    print("\n开始流式对话测试...\n" + "="*40)
    stream = team.run_stream(task=task)
    # 打字机流式输出
    async for chunk in stream:
        if hasattr(chunk, 'content'):
            print(chunk.content, end='', flush=True)
        else:
            print(str(chunk), end='', flush=True)
    print()  # 换行
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
