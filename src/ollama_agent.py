#!/usr/bin/env python3
"""
Ollama Agent 示例 - 使用本地 gemma3:27b 模型
"""

import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage


async def main():
    # 初始化 Ollama 客户端，使用 gemma3:27b 模型
    # 与 notebook/test.ipynb 中的配置保持一致
    ollama_client = OllamaChatCompletionClient(
        model="gemma3:27b",
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
            'structured_output': False,  # 添加缺失的structured_output字段
        },
        options={
            'num_ctx': 60000,
        }
    )
    
    try:
        # 发送示例请求
        response = await ollama_client.create([
            UserMessage(content="中国首都是?", source="user")
        ])
        
        print("模型响应:")
        print(response.content)
        
    except Exception as e:
        print(f"发生错误: {e}")
    
    finally:
        # 关闭客户端连接
        await ollama_client.close()


if __name__ == "__main__":
    asyncio.run(main())