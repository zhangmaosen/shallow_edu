#!/usr/bin/env python3
"""
测试FileHandlerAgent在团队环境中的工具调用
"""

import asyncio
import os
import sys
import tempfile

# 添加src目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from unittest.mock import AsyncMock, MagicMock

from teaching_team import FileHandlerAgent


async def test_file_handler_in_team():
    """测试FileHandlerAgent在团队中的工具调用"""
    print("开始测试FileHandlerAgent在团队环境中的工具调用...")
    
    # 创建 OllamaChatCompletionClient，使用 qwen3 30b 模型
    model_client = OllamaChatCompletionClient(
        model="qwen3:30b",
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
            'structured_output': False,
            'family': "qwen", 
        },
        options={
            'num_ctx': 6000,
        }
    )
    
    # 创建FileHandlerAgent
    file_handler_agent = FileHandlerAgent(model_client)
    
    # 创建一个简单的测试代理，用于发送文件操作请求
    test_agent = AssistantAgent(
        "test_agent",
        model_client=model_client,
        system_message="你是一个测试代理，专门用于测试FileHandlerAgent的功能。"
    )
    
    # 创建用户代理
    user_proxy = UserProxyAgent("user", input_func=lambda prompt: "APPROVE")
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    file_handler_agent._base_path = temp_dir
    
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 定义终止条件
        termination_condition = TextMentionTermination("APPROVE")
        
        # 创建团队
        team = SelectorGroupChat(
            [user_proxy, test_agent, file_handler_agent],
            model_client=model_client,
            termination_condition=termination_condition,
            max_turns=10
        )
        
        # 测试任务：让test_agent请求file_handler保存文件
        task = """请执行以下操作：
        1. 向FileHandlerAgent发送请求，让它保存一个文件
        2. 文件名：integration_test.txt
        3. 文件内容：这是集成测试的内容，用于验证FileHandlerAgent在团队中的工具调用功能。
        4. 请求格式：FileHandlerAgent，请调用save_content_to_file工具保存文件，文件名：integration_test.txt，内容：这是集成测试的内容，用于验证FileHandlerAgent在团队中的工具调用功能。

        """
        
        print("\n开始团队对话...")
        # 运行团队任务
        await team.reset()
        result = await team.run(task=task)
        
        print(f"团队运行结果: {result}")
        
        # 检查文件是否被正确创建
        file_path = os.path.join(temp_dir, "integration_test.txt")
        if os.path.exists(file_path):
            print("✓ FileHandlerAgent成功保存了文件")
            # 读取并验证文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                expected_content = "这是集成测试的内容，用于验证FileHandlerAgent在团队中的工具调用功能。"
                if content == expected_content:
                    print("✓ 文件内容正确")
                else:
                    print(f"✗ 文件内容不正确，期望: {expected_content}, 实际: {content}")
        else:
            print("✗ FileHandlerAgent未能保存文件")
        
        # 测试读取文件功能
        print("\n测试读取文件功能...")
        read_task = """请执行以下操作：
        1. 向FileHandlerAgent发送请求，让它读取刚才保存的文件
        2. 请求格式：FileHandlerAgent，请调用read_file_content工具读取文件，文件名：integration_test.txt
        3. 读取完成后，验证内容是否正确
        4. 完成后回复"APPROVE"表示任务完成
        """
        
        await team.reset()
        result = await team.run(task=read_task)
        print(f"读取文件任务结果: {result}")
        
        print("\n集成测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        print("\n清理临时文件...")
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            print("临时文件清理完成")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")


if __name__ == "__main__":
    asyncio.run(test_file_handler_in_team())