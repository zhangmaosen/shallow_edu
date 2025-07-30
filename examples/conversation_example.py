#!/usr/bin/env python3
"""
使用 autogen 和 Ollama gemma3:27b 模型进行对话的完整示例
"""

import asyncio
from autogen_core.models import UserMessage, SystemMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def chat_with_model():
    # 初始化 Ollama 客户端，使用 gemma3:27b 模型
    ollama_client = OllamaChatCompletionClient(
        model="gemma3:27b",
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
        },
        options={
            'num_ctx': 60000,
        }
    )
    
    try:
        # 系统提示词
        system_message = SystemMessage(content="你是一个有帮助的AI助手。请用简体中文回答问题。")
        
        # 用户消息
        user_message = UserMessage(content="请解释一下什么是大型语言模型？", source="user")
        
        print("发送消息到 gemma3:27b 模型...")
        response = await ollama_client.create([system_message, user_message])
        
        print("\n模型响应:")
        print(response.content)
        print(f"\n使用情况: {response.usage}")
        print(f"完成原因: {response.finish_reason}")
        
    except Exception as e:
        print(f"与模型通信时发生错误: {e}")
    
    finally:
        # 关闭客户端连接
        await ollama_client.close()


async def multi_turn_conversation():
    # 多轮对话示例
    ollama_client = OllamaChatCompletionClient(
        model="gemma3:27b",
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
        },
        options={
            'num_ctx': 60000,
        }
    )
    
    conversation_history = [
        SystemMessage(content="你是一个有帮助的AI助手。请用简体中文回答问题。"),
    ]
    
    try:
        # 第一轮对话
        user_question1 = UserMessage(content="什么是机器学习？", source="user")
        conversation_history.append(user_question1)
        
        print("第一轮对话:")
        print(f"用户: {user_question1.content}")
        
        response1 = await ollama_client.create(conversation_history)
        conversation_history.append(response1)
        print(f"助手: {response1.content}\n")
        
        # 第二轮对话
        user_question2 = UserMessage(content="它与深度学习有什么区别？", source="user")
        conversation_history.append(user_question2)
        
        print("第二轮对话:")
        print(f"用户: {user_question2.content}")
        
        response2 = await ollama_client.create(conversation_history)
        print(f"助手: {response2.content}")
        print(f"\n使用情况: {response2.usage}")
        
    except Exception as e:
        print(f"多轮对话中发生错误: {e}")
    
    finally:
        await ollama_client.close()


# 添加基于notebook的agent团队功能
async def agent_team_example():
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.ui import Console
        
        # 创建 Ollama 客户端
        ollama_client = OllamaChatCompletionClient(
            model="gemma3:27b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
            },
            options={
                'num_ctx': 60000,
            }
        )

        # 创建主要助手agent
        primary_agent = AssistantAgent(
            "primary",
            model_client=ollama_client,
            system_message="你是一个有帮助的AI助手。请用简体中文回答问题。",
        )

        # 创建批评者agent
        critic_agent = AssistantAgent(
            "critic",
            model_client=ollama_client,
            system_message="提供有建设性的反馈。当你的反馈得到解决时，请回复'APPROVE'。",
        )

        # 定义终止条件
        text_termination = TextMentionTermination("APPROVE")

        # 创建团队
        team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)
        
        # 运行团队任务
        print("=== Agent 团队示例 ===")
        await team.reset()  # 为新任务重置团队
        print("开始执行任务: 写一首五言绝句，关于夏天的热")
        
        # 注意：这里为了演示目的，我们不会实际运行流式输出
        # 如果需要完整功能，取消下面一行的注释
        # await Console(team.run_stream(task="写一首五言绝句，关于夏天的热"))
        
        print("任务完成")
        
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        print("请确保安装了 autogen-agentchat 包")
    except Exception as e:
        print(f"Agent团队示例中发生错误: {e}")


async def main():
    print("=== 单轮对话示例 ===")
    await chat_with_model()
    
    print("\n" + "="*50)
    print("=== 多轮对话示例 ===")
    await multi_turn_conversation()
    
    print("\n" + "="*50)
    await agent_team_example()


if __name__ == "__main__":
    asyncio.run(main())